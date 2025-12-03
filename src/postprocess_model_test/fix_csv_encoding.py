#!/usr/bin/env python3
import argparse
import concurrent.futures
import io
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List


@dataclass
class EncodingResult:
    file_path: Path
    original_encoding: Optional[str]
    decoding_used: Optional[str]
    target_encoding: str
    status: str  # converted | already_utf8 | skipped | error | removed_bom
    error: Optional[str]
    size_bytes: int


def normalize_encoding_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    enc = label.strip().lower()
    # Normalize common variants
    mapping = {
        "utf_8": "utf-8",
        "utf8": "utf-8",
        "utf-8-sig": "utf-8-sig",
        "utf_8_sig": "utf-8-sig",
        "euc-kr": "cp949",  # cp949 is a superset
        "ks_c_5601-1987": "cp949",
        "x-windows-949": "cp949",
        "windows-949": "cp949",
        "cp-949": "cp949",
    }
    return mapping.get(enc, enc)


def detect_with_charset_normalizer(path: Path, max_bytes_scan: int) -> Optional[str]:
    try:
        from charset_normalizer import from_path  # type: ignore
    except Exception:
        return None
    try:
        matches = from_path(str(path))
        best = matches.best()
        if best and getattr(best, "encoding", None):
            return normalize_encoding_label(best.encoding)
    except Exception:
        return None
    return None


def detect_with_chardet(path: Path, max_bytes_scan: int) -> Optional[str]:
    try:
        import chardet  # type: ignore
    except Exception:
        return None
    try:
        with open(path, "rb") as f:
            raw = f.read(max_bytes_scan)
        guess = chardet.detect(raw) or {}
        enc = guess.get("encoding")
        return normalize_encoding_label(enc)
    except Exception:
        return None


def detect_file_encoding(path: Path, max_bytes_scan: int = 2_000_000) -> Optional[str]:
    enc = detect_with_charset_normalizer(path, max_bytes_scan)
    if enc:
        return enc
    enc = detect_with_chardet(path, max_bytes_scan)
    if enc:
        return enc
    return None


def probe_utf8(path: Path) -> Tuple[bool, bool]:
    has_bom = False
    try:
        with open(path, "rb") as f:
            start = f.read(3)
            has_bom = start.startswith(b"\xef\xbb\xbf")
        with open(path, "rb") as f:
            data = f.read()
        data.decode("utf-8")
        return True, has_bom
    except UnicodeDecodeError:
        return False, has_bom
    except Exception:
        return False, has_bom


def iter_csv_files(root: Path, only_substring: Optional[str]) -> List[Path]:
    files = []
    for p in root.rglob("*.csv"):
        if only_substring and only_substring not in str(p):
            continue
        files.append(p)
    return files


def convert_csv_file(
    file_path: Path,
    target_encoding: str,
    write_bom: bool,
    backup: bool,
    dry_run: bool,
    max_bytes_scan: int,
) -> EncodingResult:
    size_bytes = 0
    try:
        size_bytes = file_path.stat().st_size
    except Exception:
        pass

    # Quick check for UTF-8
    is_utf8, has_bom = probe_utf8(file_path)
    if is_utf8:
        if (target_encoding.lower() == "utf-8" and not write_bom) and not has_bom:
            return EncodingResult(
                file_path=file_path,
                original_encoding="utf-8" + ("-sig" if has_bom else ""),
                decoding_used="utf-8",
                target_encoding=target_encoding,
                status="already_utf8",
                error=None,
                size_bytes=size_bytes,
            )
        # Remove BOM if target is plain UTF-8
        if (target_encoding.lower() == "utf-8" and not write_bom) and has_bom:
            if dry_run:
                return EncodingResult(
                    file_path=file_path,
                    original_encoding="utf-8-sig",
                    decoding_used="utf-8-sig",
                    target_encoding="utf-8",
                    status="removed_bom",
                    error=None,
                    size_bytes=size_bytes,
                )
            try:
                with open(file_path, "rb") as src:
                    raw = src.read()
                # Strip BOM bytes
                if raw.startswith(b"\xef\xbb\xbf"):
                    raw = raw[3:]
                with open(file_path, "wb") as dst:
                    dst.write(raw)
                return EncodingResult(
                    file_path=file_path,
                    original_encoding="utf-8-sig",
                    decoding_used="utf-8-sig",
                    target_encoding="utf-8",
                    status="removed_bom",
                    error=None,
                    size_bytes=size_bytes,
                )
            except Exception as e:
                return EncodingResult(
                    file_path=file_path,
                    original_encoding="utf-8-sig",
                    decoding_used="utf-8-sig",
                    target_encoding="utf-8",
                    status="error",
                    error=str(e),
                    size_bytes=size_bytes,
                )

    # Detect non-UTF8 encodings
    detected = detect_file_encoding(file_path, max_bytes_scan=max_bytes_scan)
    decoding_candidates: List[str] = []
    if detected:
        decoding_candidates.append(detected)
    # Common Korean fallbacks
    for cand in ("cp949", "euc-kr", "utf-8"):
        norm = normalize_encoding_label(cand)
        if norm and norm not in decoding_candidates:
            decoding_candidates.append(norm)

    if dry_run:
        return EncodingResult(
            file_path=file_path,
            original_encoding=detected,
            decoding_used=decoding_candidates[0] if decoding_candidates else None,
            target_encoding=target_encoding if not write_bom else "utf-8-sig",
            status="converted",
            error=None,
            size_bytes=size_bytes,
        )

    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    decoding_used: Optional[str] = None
    last_error: Optional[str] = None

    for enc in decoding_candidates:
        try:
            # Streamed decode -> re-encode
            with open(file_path, "rb") as src, open(
                temp_path,
                "w",
                encoding=("utf-8-sig" if write_bom else target_encoding),
                newline="",
            ) as dst:
                wrapper = io.TextIOWrapper(src, encoding=enc, errors="strict", newline="")
                for line in wrapper:
                    dst.write(line)
            decoding_used = enc
            break
        except UnicodeDecodeError as e:
            last_error = f"UnicodeDecodeError using {enc}: {e}"
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
            continue
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass
            continue

    if decoding_used is None:
        return EncodingResult(
            file_path=file_path,
            original_encoding=detected,
            decoding_used=None,
            target_encoding=target_encoding if not write_bom else "utf-8-sig",
            status="error",
            error=last_error or "Failed to decode with all candidates",
            size_bytes=size_bytes,
        )

    try:
        if backup:
            backup_path = file_path.with_suffix(file_path.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(file_path, backup_path)
        os.replace(temp_path, file_path)
    except Exception as e:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        return EncodingResult(
            file_path=file_path,
            original_encoding=detected,
            decoding_used=decoding_used,
            target_encoding=target_encoding if not write_bom else "utf-8-sig",
            status="error",
            error=str(e),
            size_bytes=size_bytes,
        )

    return EncodingResult(
        file_path=file_path,
        original_encoding=detected,
        decoding_used=decoding_used,
        target_encoding=target_encoding if not write_bom else "utf-8-sig",
        status="converted",
        error=None,
        size_bytes=size_bytes,
    )


def write_report(results: List[EncodingResult], report_path: Path) -> None:
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8", newline="") as f:
            f.write(
                "file_path,original_encoding,decoding_used,target_encoding,status,error,size_bytes\n"
            )
            for r in results:
                row = [
                    str(r.file_path),
                    r.original_encoding or "",
                    r.decoding_used or "",
                    r.target_encoding,
                    r.status,
                    (r.error or "").replace("\n", " ").replace(",", " "),
                    str(r.size_bytes),
                ]
                f.write(",".join(row) + "\n")
    except Exception:
        pass


def main() -> None:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Detect and convert CSV encodings to UTF-8")
    parser.add_argument(
        "--root",
        type=str,
        default=str(default_root),
        help="루트 디렉터리 (기본값: 데이터셋 루트)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="파일 경로에 포함되어야 하는 부분 문자열 (옵션)",
    )
    parser.add_argument(
        "--target-encoding",
        type=str,
        default="utf-8",
        help="목표 인코딩 (기본값: utf-8)",
    )
    parser.add_argument(
        "--write-bom",
        action="store_true",
        help="UTF-8로 저장 시 BOM을 포함하여 저장",
    )
    parser.add_argument("--backup", action="store_true", help="원본을 .bak으로 백업")
    parser.add_argument("--dry-run", action="store_true", help="변경 없이 시뮬레이션")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4))),
        help="동시 작업 수 (기본: 최대 8)",
    )
    parser.add_argument(
        "--max-bytes-scan",
        type=int,
        default=2_000_000,
        help="인코딩 추정 시 읽을 최대 바이트 수",
    )

    args = parser.parse_args()
    root = Path(args.root).resolve()
    only = args.only if args.only else None

    csv_files = iter_csv_files(root, only)
    if not csv_files:
        print("CSV 파일이 없습니다.")
        return

    print(f"대상 파일 수: {len(csv_files)}")
    results: List[EncodingResult] = []

    def _work(p: Path) -> EncodingResult:
        return convert_csv_file(
            file_path=p,
            target_encoding=args.target_encoding,
            write_bom=bool(args.write_bom),
            backup=bool(args.backup),
            dry_run=bool(args.dry_run),
            max_bytes_scan=int(args.max_bytes_scan),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_work, p) for p in csv_files]
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            results.append(r)
            print(
                f"[{r.status}] {r.file_path} | orig={r.original_encoding} -> used={r.decoding_used} -> target={r.target_encoding}"
            )

    report_path = Path(__file__).parent / "encoding_report.csv"
    write_report(results, report_path)

    total = len(results)
    converted = sum(1 for r in results if r.status == "converted")
    removed_bom = sum(1 for r in results if r.status == "removed_bom")
    already = sum(1 for r in results if r.status == "already_utf8")
    errors = [r for r in results if r.status == "error"]

    print("\n요약:")
    print(f"- 총 파일: {total}")
    print(f"- 변환됨: {converted}")
    print(f"- BOM 제거: {removed_bom}")
    print(f"- 이미 UTF-8: {already}")
    print(f"- 오류: {len(errors)}")
    if errors:
        print("  일부 파일에서 오류가 발생했습니다. 자세한 내용은 보고서를 확인하세요.")
    print(f"보고서: {report_path}")


if __name__ == "__main__":
    main()


