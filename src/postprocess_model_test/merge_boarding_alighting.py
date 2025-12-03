from __future__ import annotations

import argparse
import sys
from typing import List

import pandas as pd


def detect_encoding(input_path: str, preferred: str | None = None) -> str:
    """Return a usable encoding for the CSV, trying common Korean encodings."""
    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["utf-8", "cp949", "euc-kr"])  # try in order
    tried: List[str] = []
    for enc in candidates:
        try:
            pd.read_csv(input_path, nrows=0, encoding=enc)
            return enc
        except UnicodeDecodeError:
            tried.append(enc)
        except Exception:
            # Ignore other transient parser errors here; will be handled on full read
            return enc
    # Fallback
    return preferred or "utf-8"


def read_columns(input_path: str, encoding: str, sep: str) -> List[str]:
    df_head = pd.read_csv(input_path, nrows=0, encoding=encoding, sep=sep)
    return list(df_head.columns)


def merge_boarding_alighting(
    input_path: str,
    output_path: str,
    keys: List[str] | None = None,
    encoding: str | None = None,
    sep: str = ",",
    chunksize: int | None = 300_000,
) -> None:
    # Detect encoding if not given
    enc = detect_encoding(input_path, preferred=encoding)

    # Determine columns and default keys
    all_columns = read_columns(input_path, encoding=enc, sep=sep)
    missing = {"구분", "인원수"} - set(all_columns)
    if missing:
        raise ValueError(f"입력 파일에 필요한 컬럼이 없습니다: {sorted(missing)}. 실제 컬럼: {all_columns}")

    if keys is None or len(keys) == 0:
        # 기본: 식별에 불필요한 '연번'과 피벗 관련 컬럼을 제외한 모든 컬럼을 키로 사용
        keys = [c for c in all_columns if c not in ("연번", "구분", "인원수")]
        if not keys:
            raise ValueError("키 컬럼을 결정할 수 없습니다. --keys 옵션으로 키 컬럼을 지정해 주세요.")

    usecols = list(dict.fromkeys([*keys, "구분", "인원수"]))

    frames: List[pd.DataFrame] = []
    iter_kwargs = dict(
        filepath_or_buffer=input_path,
        encoding=enc,
        sep=sep,
        usecols=usecols,
        chunksize=chunksize,
        dtype={c: "string" for c in usecols if c != "인원수"},
    )

    # If chunksize is None, pandas returns a DataFrame; normalize to iterator
    reader = pd.read_csv(**iter_kwargs)
    if not hasattr(reader, "__iter__") or isinstance(reader, pd.DataFrame):
        reader = [reader]

    for idx, chunk in enumerate(reader, start=1):
        # Ensure numeric
        chunk["인원수"] = pd.to_numeric(chunk["인원수"], errors="coerce").fillna(0)

        # Aggregate in case of duplicates within chunk
        grouped = (
            chunk.groupby(keys + ["구분"], dropna=False)["인원수"].sum()
        )

        # Pivot '구분' -> columns ['승차','하차']
        pivot = grouped.unstack("구분", fill_value=0)

        # Guarantee both columns exist
        for col in ("승차", "하차"):
            if col not in pivot.columns:
                pivot[col] = 0

        pivot = pivot[["승차", "하차"]].reset_index()
        frames.append(pivot)

        # Lightweight progress
        print(f"[정보] 청크 {idx} 처리 완료: {len(pivot):,}행", file=sys.stderr)

    if not frames:
        # No data
        pd.DataFrame(columns=[*keys, "승차", "하차"]).to_csv(
            output_path, index=False, encoding="utf-8"
        )
        print(f"[완료] 결과가 비어 있습니다. {output_path} 저장", file=sys.stderr)
        return

    result = pd.concat(frames, ignore_index=True)
    # Merge across chunks if same keys appear in multiple chunks
    result = (
        result.groupby(keys, dropna=False)[["승차", "하차"]].sum().reset_index()
    )

    # Optional: sort by keys for readability
    result.sort_values(by=keys, inplace=True, kind="stable")

    # Save
    result.to_csv(output_path, index=False, encoding="utf-8")
    print(
        f"[완료] 저장됨: {output_path}  (행 {len(result):,})  인코딩={enc}",
        file=sys.stderr,
    )


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "'구분' 열의 승차/하차를 한 행으로 합치고 '승차','하차' 컬럼으로 피벗합니다."
        )
    )
    p.add_argument("--input", required=True, help="입력 CSV 경로")
    p.add_argument("--output", required=True, help="출력 CSV 경로")
    p.add_argument(
        "--keys",
        nargs="*",
        help=(
            "그룹핑 키 컬럼들. 기본값: 전체 컬럼에서 '연번','구분','인원수'를 뺀 나머지"
        ),
    )
    p.add_argument(
        "--encoding",
        default=None,
        help="입력 인코딩(자동 감지 기본). 예: utf-8, cp949",
    )
    p.add_argument("--sep", default=",", help="구분자 (기본 ',')")
    p.add_argument(
        "--chunksize",
        type=int,
        default=300_000,
        help="청크 크기 (메모리 여유 시 더 크게 조정 가능, 전체 로드하려면 0)",
    )
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    chunksize = None if (args.chunksize is not None and args.chunksize <= 0) else args.chunksize
    merge_boarding_alighting(
        input_path=args.input,
        output_path=args.output,
        keys=args.keys,
        encoding=args.encoding,
        sep=args.sep,
        chunksize=chunksize,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


