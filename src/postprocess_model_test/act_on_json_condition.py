from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import pandas as pd


def read_json_records(input_path: Path, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    """
    JSON을 레코드 리스트(list[dict])로 로드합니다.
    - NDJSON(lines=True) → 일반 JSON → json_normalize 순으로 시도
    """
    try:
        text = input_path.read_text(encoding=encoding)
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"파일 인코딩 오류: {input_path} 를 '{encoding}' 로 읽을 수 없습니다. --encoding 옵션을 확인하세요."
        ) from exc

    # NDJSON
    try:
        df = pd.read_json(io.StringIO(text), lines=True)
        return df.to_dict(orient="records")
    except ValueError:
        pass

    # 일반 JSON (배열/객체)
    try:
        df = pd.read_json(io.StringIO(text))
        return df.to_dict(orient="records")
    except ValueError:
        pass

    # 최후 폴백
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            # list[dict] 또는 list[Any]
            norm = pd.json_normalize(obj)
            return norm.to_dict(orient="records")
        else:
            # 단일 객체 → 1개 레코드로 간주
            norm = pd.json_normalize(obj)
            return norm.to_dict(orient="records")
    except Exception as exc:
        raise RuntimeError(
            f"JSON 파싱 실패: {input_path}. 파일 형식을 확인하세요."
        ) from exc


def write_json_records(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_scalar(value_str: str) -> Any:
    """문자열을 숫자/불리언/null 등으로 자동 변환 시도 후, 실패하면 원문 문자열 반환."""
    try:
        return json.loads(value_str)
    except Exception:
        return value_str


def get_nested(mapping: MutableMapping[str, Any], dotted_key: str, sep: str = ".") -> Any:
    cur: Any = mapping
    for part in dotted_key.split(sep):
        if isinstance(cur, MutableMapping) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def set_nested(mapping: MutableMapping[str, Any], dotted_key: str, value: Any, sep: str = ".") -> None:
    parts = dotted_key.split(sep)
    cur: MutableMapping[str, Any] = mapping
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cur[part] = nxt
        cur = nxt  # type: ignore[assignment]
    cur[parts[-1]] = value


def delete_nested(mapping: MutableMapping[str, Any], dotted_key: str, sep: str = ".") -> None:
    parts = dotted_key.split(sep)
    cur: MutableMapping[str, Any] = mapping
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, MutableMapping):
            return
        cur = nxt  # type: ignore[assignment]
    cur.pop(parts[-1], None)


def compute_default_output_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}_{suffix}{input_path.suffix}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "JSON 파일에서 특정 key의 value가 특정 값일 때 지정한 행동을 수행합니다.\n"
            "지원 동작: filter-keep, filter-drop, set, delete-key"
        )
    )
    p.add_argument("input", help="입력 JSON 파일 경로")
    p.add_argument("--encoding", default="utf-8", help="입력 인코딩 (기본: utf-8)")
    p.add_argument("--key", required=True, help="조건에 사용할 key (중첩은 dot 표기, 예: 장소.위도)")
    p.add_argument("--equals", required=True, help="조건 값 (자동으로 숫자/불리언/null 감지)")
    p.add_argument(
        "--action",
        required=True,
        choices=["filter-keep", "filter-drop", "set", "delete-key"],
        help="수행할 동작",
    )
    p.add_argument("--target-key", help="set/delete-key에서 대상 key (dot 표기 가능)")
    p.add_argument("--target-value", help="set에서 설정할 값 (자동 형 변환)")
    p.add_argument("--output", "-o", help="출력 JSON 경로")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"입력 파일을 찾을 수 없습니다: {input_path}")
        return 1

    try:
        records = read_json_records(input_path, encoding=args.encoding)
    except Exception as exc:
        print(str(exc))
        return 1

    cond_value = parse_scalar(args.equals)

    if args.action in {"filter-keep", "filter-drop"}:
        matched: List[Dict[str, Any]] = []
        not_matched: List[Dict[str, Any]] = []
        for rec in records:
            val = get_nested(rec, args.key)
            if val == cond_value:
                matched.append(rec)
            else:
                not_matched.append(rec)

        out_records = matched if args.action == "filter-keep" else not_matched
        suffix = "filtered_keep" if args.action == "filter-keep" else "filtered_drop"
        output_path = Path(args.output) if args.output else compute_default_output_path(input_path, suffix)
        try:
            write_json_records(out_records, output_path)
        except Exception as exc:
            print(f"출력 저장 실패: {output_path}\n{exc}")
            return 1
        print(f"저장 완료: {output_path} (총 {len(out_records)}건)")
        return 0

    if args.action == "set":
        if not args.target_key:
            print("set 동작에는 --target-key 가 필요합니다.")
            return 2
        if args.target_value is None:
            print("set 동작에는 --target-value 가 필요합니다.")
            return 2
        target_value = parse_scalar(args.target_value)
        updated = 0
        for rec in records:
            val = get_nested(rec, args.key)
            if val == cond_value:
                set_nested(rec, args.target_key, target_value)
                updated += 1
        output_path = Path(args.output) if args.output else compute_default_output_path(input_path, "mutated")
        try:
            write_json_records(records, output_path)
        except Exception as exc:
            print(f"출력 저장 실패: {output_path}\n{exc}")
            return 1
        print(f"저장 완료: {output_path} (수정 {updated}건)")
        return 0

    if args.action == "delete-key":
        if not args.target_key:
            print("delete-key 동작에는 --target-key 가 필요합니다.")
            return 2
        updated = 0
        for rec in records:
            val = get_nested(rec, args.key)
            if val == cond_value:
                delete_nested(rec, args.target_key)
                updated += 1
        output_path = Path(args.output) if args.output else compute_default_output_path(input_path, "mutated")
        try:
            write_json_records(records, output_path)
        except Exception as exc:
            print(f"출력 저장 실패: {output_path}\n{exc}")
            return 1
        print(f"저장 완료: {output_path} (삭제 {updated}건)")
        return 0

    print("지원하지 않는 동작입니다.")
    return 2


if __name__ == "__main__":
    sys.exit(main())



