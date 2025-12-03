from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Set

import pandas as pd


def compute_default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_dropped{input_path.suffix}")


def parse_selection_to_columns(selection: str, available_columns: Sequence[str]) -> List[str]:
    """
    쉼표로 구분된 입력을 컬럼명 리스트로 변환합니다.
    - 숫자는 1-based 인덱스를 의미 (예: 1,2,5)
    - 그 외는 컬럼명으로 간주 (정확 일치)
    """
    column_set: Set[str] = set()
    tokens = [tok.strip() for tok in selection.split(",") if tok.strip()]
    name_to_index = {name: idx for idx, name in enumerate(available_columns)}

    for tok in tokens:
        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < len(available_columns):
                column_set.add(available_columns[idx])
            else:
                print(f"경고: 인덱스 {tok} 는 컬럼 범위를 벗어났습니다. 무시합니다.")
        else:
            if tok in name_to_index:
                column_set.add(tok)
            else:
                print(f"경고: 컬럼명 '{tok}' 을(를) 찾을 수 없습니다. 무시합니다.")

    return [c for c in available_columns if c in column_set]


def prompt_columns_to_drop(columns: Sequence[str]) -> List[str]:
    print("\n다음은 입력 CSV에서 감지된 컬럼 목록입니다 (1-based 인덱스):")
    for i, name in enumerate(columns, start=1):
        print(f"  {i:>3}. {name}")

    print(
        "\n제거할 컬럼의 번호 또는 이름을 쉼표로 구분해 입력하세요.\n"
        "예시) 1,2,5  또는  컬럼A,컬럼B\n"
        "아무것도 입력하지 않으면 취소됩니다."
    )

    try:
        user_input = input("제거할 컬럼 선택: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 취소되었습니다.")
        return []

    if not user_input:
        return []

    return parse_selection_to_columns(user_input, columns)


def drop_columns(df: pd.DataFrame, columns_to_drop: Iterable[str]) -> pd.DataFrame:
    columns = list(columns_to_drop)
    if not columns:
        return df
    return df.drop(columns=[c for c in columns if c in df.columns], errors="ignore")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CSV 파일에서 불필요한 컬럼을 선택해 제거하고 저장합니다. 기본은 대화형 선택이며, --drop 으로 비대화형 제거도 가능합니다."
        )
    )
    parser.add_argument("input", help="입력 CSV 파일 경로")
    parser.add_argument(
        "--output",
        "-o",
        help="출력 CSV 파일 경로 (미지정 시 입력 파일명에 _dropped 접미)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="입출력 파일 인코딩 (기본: utf-8)",
    )
    parser.add_argument(
        "--sep",
        default=",",
        help="구분자 (기본: ','). 자동 감지는 'auto' 사용",
    )
    parser.add_argument(
        "--drop",
        "-d",
        nargs="+",
        help=(
            "제거할 컬럼명 또는 1-based 인덱스(공백/쉼표로 구분). 예: --drop 1 2 5  또는  --drop 컬럼A 컬럼B"
        ),
    )
    parser.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        help="강제로 대화형 모드 사용",
    )
    parser.add_argument(
        "--no-interactive",
        dest="interactive",
        action="store_false",
        help="강제로 비대화형 모드 사용 (반드시 --drop 지정)",
    )
    parser.set_defaults(interactive=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"입력 파일을 찾을 수 없습니다: {input_path}")
        return 1

    # 구분자 처리
    sep_arg = args.sep
    read_kwargs = {"encoding": args.encoding}
    if isinstance(sep_arg, str) and sep_arg.lower() == "auto":
        read_kwargs.update({"sep": None, "engine": "python"})  # 판다스 자동 추론
    else:
        read_kwargs.update({"sep": sep_arg})

    try:
        df = pd.read_csv(input_path, **read_kwargs)
    except Exception as exc:
        print(f"CSV 로드 실패: {exc}")
        return 1

    # 대화형 여부 결정
    interactive = (args.interactive if args.interactive is not None else not bool(args.drop))

    columns_to_drop: List[str] = []
    if interactive:
        columns_to_drop = prompt_columns_to_drop(list(df.columns))
    else:
        if not args.drop:
            print("비대화형 모드에서는 --drop 로 제거할 컬럼을 지정해야 합니다.")
            return 2
        user_spec = ",".join(args.drop)
        columns_to_drop = parse_selection_to_columns(user_spec, list(df.columns))

    if not columns_to_drop:
        print("제거할 컬럼이 선택되지 않았습니다. 아무 작업도 수행하지 않습니다.")
        return 0

    print("\n제거될 컬럼:")
    for name in columns_to_drop:
        print(f"  - {name}")

    df_dropped = drop_columns(df, columns_to_drop)

    output_path = Path(args.output) if args.output else compute_default_output_path(input_path)
    try:
        df_dropped.to_csv(output_path, index=False, encoding=args.encoding)
    except Exception as exc:
        print(f"출력 파일 저장 실패: {output_path}\n{exc}")
        return 1

    print(f"\n저장 완료: {output_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())




