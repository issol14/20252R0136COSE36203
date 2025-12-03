import argparse
import os
from typing import List, Optional

import pandas as pd


DEFAULT_ID_COLUMNS: List[str] = ["연번", "날짜", "호선", "역번호", "역명", "구분"]


def pick_working_encoding(csv_path: str) -> str:
    encodings_to_try: List[str] = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error: Optional[Exception] = None
    for enc in encodings_to_try:
        try:
            pd.read_csv(csv_path, nrows=0, encoding=enc)
            return enc
        except Exception as e:  # noqa: BLE001
            last_error = e
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to detect encoding for the input CSV file.")


def get_value_columns(all_columns: List[str], id_columns: List[str]) -> List[str]:
    id_set = set(id_columns)
    return [c for c in all_columns if c not in id_set]


def ensure_output_dir(output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


def compute_default_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}_long{ext or '.csv'}"


def melt_csv(
    input_path: str,
    output_path: Optional[str],
    id_columns: List[str],
    chunksize: int,
) -> str:
    encoding = pick_working_encoding(input_path)
    header_df = pd.read_csv(input_path, nrows=0, encoding=encoding)
    all_columns = header_df.columns.tolist()

    # Validate id columns are present
    missing = [c for c in id_columns if c not in all_columns]
    if missing:
        raise ValueError(f"입력 파일에 존재하지 않는 식별 컬럼: {missing}")

    value_columns = get_value_columns(all_columns, id_columns)
    if not value_columns:
        raise ValueError("변환할 시간대 컬럼을 찾을 수 없습니다. id 컬럼만 존재합니다.")

    out_path = output_path or compute_default_output_path(input_path)
    ensure_output_dir(out_path)

    # Remove existing output to avoid header duplication in re-runs
    if os.path.exists(out_path):
        os.remove(out_path)

    reader = pd.read_csv(
        input_path,
        encoding=encoding,
        chunksize=chunksize,
        low_memory=False,
    )

    is_first_chunk = True
    for chunk in reader:
        melted = chunk.melt(
            id_vars=id_columns,
            value_vars=value_columns,
            var_name="시간대",
            value_name="인원수",
        )

        # Optionally coerce to numeric without forcing fill; keeps NaN if any
        try:
            melted["인원수"] = pd.to_numeric(melted["인원수"], errors="coerce")
        except Exception:  # noqa: BLE001
            pass

        # Reorder columns for clarity
        melted = melted[id_columns + ["시간대", "인원수"]]

        melted.to_csv(
            out_path,
            mode="w" if is_first_chunk else "a",
            index=False,
            header=is_first_chunk,
            encoding="utf-8-sig",
        )
        is_first_chunk = False

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "시간대별 열이 가로로 펼쳐진 CSV를 '시간대'/'인원수'의 세로(long) 포맷으로 변환합니다."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="입력 CSV 절대경로",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="출력 CSV 절대경로 (미지정 시 *_long.csv 로 생성)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50000,
        help="청크 단위 행 수 (기본 50,000)",
    )
    parser.add_argument(
        "--id-cols",
        nargs="*",
        default=DEFAULT_ID_COLUMNS,
        help=(
            "식별 컬럼 목록. 기본값: '연번 날짜 호선 역번호 역명 구분'"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = melt_csv(
        input_path=args.input,
        output_path=args.output,
        id_columns=list(args.id_cols),
        chunksize=args.chunksize,
    )
    print(output)


if __name__ == "__main__":
    main()


