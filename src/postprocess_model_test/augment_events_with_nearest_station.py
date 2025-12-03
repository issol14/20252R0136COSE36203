from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


def read_stations_csv(
    csv_path: Path,
    encoding: str = "utf-8",
    name_col_candidates: Sequence[str] = ("역명", "station", "name"),
    line_col_candidates: Sequence[str] = ("호선", "line"),
    lat_col_candidates: Sequence[str] = ("위도", "lat", "latitude", "y", "Y"),
    lon_col_candidates: Sequence[str] = ("경도", "lng", "lon", "longitude", "x", "X"),
) -> pd.DataFrame:
    """
    역 좌표 CSV를 읽어 공통 스키마로 정리합니다.
    반환 컬럼: [station_name, line_name, latitude, longitude]
    """
    df = pd.read_csv(csv_path, encoding=encoding)

    def pick(colnames: Sequence[str]) -> Optional[str]:
        for c in colnames:
            if c in df.columns:
                return c
        return None

    name_col = pick(name_col_candidates)
    line_col = pick(line_col_candidates)
    lat_col = pick(lat_col_candidates)
    lon_col = pick(lon_col_candidates)

    missing = [label for label, c in [("역명", name_col), ("위도", lat_col), ("경도", lon_col)] if c is None]
    if missing:
        raise RuntimeError(
            f"역 CSV 필수 컬럼을 찾을 수 없습니다: {', '.join(missing)}\n"
            f"발견된 컬럼: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "station_name": df[name_col].astype(str),
            "line_name": df[line_col].astype(str) if line_col else pd.Series([None] * len(df)),
            "latitude": pd.to_numeric(df[lat_col], errors="coerce"),
            "longitude": pd.to_numeric(df[lon_col], errors="coerce"),
        }
    )

    out = out.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    return out


def try_read_json_dataframe(input_path: Path, encoding: str = "utf-8") -> pd.DataFrame:
    """
    JSON 파일을 DataFrame으로 로드합니다.
    - NDJSON(lines=True) 시도 → 일반 JSON → json.loads + json_normalize 폴백
    """
    try:
        text = input_path.read_text(encoding=encoding)
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"파일 인코딩 오류: {input_path} 를 '{encoding}' 로 읽을 수 없습니다. --encoding 옵션을 확인하세요."
        ) from exc

    # NDJSON 우선
    try:
        return pd.read_json(io.StringIO(text), lines=True)
    except ValueError:
        pass

    # 일반 JSON
    try:
        return pd.read_json(io.StringIO(text))
    except ValueError:
        pass

    # 최후 폴백: json_normalize
    try:
        obj = json.loads(text)
        return pd.json_normalize(obj)
    except Exception as exc:
        raise RuntimeError(
            f"JSON 파싱 실패: {input_path}. 파일 형식을 확인하세요."
        ) from exc


def detect_lat_lon_columns(
    df: pd.DataFrame,
    lat_field: Optional[str] = None,
    lon_field: Optional[str] = None,
) -> Tuple[str, str]:
    """
    위도/경도 컬럼명을 자동 탐지합니다. 명시되면 해당 값 사용.
    """
    if lat_field and lon_field:
        if lat_field not in df.columns or lon_field not in df.columns:
            raise RuntimeError(
                f"지정한 위도/경도 컬럼을 찾을 수 없습니다: {lat_field}, {lon_field}.\n"
                f"가용 컬럼: {list(df.columns)}"
            )
        return lat_field, lon_field

    lat_candidates = [
        "위도",
        "lat",
        "latitude",
        "Y",
        "y",
        "좌표Y",
        "좌표y",
    ]
    lon_candidates = [
        "경도",
        "lng",
        "lon",
        "longitude",
        "X",
        "x",
        "좌표X",
        "좌표x",
    ]

    lat_found = next((c for c in lat_candidates if c in df.columns), None)
    lon_found = next((c for c in lon_candidates if c in df.columns), None)

    if not lat_found or not lon_found:
        raise RuntimeError(
            "위도/경도 컬럼을 자동으로 찾을 수 없습니다. --lat-field 와 --lon-field로 지정하세요.\n"
            f"가용 컬럼: {list(df.columns)}"
        )

    return lat_found, lon_found


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Haversine 공식을 이용해 두 점 간 거리(km)를 계산합니다.
    입력은 모두 라디안이어야 합니다.
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def compute_nearest_station(
    events_lat_deg: np.ndarray,
    events_lon_deg: np.ndarray,
    stations_lat_deg: np.ndarray,
    stations_lon_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    각 이벤트 지점에 대해 최근접 역의 (index, 거리_km) 를 반환합니다.
    벡터화 계산으로 성능을 확보합니다.
    """
    # 라디안 변환 (방향 벡터화)
    ev_lat_rad = np.radians(events_lat_deg)
    ev_lon_rad = np.radians(events_lon_deg)
    st_lat_rad = np.radians(stations_lat_deg)
    st_lon_rad = np.radians(stations_lon_deg)

    # 결과 버퍼
    nearest_idx = np.empty(ev_lat_rad.shape[0], dtype=np.int64)
    nearest_dist = np.empty(ev_lat_rad.shape[0], dtype=np.float64)

    # 역 좌표는 고정 → 2D 배열 브로드캐스팅으로 한번에 계산하면 메모리 부담이 커질 수 있으므로
    # 이벤트 단위로 역 전체를 벡터화하여 계산
    for i in range(ev_lat_rad.shape[0]):
        lat1 = np.full(st_lat_rad.shape, ev_lat_rad[i])
        lon1 = np.full(st_lon_rad.shape, ev_lon_rad[i])
        dists = haversine_km(lat1, lon1, st_lat_rad, st_lon_rad)
        j = int(np.argmin(dists))
        nearest_idx[i] = j
        nearest_dist[i] = float(dists[j])

    return nearest_idx, nearest_dist


def save_json(df: pd.DataFrame, output_path: Path) -> None:
    df.to_json(output_path, orient="records", force_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "문화행사 JSON에 최근접 지하철역과 거리(km)를 계산해 병합 저장합니다."
        )
    )
    p.add_argument("events_json", help="입력 문화행사 JSON 경로")
    p.add_argument(
        "--stations-csv",
        required=True,
        help="역 위경도 CSV 경로 (예: 서울교통공사_1-8호선 지하철역 위경도 좌표정보.csv)",
    )
    p.add_argument("--output", "-o", help="출력 JSON 경로 (기본: 입력명_with_nearest_station.json)")
    p.add_argument("--encoding", default="utf-8", help="이벤트 JSON 인코딩 (기본: utf-8)")
    p.add_argument(
        "--station-encoding",
        default="utf-8",
        help="역 CSV 인코딩 (기본: utf-8)",
    )
    p.add_argument("--lat-field", help="이벤트 위도 컬럼명(자동 감지 실패 시 지정)")
    p.add_argument("--lon-field", help="이벤트 경도 컬럼명(자동 감지 실패 시 지정)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    events_path = Path(args.events_json)
    if not events_path.exists():
        print(f"입력 JSON을 찾을 수 없습니다: {events_path}")
        return 1

    stations_path = Path(args.stations_csv)
    if not stations_path.exists():
        print(f"역 CSV를 찾을 수 없습니다: {stations_path}")
        return 1

    try:
        stations_df = read_stations_csv(stations_path, encoding=args.station_encoding)
    except Exception as exc:
        print(f"역 CSV 로드 실패: {exc}")
        return 1

    try:
        events_df = try_read_json_dataframe(events_path, encoding=args.encoding)
    except Exception as exc:
        print(str(exc))
        return 1

    try:
        lat_col, lon_col = detect_lat_lon_columns(
            events_df, lat_field=args.lat_field, lon_field=args.lon_field
        )
    except Exception as exc:
        print(str(exc))
        return 1

    # 유효 좌표만 계산 대상으로 필터링
    events_df["__lat__"] = pd.to_numeric(events_df[lat_col], errors="coerce")
    events_df["__lon__"] = pd.to_numeric(events_df[lon_col], errors="coerce")

    valid_mask = events_df["__lat__"].notna() & events_df["__lon__"].notna()

    valid_events = events_df.loc[valid_mask, ["__lat__", "__lon__"]].to_numpy()

    # 역 좌표 배열
    stations_lat = stations_df["latitude"].to_numpy(dtype=float)
    stations_lon = stations_df["longitude"].to_numpy(dtype=float)

    if valid_events.shape[0] > 0 and stations_lat.size > 0:
        idxes, dists = compute_nearest_station(
            valid_events[:, 0], valid_events[:, 1], stations_lat, stations_lon
        )

        # 결과 매핑
        nearest_station_name = np.array(stations_df["station_name"], dtype=object)
        nearest_line_name = np.array(stations_df["line_name"], dtype=object)

        events_df.loc[valid_mask, "nearest_station_name"] = nearest_station_name[idxes]
        events_df.loc[valid_mask, "nearest_station_line"] = nearest_line_name[idxes]
        events_df.loc[valid_mask, "nearest_station_lat"] = stations_lat[idxes]
        events_df.loc[valid_mask, "nearest_station_lon"] = stations_lon[idxes]
        events_df.loc[valid_mask, "nearest_station_distance_km"] = dists
    else:
        print("경고: 유효한 이벤트 좌표 또는 역 좌표가 없습니다. 결과 컬럼은 생성되지만 모두 결측일 수 있습니다.")

    # 내부 작업 컬럼 제거
    events_df = events_df.drop(columns=["__lat__", "__lon__"], errors="ignore")

    output_path = (
        Path(args.output)
        if args.output
        else events_path.with_name(f"{events_path.stem}_with_nearest_station.json")
    )

    try:
        save_json(events_df, output_path)
    except Exception as exc:
        print(f"출력 저장 실패: {output_path}\n{exc}")
        return 1

    print(f"저장 완료: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())



