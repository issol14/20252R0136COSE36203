"""
날씨 데이터 전처리 모듈

기상청 데이터를 학습에 사용 가능한 형태로 변환합니다.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def preprocess_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    날씨 데이터를 전처리합니다.
    
    전처리 내용:
    1. 컬럼명 영문 변환 (특수문자 제거)
    2. 일시 → Date, Hour 분리
    3. 결측치 처리
       - 강수량, 적설: 0으로 대체 (없음 = 0mm/cm)
       - 기온, 습도, 풍속: 선형 보간
    4. 불쾌지수(Discomfort Index) 계산
    
    Parameters
    ----------
    weather_df : pd.DataFrame
        원본 날씨 데이터
    
    Returns
    -------
    pd.DataFrame
        전처리된 날씨 데이터
    """
    df = weather_df.copy()
    
    # 1. 컬럼명 정리
    column_mapping = {
        "지점": "station_id",
        "지점명": "station_name", 
        "일시": "datetime",
        "기온(°C)": "temperature",
        "강수량(mm)": "rainfall",
        "풍속(m/s)": "wind_speed",
        "습도(%)": "humidity",
        "적설(cm)": "snowfall",
        "지면상태(지면상태코드)": "ground_state"
    }
    df = df.rename(columns=column_mapping)
    
    # 2. 일시 파싱 → Date, Hour
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["Date"] = df["datetime"].dt.date
    df["Date"] = pd.to_datetime(df["Date"])
    df["Hour"] = df["datetime"].dt.hour
    
    # 3. 결측치 처리
    # 3.1 강수량, 적설: 결측 = 없음 = 0
    df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce").fillna(0)
    df["snowfall"] = pd.to_numeric(df["snowfall"], errors="coerce").fillna(0)
    
    # 3.2 기온, 습도, 풍속: 선형 보간
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
    df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")
    
    # 시간순 정렬 후 보간
    df = df.sort_values("datetime")
    df["temperature"] = df["temperature"].interpolate(method="linear")
    df["humidity"] = df["humidity"].interpolate(method="linear")
    df["wind_speed"] = df["wind_speed"].interpolate(method="linear")
    
    # 보간 후에도 남은 결측치(양 끝단)는 forward/backward fill
    df["temperature"] = df["temperature"].ffill().bfill()
    df["humidity"] = df["humidity"].ffill().bfill()
    df["wind_speed"] = df["wind_speed"].ffill().bfill()
    
    # 4. 불쾌지수 계산
    # DI = (9/5)*Ta - 0.55*(1-RH)*((9/5)*Ta - 26) + 32
    # Ta: 기온(°C), RH: 상대습도(0~1)
    df["discomfort_index"] = calculate_discomfort_index(
        df["temperature"], df["humidity"]
    )
    
    # 5. 필요한 컬럼만 선택
    result_columns = [
        "Date", "Hour", "temperature", "rainfall", 
        "wind_speed", "humidity", "snowfall", "discomfort_index"
    ]
    result = df[result_columns].copy()
    
    # 결과 요약 출력
    print(f"[날씨 전처리] 완료")
    print(f"  - 결과 행 수: {len(result):,}")
    print(f"  - 날짜 범위: {result['Date'].min()} ~ {result['Date'].max()}")
    print(f"  - 기온 범위: {result['temperature'].min():.1f}°C ~ {result['temperature'].max():.1f}°C")
    print(f"  - 강수일 수: {(result['rainfall'] > 0).sum():,}")
    print(f"  - 적설일 수: {(result['snowfall'] > 0).sum():,}")
    
    return result


def calculate_discomfort_index(
    temperature: pd.Series, 
    humidity: pd.Series
) -> pd.Series:
    """
    불쾌지수(Discomfort Index)를 계산합니다.
    
    공식: DI = (9/5)*Ta - 0.55*(1-RH)*((9/5)*Ta - 26) + 32
    - Ta: 기온(°C)
    - RH: 상대습도 (0~1)
    
    Parameters
    ----------
    temperature : pd.Series
        기온 (°C)
    humidity : pd.Series
        상대습도 (%)
    
    Returns
    -------
    pd.Series
        불쾌지수
    """
    ta = temperature
    rh = humidity / 100  # % → 비율
    
    term1 = (9 / 5) * ta
    term2 = 0.55 * (1 - rh) * ((9 / 5) * ta - 26)
    di = term1 - term2 + 32
    
    return di


if __name__ == "__main__":
    from pathlib import Path
    from load_data import load_weather_data
    
    data_dir = Path(__file__).parent.parent.parent / "data"
    
    print("=" * 60)
    weather_raw = load_weather_data(data_dir)
    print()
    
    print("=" * 60)
    weather_processed = preprocess_weather(weather_raw)
    print()
    print(weather_processed.head(10))







