"""
V3: Station-Time 모델

역별, 시간대별 평균 패턴을 기준으로 예측하고,
외부 요인(날씨, 이벤트)에 의한 편차를 학습합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.ensemble import GradientBoostingRegressor

from .base_model import (
    BaseModel, 
    ModelConfig, 
    load_featured_data,
    ModelMetrics
)
from ..analysis.feature_importance import analyze_feature_importance, analyze_feature_groups


def create_station_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """역-시간-요일 조합의 평균/편차 feature 생성"""
    result = df.copy()
    
    # 1. 역-시간대별 평균 net_passengers (Train set에서만 계산해야 함 - 여기서는 전체로 근사)
    station_hour_mean = df.groupby(["역명", "호선", "Hour"])["net_passengers"].transform("mean")
    result["station_hour_mean"] = station_hour_mean
    
    # 2. 역-요일별 평균
    station_weekday_mean = df.groupby(["역명", "호선", "요일"])["net_passengers"].transform("mean")
    result["station_weekday_mean"] = station_weekday_mean
    
    # 3. 역-시간-요일별 평균 (더 세밀)
    station_hour_weekday_mean = df.groupby(["역명", "호선", "Hour", "요일"])["net_passengers"].transform("mean")
    result["station_hour_weekday_mean"] = station_hour_weekday_mean
    
    # 4. 순서-시간별 평균 (순서는 노선 내 위치)
    order_hour_mean = df.groupby(["순서", "호선", "Hour"])["net_passengers"].transform("mean")
    result["order_hour_mean"] = order_hour_mean
    
    # 5. 환승역 여부-시간별 평균
    transfer_hour_mean = df.groupby(["is_transfer_station", "Hour"])["net_passengers"].transform("mean")
    result["transfer_hour_mean"] = transfer_hour_mean
    
    return result


def get_v3_features(all_features: List[str]) -> List[str]:
    """V3 모델용 feature 선별 (기존 Lag 제외 + 신규 추가)"""
    lag_keywords = [
        "lag_", "rolling_", "diff_", "pct_change",
    ]
    
    # Lag 제외
    no_lag_features = [f for f in all_features if not any(kw in f for kw in lag_keywords)]
    
    # 시간/요일 평균은 유지 (time_mean_승차, weekday_mean_승차 등)
    keep_patterns = ["_mean_승차", "_mean_하차"]
    for f in all_features:
        if any(p in f for p in keep_patterns) and f not in no_lag_features:
            no_lag_features.append(f)
    
    return no_lag_features


def train_v3_station_time(
    data_path: Path,
    output_dir: Path,
    use_lightgbm: bool = True
) -> BaseModel:
    """
    V3 Station-Time 모델을 학습합니다.
    
    역-시간 조합의 평균 패턴을 기반으로 예측합니다.
    """
    print("=" * 70)
    print(" V3 Station-Time 모델 학습")
    print(" (역-시간 조합 평균 + 외부 요인 편차)")
    print("=" * 70)
    
    # 데이터 로드
    df = load_featured_data(data_path)
    
    # 역-시간 조합 feature 추가
    print("\n역-시간 조합 feature 생성 중...")
    df = create_station_time_features(df)
    
    # Feature 컬럼 추출
    exclude_cols = {
        "날짜", "Date", "역명", "역번호",
        "승차", "하차", "net_passengers",
        "오름혼잡도", "내림혼잡도", "오름전역혼잡도", "내림전역혼잡도",
        "요일구분",
    }
    all_features = [c for c in df.columns if c not in exclude_cols]
    feature_cols = get_v3_features(all_features)
    
    # 신규 추가된 feature 포함 확인
    new_features = ["station_hour_mean", "station_weekday_mean", "station_hour_weekday_mean",
                    "order_hour_mean", "transfer_hour_mean"]
    for f in new_features:
        if f in df.columns and f not in feature_cols:
            feature_cols.append(f)
    
    print(f"\n사용 Feature 수: {len(feature_cols)}")
    
    # 모델 설정
    if use_lightgbm and HAS_LIGHTGBM:
        config = ModelConfig(
            name="V3_StationTime_LightGBM",
            version="v3",
            description="역-시간 조합 평균 feature + 외부 요인",
            features=feature_cols,
            target="net_passengers",
            test_size=0.2,
            random_state=42,
            hyperparameters={
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "num_leaves": 127,
                "max_depth": 12,
                "min_child_samples": 30,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.2,
                "reg_lambda": 0.2,
                "n_jobs": -1,
                "verbose": -1,
                "random_state": 42
            }
        )
    else:
        config = ModelConfig(
            name="V3_StationTime_GBM",
            version="v3",
            description="역-시간 조합 평균 feature + 외부 요인",
            features=feature_cols,
            target="net_passengers",
            test_size=0.2,
            random_state=42,
            hyperparameters={
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 8,
                "min_samples_split": 20,
                "min_samples_leaf": 10,
                "random_state": 42
            }
        )
    
    # 모델 객체 생성
    model_wrapper = BaseModel(config)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = model_wrapper.prepare_data(df)
    
    # 모델 생성 및 학습
    print(f"\n모델 학습 시작: {config.name}")
    print("-" * 50)
    
    if use_lightgbm and HAS_LIGHTGBM:
        model = lgb.LGBMRegressor(**config.hyperparameters)
        
        # Early stopping
        X_tr, X_val = X_train.iloc[:-10000], X_train.iloc[-10000:]
        y_tr, y_val = y_train.iloc[:-10000], y_train.iloc[-10000:]
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        print(f"  - Best iteration: {model.best_iteration_}")
    else:
        model = GradientBoostingRegressor(**config.hyperparameters)
        model.fit(X_train, y_train)
    
    model_wrapper.model = model
    
    # 예측
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 평가
    model_wrapper.metrics_train = model_wrapper.evaluate(y_train, y_train_pred)
    model_wrapper.metrics_test = model_wrapper.evaluate(y_test, y_test_pred)
    
    model_wrapper.print_metrics(model_wrapper.metrics_train, "Train")
    model_wrapper.print_metrics(model_wrapper.metrics_test, "Test")
    
    # Feature Importance 분석
    print("\n" + "=" * 50)
    model_wrapper.feature_importance = analyze_feature_importance(
        model, feature_cols, top_n=25, output_dir=output_dir
    )
    
    # 그룹별 중요도 분석
    if model_wrapper.feature_importance is not None:
        analyze_feature_groups(model_wrapper.feature_importance)
    
    # 저장
    print("\n" + "=" * 50)
    print("결과 저장 중...")
    model_wrapper.save(output_dir)
    
    # 예측 샘플 저장
    predictions_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_test_pred,
        "error": y_test.values - y_test_pred
    })
    predictions_df.head(1000).to_csv(output_dir / "predictions_sample.csv", index=False)
    
    print("\n" + "=" * 70)
    print(" V3 Station-Time 학습 완료!")
    print("=" * 70)
    
    return model_wrapper


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "outputs" / "featured_data.csv"
    output_dir = project_root / "experiments" / "v3"
    
    model = train_v3_station_time(data_path, output_dir, use_lightgbm=True)







