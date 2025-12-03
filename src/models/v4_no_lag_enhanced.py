"""
V4: No-Lag Enhanced 모델

Lag/Rolling feature를 완전히 제외하고,
역사적 통계 feature (Target Encoding 방식)로 대체하여 높은 성능을 달성합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import KFold

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
)
from ..analysis.feature_importance import analyze_feature_importance, analyze_feature_groups


def create_target_encoding_features(
    df: pd.DataFrame, 
    target_col: str = "net_passengers",
    n_folds: int = 5
) -> pd.DataFrame:
    """
    Target Encoding 방식으로 역-시간-요일 통계 feature를 생성합니다.
    
    K-Fold 방식으로 data leakage를 방지합니다.
    각 fold에서 자신을 제외한 데이터로 평균을 계산합니다.
    
    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터프레임
    target_col : str
        타겟 컬럼명
    n_folds : int
        K-Fold 수
    
    Returns
    -------
    pd.DataFrame
        Target encoding feature가 추가된 데이터프레임
    """
    result = df.copy()
    
    # 인코딩할 그룹 조합 정의
    encoding_groups = [
        # (그룹 컬럼들, feature 이름, 통계 함수)
        (["역명", "호선", "Hour"], "station_hour_mean_te", "mean"),
        (["역명", "호선", "요일"], "station_weekday_mean_te", "mean"),
        (["역명", "호선", "Hour", "요일"], "station_hour_weekday_mean_te", "mean"),
        (["역명", "호선", "Month"], "station_month_mean_te", "mean"),
        (["역명", "호선", "Hour", "Is_Weekend"], "station_hour_weekend_mean_te", "mean"),
        (["순서", "호선", "Hour"], "order_hour_mean_te", "mean"),
        (["순서", "호선", "Hour", "요일"], "order_hour_weekday_mean_te", "mean"),
        (["is_transfer_station", "Hour"], "transfer_hour_mean_te", "mean"),
        (["is_transfer_station", "Hour", "요일"], "transfer_hour_weekday_mean_te", "mean"),
        # 표준편차 feature (변동성)
        (["역명", "호선", "Hour"], "station_hour_std_te", "std"),
        (["역명", "호선", "Hour", "요일"], "station_hour_weekday_std_te", "std"),
        # 중앙값 feature (이상치에 강건)
        (["역명", "호선", "Hour", "요일"], "station_hour_weekday_median_te", "median"),
    ]
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for group_cols, feature_name, agg_func in encoding_groups:
        # 모든 그룹 컬럼이 존재하는지 확인
        if not all(col in result.columns for col in group_cols):
            print(f"  [Skip] {feature_name}: 필요한 컬럼 없음")
            continue
            
        result[feature_name] = np.nan
        
        # K-Fold Target Encoding
        for train_idx, val_idx in kf.split(result):
            train_data = result.iloc[train_idx]
            
            # train에서 그룹별 통계 계산
            if agg_func == "mean":
                group_stats = train_data.groupby(group_cols)[target_col].mean()
            elif agg_func == "std":
                group_stats = train_data.groupby(group_cols)[target_col].std()
            elif agg_func == "median":
                group_stats = train_data.groupby(group_cols)[target_col].median()
            
            # val에 적용
            for idx in val_idx:
                key = tuple(result.iloc[idx][col] for col in group_cols)
                if key in group_stats.index:
                    result.loc[result.index[idx], feature_name] = group_stats[key]
        
        # 결측치를 전체 평균으로 대체
        if agg_func == "mean":
            global_stat = result[target_col].mean()
        elif agg_func == "std":
            global_stat = result[target_col].std()
        elif agg_func == "median":
            global_stat = result[target_col].median()
            
        result[feature_name] = result[feature_name].fillna(global_stat)
        
        print(f"  ✓ {feature_name} 생성 완료")
    
    return result


def create_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    역사적 패턴 기반 feature를 생성합니다.
    (lag/rolling 없이 시간 패턴을 잡기 위함)
    """
    result = df.copy()
    
    # 1. 시간대 그룹별 전체 평균 (역에 무관하게)
    hour_global_mean = df.groupby("Hour")["net_passengers"].transform("mean")
    result["hour_global_mean"] = hour_global_mean
    
    # 2. 요일별 전체 평균
    weekday_global_mean = df.groupby("요일")["net_passengers"].transform("mean")
    result["weekday_global_mean"] = weekday_global_mean
    
    # 3. 호선별 평균
    line_mean = df.groupby("호선")["net_passengers"].transform("mean")
    result["line_mean"] = line_mean
    
    # 4. 호선-시간 평균
    line_hour_mean = df.groupby(["호선", "Hour"])["net_passengers"].transform("mean")
    result["line_hour_mean"] = line_hour_mean
    
    # 5. 순서 구간별 평균 (노선 시작/중간/끝)
    result["order_segment"] = pd.cut(
        result["순서"], 
        bins=[0, 15, 30, 45, 100], 
        labels=[0, 1, 2, 3]
    ).astype(float)
    
    # 6. 시간대-요일 교차 전체 평균
    hour_weekday_global_mean = df.groupby(["Hour", "요일"])["net_passengers"].transform("mean")
    result["hour_weekday_global_mean"] = hour_weekday_global_mean
    
    # 7. 월-요일 교차 평균
    month_weekday_mean = df.groupby(["Month", "요일"])["net_passengers"].transform("mean")
    result["month_weekday_mean"] = month_weekday_mean
    
    # 8. 공휴일-시간 평균
    if "Is_Holiday" in result.columns:
        holiday_hour_mean = df.groupby(["Is_Holiday", "Hour"])["net_passengers"].transform("mean")
        result["holiday_hour_mean"] = holiday_hour_mean
    
    # 9. 러시아워 특성
    if "Is_RushHour" in result.columns:
        rush_line_mean = df.groupby(["Is_RushHour", "호선"])["net_passengers"].transform("mean")
        result["rush_line_mean"] = rush_line_mean
    
    return result


def create_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    비율 기반 feature 생성 (현재값/평균 대비 등)
    """
    result = df.copy()
    
    # 평균 대비 비율 계산용 (나중에 추론 시에도 사용 가능하도록)
    # 이 feature들은 target을 직접 사용하지 않음
    
    # 1. 환승역 여부와 시간대의 interaction 강화
    if "is_transfer_station" in result.columns and "Hour" in result.columns:
        result["transfer_hour_interaction"] = result["is_transfer_station"] * result["Hour"]
    
    # 2. 순서와 시간의 polynomial interaction
    if "순서" in result.columns and "Hour" in result.columns:
        result["order_hour_interaction"] = result["순서"] * result["Hour"]
        result["order_squared"] = result["순서"] ** 2
    
    # 3. 시간대 세부 분류
    if "Hour" in result.columns:
        result["hour_sin_squared"] = np.sin(2 * np.pi * result["Hour"] / 24) ** 2
        result["hour_cos_squared"] = np.cos(2 * np.pi * result["Hour"] / 24) ** 2
    
    return result


def get_v4_features(all_features: List[str]) -> List[str]:
    """V4 모델용 feature 선별 (lag/rolling 완전 제외)"""
    # 완전히 제외할 키워드
    exclude_keywords = [
        "lag_", "rolling_", "diff_", "pct_change",
        # 기존 time_mean, weekday_mean 등도 제외 (새로운 TE feature로 대체)
        # "time_mean_승차", "weekday_mean_승차", "month_mean_승차"
    ]
    
    # 필터링
    selected_features = [
        f for f in all_features 
        if not any(kw in f for kw in exclude_keywords)
    ]
    
    return selected_features


def train_v4_no_lag_enhanced(
    data_path: Path,
    output_dir: Path,
    use_lightgbm: bool = True,
    use_target_encoding: bool = True
) -> BaseModel:
    """
    V4 No-Lag Enhanced 모델을 학습합니다.
    
    Lag/Rolling을 완전히 제외하고, Target Encoding + 역사적 통계로 대체합니다.
    
    Parameters
    ----------
    data_path : Path
        featured_data.csv 경로
    output_dir : Path
        결과 저장 디렉토리
    use_lightgbm : bool
        LightGBM 사용 여부
    use_target_encoding : bool
        Target Encoding 사용 여부 (False면 단순 통계만 사용)
    
    Returns
    -------
    BaseModel
        학습된 모델
    """
    print("=" * 70)
    print(" V4 No-Lag Enhanced 모델 학습")
    print(" (Lag/Rolling 완전 제외 + Target Encoding)")
    print("=" * 70)
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드
    df = load_featured_data(data_path)
    
    # 1. 역사적 패턴 feature 추가
    print("\n[Step 1] 역사적 패턴 feature 생성...")
    df = create_historical_features(df)
    
    # 2. Target Encoding feature 추가
    if use_target_encoding:
        print("\n[Step 2] Target Encoding feature 생성 (K-Fold)...")
        df = create_target_encoding_features(df, n_folds=5)
    
    # 3. 비율/교차 feature 추가
    print("\n[Step 3] 비율/교차 feature 생성...")
    df = create_ratio_features(df)
    
    # Feature 컬럼 추출
    exclude_cols = {
        "날짜", "Date", "역명", "역번호",
        "승차", "하차", "net_passengers",
        "오름혼잡도", "내림혼잡도", "오름전역혼잡도", "내림전역혼잡도",
        "요일구분",
    }
    all_features = [c for c in df.columns if c not in exclude_cols]
    
    # Lag/Rolling 완전 제외
    feature_cols = get_v4_features(all_features)
    
    print(f"\n전체 컬럼 수: {len(df.columns)}")
    print(f"사용 Feature 수 (Lag 완전 제외): {len(feature_cols)}")
    
    # Feature 그룹별 개수 출력
    te_features = [f for f in feature_cols if "_te" in f]
    hist_features = [f for f in feature_cols if "_global_" in f or "_mean" in f]
    print(f"  - Target Encoding: {len(te_features)}개")
    print(f"  - 역사적 통계: {len(hist_features)}개")
    
    # 모델 설정
    if use_lightgbm and HAS_LIGHTGBM:
        config = ModelConfig(
            name="V4_NoLag_Enhanced_LightGBM",
            version="v4",
            description="Lag/Rolling 완전 제외, Target Encoding + 역사적 통계로 대체",
            features=feature_cols,
            target="net_passengers",
            test_size=0.2,
            random_state=42,
            hyperparameters={
                "n_estimators": 1500,
                "learning_rate": 0.03,
                "num_leaves": 127,
                "max_depth": 15,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "n_jobs": -1,
                "verbose": -1,
                "random_state": 42
            }
        )
    else:
        config = ModelConfig(
            name="V4_NoLag_Enhanced_GBM",
            version="v4",
            description="Lag/Rolling 완전 제외, Target Encoding + 역사적 통계로 대체",
            features=feature_cols,
            target="net_passengers",
            test_size=0.2,
            random_state=42,
            hyperparameters={
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 10,
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
        split_idx = int(len(X_train) * 0.9)
        X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
        y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
        
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
        model, feature_cols, top_n=30, output_dir=output_dir
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
    print(" V4 No-Lag Enhanced 학습 완료!")
    print("=" * 70)
    
    return model_wrapper


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "outputs" / "featured_data.csv"
    output_dir = project_root / "experiments" / "v4"
    
    model = train_v4_no_lag_enhanced(data_path, output_dir, use_lightgbm=True)







