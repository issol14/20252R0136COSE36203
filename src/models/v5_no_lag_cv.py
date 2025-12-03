"""
V5: No-Lag with Proper Time Series CV

Lag/Rolling feature를 완전히 제외하고,
적절한 Time Series Cross Validation과 Leak-free Target Encoding을 적용합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.ensemble import GradientBoostingRegressor

from .base_model import load_featured_data


def create_leak_free_target_encoding(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str = "net_passengers"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Leak-free Target Encoding: Train 데이터만으로 통계 계산 후 Val에 적용
    
    Parameters
    ----------
    train_df : pd.DataFrame
        학습 데이터 (이 데이터로만 통계 계산)
    val_df : pd.DataFrame
        검증 데이터 (통계 적용만)
    target_col : str
        타겟 컬럼명
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Target Encoding이 적용된 (train, val) 데이터프레임
    """
    train_result = train_df.copy()
    val_result = val_df.copy()
    
    # 인코딩할 그룹 조합 정의 (Train에서만 계산)
    encoding_configs = [
        (["역명", "호선", "Hour"], "station_hour_mean_te"),
        (["역명", "호선", "요일"], "station_weekday_mean_te"),
        (["역명", "호선", "Hour", "요일"], "station_hour_weekday_mean_te"),
        (["순서", "호선", "Hour"], "order_hour_mean_te"),
        (["is_transfer_station", "Hour"], "transfer_hour_mean_te"),
    ]
    
    global_mean = train_df[target_col].mean()
    
    for group_cols, feature_name in encoding_configs:
        # 모든 그룹 컬럼이 존재하는지 확인
        if not all(col in train_df.columns for col in group_cols):
            continue
        
        # Train에서 그룹별 평균 계산
        group_means = train_df.groupby(group_cols)[target_col].mean()
        
        # Train에 적용 (자기 자신 제외 - Leave-One-Out 방식)
        # 각 그룹의 합과 개수를 미리 계산
        group_sum = train_df.groupby(group_cols)[target_col].transform("sum")
        group_count = train_df.groupby(group_cols)[target_col].transform("count")
        
        # LOO 평균: (그룹합 - 자기값) / (그룹개수 - 1)
        train_result[feature_name] = (group_sum - train_df[target_col]) / (group_count - 1)
        # 그룹에 1개만 있는 경우 global mean 사용
        train_result[feature_name] = train_result[feature_name].fillna(global_mean)
        
        # Val에 적용 (Train에서 계산된 그룹 평균 사용)
        val_result[feature_name] = val_df.set_index(group_cols).index.map(
            lambda x: group_means.get(x, global_mean)
        ).values
        val_result[feature_name] = val_result[feature_name].fillna(global_mean)
    
    return train_result, val_result


def create_historical_features_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str = "net_passengers"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    역사적 통계 feature 생성 (Train에서만 계산)
    """
    train_result = train_df.copy()
    val_result = val_df.copy()
    
    global_mean = train_df[target_col].mean()
    
    # 1. 시간대별 전체 평균
    hour_mean = train_df.groupby("Hour")[target_col].mean()
    train_result["hour_global_mean"] = train_df["Hour"].map(hour_mean)
    val_result["hour_global_mean"] = val_df["Hour"].map(hour_mean).fillna(global_mean)
    
    # 2. 요일별 전체 평균
    weekday_mean = train_df.groupby("요일")[target_col].mean()
    train_result["weekday_global_mean"] = train_df["요일"].map(weekday_mean)
    val_result["weekday_global_mean"] = val_df["요일"].map(weekday_mean).fillna(global_mean)
    
    # 3. 호선별 평균
    line_mean = train_df.groupby("호선")[target_col].mean()
    train_result["line_mean"] = train_df["호선"].map(line_mean)
    val_result["line_mean"] = val_df["호선"].map(line_mean).fillna(global_mean)
    
    # 4. 호선-시간 평균
    line_hour_mean = train_df.groupby(["호선", "Hour"])[target_col].mean()
    train_result["line_hour_mean"] = train_df.set_index(["호선", "Hour"]).index.map(
        lambda x: line_hour_mean.get(x, global_mean)
    ).values
    val_result["line_hour_mean"] = val_df.set_index(["호선", "Hour"]).index.map(
        lambda x: line_hour_mean.get(x, global_mean)
    ).values
    
    # 5. 시간-요일 평균
    hour_weekday_mean = train_df.groupby(["Hour", "요일"])[target_col].mean()
    train_result["hour_weekday_global_mean"] = train_df.set_index(["Hour", "요일"]).index.map(
        lambda x: hour_weekday_mean.get(x, global_mean)
    ).values
    val_result["hour_weekday_global_mean"] = val_df.set_index(["Hour", "요일"]).index.map(
        lambda x: hour_weekday_mean.get(x, global_mean)
    ).values
    
    return train_result, val_result


def get_no_lag_features(all_features: List[str]) -> List[str]:
    """Lag/Rolling feature 완전 제외"""
    exclude_keywords = [
        "lag_", "rolling_", "diff_", "pct_change",
        "time_mean_", "weekday_mean_", "month_mean_"  # 기존 leak 가능한 feature도 제외
    ]
    
    return [
        f for f in all_features 
        if not any(kw in f for kw in exclude_keywords)
    ]


def train_v5_no_lag_cv(
    data_path: Path,
    output_dir: Path,
    n_splits: int = 5,
    use_lightgbm: bool = True
) -> Dict:
    """
    V5 No-Lag with Time Series CV 모델 학습
    
    Parameters
    ----------
    data_path : Path
        featured_data.csv 경로
    output_dir : Path
        결과 저장 디렉토리
    n_splits : int
        Time Series CV fold 수
    use_lightgbm : bool
        LightGBM 사용 여부
    
    Returns
    -------
    Dict
        CV 결과 및 최종 모델
    """
    print("=" * 70)
    print(" V5 No-Lag with Proper Time Series CV")
    print(" (Leak-free Target Encoding + Time Series Cross Validation)")
    print("=" * 70)
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드
    df = load_featured_data(data_path)
    
    # 시간순 정렬 확인
    df = df.sort_values(["Date", "Hour", "역명"]).reset_index(drop=True)
    
    # 기본 feature 선별 (lag 제외)
    exclude_cols = {
        "날짜", "Date", "역명", "역번호",
        "승차", "하차", "net_passengers",
        "오름혼잡도", "내림혼잡도", "오름전역혼잡도", "내림전역혼잡도",
        "요일구분",
    }
    all_features = [c for c in df.columns if c not in exclude_cols]
    base_features = get_no_lag_features(all_features)
    
    print(f"\n기본 Feature 수 (Lag 제외): {len(base_features)}")
    
    # Time Series Cross Validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_results = []
    feature_importances_list = []
    
    print(f"\n{'='*50}")
    print(f" Time Series {n_splits}-Fold Cross Validation")
    print(f"{'='*50}")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        print(f"\n[Fold {fold + 1}/{n_splits}]")
        print(f"  Train: {len(train_idx):,}행 (idx {train_idx[0]:,} ~ {train_idx[-1]:,})")
        print(f"  Val:   {len(val_idx):,}행 (idx {val_idx[0]:,} ~ {val_idx[-1]:,})")
        
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        
        # Leak-free Target Encoding 적용
        train_df, val_df = create_leak_free_target_encoding(
            train_df, val_df, target_col="net_passengers"
        )
        
        # 역사적 통계 feature 적용
        train_df, val_df = create_historical_features_split(
            train_df, val_df, target_col="net_passengers"
        )
        
        # 최종 feature 목록 (기본 + TE + 역사적)
        te_features = [c for c in train_df.columns if "_te" in c or "_mean" in c or "_global_" in c]
        te_features = [f for f in te_features if f in val_df.columns]
        
        # 기존 leak 가능한 feature 제외하고 새로 생성한 것만 사용
        final_features = base_features + [f for f in te_features if f not in base_features]
        final_features = [f for f in final_features if f in train_df.columns and f in val_df.columns]
        
        X_train = train_df[final_features].fillna(0).replace([np.inf, -np.inf], 0)
        X_val = val_df[final_features].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_df["net_passengers"]
        y_val = val_df["net_passengers"]
        
        if fold == 0:
            print(f"  Feature 수: {len(final_features)}")
        
        # 모델 학습
        if use_lightgbm and HAS_LIGHTGBM:
            model = lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=63,
                max_depth=10,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                n_jobs=-1,
                verbose=-1,
                random_state=42
            )
            
            # Early stopping용 분할
            es_split = int(len(X_train) * 0.9)
            X_tr, X_es = X_train.iloc[:es_split], X_train.iloc[es_split:]
            y_tr, y_es = y_train.iloc[:es_split], y_train.iloc[es_split:]
            
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_es, y_es)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                random_state=42
            )
            model.fit(X_train, y_train)
        
        # 예측 및 평가
        y_val_pred = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        mae = mean_absolute_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)
        
        print(f"  → RMSE: {rmse:,.2f}, MAE: {mae:,.2f}, R²: {r2:.4f}")
        
        cv_results.append({
            "fold": fold + 1,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })
        
        # Feature Importance 저장
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({
                "feature": final_features,
                "importance": model.feature_importances_,
                "fold": fold + 1
            })
            feature_importances_list.append(fi)
    
    # CV 결과 요약
    print(f"\n{'='*50}")
    print(" Cross Validation 결과 요약")
    print(f"{'='*50}")
    
    cv_df = pd.DataFrame(cv_results)
    
    mean_rmse = cv_df["rmse"].mean()
    std_rmse = cv_df["rmse"].std()
    mean_mae = cv_df["mae"].mean()
    std_mae = cv_df["mae"].std()
    mean_r2 = cv_df["r2"].mean()
    std_r2 = cv_df["r2"].std()
    
    print(f"\n  RMSE: {mean_rmse:,.2f} ± {std_rmse:,.2f}")
    print(f"  MAE:  {mean_mae:,.2f} ± {std_mae:,.2f}")
    print(f"  R²:   {mean_r2:.4f} ± {std_r2:.4f}")
    
    # Feature Importance 평균
    if feature_importances_list:
        all_fi = pd.concat(feature_importances_list)
        mean_fi = all_fi.groupby("feature")["importance"].mean().sort_values(ascending=False)
        
        print(f"\n{'='*50}")
        print(" Feature Importance (평균)")
        print(f"{'='*50}")
        
        total_imp = mean_fi.sum()
        cumsum = 0
        for i, (feat, imp) in enumerate(mean_fi.head(25).items()):
            pct = imp / total_imp * 100
            cumsum += pct
            print(f"  {feat:45} {pct:5.2f}% (누적: {cumsum:5.1f}%)")
        
        # Feature Importance 저장
        fi_df = pd.DataFrame({
            "feature": mean_fi.index,
            "importance": mean_fi.values,
            "importance_pct": mean_fi.values / total_imp * 100
        })
        fi_df.to_csv(output_dir / "feature_importance_cv.csv", index=False)
        
        # 그룹별 중요도
        print(f"\n[그룹별 Feature 중요도]")
        print("-" * 40)
        
        groups = {
            "Target Encoding": [f for f in mean_fi.index if "_te" in f],
            "역사적 통계": [f for f in mean_fi.index if "_mean" in f or "_global_" in f],
            "시간": [f for f in mean_fi.index if any(kw in f for kw in 
                    ["Hour", "Time", "요일", "Day", "Month", "Week", "Season", "Rush", "Holiday"])],
            "날씨": [f for f in mean_fi.index if any(kw in f for kw in 
                    ["temp", "rain", "snow", "humid", "wind", "Weather", "DI", "Feels"])],
            "이벤트": [f for f in mean_fi.index if "event" in f.lower()],
        }
        
        for group_name, group_features in groups.items():
            group_imp = mean_fi[mean_fi.index.isin(group_features)].sum()
            group_pct = group_imp / total_imp * 100
            print(f"  {group_name:15} {group_pct:6.2f}%")
    
    # 결과 저장
    cv_df.to_csv(output_dir / "cv_results.csv", index=False)
    
    summary = {
        "n_splits": n_splits,
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
        "mean_mae": mean_mae,
        "std_mae": std_mae,
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "feature_count": len(final_features) if 'final_features' in dir() else 0
    }
    
    import json
    with open(output_dir / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(" V5 No-Lag CV 학습 완료!")
    print(f"{'='*70}")
    
    return {
        "cv_results": cv_df,
        "summary": summary,
        "feature_importance": fi_df if feature_importances_list else None
    }


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "outputs" / "featured_data.csv"
    output_dir = project_root / "experiments" / "v5_cv"
    
    results = train_v5_no_lag_cv(data_path, output_dir, n_splits=5, use_lightgbm=True)







