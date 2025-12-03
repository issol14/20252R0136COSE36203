"""
V4: 다중 모델 비교 실험

LightGBM, XGBoost, RandomForest, CatBoost를 비교합니다.
TimeSeriesSplit을 사용한 K-Fold 교차검증을 적용합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from .base_model import load_featured_data, get_feature_columns


def get_models(random_state: int = 42) -> Dict[str, Any]:
    """비교할 모델들을 반환"""
    models = {
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=10,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=-1,
            verbose=-1,
            random_state=random_state
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=10,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=-1,
            verbosity=0,
            random_state=random_state
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=random_state
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state
        ),
    }
    
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            random_seed=random_state,
            verbose=False
        )
    
    return models


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """모델 평가 지표 계산"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (0이 아닌 값에 대해)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def run_timeseries_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model: Any,
    n_splits: int = 5
) -> Tuple[List[Dict], Dict]:
    """TimeSeriesSplit을 사용한 교차검증"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_results = []
    all_predictions = []
    all_actuals = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 모델 학습
        model_copy = model.__class__(**model.get_params())
        model_copy.fit(X_train, y_train)
        
        # 예측
        y_pred = model_copy.predict(X_val)
        
        # 평가
        metrics = evaluate_model(y_val.values, y_pred)
        metrics["fold"] = fold + 1
        fold_results.append(metrics)
        
        all_predictions.extend(y_pred)
        all_actuals.extend(y_val.values)
    
    # 전체 평가
    overall_metrics = evaluate_model(np.array(all_actuals), np.array(all_predictions))
    
    return fold_results, overall_metrics


def train_v4_model_comparison(
    data_path: Path,
    output_dir: Path,
    n_splits: int = 5,
    sample_size: int = None  # None이면 전체 사용
) -> Dict[str, Any]:
    """
    V4 다중 모델 비교 실험을 실행합니다.
    
    TimeSeriesSplit K-Fold 교차검증으로 여러 모델을 비교합니다.
    """
    print("=" * 70)
    print(" V4: 다중 모델 비교 실험")
    print(f" TimeSeriesSplit {n_splits}-Fold 교차검증")
    print("=" * 70)
    
    # 데이터 로드
    df = load_featured_data(data_path)
    
    # 샘플링 (선택적)
    if sample_size and sample_size < len(df):
        df = df.iloc[:sample_size]
        print(f"\n샘플링: {sample_size:,}행 사용")
    
    # Feature 추출
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df["net_passengers"]
    
    print(f"\n데이터: {len(X):,}행, {len(feature_cols)} Features")
    print(f"교차검증: {n_splits}-Fold TimeSeriesSplit")
    
    # 모델 정의
    models = get_models()
    
    # 결과 저장
    results = {}
    
    print("\n" + "=" * 70)
    print(" 모델별 학습 및 평가")
    print("=" * 70)
    
    for name, model in models.items():
        print(f"\n[{name}] 학습 중...")
        
        try:
            fold_results, overall_metrics = run_timeseries_cv(X, y, model, n_splits)
            
            results[name] = {
                "fold_results": fold_results,
                "overall": overall_metrics
            }
            
            print(f"  RMSE: {overall_metrics['rmse']:,.2f}")
            print(f"  MAE:  {overall_metrics['mae']:,.2f}")
            print(f"  R²:   {overall_metrics['r2']:.4f}")
            
        except Exception as e:
            print(f"  [오류] {e}")
            results[name] = {"error": str(e)}
    
    # 결과 비교 테이블
    print("\n" + "=" * 70)
    print(" 모델 비교 결과 (Overall)")
    print("=" * 70)
    print(f"{'모델':<20} {'RMSE':>12} {'MAE':>12} {'R²':>10}")
    print("-" * 60)
    
    comparison_data = []
    for name, result in results.items():
        if "overall" in result:
            m = result["overall"]
            print(f"{name:<20} {m['rmse']:>12,.2f} {m['mae']:>12,.2f} {m['r2']:>10.4f}")
            comparison_data.append({
                "model": name,
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
                "mape": m["mape"]
            })
    
    # 결과 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 비교 결과 CSV
    comparison_df = pd.DataFrame(comparison_data).sort_values("r2", ascending=False)
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    
    # 상세 결과 JSON
    with open(output_dir / "detailed_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    # 설정 저장
    config = {
        "experiment": "V4_Model_Comparison",
        "n_splits": n_splits,
        "sample_size": sample_size or len(df),
        "n_features": len(feature_cols),
        "models": list(models.keys()),
        "timestamp": datetime.now().isoformat()
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과 저장: {output_dir}")
    
    # 최고 모델
    if comparison_data:
        best = comparison_df.iloc[0]
        print(f"\n🏆 최고 모델: {best['model']} (R² = {best['r2']:.4f})")
    
    return results


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "outputs" / "featured_data.csv"
    output_dir = project_root / "experiments" / "v4_model_comparison"
    
    # 빠른 테스트를 위해 샘플 사용 (전체는 None)
    results = train_v4_model_comparison(data_path, output_dir, n_splits=5, sample_size=100000)







