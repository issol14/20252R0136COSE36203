"""
V5: Feature Ablation Study

Lag Only vs Lag + 외부요인 성능 비교를 통해
외부요인(날씨, 이벤트)의 기여도를 분석합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import lightgbm as lgb

from .base_model import load_featured_data


def get_feature_groups(all_features: List[str]) -> Dict[str, List[str]]:
    """Feature를 그룹별로 분류"""
    
    lag_keywords = ["lag_", "rolling_", "diff_", "pct_change_"]
    time_keywords = ["Time_", "Hour", "요일", "Month", "Season", "Week", "Day", 
                     "Rush", "Night", "Holiday", "Quarter", "시간대"]
    weather_keywords = ["Temp", "Rain", "Snow", "Humid", "Wind", "Weather", 
                        "DI_", "Feel", "Extreme", "Precip", "temperature", 
                        "rainfall", "humidity", "snowfall", "discomfort"]
    event_keywords = ["event", "Event"]
    interaction_keywords = ["_x_"]
    location_keywords = ["순서", "호선", "환승역", "is_transfer", "station_"]
    
    groups = {
        "lag": [],
        "time": [],
        "weather": [],
        "event": [],
        "interaction": [],
        "location": [],
        "other": []
    }
    
    for f in all_features:
        assigned = False
        
        if any(kw in f for kw in lag_keywords):
            groups["lag"].append(f)
            assigned = True
        elif any(kw in f for kw in weather_keywords):
            groups["weather"].append(f)
            assigned = True
        elif any(kw in f for kw in event_keywords):
            groups["event"].append(f)
            assigned = True
        elif any(kw in f for kw in interaction_keywords):
            groups["interaction"].append(f)
            assigned = True
        elif any(kw in f for kw in time_keywords):
            groups["time"].append(f)
            assigned = True
        elif any(kw in f for kw in location_keywords):
            groups["location"].append(f)
            assigned = True
        
        if not assigned:
            groups["other"].append(f)
    
    return groups


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """모델 평가"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: List[str]
) -> Tuple[Dict[str, float], Any]:
    """모델 학습 및 평가"""
    
    X_tr = X_train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    X_te = X_test[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=10,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        verbose=-1,
        random_state=42
    )
    
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    
    metrics = evaluate_model(y_test.values, y_pred)
    
    return metrics, model


def train_v5_feature_ablation(
    data_path: Path,
    output_dir: Path,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    V5 Feature Ablation Study를 실행합니다.
    
    다양한 Feature 조합으로 학습하여 각 그룹의 기여도를 분석합니다.
    """
    print("=" * 70)
    print(" V5: Feature Ablation Study")
    print(" (Lag Only vs Lag + 외부요인 비교)")
    print("=" * 70)
    
    # 데이터 로드
    df = load_featured_data(data_path)
    
    # Feature 추출
    exclude_cols = {
        "날짜", "Date", "역명", "역번호",
        "승차", "하차", "net_passengers",
        "오름혼잡도", "내림혼잡도", "오름전역혼잡도", "내림전역혼잡도",
        "요일구분",
    }
    all_features = [c for c in df.columns if c not in exclude_cols]
    
    # Feature 그룹 분류
    feature_groups = get_feature_groups(all_features)
    
    print("\n[Feature 그룹 분포]")
    for group, features in feature_groups.items():
        print(f"  {group}: {len(features)}개")
    
    # 데이터 분할 (시간순)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    y_train = train_df["net_passengers"]
    y_test = test_df["net_passengers"]
    
    print(f"\n[데이터 분할]")
    print(f"  Train: {len(train_df):,}행")
    print(f"  Test: {len(test_df):,}행")
    
    # 실험 설계
    experiments = {
        "1. Lag Only": feature_groups["lag"],
        "2. Lag + Time": feature_groups["lag"] + feature_groups["time"],
        "3. Lag + Time + Location": feature_groups["lag"] + feature_groups["time"] + feature_groups["location"],
        "4. Lag + Time + Location + Weather": feature_groups["lag"] + feature_groups["time"] + feature_groups["location"] + feature_groups["weather"],
        "5. Lag + Time + Location + Event": feature_groups["lag"] + feature_groups["time"] + feature_groups["location"] + feature_groups["event"],
        "6. Lag + Time + Location + Weather + Event": feature_groups["lag"] + feature_groups["time"] + feature_groups["location"] + feature_groups["weather"] + feature_groups["event"],
        "7. All Features": all_features,
        "8. No Lag (외부요인만)": feature_groups["time"] + feature_groups["location"] + feature_groups["weather"] + feature_groups["event"] + feature_groups["interaction"] + feature_groups["other"],
    }
    
    # 실험 실행
    print("\n" + "=" * 70)
    print(" 실험 실행")
    print("=" * 70)
    
    results = {}
    
    for exp_name, features in experiments.items():
        if not features:
            print(f"\n[{exp_name}] Feature 없음, 건너뜀")
            continue
            
        print(f"\n[{exp_name}] ({len(features)} features)")
        
        metrics, model = train_and_evaluate(
            train_df, y_train, test_df, y_test, features
        )
        
        results[exp_name] = {
            "n_features": len(features),
            "metrics": metrics,
            "features": features
        }
        
        print(f"  RMSE: {metrics['rmse']:,.2f}")
        print(f"  MAE:  {metrics['mae']:,.2f}")
        print(f"  R²:   {metrics['r2']:.4f}")
    
    # 결과 비교 테이블
    print("\n" + "=" * 70)
    print(" 실험 결과 비교")
    print("=" * 70)
    print(f"{'실험':<45} {'Features':>8} {'RMSE':>12} {'R²':>10}")
    print("-" * 80)
    
    comparison_data = []
    for exp_name, result in results.items():
        m = result["metrics"]
        n = result["n_features"]
        print(f"{exp_name:<45} {n:>8} {m['rmse']:>12,.2f} {m['r2']:>10.4f}")
        comparison_data.append({
            "experiment": exp_name,
            "n_features": n,
            "rmse": m["rmse"],
            "mae": m["mae"],
            "r2": m["r2"]
        })
    
    # 개선율 계산
    print("\n" + "=" * 70)
    print(" 외부요인 추가 효과 분석")
    print("=" * 70)
    
    if "1. Lag Only" in results and "7. All Features" in results:
        lag_only_r2 = results["1. Lag Only"]["metrics"]["r2"]
        all_features_r2 = results["7. All Features"]["metrics"]["r2"]
        improvement = (all_features_r2 - lag_only_r2) / (1 - lag_only_r2) * 100
        
        print(f"  Lag Only R²: {lag_only_r2:.4f}")
        print(f"  All Features R²: {all_features_r2:.4f}")
        print(f"  외부요인 추가로 인한 R² 개선: {all_features_r2 - lag_only_r2:.4f}")
        print(f"  남은 오차 대비 개선율: {improvement:.2f}%")
    
    # 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / "ablation_results.csv", index=False)
    
    # 상세 결과 (features 제외)
    results_summary = {k: {"n_features": v["n_features"], "metrics": v["metrics"]} 
                       for k, v in results.items()}
    with open(output_dir / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    
    config = {
        "experiment": "V5_Feature_Ablation",
        "test_size": test_size,
        "feature_groups": {k: len(v) for k, v in feature_groups.items()},
        "timestamp": datetime.now().isoformat()
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과 저장: {output_dir}")
    
    return results


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "outputs" / "featured_data.csv"
    output_dir = project_root / "experiments" / "v5_feature_ablation"
    
    results = train_v5_feature_ablation(data_path, output_dir)







