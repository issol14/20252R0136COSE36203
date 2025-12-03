"""
V6 기반 앙상블 실험

V6 Pure External 모델을 기준으로 다양한 앙상블 기법을 시도합니다.
- 개별 모델: LightGBM, XGBoost, RandomForest, Ridge, CatBoost
- 앙상블: Voting, Stacking, Weighted Average
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[경고] XGBoost 미설치")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("[경고] CatBoost 미설치")

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent


def get_pure_external_features(df):
    """순수 외부 요인 feature만 선별 (V6와 동일)"""
    exclude_exact = {
        "날짜", "Date", "역명", "역번호",
        "승차", "하차", "net_passengers",
        "오름혼잡도", "내림혼잡도", "오름전역혼잡도", "내림전역혼잡도",
        "요일구분",
    }
    
    exclude_keywords = [
        "lag_", "rolling_", "diff_", "pct_change",
        "_mean_승차", "_mean_하차", "_mean_net",
        "_std_승차", "_std_하차", "_std_net",
        "time_mean", "weekday_mean", "month_mean",
        "_te", "승차", "하차", "net_passengers"
    ]
    
    selected_features = []
    for col in df.columns:
        if col in exclude_exact:
            continue
        if any(kw in col for kw in exclude_keywords):
            continue
        selected_features.append(col)
    
    return selected_features


def get_models():
    """실험할 모델 정의"""
    models = {}
    
    # 1. LightGBM (V6 기준)
    if HAS_LIGHTGBM:
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=10,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=0.5,
            n_jobs=-1,
            verbose=-1,
            random_state=42
        )
    
    # 2. XGBoost
    if HAS_XGBOOST:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=0.5,
            n_jobs=-1,
            verbosity=0,
            random_state=42
        )
    
    # 3. RandomForest
    models['RandomForest'] = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42
    )
    
    # 4. Ridge Regression
    models['Ridge'] = Ridge(alpha=1.0, random_state=42)
    
    # 5. CatBoost
    if HAS_CATBOOST:
        models['CatBoost'] = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        )
    
    return models


def train_and_evaluate_models(X_train, X_val, y_train, y_val, models):
    """각 모델 학습 및 평가"""
    results = {}
    predictions = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"    Training {name}...", end=" ")
        
        try:
            # Early stopping for boosting models
            if name == 'LightGBM':
                es_split = int(len(X_train) * 0.9)
                X_tr, X_es = X_train.iloc[:es_split], X_train.iloc[es_split:]
                y_tr, y_es = y_train.iloc[:es_split], y_train.iloc[es_split:]
                
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_es, y_es)],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
            elif name == 'XGBoost':
                es_split = int(len(X_train) * 0.9)
                X_tr, X_es = X_train.iloc[:es_split], X_train.iloc[es_split:]
                y_tr, y_es = y_train.iloc[:es_split], y_train.iloc[es_split:]
                
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_es, y_es)],
                    verbose=False
                )
            elif name == 'CatBoost':
                es_split = int(len(X_train) * 0.9)
                X_tr, X_es = X_train.iloc[:es_split], X_train.iloc[es_split:]
                y_tr, y_es = y_train.iloc[:es_split], y_train.iloc[es_split:]
                
                model.fit(
                    X_tr, y_tr,
                    eval_set=(X_es, y_es),
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            
            # 예측
            y_pred = model.predict(X_val)
            
            # 평가
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            results[name] = {'rmse': rmse, 'mae': mae, 'r2': r2}
            predictions[name] = y_pred
            trained_models[name] = model
            
            print(f"R²={r2:.4f}")
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    return results, predictions, trained_models


def create_ensemble_predictions(predictions, y_val):
    """앙상블 예측 생성"""
    ensemble_results = {}
    
    pred_names = list(predictions.keys())
    pred_values = np.array([predictions[name] for name in pred_names])
    
    # 1. Simple Average
    avg_pred = np.mean(pred_values, axis=0)
    ensemble_results['Simple_Average'] = {
        'rmse': np.sqrt(mean_squared_error(y_val, avg_pred)),
        'mae': mean_absolute_error(y_val, avg_pred),
        'r2': r2_score(y_val, avg_pred),
        'prediction': avg_pred
    }
    
    # 2. Weighted Average (R² 기반 가중치)
    if len(predictions) > 1:
        # 각 모델의 R²를 가중치로 사용
        weights = []
        for name in pred_names:
            r2 = r2_score(y_val, predictions[name])
            weights.append(max(0, r2))  # 음수 R²는 0으로
        
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        weighted_pred = np.average(pred_values, axis=0, weights=weights)
        ensemble_results['Weighted_Average'] = {
            'rmse': np.sqrt(mean_squared_error(y_val, weighted_pred)),
            'mae': mean_absolute_error(y_val, weighted_pred),
            'r2': r2_score(y_val, weighted_pred),
            'prediction': weighted_pred,
            'weights': dict(zip(pred_names, weights))
        }
    
    # 3. Best 3 Average (상위 3개 모델 평균)
    if len(predictions) >= 3:
        r2_scores = {name: r2_score(y_val, pred) for name, pred in predictions.items()}
        top3 = sorted(r2_scores.keys(), key=lambda x: r2_scores[x], reverse=True)[:3]
        top3_pred = np.mean([predictions[name] for name in top3], axis=0)
        ensemble_results['Top3_Average'] = {
            'rmse': np.sqrt(mean_squared_error(y_val, top3_pred)),
            'mae': mean_absolute_error(y_val, top3_pred),
            'r2': r2_score(y_val, top3_pred),
            'prediction': top3_pred,
            'models': top3
        }
    
    # 4. Median Ensemble
    median_pred = np.median(pred_values, axis=0)
    ensemble_results['Median'] = {
        'rmse': np.sqrt(mean_squared_error(y_val, median_pred)),
        'mae': mean_absolute_error(y_val, median_pred),
        'r2': r2_score(y_val, median_pred),
        'prediction': median_pred
    }
    
    return ensemble_results


def run_experiment():
    """전체 실험 실행"""
    print("=" * 70)
    print(" V6 기반 앙상블 실험")
    print("=" * 70)
    
    # 데이터 로드
    data_path = PROJECT_ROOT / "outputs" / "featured_data.csv"
    print(f"\n데이터 로드: {data_path}")
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Hour", "역명"]).reset_index(drop=True)
    
    print(f"  - 행 수: {len(df):,}")
    
    # Feature 선별 (V6와 동일)
    feature_cols = get_pure_external_features(df)
    print(f"  - Feature 수: {len(feature_cols)}")
    
    # 모델 정의
    models = get_models()
    print(f"\n실험 모델: {list(models.keys())}")
    
    # Time Series CV
    tscv = TimeSeriesSplit(n_splits=5)
    
    all_results = {name: [] for name in models.keys()}
    all_ensemble_results = {}
    
    print(f"\n{'='*70}")
    print(" Time Series 5-Fold Cross Validation")
    print(f"{'='*70}")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        print(f"\n[Fold {fold + 1}/5]")
        print(f"  Train: {len(train_idx):,}행, Val: {len(val_idx):,}행")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_val = val_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_df["net_passengers"]
        y_val = val_df["net_passengers"]
        
        # 개별 모델 학습
        results, predictions, trained_models = train_and_evaluate_models(
            X_train, X_val, y_train, y_val, models
        )
        
        # 결과 저장
        for name, metrics in results.items():
            all_results[name].append(metrics)
        
        # 앙상블 예측
        if len(predictions) > 1:
            ensemble_results = create_ensemble_predictions(predictions, y_val)
            
            for ens_name, ens_metrics in ensemble_results.items():
                if ens_name not in all_ensemble_results:
                    all_ensemble_results[ens_name] = []
                all_ensemble_results[ens_name].append({
                    'rmse': ens_metrics['rmse'],
                    'mae': ens_metrics['mae'],
                    'r2': ens_metrics['r2']
                })
    
    # 결과 요약
    print(f"\n{'='*70}")
    print(" 실험 결과 요약")
    print(f"{'='*70}")
    
    summary = {}
    
    # 개별 모델 결과
    print("\n[개별 모델 성능]")
    print("-" * 60)
    print(f"{'Model':<20} {'RMSE':>12} {'MAE':>12} {'R²':>12}")
    print("-" * 60)
    
    for name, results_list in all_results.items():
        if len(results_list) == 0:
            continue
        
        mean_rmse = np.mean([r['rmse'] for r in results_list])
        std_rmse = np.std([r['rmse'] for r in results_list])
        mean_mae = np.mean([r['mae'] for r in results_list])
        mean_r2 = np.mean([r['r2'] for r in results_list])
        std_r2 = np.std([r['r2'] for r in results_list])
        
        summary[name] = {
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'mean_mae': mean_mae,
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'type': 'individual'
        }
        
        print(f"{name:<20} {mean_rmse:>8.2f}±{std_rmse:<3.0f} {mean_mae:>8.2f} {mean_r2:>8.4f}±{std_r2:.3f}")
    
    # 앙상블 결과
    print("\n[앙상블 성능]")
    print("-" * 60)
    
    for ens_name, results_list in all_ensemble_results.items():
        if len(results_list) == 0:
            continue
        
        mean_rmse = np.mean([r['rmse'] for r in results_list])
        std_rmse = np.std([r['rmse'] for r in results_list])
        mean_mae = np.mean([r['mae'] for r in results_list])
        mean_r2 = np.mean([r['r2'] for r in results_list])
        std_r2 = np.std([r['r2'] for r in results_list])
        
        summary[ens_name] = {
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'mean_mae': mean_mae,
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'type': 'ensemble'
        }
        
        print(f"{ens_name:<20} {mean_rmse:>8.2f}±{std_rmse:<3.0f} {mean_mae:>8.2f} {mean_r2:>8.4f}±{std_r2:.3f}")
    
    # 최고 성능 모델 선정
    print(f"\n{'='*70}")
    print(" 최종 결론")
    print(f"{'='*70}")
    
    # R² 기준 정렬
    sorted_models = sorted(summary.items(), key=lambda x: x[1]['mean_r2'], reverse=True)
    
    print("\n[R² 기준 순위]")
    for i, (name, metrics) in enumerate(sorted_models[:5], 1):
        model_type = "앙상블" if metrics['type'] == 'ensemble' else "개별"
        print(f"  {i}. {name} ({model_type}): R²={metrics['mean_r2']:.4f}")
    
    best_model = sorted_models[0]
    baseline_r2 = summary.get('LightGBM', {}).get('mean_r2', 0)
    best_r2 = best_model[1]['mean_r2']
    
    print(f"\n[결론]")
    print(f"  - 기준 모델 (LightGBM): R² = {baseline_r2:.4f}")
    print(f"  - 최고 성능 ({best_model[0]}): R² = {best_r2:.4f}")
    
    improvement = (best_r2 - baseline_r2) * 100
    
    if best_model[1]['type'] == 'ensemble' and improvement > 0.1:
        print(f"  - 앙상블 채택: {best_model[0]} (+{improvement:.2f}%p 향상)")
        decision = f"ADOPT: {best_model[0]}"
    else:
        print(f"  - 앙상블 미채택: 개선폭 미미 ({improvement:+.2f}%p)")
        decision = "REJECT: Keep LightGBM (V6)"
    
    # 결과 저장
    results_df = pd.DataFrame([
        {
            'model': name,
            'type': metrics['type'],
            'mean_rmse': metrics['mean_rmse'],
            'std_rmse': metrics['std_rmse'],
            'mean_mae': metrics['mean_mae'],
            'mean_r2': metrics['mean_r2'],
            'std_r2': metrics['std_r2']
        }
        for name, metrics in summary.items()
    ])
    results_df = results_df.sort_values('mean_r2', ascending=False)
    results_df.to_csv(OUTPUT_DIR / 'ensemble_comparison.csv', index=False)
    
    # 요약 JSON
    final_summary = {
        'baseline': {
            'model': 'LightGBM (V6)',
            'r2': baseline_r2
        },
        'best': {
            'model': best_model[0],
            'r2': best_r2,
            'type': best_model[1]['type']
        },
        'improvement': improvement,
        'decision': decision,
        'all_models': {name: {'r2': m['mean_r2'], 'rmse': m['mean_rmse']} 
                       for name, m in summary.items()}
    }
    
    with open(OUTPUT_DIR / 'ensemble_summary.json', 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n결과 저장: {OUTPUT_DIR}")
    print(f"  - ensemble_comparison.csv")
    print(f"  - ensemble_summary.json")
    
    return summary, decision


if __name__ == "__main__":
    summary, decision = run_experiment()







