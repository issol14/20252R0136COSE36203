"""
Top3_Average 앙상블 기반 최종 시각화

LightGBM + XGBoost + RandomForest 앙상블의 결과를 시각화합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import json
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent

# 색상
COLORS = {
    'primary': '#2563EB',
    'secondary': '#7C3AED',
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'train': '#3B82F6',
    'valid': '#EF4444',
}

GROUP_COLORS = {
    'Time': '#3B82F6',
    'Location': '#10B981',
    'Weather': '#F59E0B',
    'Event': '#EF4444',
    'Interaction': '#8B5CF6',
    'Other': '#6B7280',
}

FEATURE_NAME_MAP = {
    '순서': 'station_order',
    '시간대': 'time_slot',
    '요일': 'weekday',
    '호선': 'line',
    '환승역': 'transfer_station',
    '요일_Sin': 'weekday_Sin',
    '요일_Cos': 'weekday_Cos',
    '월_Sin': 'month_Sin',
    '월_Cos': 'month_Cos',
    '시간대_그룹': 'time_group',
}


def translate_feature(name):
    return FEATURE_NAME_MAP.get(name, name)


def categorize_feature(feature_name):
    time_keywords = ["Hour", "Time", "Day", "Month", "Week", "Season", 
                     "Rush", "Holiday", "Quarter", "Night", "weekday", "time"]
    location_keywords = ["order", "line", "transfer", "station"]
    weather_keywords = ["temp", "rain", "snow", "humid", "wind", "Weather", 
                        "DI", "Feels", "discomfort", "Precip", "Cold", "Hot", 
                        "Freez", "Extreme"]
    event_keywords = ["event", "Event"]
    interaction_keywords = ["_x_"]
    
    if any(kw.lower() in feature_name.lower() for kw in event_keywords):
        return 'Event'
    elif any(kw in feature_name for kw in interaction_keywords):
        return 'Interaction'
    elif any(kw.lower() in feature_name.lower() for kw in location_keywords):
        return 'Location'
    elif any(kw.lower() in feature_name.lower() for kw in weather_keywords):
        return 'Weather'
    elif any(kw.lower() in feature_name.lower() for kw in time_keywords):
        return 'Time'
    else:
        return 'Other'


def get_pure_external_features(df):
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


def train_ensemble_with_history():
    """Top3 앙상블 학습 및 결과 수집"""
    print("=" * 60)
    print(" Top3_Average Ensemble Training")
    print("=" * 60)
    
    # 데이터 로드
    data_path = PROJECT_ROOT / "outputs" / "featured_data.csv"
    print(f"\nLoading data: {data_path}")
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Hour", "역명"]).reset_index(drop=True)
    
    feature_cols = get_pure_external_features(df)
    print(f"  - Rows: {len(df):,}")
    print(f"  - Features: {len(feature_cols)}")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_results = []
    all_histories = []
    all_feature_importances = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        print(f"\n[Fold {fold + 1}/5]")
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        X_train = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        X_val = val_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y_train = train_df["net_passengers"]
        y_val = val_df["net_passengers"]
        
        # Models
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=63, max_depth=10,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=0.5, n_jobs=-1, verbose=-1, random_state=42
        )
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=8, min_child_weight=50,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=0.5,
            n_jobs=-1, verbosity=0, random_state=42
        )
        
        rf_model = RandomForestRegressor(
            n_estimators=300, max_depth=15, min_samples_split=20,
            min_samples_leaf=10, n_jobs=-1, random_state=42
        )
        
        # LightGBM with history
        es_split = int(len(X_train) * 0.9)
        X_tr, X_es = X_train.iloc[:es_split], X_train.iloc[es_split:]
        y_tr, y_es = y_train.iloc[:es_split], y_train.iloc[es_split:]
        
        evals_result = {}
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_es, y_es)],
            eval_names=['train', 'valid'],
            eval_metric='rmse',
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.record_evaluation(evals_result)
            ]
        )
        
        all_histories.append({
            'fold': fold + 1,
            'train_rmse': evals_result['train']['rmse'],
            'valid_rmse': evals_result['valid']['rmse'],
            'best_iteration': lgb_model.best_iteration_
        })
        
        # XGBoost
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
        
        # RandomForest
        rf_model.fit(X_train, y_train)
        
        # Predictions
        lgb_pred = lgb_model.predict(X_val)
        xgb_pred = xgb_model.predict(X_val)
        rf_pred = rf_model.predict(X_val)
        
        # Ensemble prediction
        ensemble_pred = (lgb_pred + xgb_pred + rf_pred) / 3
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        mae = mean_absolute_error(y_val, ensemble_pred)
        r2 = r2_score(y_val, ensemble_pred)
        
        cv_results.append({
            'fold': fold + 1,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })
        
        print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        # Feature importance (average of 3 models)
        lgb_imp = lgb_model.feature_importances_
        xgb_imp = xgb_model.feature_importances_
        rf_imp = rf_model.feature_importances_
        
        # Normalize
        lgb_imp = lgb_imp / lgb_imp.sum()
        xgb_imp = xgb_imp / xgb_imp.sum()
        rf_imp = rf_imp / rf_imp.sum()
        
        avg_imp = (lgb_imp + xgb_imp + rf_imp) / 3
        
        fi = pd.DataFrame({
            'feature': feature_cols,
            'importance': avg_imp,
            'fold': fold + 1
        })
        all_feature_importances.append(fi)
    
    # Summary
    cv_df = pd.DataFrame(cv_results)
    
    mean_rmse = cv_df['rmse'].mean()
    std_rmse = cv_df['rmse'].std()
    mean_mae = cv_df['mae'].mean()
    std_mae = cv_df['mae'].std()
    mean_r2 = cv_df['r2'].mean()
    std_r2 = cv_df['r2'].std()
    
    print(f"\n{'='*60}")
    print(" CV Results Summary")
    print(f"{'='*60}")
    print(f"  RMSE: {mean_rmse:.2f} ± {std_rmse:.2f}")
    print(f"  MAE:  {mean_mae:.2f} ± {std_mae:.2f}")
    print(f"  R²:   {mean_r2:.4f} ± {std_r2:.4f}")
    
    # Feature importance average
    all_fi = pd.concat(all_feature_importances)
    mean_fi = all_fi.groupby('feature')['importance'].mean().sort_values(ascending=False)
    
    fi_df = pd.DataFrame({
        'feature': mean_fi.index,
        'importance': mean_fi.values,
        'importance_pct': mean_fi.values / mean_fi.sum() * 100
    })
    fi_df['cumulative_pct'] = fi_df['importance_pct'].cumsum()
    fi_df['feature_en'] = fi_df['feature'].apply(translate_feature)
    
    return cv_df, fi_df, all_histories, {
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'mean_mae': mean_mae,
        'std_mae': std_mae,
        'mean_r2': mean_r2,
        'std_r2': std_r2
    }


def plot_feature_importance_top20(fi_df, output_dir):
    """상위 20개 Feature Importance"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    top20 = fi_df.head(20).copy()
    top20['group'] = top20['feature_en'].apply(categorize_feature)
    top20 = top20.iloc[::-1]
    
    colors = [GROUP_COLORS.get(g, '#6B7280') for g in top20['group']]
    
    bars = ax.barh(range(len(top20)), top20['importance_pct'], color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['feature_en'], fontsize=11)
    ax.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Feature Importance\n(Top3_Average Ensemble: LightGBM + XGBoost + RandomForest)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    for bar, pct in zip(bars, top20['importance_pct']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                f'{pct:.2f}%', va='center', fontsize=10)
    
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=group) 
                       for group, color in GROUP_COLORS.items() if group in top20['group'].values]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    ax.set_xlim(0, max(top20['importance_pct']) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_top20.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ feature_importance_top20.png saved")


def plot_group_importance(fi_df, output_dir):
    """그룹별 중요도"""
    fi_df['group'] = fi_df['feature_en'].apply(categorize_feature)
    group_importance = fi_df.groupby('group')['importance_pct'].sum().sort_values(ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = [GROUP_COLORS.get(g, '#6B7280') for g in group_importance.index]
    wedges, texts, autotexts = axes[0].pie(
        group_importance.values, labels=group_importance.index,
        autopct='%1.1f%%', colors=colors,
        explode=[0.05 if g == 'Time' else 0 for g in group_importance.index],
        startangle=90
    )
    axes[0].set_title('Feature Group Importance\n(Top3_Average Ensemble)', fontsize=14, fontweight='bold')
    
    bars = axes[1].bar(group_importance.index, group_importance.values, color=colors, edgecolor='white', linewidth=1)
    axes[1].set_ylabel('Total Importance (%)', fontsize=12)
    axes[1].set_title('Feature Group Importance\n(Bar Chart)', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    
    for bar, val in zip(bars, group_importance.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                     f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'group_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ group_importance.png saved")


def plot_cv_results(cv_df, output_dir):
    """CV 결과"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['rmse', 'mae', 'r2']
    titles = ['RMSE', 'MAE', 'R² Score']
    colors_list = [COLORS['danger'], COLORS['warning'], COLORS['success']]
    
    for ax, metric, title, color in zip(axes, metrics, titles, colors_list):
        values = cv_df[metric].values
        folds = cv_df['fold'].values
        
        bars = ax.bar(folds, values, color=color, edgecolor='white', linewidth=1, alpha=0.8)
        
        mean_val = values.mean()
        ax.axhline(y=mean_val, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        
        ax.set_xlabel('Fold', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title} by Fold\n(Top3_Average Ensemble)', fontsize=13, fontweight='bold')
        ax.set_xticks(folds)
        ax.legend(loc='upper right')
        
        for bar, val in zip(bars, values):
            if metric == 'r2':
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                        f'{val:.4f}', ha='center', fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                        f'{val:.1f}', ha='center', fontsize=9)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cv_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ cv_results.png saved")


def plot_model_comparison(output_dir):
    """모델 비교 (앙상블 포함)"""
    models = ['LightGBM\n(Single)', 'XGBoost\n(Single)', 'RandomForest\n(Single)', 
              'Top3_Average\n(Ensemble)']
    r2_scores = [0.9243, 0.9271, 0.9324, 0.9379]
    
    colors = [COLORS['primary'], COLORS['primary'], COLORS['primary'], COLORS['success']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, r2_scores, color=colors, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars, r2_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.4f}', ha='center', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison\n(Single Models vs Ensemble)', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0.9, 0.95)
    
    ax.axhline(y=0.93, color='gray', linestyle='--', alpha=0.5, label='R² = 0.93')
    ax.legend()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.annotate('Best: Ensemble\n+1.35%p vs LightGBM', xy=(3, 0.9379), xytext=(2.2, 0.925),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color=COLORS['success']))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ model_comparison.png saved")


def plot_learning_curves(histories, output_dir):
    """학습 곡선"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, hist in enumerate(histories):
        ax = axes[i]
        train_rmse = hist['train_rmse']
        valid_rmse = hist['valid_rmse']
        best_iter = hist['best_iteration']
        
        epochs = range(1, len(train_rmse) + 1)
        
        ax.plot(epochs, train_rmse, color=COLORS['train'], label='Train', linewidth=1.5)
        ax.plot(epochs, valid_rmse, color=COLORS['valid'], label='Valid', linewidth=1.5)
        ax.axvline(x=best_iter, color='gray', linestyle='--', alpha=0.7, label=f'Best: {best_iter}')
        
        ax.set_xlabel('Iterations')
        ax.set_ylabel('RMSE')
        ax.set_title(f'Fold {i + 1}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Average
    ax = axes[5]
    min_len = min(len(h['train_rmse']) for h in histories)
    
    avg_train = np.mean([h['train_rmse'][:min_len] for h in histories], axis=0)
    avg_valid = np.mean([h['valid_rmse'][:min_len] for h in histories], axis=0)
    
    epochs = range(1, min_len + 1)
    ax.plot(epochs, avg_train, color=COLORS['train'], label='Train (Mean)', linewidth=2)
    ax.plot(epochs, avg_valid, color=COLORS['valid'], label='Valid (Mean)', linewidth=2)
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('RMSE')
    ax.set_title('Average (All Folds)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.suptitle('Learning Curves - Top3_Average Ensemble (LightGBM component)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves_all_folds.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ learning_curves_all_folds.png saved")


def plot_key_insights(fi_df, output_dir):
    """핵심 인사이트"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    fi_df['group'] = fi_df['feature_en'].apply(categorize_feature)
    
    # 1. 그룹별 중요도
    ax1 = axes[0, 0]
    group_imp = fi_df.groupby('group')['importance_pct'].sum().sort_values(ascending=False)
    colors = [GROUP_COLORS.get(g, '#6B7280') for g in group_imp.index]
    
    wedges, texts, autotexts = ax1.pie(group_imp.values, labels=group_imp.index, autopct='%1.1f%%', 
                                        colors=colors, wedgeprops=dict(width=0.5), startangle=90)
    ax1.set_title('Feature Group Importance', fontsize=13, fontweight='bold')
    
    # 2. 주요 Feature
    ax2 = axes[0, 1]
    top_features = fi_df.head(8)
    feature_colors = [GROUP_COLORS.get(categorize_feature(f), '#6B7280') for f in top_features['feature_en']]
    
    bars = ax2.barh(top_features['feature_en'].iloc[::-1], top_features['importance_pct'].iloc[::-1], 
                    color=feature_colors[::-1])
    ax2.set_xlabel('Importance (%)', fontsize=11)
    ax2.set_title('Top 8 Features', fontsize=13, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. 이벤트 Feature
    ax3 = axes[1, 0]
    event_df = fi_df[fi_df['group'] == 'Event'].head(8)
    
    if len(event_df) > 0:
        bars = ax3.barh(event_df['feature_en'].iloc[::-1], event_df['importance_pct'].iloc[::-1], 
                        color=GROUP_COLORS['Event'], alpha=0.8)
        ax3.set_xlabel('Importance (%)', fontsize=11)
        ax3.set_title('Event-related Features', fontsize=13, fontweight='bold')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
    
    # 4. 날씨 Feature
    ax4 = axes[1, 1]
    weather_df = fi_df[fi_df['group'] == 'Weather'].head(8)
    
    if len(weather_df) > 0:
        bars = ax4.barh(weather_df['feature_en'].iloc[::-1], weather_df['importance_pct'].iloc[::-1], 
                        color=GROUP_COLORS['Weather'], alpha=0.8)
        ax4.set_xlabel('Importance (%)', fontsize=11)
        ax4.set_title('Weather-related Features', fontsize=13, fontweight='bold')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'key_insights.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ key_insights.png saved")


def plot_cumulative_importance(fi_df, output_dir):
    """누적 중요도"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(1, len(fi_df) + 1)
    y = fi_df['cumulative_pct'].values
    
    ax.plot(x, y, color=COLORS['primary'], linewidth=2)
    ax.fill_between(x, y, alpha=0.3, color=COLORS['primary'])
    
    thresholds = [50, 80, 90, 95]
    for thresh in thresholds:
        idx = np.argmax(y >= thresh)
        ax.axhline(y=thresh, color='gray', linestyle='--', alpha=0.5)
        ax.scatter([idx + 1], [y[idx]], color=COLORS['danger'], s=100, zorder=5)
        ax.annotate(f'{idx + 1} features\nfor {thresh}%', xy=(idx + 1, y[idx]), 
                   xytext=(idx + 10, y[idx] - 5), fontsize=9,
                   arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Cumulative Importance (%)', fontsize=12)
    ax.set_title('Cumulative Feature Importance\n(Top3_Average Ensemble)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ cumulative_importance.png saved")


def plot_feature_engineering_strategy(output_dir):
    """Feature Engineering 전략도"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(7, 9.5, 'Feature Engineering Strategy', fontsize=18, fontweight='bold', ha='center')
    
    def draw_box(x, y, w, h, text, color, fontsize=10):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', wrap=True)
    
    draw_box(0.5, 7, 3, 1.5, 'Raw Data\n(Boarding, Weather,\nEvents)', '#E5E7EB', 11)
    
    draw_box(5, 7.5, 2.5, 1, 'Time Features\n(35)', GROUP_COLORS['Time'], 10)
    draw_box(8, 7.5, 2.5, 1, 'Location Features\n(4)', GROUP_COLORS['Location'], 10)
    draw_box(11, 7.5, 2.5, 1, 'Weather Features\n(17)', GROUP_COLORS['Weather'], 10)
    draw_box(5, 6, 2.5, 1, 'Event Features\n(13)', GROUP_COLORS['Event'], 10)
    draw_box(8, 6, 2.5, 1, 'Interaction\nFeatures (14)', GROUP_COLORS['Interaction'], 10)
    
    ax.annotate('', xy=(4.5, 7.5), xytext=(3.5, 7.75),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    draw_box(5.5, 4, 4, 1.2, 'Total: 75 Pure External Features\n(No Lag, No Target Info)', '#DBEAFE', 12)
    
    ax.annotate('', xy=(7.5, 4), xytext=(7.5, 5.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    draw_box(4.5, 2, 6, 1.2, 'Top3_Average Ensemble\n(LightGBM + XGBoost + RandomForest)', '#FEF3C7', 11)
    
    ax.annotate('', xy=(7.5, 2), xytext=(7.5, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    draw_box(5.5, 0.3, 4, 1.2, 'R² = 0.9379\n(93.79% Explained)', '#D1FAE5', 13)
    
    ax.annotate('', xy=(7.5, 0.3), xytext=(7.5, 1.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax.text(1, 5.5, 'Excluded Features:\n- lag_*\n- rolling_*\n- diff_*\n- pct_*\n- target statistics', 
            fontsize=9, bbox=dict(boxstyle='round', facecolor='#FEE2E2', alpha=0.8))
    
    ax.text(11.5, 3.5, 'Ensemble Strategy:\n- 3 diverse models\n- Simple average\n- Time Series CV', 
            fontsize=9, bbox=dict(boxstyle='round', facecolor='#E0E7FF', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_engineering_strategy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ feature_engineering_strategy.png saved")


def main():
    print("=" * 60)
    print(" Top3_Average Ensemble Visualization")
    print("=" * 60)
    
    # Train and get results
    cv_df, fi_df, histories, summary = train_ensemble_with_history()
    
    # Save data
    cv_df.to_csv(OUTPUT_DIR / 'cv_results.csv', index=False)
    fi_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
    
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nGenerating visualizations...")
    
    plot_feature_importance_top20(fi_df, OUTPUT_DIR)
    plot_group_importance(fi_df, OUTPUT_DIR)
    plot_cv_results(cv_df, OUTPUT_DIR)
    plot_model_comparison(OUTPUT_DIR)
    plot_learning_curves(histories, OUTPUT_DIR)
    plot_key_insights(fi_df, OUTPUT_DIR)
    plot_cumulative_importance(fi_df, OUTPUT_DIR)
    plot_feature_engineering_strategy(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print(" All visualizations completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()







