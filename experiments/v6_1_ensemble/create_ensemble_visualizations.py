"""
앙상블 실험 결과 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

OUTPUT_DIR = Path(__file__).parent

# 색상
COLORS = {
    'individual': '#3B82F6',  # Blue
    'ensemble': '#10B981',    # Green
    'baseline': '#EF4444',    # Red
    'best': '#F59E0B',        # Amber
}


def load_data():
    """데이터 로드"""
    results_df = pd.read_csv(OUTPUT_DIR / 'ensemble_comparison.csv')
    
    with open(OUTPUT_DIR / 'ensemble_summary.json', 'r') as f:
        summary = json.load(f)
    
    return results_df, summary


def plot_model_comparison(results_df, summary, output_dir):
    """모델별 R² 비교"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 정렬
    results_df = results_df.sort_values('mean_r2', ascending=True)
    
    # 색상 설정
    colors = []
    for _, row in results_df.iterrows():
        if row['model'] == 'LightGBM':
            colors.append(COLORS['baseline'])
        elif row['model'] == summary['best']['model']:
            colors.append(COLORS['best'])
        elif row['type'] == 'ensemble':
            colors.append(COLORS['ensemble'])
        else:
            colors.append(COLORS['individual'])
    
    # 막대 그래프
    bars = ax.barh(results_df['model'], results_df['mean_r2'], color=colors, 
                   edgecolor='white', linewidth=1)
    
    # 에러바
    ax.errorbar(results_df['mean_r2'], results_df['model'], 
                xerr=results_df['std_r2'], fmt='none', color='black', capsize=3)
    
    # 값 표시
    for bar, r2, model in zip(bars, results_df['mean_r2'], results_df['model']):
        label = f'{r2:.4f}'
        if model == summary['best']['model']:
            label += ' ★'
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                label, va='center', fontsize=10)
    
    # 기준선
    baseline_r2 = summary['baseline']['r2']
    ax.axvline(x=baseline_r2, color=COLORS['baseline'], linestyle='--', 
               linewidth=2, label=f"Baseline (LightGBM): {baseline_r2:.4f}")
    
    ax.set_xlabel('R² Score (Mean ± Std)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison\n(Individual vs Ensemble)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.legend(loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 범례 추가
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['individual'], label='Individual Model'),
        Patch(facecolor=COLORS['ensemble'], label='Ensemble'),
        Patch(facecolor=COLORS['baseline'], label='Baseline (LightGBM)'),
        Patch(facecolor=COLORS['best'], label='Best Model'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_ensemble.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ model_comparison_ensemble.png saved")


def plot_rmse_comparison(results_df, summary, output_dir):
    """RMSE 비교"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # RMSE 기준 정렬 (낮을수록 좋음)
    results_df_sorted = results_df.sort_values('mean_rmse', ascending=False)
    
    # 색상
    colors = []
    for _, row in results_df_sorted.iterrows():
        if row['model'] == 'LightGBM':
            colors.append(COLORS['baseline'])
        elif row['type'] == 'ensemble':
            colors.append(COLORS['ensemble'])
        else:
            colors.append(COLORS['individual'])
    
    bars = ax.barh(results_df_sorted['model'], results_df_sorted['mean_rmse'], 
                   color=colors, edgecolor='white', linewidth=1)
    
    # 에러바
    ax.errorbar(results_df_sorted['mean_rmse'], results_df_sorted['model'], 
                xerr=results_df_sorted['std_rmse'], fmt='none', color='black', capsize=3)
    
    # 값 표시
    for bar, rmse in zip(bars, results_df_sorted['mean_rmse']):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2, 
                f'{rmse:.1f}', va='center', fontsize=10)
    
    ax.set_xlabel('RMSE (Mean ± Std)', fontsize=12, fontweight='bold')
    ax.set_title('RMSE Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rmse_comparison_ensemble.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ rmse_comparison_ensemble.png saved")


def plot_improvement_analysis(summary, output_dir):
    """개선 분석"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Baseline vs Best 비교
    ax1 = axes[0]
    models = ['Baseline\n(LightGBM)', f"Best\n({summary['best']['model']})"]
    r2_values = [summary['baseline']['r2'], summary['best']['r2']]
    colors = [COLORS['baseline'], COLORS['best']]
    
    bars = ax1.bar(models, r2_values, color=colors, edgecolor='white', linewidth=2)
    
    for bar, r2 in zip(bars, r2_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{r2:.4f}', ha='center', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax1.set_title('Baseline vs Best Model', fontsize=13, fontweight='bold')
    ax1.set_ylim(0.9, 0.95)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 개선율 표시
    improvement = summary['improvement']
    ax1.annotate(f'+{improvement:.2f}%p', 
                xy=(1, summary['best']['r2']), 
                xytext=(0.5, (summary['baseline']['r2'] + summary['best']['r2'])/2),
                fontsize=14, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    # 2. 모델 유형별 분포
    ax2 = axes[1]
    all_models = summary['all_models']
    
    individual = [(k, v['r2']) for k, v in all_models.items() 
                  if k in ['LightGBM', 'XGBoost', 'RandomForest', 'Ridge', 'CatBoost']]
    ensemble = [(k, v['r2']) for k, v in all_models.items() 
                if k not in ['LightGBM', 'XGBoost', 'RandomForest', 'Ridge', 'CatBoost']]
    
    # Ridge 제외 (이상치)
    individual = [(k, v) for k, v in individual if k != 'Ridge']
    
    ind_r2 = [v for k, v in individual]
    ens_r2 = [v for k, v in ensemble]
    
    bp1 = ax2.boxplot([ind_r2], positions=[1], widths=0.6, patch_artist=True)
    bp2 = ax2.boxplot([ens_r2], positions=[2], widths=0.6, patch_artist=True)
    
    bp1['boxes'][0].set_facecolor(COLORS['individual'])
    bp2['boxes'][0].set_facecolor(COLORS['ensemble'])
    
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Individual\n(excl. Ridge)', 'Ensemble'])
    ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_title('Individual vs Ensemble Distribution', fontsize=13, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ improvement_analysis.png saved")


def plot_decision_summary(summary, output_dir):
    """결정 요약"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # 제목
    ax.text(0.5, 0.95, 'Ensemble Experiment Summary', fontsize=18, fontweight='bold', 
            ha='center', transform=ax.transAxes)
    
    # 결과 박스
    decision = summary['decision']
    is_adopt = 'ADOPT' in decision
    
    box_color = '#D1FAE5' if is_adopt else '#FEE2E2'
    text_color = '#065F46' if is_adopt else '#991B1B'
    
    # 박스 그리기
    rect = plt.Rectangle((0.1, 0.3), 0.8, 0.5, facecolor=box_color, 
                          edgecolor=text_color, linewidth=3, transform=ax.transAxes)
    ax.add_patch(rect)
    
    # 내용
    ax.text(0.5, 0.7, f"Decision: {'✅ ADOPT ENSEMBLE' if is_adopt else '❌ KEEP BASELINE'}", 
            fontsize=16, fontweight='bold', ha='center', color=text_color, transform=ax.transAxes)
    
    ax.text(0.5, 0.55, f"Baseline (LightGBM): R² = {summary['baseline']['r2']:.4f}", 
            fontsize=12, ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.45, f"Best ({summary['best']['model']}): R² = {summary['best']['r2']:.4f}", 
            fontsize=12, ha='center', transform=ax.transAxes)
    
    improvement_text = f"+{summary['improvement']:.2f}%p" if summary['improvement'] > 0 else f"{summary['improvement']:.2f}%p"
    ax.text(0.5, 0.35, f"Improvement: {improvement_text}", 
            fontsize=14, fontweight='bold', ha='center', 
            color='green' if summary['improvement'] > 0 else 'red', transform=ax.transAxes)
    
    # 권장사항
    if is_adopt:
        recommendation = f"Recommendation: Use {summary['best']['model']} ensemble\n(Top 3 models: LightGBM, XGBoost, RandomForest)"
    else:
        recommendation = "Recommendation: Keep LightGBM (V6) as final model"
    
    ax.text(0.5, 0.15, recommendation, fontsize=11, ha='center', 
            style='italic', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'decision_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ decision_summary.png saved")


def main():
    print("=" * 50)
    print(" Ensemble Experiment Visualization")
    print("=" * 50)
    
    results_df, summary = load_data()
    
    print(f"\nBest model: {summary['best']['model']}")
    print(f"Improvement: {summary['improvement']:.2f}%p")
    print(f"Decision: {summary['decision']}")
    
    print("\nGenerating visualizations...")
    
    plot_model_comparison(results_df, summary, OUTPUT_DIR)
    plot_rmse_comparison(results_df, summary, OUTPUT_DIR)
    plot_improvement_analysis(summary, OUTPUT_DIR)
    plot_decision_summary(summary, OUTPUT_DIR)
    
    print("\n" + "=" * 50)
    print(" All visualizations completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()







