import os
import matplotlib.pyplot    as plt
import matplotlib.ticker    as mticker
import matplotlib.gridspec  as gridspec
import seaborn              as sns
import numpy                as np
import pandas               as pd

sns.set_theme(style='whitegrid', font_scale=1.0)
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'font.family':       'sans-serif',
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    11,
})

PALETTE = {
    'blue':  '#2563EB',
    'green': '#16A34A',
    'red':   '#DC2626',
    'amber': '#D97706',
    'gray':  '#6B7280',
    'light_blue': '#BFDBFE',
}

CATEGORY_COLORS = ['#2563EB','#16A34A','#DC2626','#D97706','#7C3AED','#0891B2','#DB2777']
REGION_COLORS   = ['#2563EB','#16A34A','#DC2626','#D97706','#DB2777']
RFM_COLORS      = ['#16A34A','#2563EB','#D97706','#DC2626']


def _save(fig, output_dir: str, filename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_monthly_revenue(monthly_df: pd.DataFrame, output_dir: str = 'outputs') -> None:

    fig, ax = plt.subplots(figsize=(14, 5))

    x = range(len(monthly_df))

    ax.plot(x, monthly_df['revenue'], color=PALETTE['light_blue'],
            linewidth=1.5, alpha=0.8, label='Monthly Revenue')
    ax.fill_between(x, monthly_df['revenue'], alpha=0.15, color=PALETTE['blue'])

    ax.plot(x, monthly_df['rolling_avg'], color=PALETTE['blue'],
            linewidth=2.5, label='3-Month Rolling Avg')

    for i, period in enumerate(monthly_df['period']):
        if period.endswith('-01'):
            year = period[:4]
            ax.axvline(x=i, color='#CBD5E1', linewidth=1.2, linestyle='--')
            ax.text(i + 0.3, monthly_df['revenue'].max() * 1.02, year,
                    fontsize=9, color=PALETTE['gray'])

    tick_positions = list(range(0, len(monthly_df), 3))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([monthly_df['period'].iloc[i] for i in tick_positions], rotation=45, ha='right')

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))
    ax.set_title('Monthly Revenue Trend (2022–2024)')
    ax.set_ylabel('Revenue ($)')
    ax.legend()
    fig.tight_layout()

    _save(fig, output_dir, 'monthly_revenue_trend.png')

def plot_category_performance(cat_df: pd.DataFrame, output_dir: str = 'outputs') -> None:

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    bars = axes[0].barh(
        cat_df['category'], cat_df['revenue'],
        color=CATEGORY_COLORS[:len(cat_df)], edgecolor='none'
    )
    axes[0].set_title('Revenue by Category')
    axes[0].set_xlabel('Total Revenue ($)')
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v/1e6:.1f}M'))
    for bar, val in zip(bars, cat_df['revenue']):
        axes[0].text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2,
                     f'${val/1e3:.0f}K', va='center', fontsize=9)

    bars2 = axes[1].barh(
        cat_df['category'], cat_df['orders'],
        color=CATEGORY_COLORS[:len(cat_df)], edgecolor='none', alpha=0.85
    )
    axes[1].set_title('Order Count by Category')
    axes[1].set_xlabel('Number of Orders')
    for bar, val in zip(bars2, cat_df['orders']):
        axes[1].text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                     f'{val:,}', va='center', fontsize=9)

    fig.tight_layout()
    _save(fig, output_dir, 'category_performance.png')


def plot_regional_sales(reg_df: pd.DataFrame, output_dir: str = 'outputs') -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    wedges, texts, autotexts = ax.pie(
        reg_df['revenue'],
        labels=reg_df['region'],
        colors=REGION_COLORS[:len(reg_df)],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.55),  # Doughnut hole
        pctdistance=0.8,
    )
    for t in texts:     t.set_fontsize(11)
    for a in autotexts: a.set_fontsize(10)

    ax.set_title('Revenue Distribution by Region', pad=20)
    total = reg_df['revenue'].sum()
    ax.text(0, 0, f'${total/1e6:.2f}M\nTotal', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#1E293B')

    _save(fig, output_dir, 'regional_sales.png')

def plot_weekday_sales(wd_df: pd.DataFrame, output_dir: str = 'outputs') -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    max_val = wd_df['revenue'].max()

    colors = [PALETTE['blue'] if v == max_val else PALETTE['light_blue']
              for v in wd_df['revenue']]

    bars = ax.bar(wd_df['weekday'], wd_df['revenue'], color=colors, edgecolor='none', width=0.65)

    for bar, val in zip(bars, wd_df['revenue']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
                f'${val/1e3:.0f}K', ha='center', fontsize=9)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))
    ax.set_title('Revenue by Day of Week')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Total Revenue ($)')
    fig.tight_layout()
    _save(fig, output_dir, 'weekday_sales.png')

def plot_discount_impact(disc_df: pd.DataFrame, output_dir: str = 'outputs') -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    x   = disc_df['discount_pct'].astype(str) + '%'
    pos = range(len(x))

    ax2.bar(pos, disc_df['total_orders'], color=PALETTE['light_blue'],
            edgecolor='none', width=0.5, label='Order Count', alpha=0.7, zorder=1)
    ax2.set_ylabel('Number of Orders', color=PALETTE['blue'])
    ax2.tick_params(axis='y', labelcolor=PALETTE['blue'])
  
    ax1.plot(pos, disc_df['avg_revenue'], color=PALETTE['red'], linewidth=2.5,
             marker='o', markersize=8, label='Avg Revenue', zorder=2)
    ax1.set_xticks(pos)
    ax1.set_xticklabels(x)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    ax1.set_ylabel('Avg Order Revenue ($)', color=PALETTE['red'])
    ax1.tick_params(axis='y', labelcolor=PALETTE['red'])

    for i, (p, rev) in enumerate(zip(pos, disc_df['avg_revenue'])):
        ax1.text(p, rev + 8, f'${rev:.0f}', ha='center', fontsize=9, color=PALETTE['red'])

    ax1.set_title('Discount Level vs Avg Revenue & Order Volume')
    ax1.set_xlabel('Discount Applied')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig.tight_layout()
    _save(fig, output_dir, 'discount_impact.png')

def plot_model_comparison(results: list, output_dir: str = 'outputs') -> None:

    names  = [r['name'] for r in results]
    r2s    = [r['test_r2'] for r in results]
    cv_r2s = [r['cv_mean'] for r in results]
    maes   = [r['mae'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    bar_colors = [PALETTE['green'] if v == max(r2s) else PALETTE['blue'] for v in r2s]

    b1 = axes[0].bar(names, r2s, color=bar_colors, edgecolor='none', width=0.55)
    axes[0].set_title('Test R² (higher = better)')
    axes[0].set_ylim(0.9, 1.0)
    for b, v in zip(b1, r2s):
        axes[0].text(b.get_x() + b.get_width()/2, v + 0.001, f'{v:.4f}', ha='center', fontsize=10)
    axes[0].tick_params(axis='x', rotation=15)


    cv_colors = [PALETTE['green'] if v == max(cv_r2s) else PALETTE['blue'] for v in cv_r2s]
    b2 = axes[1].bar(names, cv_r2s, color=cv_colors, edgecolor='none', width=0.55)
    axes[1].set_title('CV R² (higher = better)')
    axes[1].set_ylim(0.88, 1.0)
    for b, v in zip(b2, cv_r2s):
        axes[1].text(b.get_x() + b.get_width()/2, v + 0.001, f'{v:.4f}', ha='center', fontsize=10)
    axes[1].tick_params(axis='x', rotation=15)

    mae_colors = [PALETTE['green'] if v == min(maes) else PALETTE['red'] for v in maes]
    b3 = axes[2].bar(names, maes, color=mae_colors, edgecolor='none', width=0.55)
    axes[2].set_title('MAE in $ (lower = better)')
    for b, v in zip(b3, maes):
        axes[2].text(b.get_x() + b.get_width()/2, v + 5, f'${v:,.0f}', ha='center', fontsize=10)
    axes[2].tick_params(axis='x', rotation=15)

    fig.suptitle('ML Model Comparison — CLV Prediction', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, output_dir, 'model_comparison.png')

def plot_feature_importance(feat_imp: list, output_dir: str = 'outputs') -> None:
    features    = [f['feature']    for f in feat_imp]
    importances = [f['importance'] for f in feat_imp]

    colors = [PALETTE['blue'] if i < 3 else PALETTE['light_blue']
              for i in range(len(features))]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(features[::-1], importances[::-1], color=colors[::-1], edgecolor='none')

    for bar, val in zip(bars, importances[::-1]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.set_title('Feature Importance — Random Forest (CLV Prediction)')
    ax.set_xlabel('Importance Score')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0%}'))
    fig.tight_layout()
    _save(fig, output_dir, 'feature_importance.png')

def plot_actual_vs_predicted(predictions: dict, output_dir: str = 'outputs') -> None:

    actual    = np.array(predictions['actual'])
    predicted = np.array(predictions['predicted'])
    residuals = np.array(predictions['residuals'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(actual, predicted, alpha=0.65, s=40,
                    color=PALETTE['blue'], edgecolors='none', label='Predictions')

    lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axes[0].plot(lims, lims, color=PALETTE['red'], linewidth=1.5, linestyle='--', label='Perfect Fit')
    axes[0].set_title('Actual vs Predicted CLV')
    axes[0].set_xlabel('Actual CLV ($)')
    axes[0].set_ylabel('Predicted CLV ($)')
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    axes[0].legend()

    axes[1].hist(residuals, bins=20, color=PALETTE['blue'], edgecolor='white', alpha=0.8)
    axes[1].axvline(0, color=PALETTE['red'], linewidth=1.5, linestyle='--', label='Zero Error')
    axes[1].set_title('Prediction Residuals Distribution')
    axes[1].set_xlabel('Residual (Actual − Predicted) ($)')
    axes[1].set_ylabel('Count')
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v:,.0f}'))
    axes[1].legend()

    fig.suptitle('Gradient Boosting — CLV Prediction Performance', fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, output_dir, 'actual_vs_predicted.png')


def plot_rfm_segments(rfm_summary: pd.DataFrame, output_dir: str = 'outputs') -> None:
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    seg_order  = ['Champions', 'Loyal', 'At Risk', 'Lost']
    rfm_sorted = rfm_summary.set_index('segment').reindex(seg_order).reset_index()

    b1 = axes[0].bar(rfm_sorted['segment'], rfm_sorted['customer_count'],
                     color=RFM_COLORS, edgecolor='none', width=0.6)
    axes[0].set_title('Customers per RFM Segment')
    axes[0].set_ylabel('Number of Customers')
    for b, v in zip(b1, rfm_sorted['customer_count']):
        axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                     f'{v}  ({rfm_sorted.loc[rfm_sorted["customer_count"]==v,"pct_customers"].values[0]}%)',
                     ha='center', fontsize=9)

    wedges, texts, autotexts = axes[1].pie(
        rfm_sorted['total_revenue'],
        labels=rfm_sorted['segment'],
        colors=RFM_COLORS,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.55),
        pctdistance=0.8,
    )
    for t in texts:     t.set_fontsize(11)
    for a in autotexts: a.set_fontsize(10)
    axes[1].set_title('Revenue Contribution by Segment')

    fig.suptitle('RFM Customer Segmentation Analysis', fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, output_dir, 'rfm_segments.png')


def plot_summary_dashboard(eda_results: dict, ml_results: dict,
                           rfm_results: dict, output_dir: str = 'outputs') -> None:
  
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Retail Sales Analytics — Executive Dashboard', fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0:2])  # spans 2 columns
    monthly = eda_results['monthly_trend']
    ax1.plot(range(len(monthly)), monthly['revenue'],
             color=PALETTE['light_blue'], linewidth=1.5, alpha=0.7)
    ax1.plot(range(len(monthly)), monthly['rolling_avg'],
             color=PALETTE['blue'], linewidth=2.5)
    ax1.fill_between(range(len(monthly)), monthly['revenue'], alpha=0.1, color=PALETTE['blue'])
    ax1.set_title('Monthly Revenue Trend')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v/1000:.0f}K'))
    ax1.set_xticks([])

    ax2 = fig.add_subplot(gs[0, 2])
    cat = eda_results['category_perf']
    ax2.barh(cat['category'], cat['revenue'],
             color=CATEGORY_COLORS[:len(cat)], edgecolor='none')
    ax2.set_title('Revenue by Category')
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'${v/1e3:.0f}K'))

    ax3 = fig.add_subplot(gs[1, 0])
    reg = eda_results['regional_sales']
    ax3.pie(reg['revenue'], labels=reg['region'],
            colors=REGION_COLORS, autopct='%1.0f%%',
            startangle=90, wedgeprops=dict(width=0.55))
    ax3.set_title('Revenue by Region')

    ax4 = fig.add_subplot(gs[1, 1])
    model_names = [r['name'].replace(' ', '\n') for r in ml_results['results']]
    cv_r2s      = [r['cv_mean'] for r in ml_results['results']]
    colors_mc   = [PALETTE['green'] if v == max(cv_r2s) else PALETTE['blue'] for v in cv_r2s]
    ax4.bar(model_names, cv_r2s, color=colors_mc, edgecolor='none', width=0.55)
    ax4.set_title('Model CV R² Comparison')
    ax4.set_ylim(0.88, 1.0)
    for i, v in enumerate(cv_r2s):
        ax4.text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=9)

    ax5 = fig.add_subplot(gs[1, 2])
    seg_order = ['Champions', 'Loyal', 'At Risk', 'Lost']
    rfm_sum = rfm_results['summary'].set_index('segment').reindex(seg_order).reset_index()
    ax5.bar(rfm_sum['segment'], rfm_sum['customer_count'],
            color=RFM_COLORS, edgecolor='none', width=0.6)
    ax5.set_title('RFM Customer Segments')
    ax5.set_ylabel('Customers')
    for i, v in enumerate(rfm_sum['customer_count']):
        ax5.text(i, v + 1, str(v), ha='center', fontsize=9)

    _save(fig, output_dir, 'executive_dashboard.png')


def generate_all_charts(eda_results: dict, ml_results: dict,
                        rfm_results: dict, output_dir: str = 'outputs') -> None:
 
    print("\n" + "█" * 55)
    print("  GENERATING VISUALIZATIONS")
    print("█" * 55)

    plot_monthly_revenue(    eda_results['monthly_trend'],    output_dir)
    plot_category_performance(eda_results['category_perf'],   output_dir)
    plot_regional_sales(     eda_results['regional_sales'],   output_dir)
    plot_weekday_sales(      eda_results['weekday_pattern'],  output_dir)
    plot_discount_impact(    eda_results['discount_impact'],  output_dir)
    plot_model_comparison(   ml_results['results'],           output_dir)
    plot_feature_importance( ml_results['feat_imp'],          output_dir)
    plot_actual_vs_predicted(ml_results['predictions'],       output_dir)
    plot_rfm_segments(       rfm_results['summary'],          output_dir)
    plot_summary_dashboard(  eda_results, ml_results, rfm_results, output_dir)

    print(f"\n✅ All charts saved to: {output_dir}/")
