

import pandas as pd
import numpy as np

def data_quality_report(df: pd.DataFrame) -> dict:
  
    report = {
        'shape':         df.shape,
        'columns':       list(df.columns),
        'dtypes':        df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_pct':   (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'duplicate_rows': int(df.duplicated().sum()),
        'numeric_stats': df.describe().round(2).to_dict(),
    }

    print("=" * 55)
    print("DATA QUALITY REPORT")
    print("=" * 55)
    print(f"Shape         : {report['shape'][0]:,} rows × {report['shape'][1]} columns")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print("\nMissing Values:")
    for col, cnt in report['missing_values'].items():
        if cnt > 0:
            print(f"  {col:<20} {cnt:>5} ({report['missing_pct'][col]:.1f}%)")
        else:
            print(f"  {col:<20} {'0':>5} — clean ✓")
    return report


def compute_kpis(df: pd.DataFrame) -> dict:
 
    total_revenue    = df['revenue'].sum()
    total_orders     = len(df)
    aov              = df['revenue'].mean()
    unique_customers = df['customer_id'].nunique()
    avg_quantity     = df['quantity'].mean()
    avg_discount     = df['discount'].mean() * 100  # convert to pct

   
    yoy = df.groupby('year')['revenue'].sum()
    yoy_growth = None
    if 2024 in yoy.index and 2023 in yoy.index:
        yoy_growth = (yoy[2024] - yoy[2023]) / yoy[2023] * 100

    kpis = {
        'total_revenue':      round(total_revenue, 2),
        'total_orders':       total_orders,
        'avg_order_value':    round(aov, 2),
        'unique_customers':   unique_customers,
        'orders_per_customer': round(total_orders / unique_customers, 1),
        'avg_discount_pct':   round(avg_discount, 1),
        'yoy_growth_2024':    round(yoy_growth, 2) if yoy_growth else None,
    }

    print("\n" + "=" * 55)
    print("KEY PERFORMANCE INDICATORS")
    print("=" * 55)
    for k, v in kpis.items():
        label = k.replace('_', ' ').title()
        if 'revenue' in k or 'value' in k:
            print(f"  {label:<30} ${v:>12,.2f}")
        elif 'pct' in k or 'growth' in k:
            print(f"  {label:<30} {v:>11.1f}%")
        else:
            print(f"  {label:<30} {str(v):>12}")

    return kpis


def monthly_revenue_trend(df: pd.DataFrame) -> pd.DataFrame:

    monthly = (
        df.groupby(['year', 'month'])['revenue']
        .sum()
        .reset_index()
    )
  
    monthly['period'] = monthly.apply(
        lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1
    )
    monthly = monthly.sort_values('period').reset_index(drop=True)
    monthly['revenue'] = monthly['revenue'].round(2)

    monthly['rolling_avg'] = monthly['revenue'].rolling(window=3, min_periods=1).mean().round(2)

    print(f"\nMonthly Revenue — {len(monthly)} periods")
    print(f"  Range: {monthly['period'].iloc[0]} → {monthly['period'].iloc[-1]}")
    print(f"  Min:   ${monthly['revenue'].min():>10,.2f}  ({monthly.loc[monthly['revenue'].idxmin(),'period']})")
    print(f"  Max:   ${monthly['revenue'].max():>10,.2f}  ({monthly.loc[monthly['revenue'].idxmax(),'period']})")
    return monthly


def yoy_comparison(df: pd.DataFrame) -> pd.DataFrame:
    
    yoy = df.groupby('year')['revenue'].sum().reset_index()
    yoy['revenue']     = yoy['revenue'].round(2)
    yoy['growth_pct']  = yoy['revenue'].pct_change() * 100
    yoy['growth_pct']  = yoy['growth_pct'].round(2)

    print("\nYear-over-Year Revenue:")
    for _, row in yoy.iterrows():
        g = f"{row['growth_pct']:+.1f}%" if pd.notna(row['growth_pct']) else "—"
        print(f"  {int(row['year'])}: ${row['revenue']:>12,.2f}  {g}")
    return yoy

def category_performance(df: pd.DataFrame) -> pd.DataFrame:
    cat = (
        df.groupby('category')
        .agg(
            revenue=('revenue', 'sum'),
            orders=('transaction_id', 'count'),
            avg_order_value=('revenue', 'mean'),
            total_units=('quantity', 'sum'),
        )
        .reset_index()
    )
    cat['revenue_share_pct'] = (cat['revenue'] / cat['revenue'].sum() * 100).round(1)
    cat['avg_order_value']   = cat['avg_order_value'].round(2)
    cat['revenue']           = cat['revenue'].round(2)
    cat = cat.sort_values('revenue', ascending=False).reset_index(drop=True)

    print("\nCategory Performance (sorted by revenue):")
    print(f"  {'Category':<15} {'Revenue':>12} {'Share':>7} {'Orders':>7} {'AOV':>8}")
    print("  " + "-" * 52)
    for _, r in cat.iterrows():
        print(f"  {r['category']:<15} ${r['revenue']:>11,.0f} {r['revenue_share_pct']:>6.1f}% "
              f"{r['orders']:>7,} ${r['avg_order_value']:>7,.0f}")
    return cat

def regional_sales(df: pd.DataFrame) -> pd.DataFrame:
    reg = (
        df.groupby('region')
        .agg(
            revenue=('revenue', 'sum'),
            orders=('transaction_id', 'count'),
            unique_customers=('customer_id', 'nunique'),
            avg_order_value=('revenue', 'mean'),
        )
        .reset_index()
    )
    reg['revenue_share_pct']       = (reg['revenue'] / reg['revenue'].sum() * 100).round(1)
    reg['revenue_per_customer']    = (reg['revenue'] / reg['unique_customers']).round(2)
    reg = reg.sort_values('revenue', ascending=False).reset_index(drop=True)

    print("\nRegional Sales:")
    for _, r in reg.iterrows():
        print(f"  {r['region']:<10} ${r['revenue']:>10,.0f}  ({r['revenue_share_pct']}%)")
    return reg

def weekday_sales_pattern(df: pd.DataFrame) -> pd.DataFrame:
  
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    wd = (
        df.groupby('weekday')
        .agg(revenue=('revenue', 'sum'), orders=('transaction_id', 'count'))
        .reset_index()
    )
    wd['weekday']    = pd.Categorical(wd['weekday'], categories=day_order, ordered=True)
    wd               = wd.sort_values('weekday').reset_index(drop=True)
    wd['revenue']    = wd['revenue'].round(2)

    peak = wd.loc[wd['revenue'].idxmax(), 'weekday']
    print(f"\nWeekday Sales — Peak day: {peak}")
    return wd


def quarterly_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    q = (
        df.groupby(['year', 'quarter'])['revenue']
        .sum()
        .reset_index()
        .sort_values(['year', 'quarter'])
    )
    q_avg = q.groupby('quarter')['revenue'].mean().reset_index()
    q_avg.columns = ['quarter', 'avg_revenue']
    q_avg['avg_revenue'] = q_avg['avg_revenue'].round(2)
    q_avg['quarter_label'] = q_avg['quarter'].map({1: 'Q1 (Jan-Mar)', 2: 'Q2 (Apr-Jun)',
                                                    3: 'Q3 (Jul-Sep)', 4: 'Q4 (Oct-Dec)'})
    print("\nAvg Revenue by Quarter:")
    for _, r in q_avg.iterrows():
        print(f"  {r['quarter_label']}: ${r['avg_revenue']:>10,.2f}")
    return q_avg

def discount_impact_analysis(df: pd.DataFrame) -> pd.DataFrame:
    disc = (
        df.groupby('discount')
        .agg(
            avg_revenue=('revenue', 'mean'),
            total_orders=('transaction_id', 'count'),
            total_revenue=('revenue', 'sum'),
        )
        .reset_index()
    )
    disc['discount_pct'] = (disc['discount'] * 100).astype(int)
    disc['avg_revenue']  = disc['avg_revenue'].round(2)
    disc['revenue_vs_nodiscount'] = (
        (disc['avg_revenue'] / disc.loc[disc['discount'] == 0, 'avg_revenue'].values[0] - 1) * 100
    ).round(1)

    print("\nDiscount Impact on Avg Order Revenue:")
    print(f"  {'Discount':>9} {'Avg Revenue':>12} {'vs No Discount':>15} {'Orders':>8}")
    print("  " + "-" * 48)
    for _, r in disc.iterrows():
        print(f"  {r['discount_pct']:>8}%  ${r['avg_revenue']:>10,.2f}  "
              f"{r['revenue_vs_nodiscount']:>+13.1f}%  {r['total_orders']:>8,}")
    return disc


def customer_segment_analysis(df: pd.DataFrame) -> pd.DataFrame:

    seg = (
        df.groupby('segment')
        .agg(
            revenue=('revenue', 'sum'),
            orders=('transaction_id', 'count'),
            unique_customers=('customer_id', 'nunique'),
            avg_order_value=('revenue', 'mean'),
        )
        .reset_index()
    )
    seg['revenue_per_customer'] = (seg['revenue'] / seg['unique_customers']).round(2)
    seg['revenue_share_pct']    = (seg['revenue'] / seg['revenue'].sum() * 100).round(1)

    print("\nCustomer Segment Analysis:")
    for _, r in seg.iterrows():
        print(f"  {r['segment']:<10} ${r['revenue']:>10,.0f}  "
              f"{r['unique_customers']:>4} customers  "
              f"${r['revenue_per_customer']:>7,.2f}/customer")
    return seg


def top_customers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Identifies the highest-value customers by total lifetime revenue.

    In retail, the Pareto principle often holds:
    ~20% of customers generate ~80% of revenue.
    This function quantifies that concentration.
    """
    top = (
        df.groupby('customer_id')
        .agg(
            total_revenue=('revenue', 'sum'),
            total_orders=('transaction_id', 'count'),
            avg_order_value=('revenue', 'mean'),
            segment=('segment', 'first'),
            region=('region', 'first'),
        )
        .reset_index()
        .sort_values('total_revenue', ascending=False)
        .head(n)
    )
    top['total_revenue']    = top['total_revenue'].round(2)
    top['avg_order_value']  = top['avg_order_value'].round(2)

    total_rev = df['revenue'].sum()
    top['pct_of_total'] = (top['total_revenue'] / total_rev * 100).round(2)

    print(f"\nTop {n} Customers (% of ${total_rev/1e6:.2f}M total revenue):")
    for i, r in top.iterrows():
        print(f"  {r['customer_id']}  ${r['total_revenue']:>8,.2f}  "
              f"({r['pct_of_total']}%)  {r['segment']:<10}  {r['region']}")
    return top


def run_full_eda(df: pd.DataFrame) -> dict:

    print("\n" + "█" * 55)
    print("  EXPLORATORY DATA ANALYSIS")
    print("█" * 55)
    return {
        'quality_report':      data_quality_report(df),
        'kpis':                compute_kpis(df),
        'monthly_trend':       monthly_revenue_trend(df),
        'yoy':                 yoy_comparison(df),
        'category_perf':       category_performance(df),
        'regional_sales':      regional_sales(df),
        'weekday_pattern':     weekday_sales_pattern(df),
        'quarterly_seasonal':  quarterly_seasonality(df),
        'discount_impact':     discount_impact_analysis(df),
        'segment_analysis':    customer_segment_analysis(df),
        'top_customers':       top_customers(df),
    }
