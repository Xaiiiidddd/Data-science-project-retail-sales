import pandas as pd
import numpy as np


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
  
    reference_date = df['date'].max()

    rfm = (
        df.groupby('customer_id')
        .agg(
            recency   = ('date',           lambda x: (reference_date - x.max()).days),
         
            frequency = ('transaction_id', 'count'),

            monetary  = ('revenue',        'sum'),
        )
        .reset_index()
    )

    rfm['monetary'] = rfm['monetary'].round(2)
    print(f"RFM computed for {len(rfm):,} customers")
    print(f"  Recency range:    {rfm['recency'].min()} – {rfm['recency'].max()} days")
    print(f"  Frequency range:  {rfm['frequency'].min()} – {rfm['frequency'].max()} orders")
    print(f"  Monetary range:   ${rfm['monetary'].min():,.2f} – ${rfm['monetary'].max():,.2f}")
    return rfm


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    rfm = rfm.copy()

   
    rfm['R_score'] = pd.qcut(
        rfm['recency'],
        q=4,
        labels=[4, 3, 2, 1],     
        duplicates='drop'
    ).astype(int)

    rfm['F_score'] = pd.qcut(
        rfm['frequency'].rank(method='first'),
        q=4,
        labels=[1, 2, 3, 4],      
    ).astype(int)

   
    rfm['M_score'] = pd.qcut(
        rfm['monetary'].rank(method='first'),
        q=4,
        labels=[1, 2, 3, 4],      
    ).astype(int)


    rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

    return rfm


def assign_segments(rfm: pd.DataFrame) -> pd.DataFrame:
 
    def classify(score: int) -> str:
        if score >= 10:  return 'Champions'
        elif score >= 8: return 'Loyal'
        elif score >= 6: return 'At Risk'
        else:            return 'Lost'

    rfm['segment'] = rfm['RFM_score'].apply(classify)
    return rfm


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:

    summary = (
        rfm.groupby('segment')
        .agg(
            customer_count=('customer_id', 'count'),
            avg_recency   =('recency',      'mean'),
            avg_frequency =('frequency',    'mean'),
            avg_monetary  =('monetary',     'mean'),
            total_revenue =('monetary',     'sum'),
        )
        .reset_index()
    )
    summary['pct_customers']   = (summary['customer_count'] / summary['customer_count'].sum() * 100).round(1)
    summary['pct_revenue']     = (summary['total_revenue']  / summary['total_revenue'].sum()  * 100).round(1)
    summary['avg_recency']     = summary['avg_recency'].round(1)
    summary['avg_frequency']   = summary['avg_frequency'].round(1)
    summary['avg_monetary']    = summary['avg_monetary'].round(2)

    seg_order = ['Champions', 'Loyal', 'At Risk', 'Lost']
    summary['segment'] = pd.Categorical(summary['segment'], categories=seg_order, ordered=True)
    summary = summary.sort_values('segment').reset_index(drop=True)

    print("\nRFM Segment Summary:")
    print(f"  {'Segment':<12} {'Count':>6} {'%Cust':>7} {'Avg Days':>9} "
          f"{'Avg Orders':>11} {'Avg $':>9} {'%Rev':>7}")
    print("  " + "-" * 65)
    for _, r in summary.iterrows():
        print(f"  {r['segment']:<12} {r['customer_count']:>6} "
              f"{r['pct_customers']:>6.1f}%  {r['avg_recency']:>8.0f}d "
              f"{r['avg_frequency']:>10.1f}  ${r['avg_monetary']:>8,.2f} "
              f"{r['pct_revenue']:>6.1f}%")

    return summary


def run_rfm_segmentation(df: pd.DataFrame) -> dict:
  
    print("\n" + "█" * 55)
    print("  RFM CUSTOMER SEGMENTATION")
    print("█" * 55)

    rfm     = compute_rfm(df)
    rfm     = score_rfm(rfm)
    rfm     = assign_segments(rfm)
    summary = segment_summary(rfm)

    counts = rfm['segment'].value_counts().to_dict()

    return {
        'rfm_table':   rfm,
        'summary':     summary,
        'counts':      counts,
    }
