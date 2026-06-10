import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
   
    txn_features = (
        df.groupby('customer_id')
        .agg(

            total_revenue=('revenue', 'sum'),

            total_orders=('transaction_id', 'count'),

            avg_order_value=('revenue', 'mean'),

            avg_quantity=('quantity', 'mean'),

            avg_discount=('discount', 'mean'),

            unique_categories=('category', 'nunique'),

            max_order_value=('revenue', 'max'),

            std_order_value=('revenue', 'std'),

            last_purchase_date=('date', 'max'),

            first_purchase_date=('date', 'min'),
        )
        .reset_index()
    )

    txn_features['std_order_value'] = txn_features['std_order_value'].fillna(0)

    txn_features['revenue_per_order'] = (
        txn_features['total_revenue'] / txn_features['total_orders']
    ).round(2)

    txn_features['tenure_days'] = (
        txn_features['last_purchase_date'] - txn_features['first_purchase_date']
    ).dt.days
  
    txn_features['purchase_rate'] = np.where(
        txn_features['tenure_days'] > 0,
        txn_features['total_orders'] / txn_features['tenure_days'],
        txn_features['total_orders']
    ).round(4)
  
    demographics = df[['customer_id', 'age', 'region', 'segment', 'loyalty_years']].drop_duplicates('customer_id')
    customer_features = txn_features.merge(demographics, on='customer_id', how='left')

    le_region  = LabelEncoder()
    le_segment = LabelEncoder()

    customer_features['region_enc']  = le_region.fit_transform(customer_features['region'])
    customer_features['segment_enc'] = le_segment.fit_transform(customer_features['segment'])

    for col in ['total_revenue', 'avg_order_value', 'max_order_value',
                'std_order_value', 'avg_discount', 'loyalty_years']:
        customer_features[col] = customer_features[col].round(2)

    print(f"✅ Feature engineering complete: {len(customer_features)} customers × "
          f"{len(customer_features.columns)} columns")

    return customer_features


def get_feature_columns() -> list:
 
    return [
        'total_orders',
        'avg_order_value',
        'avg_quantity',
        'avg_discount',
        'unique_categories',
        'max_order_value',
        'std_order_value',
        'age',
        'loyalty_years',
        'region_enc',
        'segment_enc',
    ]


def prepare_X_y(customer_features: pd.DataFrame):
  
    feature_cols = get_feature_columns()
    X = customer_features[feature_cols]
    y = customer_features['total_revenue']

    print(f"  X shape: {X.shape}  (samples × features)")
    print(f"  y range: ${y.min():,.2f} → ${y.max():,.2f}  (mean: ${y.mean():,.2f})")

    return X, y
