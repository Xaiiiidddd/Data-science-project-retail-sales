import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)


N_CUSTOMERS    = 500
N_TRANSACTIONS = 5000
START_DATE     = datetime(2022, 1, 1)
END_DATE       = datetime(2024, 12, 31)

CATEGORIES = ['Electronics', 'Clothing', 'Groceries',
              'Home & Garden', 'Sports', 'Books', 'Beauty']
REGIONS    = ['North', 'South', 'East', 'West', 'Central']
SEGMENTS   = ['Premium', 'Regular', 'Budget']


CATEGORY_WEIGHTS = [0.15, 0.20, 0.25, 0.15, 0.10, 0.08, 0.07]


SEGMENT_WEIGHTS = [0.20, 0.50, 0.30]

CATEGORY_PRICE_RANGE = {
    'Electronics':   (80,  1200),
    'Clothing':      (20,  300),
    'Groceries':     (10,  150),
    'Home & Garden': (30,  500),
    'Sports':        (25,  400),
    'Books':         (10,  80),
    'Beauty':        (15,  200),
}


def generate_customers() -> pd.DataFrame:

    return pd.DataFrame({
        'customer_id': [f'C{str(i).zfill(4)}' for i in range(1, N_CUSTOMERS + 1)],
        'age':          np.random.randint(18, 70, N_CUSTOMERS),
        'region':       np.random.choice(REGIONS, N_CUSTOMERS),
        'segment':      np.random.choice(SEGMENTS, N_CUSTOMERS, p=SEGMENT_WEIGHTS),
        'loyalty_years': np.random.exponential(3, N_CUSTOMERS).clip(0.1, 15).round(1),
    })


def generate_transactions(customers: pd.DataFrame) -> pd.DataFrame:
  
    customer_ids = np.random.choice(customers['customer_id'], N_TRANSACTIONS)


    cats = np.random.choice(CATEGORIES, N_TRANSACTIONS, p=CATEGORY_WEIGHTS)

    total_days = (END_DATE - START_DATE).days
    dates = [
        START_DATE + timedelta(days=int(d))
        for d in np.random.randint(0, total_days, N_TRANSACTIONS)
    ]


    prices = np.array([
        round(np.random.uniform(*CATEGORY_PRICE_RANGE[c]), 2)
        for c in cats
    ])

    quantities = np.random.randint(1, 6, N_TRANSACTIONS)

   
    discounts = np.random.choice(
        [0.0, 0.05, 0.10, 0.15, 0.20],
        N_TRANSACTIONS,
        p=[0.50, 0.20, 0.15, 0.10, 0.05]
    )

    revenue = (prices * quantities * (1 - discounts)).round(2)

    df = pd.DataFrame({
        'transaction_id': [f'T{str(i).zfill(5)}' for i in range(1, N_TRANSACTIONS + 1)],
        'customer_id':    customer_ids,
        'date':           pd.to_datetime(dates),
        'category':       cats,
        'quantity':       quantities,
        'unit_price':     prices,
        'discount':       discounts,
        'revenue':        revenue,
    })

    df['month']   = df['date'].dt.month
    df['year']    = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['weekday'] = df['date'].dt.day_name()

    return df


def build_dataset(customers: pd.DataFrame,
                  transactions: pd.DataFrame) -> pd.DataFrame:
   
    return transactions.merge(customers, on='customer_id', how='left')


def generate_and_save(output_dir: str = '.') -> pd.DataFrame:
   
    os.makedirs(output_dir, exist_ok=True)

    customers    = generate_customers()
    transactions = generate_transactions(customers)
    df           = build_dataset(customers, transactions)

    customers.to_csv(os.path.join(output_dir, 'customers.csv'), index=False)
    transactions.to_csv(os.path.join(output_dir, 'transactions.csv'), index=False)
    df.to_csv(os.path.join(output_dir, 'retail_data.csv'), index=False)

    print(f"✅ Dataset generated: {len(df):,} rows × {len(df.columns)} columns")
    print(f"   Saved to: {output_dir}/")
    return df


if __name__ == '__main__':
    generate_and_save(output_dir='.')
