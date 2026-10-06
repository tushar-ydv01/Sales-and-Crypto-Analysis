"""
Sales Data Generator and Processor module.
Generates rich multi-channel transaction datasets including crypto vs fiat payment flows,
customer cohorts, and regional sales distributions.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

PRODUCT_CATEGORIES = {
    'Enterprise Cloud': {'base_price': 1200, 'margin': 0.78},
    'AI Analytics Suite': {'base_price': 850, 'margin': 0.82},
    'Hardware Nodes': {'base_price': 2400, 'margin': 0.35},
    'Developer Tools': {'base_price': 250, 'margin': 0.88},
    'Consulting & Support': {'base_price': 3500, 'margin': 0.65},
    'Web3 Gateway Pass': {'base_price': 450, 'margin': 0.90}
}

PAYMENT_METHODS = ['Credit Card', 'Crypto (BTC)', 'Crypto (ETH)', 'Crypto (USDT)', 'Bank Wire', 'PayPal']
REGIONS = ['North America', 'Europe', 'Asia-Pacific', 'Latin America', 'Middle East & Africa']
CUSTOMER_TIERS = ['Enterprise', 'Mid-Market', 'Startup', 'Individual']

def generate_sales_data(num_records: int = 5000, start_date: str = None) -> pd.DataFrame:
    """Generate realistic sales transactions data over a 1-year timeline."""
    np.random.seed(42)
    
    if start_date is None:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365)
    else:
        start_dt = pd.to_datetime(start_date)
        end_dt = start_dt + timedelta(days=365)
        
    # Generate dates with weekend variations and holiday quarter bumps
    days_range = (end_dt - start_dt).days
    random_days = np.random.exponential(scale=days_range / 2.5, size=num_records)
    random_days = np.clip(random_days, 0, days_range)
    transaction_dates = [start_dt + timedelta(days=float(d), hours=float(np.random.randint(0, 24))) for d in random_days]
    
    customer_ids = [f"CUST-{np.random.randint(1000, 1800):04d}" for _ in range(num_records)]
    categories = list(PRODUCT_CATEGORIES.keys())
    selected_cats = np.random.choice(categories, size=num_records, p=[0.25, 0.22, 0.15, 0.18, 0.10, 0.10])
    
    quantities = np.random.choice([1, 2, 3, 5, 10], size=num_records, p=[0.60, 0.20, 0.10, 0.07, 0.03])
    
    unit_prices = []
    margins = []
    for cat in selected_cats:
        base = PRODUCT_CATEGORIES[cat]['base_price']
        margin = PRODUCT_CATEGORIES[cat]['margin']
        jitter = np.random.normal(1.0, 0.05)
        unit_prices.append(round(base * jitter, 2))
        margins.append(margin)
        
    unit_prices = np.array(unit_prices)
    quantities = np.array(quantities)
    discounts = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.25], size=num_records, p=[0.55, 0.20, 0.12, 0.08, 0.05])
    
    total_revenue = np.round(unit_prices * quantities * (1 - discounts), 2)
    cost_of_goods = np.round(total_revenue * (1 - np.array(margins)), 2)
    gross_profit = np.round(total_revenue - cost_of_goods, 2)
    
    payment_methods = np.random.choice(
        PAYMENT_METHODS, 
        size=num_records, 
        p=[0.42, 0.15, 0.12, 0.11, 0.12, 0.08]
    )
    
    regions = np.random.choice(REGIONS, size=num_records, p=[0.40, 0.25, 0.20, 0.08, 0.07])
    customer_tiers = np.random.choice(CUSTOMER_TIERS, size=num_records, p=[0.20, 0.35, 0.30, 0.15])
    
    df = pd.DataFrame({
        'Transaction_ID': [f"TX-{100000 + i}" for i in range(num_records)],
        'Date': pd.to_datetime(transaction_dates),
        'Customer_ID': customer_ids,
        'Customer_Tier': customer_tiers,
        'Region': regions,
        'Product_Category': selected_cats,
        'Quantity': quantities,
        'Unit_Price': unit_prices,
        'Discount': discounts,
        'Revenue': total_revenue,
        'COGS': cost_of_goods,
        'Gross_Profit': gross_profit,
        'Payment_Method': payment_methods,
        'Is_Crypto_Payment': [1 if 'Crypto' in p else 0 for p in payment_methods]
    }).sort_values('Date').reset_index(drop=True)
    
    return df

def calculate_rfm_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Customer Recency, Frequency, and Monetary (RFM) Segmentation."""
    if df.empty or 'Customer_ID' not in df.columns:
        return pd.DataFrame()
        
    reference_date = pd.to_datetime(df['Date']).max() + timedelta(days=1)
    
    # Recency and Monetary
    rfm = df.groupby('Customer_ID').agg(
        Recency=('Date', lambda x: (reference_date - pd.to_datetime(x).max()).days),
        Frequency=('Date', 'count'),
        Monetary=('Revenue', 'sum') if 'Revenue' in df.columns else ('Date', 'count')
    ).reset_index()
    
    # Quantile scores (1 to 4)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1], duplicates='drop')
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    
    rfm['RFM_Segment_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    # Segment labels
    def assign_segment(row):
        r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
        if r >= 3 and f >= 3 and m >= 3:
            return 'Champions / VIP'
        elif r >= 3 and f >= 2:
            return 'Loyal Customers'
        elif r >= 3 and f == 1:
            return 'Recent New Buyers'
        elif r <= 2 and f >= 3:
            return 'At Risk / Need Attention'
        elif r <= 2 and f <= 2 and m <= 2:
            return 'Hibernating / Lost'
        else:
            return 'Potential Loyalists'
            
    rfm['Customer_Segment'] = rfm.apply(assign_segment, axis=1)
    return rfm
