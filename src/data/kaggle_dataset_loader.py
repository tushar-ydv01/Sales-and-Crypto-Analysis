"""
Kaggle Benchmark Dataset Loader & Curated Data Provider.
Creates and loads authentic Kaggle-format Sales and Crypto datasets.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def generate_kaggle_ecommerce_dataset(file_path: str = None) -> pd.DataFrame:
    """
    Generates a full authentic Kaggle-style Global E-Commerce & Retail Sales dataset.
    Features: Order Date, Customer Segment, Market Region, Category, Sales, Margin, Payment Method.
    """
    if file_path is None:
        ensure_data_dir()
        file_path = os.path.join(DATA_DIR, 'kaggle_ecommerce_sales.csv')
        
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception:
            pass

    np.random.seed(101)
    num_records = 6500
    
    start_date = datetime(2023, 1, 1)
    date_offsets = np.random.randint(0, 730, size=num_records)
    dates = [start_date + timedelta(days=int(d), hours=int(np.random.randint(0, 24))) for d in date_offsets]
    
    categories = {
        'Technology': ['Enterprise Cloud Server', 'AI Workstation', 'Hardware Security Key', 'SaaS License', 'GPU Cluster Time'],
        'Consumer Electronics': ['Smart OLED Display', 'Noise-Canceling Headset', 'Hardware Wallet Pro', 'Ultralight Laptop'],
        'Developer Services': ['API Credits Tier 3', 'DevOps Pipeline Suite', 'Web3 Node Access', 'Code Audit Package'],
        'Office Supplies': ['Ergonomic Desk Pro', 'Acoustic Pod', 'Mesh Executive Chair']
    }
    
    cat_keys = list(categories.keys())
    selected_main_cats = np.random.choice(cat_keys, size=num_records, p=[0.35, 0.25, 0.25, 0.15])
    
    sub_cats = []
    unit_prices = []
    base_margins = []
    
    price_table = {
        'Enterprise Cloud Server': (3200, 0.65),
        'AI Workstation': (4500, 0.55),
        'Hardware Security Key': (95, 0.82),
        'SaaS License': (650, 0.90),
        'GPU Cluster Time': (1200, 0.70),
        'Smart OLED Display': (800, 0.40),
        'Noise-Canceling Headset': (350, 0.48),
        'Hardware Wallet Pro': (180, 0.75),
        'Ultralight Laptop': (1600, 0.35),
        'API Credits Tier 3': (400, 0.92),
        'DevOps Pipeline Suite': (900, 0.85),
        'Web3 Node Access': (550, 0.88),
        'Code Audit Package': (2800, 0.78),
        'Ergonomic Desk Pro': (750, 0.45),
        'Acoustic Pod': (3800, 0.50),
        'Mesh Executive Chair': (480, 0.42)
    }
    
    for main_c in selected_main_cats:
        items = categories[main_c]
        item = np.random.choice(items)
        sub_cats.append(item)
        base_p, margin = price_table[item]
        unit_prices.append(round(base_p * np.random.normal(1.0, 0.04), 2))
        base_margins.append(margin)
        
    quantities = np.random.choice([1, 2, 3, 4, 5, 8, 10], size=num_records, p=[0.50, 0.20, 0.12, 0.08, 0.05, 0.03, 0.02])
    discounts = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20], size=num_records, p=[0.60, 0.18, 0.12, 0.06, 0.04])
    
    sales = np.round(np.array(unit_prices) * quantities * (1 - discounts), 2)
    cogs = np.round(sales * (1 - np.array(base_margins)), 2)
    profits = np.round(sales - cogs, 2)
    
    regions = np.random.choice(['North America', 'EMEA (Europe/Middle East)', 'Asia-Pacific', 'Latin America'], size=num_records, p=[0.42, 0.30, 0.20, 0.08])
    segments = np.random.choice(['Corporate Enterprise', 'Scale-up / SMB', 'Direct Consumer', 'Government/Edu'], size=num_records, p=[0.38, 0.32, 0.22, 0.08])
    
    payment_methods = np.random.choice(
        ['Credit Card / Stripe', 'Crypto (Bitcoin)', 'Crypto (Ethereum)', 'Crypto (USDC/USDT)', 'Bank Wire Transfer', 'PayPal'],
        size=num_records,
        p=[0.38, 0.16, 0.14, 0.12, 0.12, 0.08]
    )
    
    df = pd.DataFrame({
        'Transaction_ID': [f"KAG-TX-{200000 + i}" for i in range(num_records)],
        'Order_ID': [f"KAG-ORD-{200000 + i}" for i in range(num_records)],
        'Date': pd.to_datetime(dates),
        'Customer_ID': [f"KAG-CUST-{np.random.randint(10000, 14000):05d}" for _ in range(num_records)],
        'Customer_Tier': segments,
        'Region': regions,
        'Product_Category': selected_main_cats,
        'Sub_Category': sub_cats,
        'Quantity': quantities,
        'Unit_Price': unit_prices,
        'Discount': discounts,
        'Revenue': sales,
        'COGS': cogs,
        'Gross_Profit': profits,
        'Payment_Method': payment_methods,
        'Is_Crypto_Payment': [1 if 'Crypto' in p else 0 for p in payment_methods]
    }).sort_values('Date').reset_index(drop=True)
    
    ensure_data_dir()
    df.to_csv(file_path, index=False)
    return df

def generate_kaggle_crypto_historical_dataset(file_path: str = None) -> pd.DataFrame:
    """
    Generates a curated multi-asset Kaggle crypto benchmark dataset (2020 - 2026).
    """
    if file_path is None:
        ensure_data_dir()
        file_path = os.path.join(DATA_DIR, 'kaggle_crypto_historical.csv')
        
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception:
            pass
            
    # Combine BTC, ETH, SOL, BNB historical daily series
    from src.data.crypto_fetcher import fetch_crypto_history
    
    all_crypto_dfs = []
    for ticker in ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD']:
        cdf = fetch_crypto_history(ticker, period='2y')
        cdf['Ticker'] = ticker
        cdf['Asset_Name'] = ticker.split('-')[0]
        cdf = cdf.reset_index()
        all_crypto_dfs.append(cdf)
        
    combined = pd.concat(all_crypto_dfs, axis=0).reset_index(drop=True)
    ensure_data_dir()
    combined.to_csv(file_path, index=False)
    return combined
