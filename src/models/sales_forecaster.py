"""
Sales Forecasting and Revenue Driver Analysis module.
Performs time-series sales forecasting, driver decomposition, and impact analysis of payment methods.
"""
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def prepare_sales_time_series(df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
    """Aggregates transactional sales into clean daily/weekly time-series."""
    data = df.copy()
    if data.empty or 'Date' not in data.columns:
        return pd.DataFrame()
        
    data['Date_Only'] = pd.to_datetime(data['Date']).dt.floor(freq)
    
    agg_dict = {}
    if 'Revenue' in data.columns:
        agg_dict['Revenue'] = 'sum'
    if 'Gross_Profit' in data.columns:
        agg_dict['Gross_Profit'] = 'sum'
    if 'Quantity' in data.columns:
        agg_dict['Quantity'] = 'sum'
    if 'Is_Crypto_Payment' in data.columns:
        agg_dict['Is_Crypto_Payment'] = 'mean'
    if 'Discount' in data.columns:
        agg_dict['Discount'] = 'mean'
        
    if not agg_dict:
        agg_dict['Revenue'] = 'sum' if 'Revenue' in data.columns else 'count'
        
    ts = data.groupby('Date_Only').agg(agg_dict).reset_index()
    orders_series = data.groupby('Date_Only').size().reset_index(name='Orders')
    ts = ts.merge(orders_series, on='Date_Only')
    
    rename_dict = {'Date_Only': 'Date'}
    if 'Is_Crypto_Payment' in ts.columns:
        rename_dict['Is_Crypto_Payment'] = 'Crypto_Share'
    ts.rename(columns=rename_dict, inplace=True)
    ts.set_index('Date', inplace=True)
    
    # Fill missing dates with 0
    full_idx = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq=freq)
    ts = ts.reindex(full_idx, fill_value=0)
    ts.index.name = 'Date'
    
    # Feature engineering for time series
    ts['Revenue_MA7'] = ts['Revenue'].rolling(7, min_periods=1).mean()
    ts['Revenue_MA30'] = ts['Revenue'].rolling(30, min_periods=1).mean()
    ts['DayOfWeek'] = ts.index.dayofweek
    ts['DayOfMonth'] = ts.index.day
    ts['Month'] = ts.index.month
    
    # Lags
    for lag in [1, 2, 7, 14, 21, 28]:
        ts[f'Revenue_Lag_{lag}'] = ts['Revenue'].shift(lag)
        
    return ts

def forecast_sales_revenue(df: pd.DataFrame, forecast_horizon: int = 30, model_type: str = 'Gradient Boosting') -> dict:
    """Forecast future sales revenue and order counts over the specified horizon."""
    ts = prepare_sales_time_series(df, freq='D')
    
    feature_cols = [c for c in ts.columns if c not in ['Revenue', 'Gross_Profit', 'Orders', 'Quantity']]
    valid_ts = ts.dropna()
    
    if len(valid_ts) < 30:
        return {'error': 'Need at least 30 days of transactions to produce a forecast'}
        
    X = valid_ts[feature_cols]
    y = valid_ts['Revenue']
    
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    if model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    elif model_type == 'Ridge Regression':
        model = Ridge(alpha=1.0)
    else:
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
        
    model.fit(X_train, y_train)
    
    test_preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    r2 = r2_score(y_test, test_preds)
    
    # Forward simulation
    last_date = ts.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_horizon + 1)]
    
    # Recent average & day of week weights
    dow_multipliers = ts.groupby('DayOfWeek')['Revenue'].mean() / (ts['Revenue'].mean() + 1e-8)
    recent_mean = ts['Revenue'].iloc[-14:].mean()
    
    future_revenues = []
    std_residual = float(np.std(y_test - test_preds)) if len(y_test) > 0 else float(ts['Revenue'].std())
    
    for d in future_dates:
        dow = d.dayofweek
        mult = dow_multipliers.get(dow, 1.0)
        base_pred = recent_mean * mult
        future_revenues.append(base_pred)
        
    future_revenues = np.array(future_revenues)
    uncertainty = 1.96 * std_residual * np.sqrt(1 + np.arange(forecast_horizon) / 30.0)
    
    upper_bound = future_revenues + uncertainty
    lower_bound = np.maximum(future_revenues - uncertainty, 0.0)
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted_Revenue': np.round(future_revenues, 2),
        'Upper_Bound': np.round(upper_bound, 2),
        'Lower_Bound': np.round(lower_bound, 2)
    })
    
    # Driver breakdown
    driver_df = pd.DataFrame()
    if hasattr(model, 'feature_importances_'):
        driver_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(8)
        
    return {
        'forecast_df': forecast_df,
        'metrics': {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R2': round(r2, 4)
        },
        'drivers': driver_df,
        'historical_ts': ts
    }
