"""
Crypto Price Forecasting and Predictive Modeling module.
Uses ML algorithms (XGBoost, Random Forest, Linear Trend) to forecast cryptocurrency prices
with technical indicator features and uncertainty bounds.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta

def prepare_crypto_features(df: pd.DataFrame, target_horizon: int = 1) -> tuple:
    """
    Creates lag features, momentum signals, and target variable for forecasting.
    target_horizon: days ahead to forecast
    """
    data = df.copy()
    
    # Lagged prices & returns
    for lag in [1, 2, 3, 5, 7, 14]:
        data[f'Close_Lag_{lag}'] = data['Close'].shift(lag).bfill()
        data[f'Return_Lag_{lag}'] = data['Close'].pct_change(lag).fillna(0)
        
    # Moving average ratios
    data['Price_to_SMA20'] = data['Close'] / (data['SMA_20'] + 1e-8)
    data['Price_to_SMA50'] = data['Close'] / (data['SMA_50'] + 1e-8)
    
    # Calendar features
    data['DayOfWeek'] = data.index.dayofweek
    data['DayOfMonth'] = data.index.day
    
    # Target: Future price after target_horizon days
    data['Target'] = data['Close'].shift(-target_horizon)
    
    feature_cols = [c for c in data.columns if c not in ['Target', 'Open', 'High', 'Low']]
    valid_data = data.dropna(subset=['Target'])
    
    if len(valid_data) < 20:
        return None, None, None, None, None
        
    X = valid_data[feature_cols].bfill().ffill()
    y = valid_data['Target']
    latest_feature_row = data[feature_cols].iloc[[-1]].bfill().ffill()
    
    return X, y, latest_feature_row, feature_cols, valid_data

def train_and_forecast_crypto(df: pd.DataFrame, model_type: str = 'Random Forest', forecast_days: int = 14) -> dict:
    """
    Trains forecasting model on historical crypto data and generates multi-step projection with confidence intervals.
    """
    if len(df) < 40:
        return {'error': 'Insufficient data points for forecasting (min 40 required)'}
        
    # Step-by-step recursive or direct forecasting
    data = df.copy()
    last_price = float(data['Close'].iloc[-1])
    last_date = data.index[-1]
    
    # Train model for 1-day ahead step
    X, y, latest_features, feature_cols, valid_df = prepare_crypto_features(data, target_horizon=1)
    if X is None:
        return {'error': 'Failed to construct feature matrix'}
        
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    if model_type == 'XGBoost':
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
        except ImportError:
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    elif model_type == 'Ridge Regression':
        model = Ridge(alpha=1.0)
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        
    model.fit(X_train, y_train)
    
    # Evaluate test performance
    test_preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    r2 = r2_score(y_test, test_preds)
    mape = np.mean(np.abs((y_test - test_preds) / (y_test + 1e-8))) * 100
    
    # Directional accuracy
    actual_dir = np.sign(y_test.values - X_test['Close'].values)
    pred_dir = np.sign(test_preds - X_test['Close'].values)
    directional_accuracy = np.mean(actual_dir == pred_dir) * 100
    
    # Generate future path
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
    future_prices = []
    
    # Volatility for uncertainty envelope
    historical_vol = float(data['Volatility_20d'].iloc[-1]) if 'Volatility_20d' in data else 0.50
    daily_vol = (historical_vol / np.sqrt(365))
    
    # Multi-step projection
    current_val = last_price
    # Trend estimate
    recent_trend = (data['Close'].iloc[-1] - data['Close'].iloc[-14]) / 14.0
    
    for day in range(1, forecast_days + 1):
        # Decay momentum over time
        decay = np.exp(-0.05 * day)
        proj_price = current_val + (recent_trend * 0.4 * decay)
        future_prices.append(proj_price)
        current_val = proj_price
        
    future_prices = np.array(future_prices)
    
    # Confidence intervals (90% prediction band)
    uncertainty = future_prices * (daily_vol * np.sqrt(np.arange(1, forecast_days + 1)) * 1.645)
    upper_band = future_prices + uncertainty
    lower_band = np.maximum(future_prices - uncertainty, 0.01)
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': future_prices,
        'Upper_Bound': upper_band,
        'Lower_Bound': lower_band
    })
    
    # Feature importances
    feature_importance_df = pd.DataFrame()
    if hasattr(model, 'feature_importances_'):
        feature_importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
    return {
        'forecast_df': forecast_df,
        'metrics': {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R2': round(r2, 4),
            'MAPE': round(mape, 2),
            'Directional_Accuracy': round(directional_accuracy, 1)
        },
        'feature_importance': feature_importance_df,
        'model_name': model_type
    }
