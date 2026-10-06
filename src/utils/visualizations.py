"""
Interactive Visualization Engine using Plotly.
Provides dark-themed, glassmorphic financial & sales charts.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np

# Premium Theme Tokens
THEME_COLORS = {
    'bg': '#0F172A',
    'card_bg': '#1E293B',
    'text': '#F8FAFC',
    'text_muted': '#94A3B8',
    'grid': '#334155',
    'accent_bull': '#10B981', # emerald
    'accent_bear': '#EF4444', # rose
    'accent_primary': '#6366F1', # indigo
    'accent_secondary': '#EC4899', # pink
    'accent_cyan': '#06B6D4',
    'accent_amber': '#F59E0B'
}

def get_base_layout(title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(family="Inter, sans-serif", size=18, color=THEME_COLORS['text'])),
        paper_bgcolor='rgba(15, 23, 42, 0.6)',
        plot_bgcolor='rgba(15, 23, 42, 0.8)',
        font=dict(family="Inter, sans-serif", color=THEME_COLORS['text_muted']),
        margin=dict(l=40, r=30, t=50, b=40),
        xaxis=dict(showgrid=True, gridcolor=THEME_COLORS['grid'], zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=THEME_COLORS['grid'], zeroline=False),
        legend=dict(font=dict(color=THEME_COLORS['text']), orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

def plot_crypto_candlestick(df: pd.DataFrame, ticker_name: str = 'BTC-USD', show_indicators: bool = True) -> go.Figure:
    """Create a candlestick chart with optional Bollinger Bands, SMA, and MACD/RSI subplots."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=(f'{ticker_name} Price Action & Indicators', 'MACD Momentum', 'RSI (14)')
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC',
        increasing_line_color=THEME_COLORS['accent_bull'],
        decreasing_line_color=THEME_COLORS['accent_bear']
    ), row=1, col=1)
    
    if show_indicators:
        if 'SMA_20' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color=THEME_COLORS['accent_cyan'], width=1.2), name='SMA 20'), row=1, col=1)
        if 'SMA_50' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color=THEME_COLORS['accent_amber'], width=1.2), name='SMA 50'), row=1, col=1)
        if 'BB_Upper' in df and 'BB_Lower' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(99, 102, 241, 0.3)', width=1, dash='dot'), name='BB Upper', showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(99, 102, 241, 0.3)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(99, 102, 241, 0.05)', name='BB Band', showlegend=False), row=1, col=1)
            
        # MACD
        if 'MACD' in df and 'MACD_Signal' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color=THEME_COLORS['accent_cyan'], width=1.5), name='MACD'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color=THEME_COLORS['accent_secondary'], width=1.5), name='Signal'), row=2, col=1)
            colors = [THEME_COLORS['accent_bull'] if val >= 0 else THEME_COLORS['accent_bear'] for val in df['MACD_Hist'].fillna(0)]
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=colors, name='MACD Hist', showlegend=False), row=2, col=1)
            
        # RSI
        if 'RSI' in df:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#A855F7', width=1.5), name='RSI'), row=3, col=1)
            fig.add_hline(y=70, line=dict(color=THEME_COLORS['accent_bear'], dash='dash', width=1), row=3, col=1)
            fig.add_hline(y=30, line=dict(color=THEME_COLORS['accent_bull'], dash='dash', width=1), row=3, col=1)
            
    fig.update_layout(
        **get_base_layout(),
        xaxis_rangeslider_visible=False,
        height=680
    )
    return fig

def plot_crypto_forecast_chart(historical_df: pd.DataFrame, forecast_df: pd.DataFrame, ticker_name: str = 'BTC-USD') -> go.Figure:
    """Plot historical price with future ML forecast and uncertainty corridor."""
    fig = go.Figure()
    
    # Recent history (last 60 days)
    recent = historical_df.tail(60)
    fig.add_trace(go.Scatter(
        x=recent.index,
        y=recent['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color=THEME_COLORS['accent_cyan'], width=2.5)
    ))
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Predicted_Price'],
        mode='lines+markers',
        name='ML Forecast',
        line=dict(color=THEME_COLORS['accent_amber'], width=2.5, dash='dash')
    ))
    
    # Confidence bounds
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Upper_Bound'],
        mode='lines',
        line=dict(color='rgba(245, 158, 11, 0.2)', width=0),
        name='Upper 90% Bound',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Lower_Bound'],
        mode='lines',
        line=dict(color='rgba(245, 158, 11, 0.2)', width=0),
        fill='tonexty',
        fillcolor='rgba(245, 158, 11, 0.15)',
        name='90% Confidence Interval'
    ))
    
    fig.update_layout(
        **get_base_layout(f'{ticker_name} Machine Learning Price Forecast'),
        height=480,
        hovermode='x unified'
    )
    return fig

def plot_sales_trend(ts_df: pd.DataFrame, forecast_df: pd.DataFrame = None) -> go.Figure:
    """Plot revenue and gross profit timeline with moving averages."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ts_df.index,
        y=ts_df['Revenue'],
        name='Daily Revenue',
        line=dict(color='rgba(99, 102, 241, 0.4)', width=1),
        mode='lines'
    ))
    
    if 'Revenue_MA7' in ts_df:
        fig.add_trace(go.Scatter(
            x=ts_df.index,
            y=ts_df['Revenue_MA7'],
            name='7-Day Moving Avg',
            line=dict(color=THEME_COLORS['accent_cyan'], width=2.5)
        ))
        
    if 'Revenue_MA30' in ts_df:
        fig.add_trace(go.Scatter(
            x=ts_df.index,
            y=ts_df['Revenue_MA30'],
            name='30-Day Moving Avg',
            line=dict(color=THEME_COLORS['accent_secondary'], width=2)
        ))
        
    if forecast_df is not None:
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecasted_Revenue'],
            name='Revenue Projection',
            line=dict(color=THEME_COLORS['accent_bull'], width=2.5, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Upper_Bound'],
            mode='lines',
            line=dict(color='rgba(16, 185, 129, 0.2)', width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Lower_Bound'],
            mode='lines',
            line=dict(color='rgba(16, 185, 129, 0.2)', width=0),
            fill='tonexty',
            fillcolor='rgba(16, 185, 129, 0.15)',
            name='Forecast Range'
        ))
        
    fig.update_layout(
        **get_base_layout('Revenue Performance & Forward Projection ($)'),
        height=450,
        hovermode='x unified'
    )
    return fig

def plot_payment_distribution(df: pd.DataFrame) -> go.Figure:
    """Donut chart for payment method share."""
    agg = df.groupby('Payment_Method')['Revenue'].sum().reset_index()
    
    fig = px.pie(
        agg, 
        values='Revenue', 
        names='Payment_Method', 
        hole=0.55,
        color_discrete_sequence=['#6366F1', '#F59E0B', '#10B981', '#06B6D4', '#EC4899', '#8B5CF6']
    )
    
    fig.update_layout(
        **get_base_layout('Revenue by Payment Channel'),
        height=380,
        showlegend=True
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_category_breakdown(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing revenue vs profit by product category."""
    agg = df.groupby('Product_Category').agg({'Revenue': 'sum', 'Gross_Profit': 'sum'}).reset_index()
    agg = agg.sort_values('Revenue', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=agg['Product_Category'],
        x=agg['Revenue'],
        name='Revenue ($)',
        orientation='h',
        marker=dict(color=THEME_COLORS['accent_primary'])
    ))
    fig.add_trace(go.Bar(
        y=agg['Product_Category'],
        x=agg['Gross_Profit'],
        name='Gross Profit ($)',
        orientation='h',
        marker=dict(color=THEME_COLORS['accent_bull'])
    ))
    
    fig.update_layout(
        **get_base_layout('Category Revenue vs Gross Profit'),
        barmode='group',
        height=380
    )
    return fig

def plot_fear_and_greed_gauge(score: int, classification: str, color: str) -> go.Figure:
    """Speedometer gauge for Fear & Greed."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{classification}</b>", 'font': {'size': 20, 'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': THEME_COLORS['text_muted']},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': 'rgba(30, 41, 59, 0.8)',
            'borderwidth': 2,
            'bordercolor': THEME_COLORS['grid'],
            'steps': [
                {'range': [0, 25], 'color': 'rgba(213, 0, 0, 0.3)'},
                {'range': [25, 45], 'color': 'rgba(239, 83, 80, 0.3)'},
                {'range': [45, 55], 'color': 'rgba(255, 167, 38, 0.3)'},
                {'range': [55, 75], 'color': 'rgba(38, 166, 154, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(0, 230, 118, 0.3)'},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=THEME_COLORS['text'], family="Inter, sans-serif"),
        height=280,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    return fig

def plot_feature_importance(importance_df: pd.DataFrame, title: str = "Key Drivers & Predictors") -> go.Figure:
    """Plot model feature importance."""
    if importance_df.empty:
        return go.Figure()
        
    df_sorted = importance_df.sort_values('Importance', ascending=True)
    fig = go.Figure(go.Bar(
        x=df_sorted['Importance'],
        y=df_sorted['Feature'],
        orientation='h',
        marker=dict(
            color=df_sorted['Importance'],
            colorscale='Viridis',
            showscale=False
        )
    ))
    layout_args = get_base_layout(title)
    layout_args['margin'] = dict(l=120, r=20, t=40, b=30)
    layout_args['height'] = 320
    fig.update_layout(**layout_args)
    return fig
