"""
NLP Sentiment Analysis Module.
Analyzes sentiment of crypto and market headlines using domain-tuned financial lexicon,
VADER / lexicon scoring, and calculates Fear & Greed Index.
"""
import re
import numpy as np
import pandas as pd

# Crypto-specific sentiment weights
CRYPTO_LEXICON = {
    'surge': 2.5, 'surges': 2.5, 'bullish': 3.0, 'rally': 2.5, 'all-time high': 3.5,
    'ath': 3.5, 'breakout': 2.5, 'inflows': 2.0, 'adoption': 2.5, 'approved': 2.5,
    'soars': 2.5, 'gain': 1.5, 'gains': 1.8, 'growth': 1.5, 'optimism': 2.0,
    'upgrade': 1.8, 'partnership': 2.0, 'institutional': 1.8, 'greed': 1.5,
    'jump': 1.5, 'jumps': 1.8, 'recovery': 1.8, 'resilience': 1.5,
    'crash': -3.5, 'crashes': -3.5, 'bearish': -3.0, 'dump': -3.0, 'plunge': -3.0,
    'plunges': -3.0, 'slump': -2.5, 'drop': -1.8, 'drops': -1.8, 'losses': -2.0,
    'ban': -3.0, 'crackdown': -2.8, 'lawsuit': -2.5, 'hack': -3.5, 'hacked': -3.5,
    'exploit': -3.0, 'scam': -3.5, 'fraud': -3.5, 'liquidation': -2.2, 'panic': -3.0,
    'fear': -2.0, 'fud': -2.5, 'outflows': -2.0, 'decline': -1.5, 'declines': -1.8
}

def analyze_text_sentiment(text: str) -> dict:
    """Analyze polarity score and classification of a text snippet."""
    if not isinstance(text, str) or not text.strip():
        return {'score': 0.0, 'label': 'Neutral', 'confidence': 0.5}
        
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return {'score': 0.0, 'label': 'Neutral', 'confidence': 0.5}
        
    score_total = 0.0
    matched_words = 0
    
    for word in words:
        if word in CRYPTO_LEXICON:
            score_total += CRYPTO_LEXICON[word]
            matched_words += 1
            
    # Check simple bigrams
    text_lower = text.lower()
    for phrase, weight in CRYPTO_LEXICON.items():
        if ' ' in phrase and phrase in text_lower:
            score_total += weight * 1.5
            matched_words += 1
            
    if matched_words > 0:
        normalized_score = np.tanh(score_total / (matched_words * 1.5))
    else:
        # Generic neutral-positive slight bias for general press
        normalized_score = 0.05
        
    if normalized_score >= 0.45:
        label = 'Strong Bullish 🚀'
    elif normalized_score >= 0.15:
        label = 'Bullish 📈'
    elif normalized_score <= -0.45:
        label = 'Strong Bearish 🩸'
    elif normalized_score <= -0.15:
        label = 'Bearish 📉'
    else:
        label = 'Neutral ⚖️'
        
    confidence = min(0.95, 0.5 + abs(normalized_score) * 0.45)
    
    return {
        'score': round(float(normalized_score), 3),
        'label': label,
        'confidence': round(float(confidence), 2)
    }

def enrich_news_with_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment metrics to a dataframe of news headlines."""
    df = news_df.copy()
    if df.empty:
        return df
        
    scores = []
    labels = []
    confidences = []
    
    for _, row in df.iterrows():
        combined_text = f"{row.get('title', '')} {row.get('summary', '')}"
        result = analyze_text_sentiment(combined_text)
        scores.append(result['score'])
        labels.append(result['label'])
        confidences.append(result['confidence'])
        
    df['Sentiment_Score'] = scores
    df['Sentiment_Label'] = labels
    df['Confidence'] = confidences
    return df

def calculate_fear_greed_index(news_df: pd.DataFrame, crypto_returns_7d: float = 0.0) -> dict:
    """
    Calculate market Fear & Greed index (0-100 scale).
    Combines news sentiment (40%) and recent momentum/price action (60%).
    """
    if not news_df.empty and 'Sentiment_Score' in news_df.columns:
        avg_sentiment = news_df['Sentiment_Score'].mean()
    else:
        avg_sentiment = 0.0
        
    # Scale sentiment [-1, 1] to [0, 100]
    sentiment_fgi = (avg_sentiment + 1.0) * 50.0
    
    # Scale returns (e.g. -15% to +15%) to [0, 100]
    momentum_fgi = np.clip(50.0 + (crypto_returns_7d * 2.5), 0, 100)
    
    overall_fgi = int(np.clip(0.40 * sentiment_fgi + 0.60 * momentum_fgi, 0, 100))
    
    if overall_fgi >= 75:
        classification = 'Extreme Greed'
        color = '#00E676'
    elif overall_fgi >= 55:
        classification = 'Greed'
        color = '#26A69A'
    elif overall_fgi >= 45:
        classification = 'Neutral'
        color = '#FFA726'
    elif overall_fgi >= 25:
        classification = 'Fear'
        color = '#EF5350'
    else:
        classification = 'Extreme Fear'
        color = '#D50000'
        
    return {
        'value': overall_fgi,
        'classification': classification,
        'color': color,
        'sentiment_component': round(sentiment_fgi, 1),
        'momentum_component': round(momentum_fgi, 1)
    }
