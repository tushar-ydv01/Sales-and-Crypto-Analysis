import src.data.crypto_fetcher as cf
import src.data.sales_data_generator as sg
import src.nlp.sentiment_analyzer as sa
import src.models.crypto_forecaster as cfc
import src.models.sales_forecaster as sfc
import src.data.news_scraper as ns

print("1. Testing Crypto Fetcher...")
df_c = cf.fetch_crypto_history('BTC-USD', period='6mo')
print(f"   Crypto data shape: {df_c.shape}")

print("2. Testing Sales Data Generator...")
df_s = sg.generate_sales_data(1000)
print(f"   Sales data shape: {df_s.shape}")

print("3. Testing News Scraper & Sentiment Analyzer...")
news = ns.fetch_crypto_news(limit=5)
news_enriched = sa.enrich_news_with_sentiment(news)
print(f"   Scraped {len(news_enriched)} news items with sentiment.")

print("4. Testing Crypto Forecaster (Random Forest)...")
fres = cfc.train_and_forecast_crypto(df_c, model_type='Random Forest', forecast_days=14)
print(f"   Forecaster metrics: {fres.get('metrics')}")

print("5. Testing Sales Forecaster (Gradient Boosting)...")
sfres = sfc.forecast_sales_revenue(df_s, forecast_horizon=30)
print(f"   Sales Forecaster metrics: {sfres.get('metrics')}")

print("\n>>> ALL SYSTEM TESTS PASSED CLEANLY! <<<")
