import pandas as pd
from src.sentiment import NewsStockCorrelation


def test_load_and_prepare():
    # create dummy news data
    news = pd.DataFrame({
        "headline": [
            "Stock rises after strong earnings",
            "Company faces lawsuit, shares drop",
        ],
        "date": ["2024-01-01", "2024-01-02"],
    })

    # create dummy stock data
    stock = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02"],
        "Close": [100, 102],
        "Open": [99, 101],
        "High": [101, 103],
        "Low": [98, 100],
        "Volume": [1000, 1100],
    })

    # instantiate class
    nsc = NewsStockCorrelation(news_df=news, stock_df=stock)

    # sentiment
    news_with_sentiment = nsc.compute_sentiment(method="vader")
    assert "sentiment_vader" in news_with_sentiment.columns

    # aggregate
    daily_sent = nsc.aggregate_daily_sentiment(method="vader")
    assert "sentiment_mean" in daily_sent.columns

    # returns
    stock_ret = nsc.compute_daily_returns()
    assert "Returns" in stock_ret.columns

    # merge
    merged = nsc.align_and_merge(daily_sent, stock_ret)
    assert "sentiment_mean" in merged.columns

    # correlation
    corr = nsc.compute_correlation(merged)
    assert "r" in corr and "p_value" in corr
