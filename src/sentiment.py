# src/sentiment.py

import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os

# Ensure necessary NLTK assets
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")


class NewsStockCorrelation:
    """
    Pipeline to compute correlation between news headline sentiment and stock daily returns.
    """

    def __init__(self, news_df: pd.DataFrame = None, stock_df: pd.DataFrame = None):
        """
        :param news_df: DataFrame with at least ['headline', 'date'].
        :param stock_df: OHLCV DataFrame with Date column (e.g., 'Date' or index).
        """
        self.news = news_df.copy() if news_df is not None else None
        self.stock = stock_df.copy() if stock_df is not None else None
        # initialize VADER
        self.vader = SentimentIntensityAnalyzer()

    # ---------------- Data loading helpers ----------------
    @staticmethod
    def load_news(path: str, date_col: str = "date") -> pd.DataFrame:
        df = pd.read_csv(path)
        if date_col not in df.columns:
            raise ValueError(f"news file must contain a '{date_col}' column")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])  # drop rows with bad dates
        df["date"] = df[date_col].dt.date  # normalize to date (no time)
        return df

    @staticmethod
    def load_stock(path: str, date_col: str = "Date") -> pd.DataFrame:
        df = pd.read_csv(path)
        if date_col not in df.columns:
            raise ValueError(f"stock file must contain a '{date_col}' column")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col)
        df = df.set_index(date_col)
        # keep core columns if present
        required = ["Open", "High", "Low", "Close", "Volume"]
        for c in required:
            if c not in df.columns:
                # not strictly fatal, but warn
                pass
        df.index = pd.DatetimeIndex(df.index)
        return df

    # ---------------- Sentiment ----------------
    def compute_sentiment(self, text_col: str = "headline", method: str = "vader"):
        """
        Adds sentiment columns to self.news:
          - 'sentiment_vader' : compound VADER score in [-1,1]
          - 'sentiment_textblob' : polarity in [-1,1]

        :param text_col: column name containing headline text
        :param method: 'vader' (default) or 'textblob' or 'both'
        """
        if self.news is None:
            raise ValueError("news dataframe not set")

        def tb_score(s):
            try:
                return TextBlob(str(s)).sentiment.polarity
            except Exception:
                return np.nan

        def vader_score(s):
            try:
                return self.vader.polarity_scores(str(s))["compound"]
            except Exception:
                return np.nan

        if method in ("vader", "both"):
            self.news["sentiment_vader"] = self.news[text_col].apply(
                vader_score)
        if method in ("textblob", "both"):
            self.news["sentiment_textblob"] = self.news[text_col].apply(
                tb_score)

        return self.news

    # ---------------- Aggregate daily sentiment ----------------
    def aggregate_daily_sentiment(self, method: str = "vader", date_col: str = "date") -> pd.DataFrame:
        """
        Aggregates sentiment per day (mean) and returns dataframe with columns:
         ['date', 'sentiment_mean', 'n_articles']
        """
        if self.news is None:
            raise ValueError("news dataframe not set")
        if method == "vader":
            score_col = "sentiment_vader"
        elif method == "textblob":
            score_col = "sentiment_textblob"
        else:
            raise ValueError("method must be 'vader' or 'textblob'")

        df = self.news[[date_col, score_col]].dropna()
        agg = df.groupby(date_col).agg(
            sentiment_mean=(score_col, "mean"),
            sentiment_median=(score_col, "median"),
            n_articles=(score_col, "count"),
        ).reset_index()
        # convert to datetime index to match stock later
        agg["date"] = pd.to_datetime(agg[date_col])
        agg = agg.set_index("date")
        return agg

    # ---------------- Stock returns ----------------
    def compute_daily_returns(self, close_col: str = "Close") -> pd.DataFrame:
        """
        Adds 'Returns' and 'Cumulative_Returns' to stock dataframe and returns a stock copy.
        """
        if self.stock is None:
            raise ValueError("stock dataframe not set")
        stock = self.stock.copy()
        stock["Returns"] = stock[close_col].pct_change()
        stock["Cumulative_Returns"] = (1 + stock["Returns"]).cumprod()
        return stock

    # ---------------- Align and merge ----------------
    def align_and_merge(self, daily_sentiment: pd.DataFrame, stock_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Merge daily_sentiment (indexed by date) with stock_returns (DatetimeIndex).
        Returns DataFrame with both 'sentiment_mean' and 'Returns' aligned by trading days.
        """
        # ensure both are datetime-indexed
        sent = daily_sentiment.copy()
        stock = stock_returns.copy()

        # Align: merge on dates (inner join)
        merged = pd.merge(
            stock,
            sent[["sentiment_mean", "sentiment_median", "n_articles"]],
            left_index=True,
            right_index=True,
            how="left",
        )

        # Option: forward/backfill to assign same-day news to next trading day if needed:
        # merged['sentiment_mean'] = merged['sentiment_mean'].fillna(method='ffill')

        return merged

    # ---------------- Correlation ----------------
    @staticmethod
    def compute_correlation(merged_df: pd.DataFrame, sentiment_col: str = "sentiment_mean", returns_col: str = "Returns"):
        """
        Compute Pearson correlation and p-value between daily sentiment and returns.
        Ignores NA pairs.
        """
        df = merged_df[[sentiment_col, returns_col]].dropna()
        if df.shape[0] < 2:
            return {"r": np.nan, "p_value": np.nan, "n": df.shape[0]}
        r, p = pearsonr(df[sentiment_col], df[returns_col])
        return {"r": float(r), "p_value": float(p), "n": int(df.shape[0])}

    # ---------------- Plot ----------------
    @staticmethod
    def plot_sentiment_vs_returns(merged_df: pd.DataFrame, sentiment_col: str = "sentiment_mean", returns_col: str = "Returns"):
        df = merged_df.dropna(subset=[sentiment_col, returns_col])
        plt.figure(figsize=(8, 6))
        plt.scatter(df[sentiment_col], df[returns_col], alpha=0.6)
        plt.xlabel("Average Daily Sentiment")
        plt.ylabel("Daily Returns")
        plt.title("Daily Sentiment vs Daily Returns")
        plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        plt.axvline(0, color="gray", linestyle="--", linewidth=0.5)
        plt.show()

    # ---------------- End-to-end runner ----------------
    def run_pipeline(
        self,
        news_path: str,
        stock_path: str,
        news_date_col: str = "date",
        stock_date_col: str = "Date",
        sentiment_method: str = "vader",
        close_col: str = "Close",
        save_merged_to: str = None,
    ):
        """
        Orchestrates the full pipeline and returns results dict:
          {'merged': merged_df, 'corr': {r,p_value,n}, 'daily_sentiment': daily_sentiment}
        """
        # load
        self.news = self.load_news(news_path, date_col=news_date_col)
        self.stock = self.load_stock(stock_path, date_col=stock_date_col)

        # sentiment
        self.compute_sentiment(text_col="headline", method=(
            "both" if sentiment_method == "both" else sentiment_method))

        # aggregate
        daily_sent = self.aggregate_daily_sentiment(
            method=sentiment_method, date_col=news_date_col)

        # returns
        stock_ret = self.compute_daily_returns(close_col=close_col)

        # merge
        merged = self.align_and_merge(
            daily_sentiment=daily_sent, stock_returns=stock_ret)

        if save_merged_to:
            dirname = os.path.dirname(save_merged_to)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            merged.to_csv(save_merged_to)

        # correlation
        corr = self.compute_correlation(
            merged, sentiment_col="sentiment_mean", returns_col="Returns")

        return {"merged": merged, "corr": corr, "daily_sentiment": daily_sent}
