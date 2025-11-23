# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from wordcloud import WordCloud


class FinancialNewsEDA:
    """
    EDA module for financial news datasets.
    Provides headline stats, publisher analysis, time-series analysis, and word frequency analysis.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the class with a dataframe
        :param dataframe: DataFrame with at least columns ['headline', 'publisher', 'date']
        """
        self.df = dataframe.copy()

    # ---------- BASIC DATA SUMMARY ----------
    def dataset_info(self):
        """Return dataset info and missing values summary."""
        info = self.df.info()
        missing = self.df.isnull().sum()
        return info, missing

    # ---------- HEADLINE LENGTH ----------
    def compute_headline_length(self) -> pd.Series:
        """Compute statistics for headline lengths."""
        self.df['headline_length'] = self.df['headline'].astype(str).apply(len)
        return self.df['headline_length'].describe()

    # ---------- PUBLISHER ANALYSIS ----------
    def top_publishers(self, n=10):
        """Return top N publishers."""
        counts = self.df['publisher'].value_counts().head(n)
        return counts

    def plot_top_publishers(self, n=10):
        """Plot top N publishers."""
        counts = self.top_publishers(n)
        counts.plot(kind="bar", figsize=(10, 5), title=f"Top {n} Publishers")
        plt.ylabel("Article Count")
        plt.show()

    # ---------- DATE PROCESSING ----------
    def process_dates(self):
        """Convert date column and set index."""
        self.df['date'] = pd.to_datetime(
            self.df['date'], format='mixed', utc=True)
        self.df.sort_values('date', inplace=True)
        self.df.set_index('date', inplace=True)
        return self.df

    def plot_article_frequency(self):
        """Plot daily and monthly article counts."""
        daily = self.df.resample('D').size()
        monthly = self.df.resample('M').size()

        plt.figure(figsize=(12, 5))
        daily.plot(title="Daily Article Counts")
        plt.show()

        plt.figure(figsize=(12, 5))
        monthly.plot(title="Monthly Article Counts")
        plt.show()

    # ---------- WORD FREQUENCY ----------
    def top_words(self, n=20):
        """Return top N most common words in headlines."""
        stop_words = stopwords.words('english')
        vectorizer = CountVectorizer(stop_words=stop_words)
        X = vectorizer.fit_transform(self.df['headline'].astype(str))

        word_counts = X.sum(axis=0)
        words_freq = [(word, word_counts[0, idx])
                      for word, idx in vectorizer.vocabulary_.items()]

        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    def plot_wordcloud(self):
        """Generate a word cloud for headlines."""
        wordcloud = WordCloud(width=800, height=400, background_color='white') \
            .generate(" ".join(self.df['headline'].astype(str)))

        plt.figure(figsize=(15, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud of Headlines")
        plt.show()

    # ---------- HOURLY DISTRIBUTION ----------
    def plot_hourly_distribution(self):
        """Plot number of articles published per hour."""
        self.df['hour'] = self.df.index.hour
        hourly_counts = self.df['hour'].value_counts().sort_index()

        plt.figure(figsize=(10, 5))
        hourly_counts.plot(kind='bar')
        plt.title("Articles Published by Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("Number of Articles")
        plt.show()

    # ---------- DOMAIN EXTRACTION ----------
    def extract_domains(self):
        """Extract domain part from publisher emails, if any."""
        if self.df['publisher'].astype(str).str.contains('@').any():
            self.df['domain'] = self.df['publisher'].str.extract(r'@([\w\.]+)')
            return self.df['domain'].value_counts().head(10)
        return None

    def plot_domains(self):
        """Plot top publisher domains."""
        dom = self.extract_domains()
        if dom is not None:
            plt.figure(figsize=(10, 5))
            dom.plot(kind='bar')
            plt.title("Top Publisher Email Domains")
            plt.ylabel("Number of Articles")
            plt.show()
