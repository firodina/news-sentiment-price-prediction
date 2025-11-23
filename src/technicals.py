# src/technicals.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import pynance as pn


class TechnicalAnalysis:
    """
    Computes technical indicators such as SMA, EMA,
    RSI, Bollinger Bands, MACD, returns, volatility, etc.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize with OHLCV dataframe.
        Expected columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        """
        self.df = dataframe.copy()

    # ---------------- DATE PROCESSING ----------------
    def process_dates(self):
        """Convert 'Date' column and set index."""
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.sort_values('Date', inplace=True)
        self.df.set_index('Date', inplace=True)
        return self.df

    # ---------------- MOVING AVERAGES ----------------
    def add_sma(self, period: int):
        """Add Simple Moving Average."""
        self.df[f'SMA_{period}'] = talib.SMA(
            self.df['Close'], timeperiod=period)
        return self.df

    def add_ema(self, period: int):
        """Add Exponential Moving Average."""
        self.df[f'EMA_{period}'] = talib.EMA(
            self.df['Close'], timeperiod=period)
        return self.df

    # ---------------- RSI ----------------
    def add_rsi(self, period: int = 14):
        """Add Relative Strength Index."""
        self.df['RSI'] = talib.RSI(self.df['Close'], timeperiod=period)
        return self.df

    # ---------------- BOLLINGER BANDS ----------------
    def add_bollinger_bands(self, period: int = 20, std: int = 2):
        """Add Bollinger Bands."""
        upper, mid, lower = talib.BBANDS(
            self.df['Close'],
            timeperiod=period,
            nbdevup=std,
            nbdevdn=std
        )
        self.df['Upper_BB'] = upper
        self.df['Middle_BB'] = mid
        self.df['Lower_BB'] = lower
        return self.df

    # ---------------- MACD ----------------
    def add_macd(self):
        """Add MACD indicators."""
        macd, signal, hist = talib.MACD(
            self.df['Close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        self.df['MACD'] = macd
        self.df['MACD_Signal'] = signal
        self.df['MACD_Hist'] = hist
        return self.df

    # ---------------- RETURNS ----------------
    def add_returns(self):
        """Add daily and cumulative returns."""
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['Cumulative_Returns'] = (1 + self.df['Returns']).cumprod()
        return self.df

    # ---------------- VOLATILITY ----------------
    def add_volatility(self, window: int = 30):
        """Add rolling volatility."""
        self.df['Volatility_30'] = self.df['Returns'].rolling(window).std()
        return self.df

    # ---------------- PLOTTING ----------------
    def plot_price_with_ma(self):
        """Plot close price + SMAs + EMAs."""
        plt.figure(figsize=(14, 7))
        plt.plot(self.df['Close'], label='Close Price')

        for col in self.df.columns:
            if col.startswith("SMA") or col.startswith("EMA"):
                plt.plot(self.df[col], label=col, alpha=0.7)

        plt.title("Price with Moving Averages")
        plt.legend()
        plt.show()

    def plot_rsi(self):
        """Plot RSI indicator."""
        plt.figure(figsize=(14, 4))
        plt.plot(self.df['RSI'], label='RSI', color='purple')
        plt.axhline(70, color='red', linestyle='--')
        plt.axhline(30, color='green', linestyle='--')
        plt.title("RSI Indicator")
        plt.show()

    def plot_macd(self):
        """Plot MACD chart."""
        plt.figure(figsize=(14, 5))
        plt.plot(self.df['MACD'], label='MACD')
        plt.plot(self.df['MACD_Signal'], label='Signal')
        plt.bar(self.df.index, self.df['MACD_Hist'], label='Histogram')
        plt.legend()
        plt.title("MACD Indicator")
        plt.show()

    def plot_bollinger(self):
        """Plot Bollinger Bands."""
        plt.figure(figsize=(14, 6))
        plt.plot(self.df['Close'], label='Close Price')
        plt.plot(self.df['Upper_BB'], label='Upper BB')
        plt.plot(self.df['Middle_BB'], label='Middle BB')
        plt.plot(self.df['Lower_BB'], label='Lower BB')
        plt.title("Bollinger Bands")
        plt.legend()
        plt.show()

    def plot_cumulative_returns(self):
        """Plot cumulative returns."""
        plt.figure(figsize=(14, 6))
        plt.plot(self.df['Cumulative_Returns'], label='Cumulative Returns')
        plt.title("Cumulative Returns Over Time")
        plt.legend()
        plt.show()
