# tests/test_technicals.py

import unittest
import pandas as pd
import numpy as np
from src.technicals import TechnicalAnalysis


class TestTechnicalAnalysis(unittest.TestCase):

    def setUp(self):
        # Create a small synthetic OHLCV dataset
        data = {
            'Date': pd.date_range(start='2021-01-01', periods=60, freq='D'),
            'Open': np.random.rand(60) * 100,
            'High': np.random.rand(60) * 100,
            'Low': np.random.rand(60) * 100,
            'Close': np.random.rand(60) * 100,
            'Volume': np.random.randint(100000, 500000, size=60),
        }
        df = pd.DataFrame(data)

        self.ta = TechnicalAnalysis(df)

    # ---------------- DATE PROCESSING ----------------
    def test_process_dates(self):
        processed = self.ta.process_dates()
        self.assertTrue(isinstance(processed.index, pd.DatetimeIndex))

    # ---------------- SMA ----------------
    def test_sma(self):
        self.ta.process_dates()
        df = self.ta.add_sma(20)
        self.assertIn('SMA_20', df.columns)

    # ---------------- EMA ----------------
    def test_ema(self):
        self.ta.process_dates()
        df = self.ta.add_ema(20)
        self.assertIn('EMA_20', df.columns)

    # ---------------- RSI ----------------
    def test_rsi(self):
        self.ta.process_dates()
        df = self.ta.add_rsi()
        self.assertIn('RSI', df.columns)

    # ---------------- BOLLINGER BANDS ----------------
    def test_bollinger_bands(self):
        self.ta.process_dates()
        df = self.ta.add_bollinger_bands()
        self.assertIn('Upper_BB', df.columns)
        self.assertIn('Middle_BB', df.columns)
        self.assertIn('Lower_BB', df.columns)

    # ---------------- MACD ----------------
    def test_macd(self):
        self.ta.process_dates()
        df = self.ta.add_macd()
        self.assertIn('MACD', df.columns)
        self.assertIn('MACD_Signal', df.columns)
        self.assertIn('MACD_Hist', df.columns)

    # ---------------- RETURNS ----------------
    def test_returns(self):
        self.ta.process_dates()
        df = self.ta.add_returns()
        self.assertIn('Returns', df.columns)
        self.assertIn('Cumulative_Returns', df.columns)

    # ---------------- VOLATILITY ----------------
    def test_volatility(self):
        self.ta.process_dates()
        self.ta.add_returns()  # volatility depends on returns
        df = self.ta.add_volatility()
        self.assertIn('Volatility_30', df.columns)


if __name__ == "__main__":
    unittest.main()
