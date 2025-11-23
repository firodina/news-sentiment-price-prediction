# tests/test_eda.py

import unittest
import pandas as pd
from src.eda import FinancialNewsEDA


class TestEDA(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'headline': ['Stocks rally', 'Market crashes'],
            'publisher': ['Reuters', 'Bloomberg'],
            'date': ['2022-01-01', '2022-01-02']
        })
        self.eda = FinancialNewsEDA(self.df)

    def test_headline_length(self):
        stats = self.eda.compute_headline_length()
        self.assertIn('mean', stats)

    def test_top_publishers(self):
        top = self.eda.top_publishers(1)
        self.assertEqual(len(top), 1)

    def test_process_dates(self):
        df_processed = self.eda.process_dates()
        self.assertTrue(isinstance(df_processed.index, pd.DatetimeIndex))


if __name__ == "__main__":
    unittest.main()
