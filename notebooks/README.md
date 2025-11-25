---

## 📝 Task 1 Requirements

### ✔ 1. **Descriptive Statistics**

* Calculate headline lengths
* Count articles per publisher
* Check missing values
* Analyze date distribution

### ✔ 2. **Text Analysis (Keyword Exploration)**

* Find most common words
* Optionally create a word cloud
* Extract bigrams/trigrams (optional extension)

### ✔ 3. **Time Series Analysis**

* Articles published per day
* Articles published per month
* Publishing time-of-day patterns

### ✔ 4. **Publisher Analysis**

* Most active publishers
* If publishers include emails → extract domains

---

## 🧪 How to Run the EDA Notebook

1. Install the virtual environment (Python 3.11 recommended):

```bash
python -m venv venv
source venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

---

## 📝 Task 2 Requirements

### ✔ 1. **Stock Data Preparation**

* Load historical stock price data (Open, High, Low, Close, Volume)
* Preprocess missing values and sort by date
* Set datetime index for time series analysis

### ✔ 2. **Technical Indicators (TA-Lib)**

* Calculate Simple Moving Averages (SMA 20, 50)
* Calculate Exponential Moving Average (EMA 20)
* Compute Relative Strength Index (RSI 14)
* Compute MACD (MACD line, signal line, histogram)
* Compute Bollinger Bands (Upper, Middle, Lower)

### ✔ 3. **Financial Metrics (PyNance)**

* Calculate daily returns and cumulative returns
* Compute rolling volatility (30-day)
* Prepare metrics for correlation with sentiment analysis

### ✔ 4. **Visualizations**

* Plot closing price with SMA & EMA
* RSI indicator with thresholds
* MACD indicator with histogram
* Bollinger Bands over closing price
* Cumulative returns over time

### 🧪 How to Run the Quantitative Analysis Notebook

```bash
python -m venv venv
source venv/Scripts/activate   # Windows
pip install -r requirements.txt
jupyter notebook notebooks/task_2_quant_analysis.ipynb
```

This notebook will execute all steps from stock data preparation, technical indicators, financial metrics calculation, and visualizations.
