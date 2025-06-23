import yfinance as yf
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt

start_date = "1990-01-02"
end_date = None  # today

# Download VIX data
vix = yf.download("^VIX", start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'VIX_Close'})

# Download S&P 500 data
sp500 = yf.download("^GSPC", start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'SP500_Close'})

# Merge on date index (inner join to keep only dates with both)
data = pd.merge(vix, sp500, left_index=True, right_index=True, how='inner')

# Save to CSV
data.to_csv("vix_sp500_merged.csv")

print(data.head())

plt.rcParams['font.family'] = 'Helvetica'  # Set Helvetica font globally

plt.figure(figsize=(12,6))
plt.plot(data.index, data['VIX_Close'], label='VIX Close', color='red')
plt.plot(data.index, data['SP500_Close'], label='S&P 500 Close', color='blue')

plt.title('VIX and S&P 500 Close Prices (1990 - Present)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()

plt.show()