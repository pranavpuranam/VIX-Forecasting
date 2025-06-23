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

plt.rcParams['font.family'] = 'Arial'

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# VIX plot
axs[0].plot(data.index, data['VIX_Close'], color ='red', linewidth=1.5)  # slightly softer red
axs[0].set_title('VIX Close Price (1990 - Present)', fontsize=16)
axs[0].set_ylabel('VIX Price', fontsize=14)
axs[0].grid(False)
axs[0].tick_params(axis='both', which='major', length=4, width=1)

# S&P 500 plot
axs[1].plot(data.index, data['SP500_Close'], color='blue', linewidth=1.5)  # softer blue
axs[1].set_title('S&P 500 Close Price (1990 - Present)', fontsize=16)
axs[1].set_xlabel('Date', fontsize=14)
axs[1].set_ylabel('S&P 500 Price', fontsize=14)
axs[1].grid(False)
axs[1].tick_params(axis='both', which='major', length=4, width=1)

plt.tight_layout()
plt.savefig("vix_sp500_plots.png", dpi=300)
plt.show()