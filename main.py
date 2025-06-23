import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import matplotlib.pyplot as plt

col_names = ['Date', 'VIX_Close', 'SP500_Close']
data = pd.read_csv('vix_sp500_merged.csv', skiprows=3, names=col_names, parse_dates=['Date'])
data.set_index('Date', inplace=True)
data.dropna(inplace=True)

endog = data['VIX_Close']
exog = data['SP500_Close']

# Because d usually is 0 or 1 for financial data, try both
p_values = range(0, 4)
d_values = range(0, 2)
q_values = range(0, 4)

best_aic = float('inf')
best_order = None
best_model = None

warnings.filterwarnings("ignore")  # to suppress convergence warnings

for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                model = SARIMAX(endog, exog=exog, order=(p,d,q))
                results = model.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p,d,q)
                    best_model = results
            except Exception:
                continue

print(f"Best ARIMAX order: {best_order} with AIC: {best_aic}")

# You can plot diagnostics on the best model
best_model.plot_diagnostics(figsize=(12,8))
plt.show()

