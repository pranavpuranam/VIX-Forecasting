import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import matplotlib.pyplot as plt

col_names = ['Date', 'VIX_Close', 'SP500_Close']
data = pd.read_csv('vix_sp500_merged.csv', skiprows=3, names=col_names, parse_dates=['Date'])
data.set_index('Date', inplace=True)
data.dropna(inplace=True)

endog = data['VIX_Close']

p_values = range(0, 4)
d_values = range(0, 2)
q_values = range(0, 4)

best_aic = float('inf')
best_order = None
best_model = None

warnings.filterwarnings("ignore")

for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                model = SARIMAX(endog, order=(p,d,q))
                results = model.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p,d,q)
                    best_model = results
            except Exception:
                continue

print(f"Best ARIMA order: {best_order} with AIC: {best_aic}")

n_steps = 20
forecast = best_model.get_forecast(steps=n_steps)
forecast_index = pd.date_range(start=endog.index[-1] + pd.Timedelta(days=1), periods=n_steps, freq='B')

forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

past_days = 30
observed_slice = endog[-past_days:]

plt.rcParams['font.family'] = 'Arial'  # Set font to Arial

plt.figure(figsize=(12,6))
plt.plot(observed_slice.index, observed_slice, label='Observed VIX (last 30 days)', color='black')
plt.plot(forecast_index, forecast_mean, label='Forecasted VIX (next 20 days)', color='red')
plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='red', alpha=0.3)

plt.xlabel('Date')
plt.ylabel('VIX_Close')
plt.title('VIX Close: Last 30 Days Observed and 20 Days Forecasted (ARIMA)')
plt.legend()
plt.show()