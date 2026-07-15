import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load data
df = pd.read_csv("/content/traffic.csv", encoding='latin-1')
df['DateTime'] = pd.to_datetime(df['DateTime'])


#  BEFORE RESAMPLING PLOT

plt.figure(figsize=(10, 5))
plt.plot(df['DateTime'], df['Vehicles'])
plt.xlabel("DateTime")
plt.ylabel("Vehicles")
plt.title("Traffic Before Resampling")
plt.show()

# DATA PREPARATION

df_prophet = df[['DateTime', 'Vehicles']]
df_prophet.columns = ['ds', 'y']
df_prophet.set_index('ds', inplace=True)
df_prophet = df_prophet.resample('D').sum()
df_prophet.reset_index(inplace=True)

# Train-test split
size = 80
train = df_prophet[:-size]
test = df_prophet[-size:]


# MODEL TRAINING

model = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.9)
model.fit(train)


# PREDICTION

future = model.make_future_dataframe(periods=size)
forecast = model.predict(future)

pred = forecast.tail(size)


#  PREDICTED OUTPUT PLOT

plt.figure(figsize=(10, 5))
plt.plot(test['ds'], test['y'], label='Actual')
plt.plot(pred['ds'], pred['yhat'], color='red', label='Predicted')
plt.fill_between(
    pred['ds'],
    pred['yhat_lower'],
    pred['yhat_upper'],
    color='orange',
    alpha=0.3,
    label='Confidence Interval'
)
plt.xlabel("Date")
plt.ylabel("Vehicles")
plt.title("Traffic Prediction Output")
plt.legend()
plt.show()
