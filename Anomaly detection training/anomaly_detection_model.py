import pandas as pd
from prophet import Prophet
import os
import joblib 

df = pd.read_csv("envai_training_dataset.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

sensors = [
    "temperature_C", "humidity_%", "CO2_ppm", "CO_ppm",
    "NO2_ppb", "VOC_ppb", "CH4_ppm", "C3H8_ppm", "H2S_ppm", "Rn_Bq_m3"
]

model_dir = "prophet_models"
os.makedirs(model_dir, exist_ok=True)

for s in sensors:
    sensor_df = df[['timestamp', s]].rename(columns={'timestamp': 'ds', s: 'y'}).dropna()

    m = Prophet(daily_seasonality=True, yearly_seasonality=False, interval_width=1, mcmc_samples=0)
    m.fit(sensor_df)

    model_path = os.path.join(model_dir, f"{s}_prophet_model.joblib")
    joblib.dump(m, model_path)

    future = sensor_df[['ds']].copy()
    forecast = m.predict(future)

    anomalies = sensor_df[
        (sensor_df['y'] < forecast['yhat_lower']) |
        (sensor_df['y'] > forecast['yhat_upper'])
    ].copy()