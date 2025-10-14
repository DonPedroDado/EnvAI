import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
import time

start_time=time.time()

# --- Load existing dataset ---
df = pd.read_csv("envai_full_mock_dataset_v2_1.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# --- Define sensors ---
sensors = ["temperature_C", "humidity_%", "CO2_ppm", "CO_ppm",
           "NO2_ppb", "VOC_ppb", "CH4_ppm", "C3H8_ppm", "H2S_ppm", "Rn_Bq_m3"]

# --- Initialize sensor-specific anomaly columns if not present ---
for s in sensors:
    sensor_anomaly_col = s + "_anomaly"
    if sensor_anomaly_col not in df.columns:
        df[sensor_anomaly_col] = 0

# --- Create folder to save charts ---
output_dir = "sensor_charts"
os.makedirs(output_dir, exist_ok=True)

# --- Loop through sensors ---
for s in sensors:
    print(f"Processing {s}...")

    # Prepare data for Prophet
    sensor_df = df[['timestamp', s]].rename(columns={'timestamp':'ds', s:'y'}).dropna()

    # Fit Prophet model
    m = Prophet(daily_seasonality=True, yearly_seasonality=False, interval_width=0.95, mcmc_samples=0)
    m.fit(sensor_df)

    # Forecast on the same timestamps
    future = sensor_df[['ds']].copy()
    forecast = m.predict(future)

    # Detect anomalies based on Prophet's confidence interval
    df[s + "_anomaly"] = ((sensor_df['y'] < forecast['yhat_lower']) |
                          (sensor_df['y'] > forecast['yhat_upper'])).astype(int)

    # Plot
    sample_step = max(1, len(sensor_df) // 10000)
    plt.figure(figsize=(15,5))
    plt.plot(sensor_df['ds'][::sample_step], sensor_df['y'][::sample_step], label='Actual', color='blue')
    plt.plot(sensor_df['ds'][::sample_step], forecast['yhat'][::sample_step], label='Forecast', color='green')
    plt.fill_between(sensor_df['ds'][::sample_step],
                     forecast['yhat_lower'][::sample_step],
                     forecast['yhat_upper'][::sample_step],
                     color='gray', alpha=0.3)
    plt.scatter(sensor_df['ds'][df[s+'_anomaly']==1][::sample_step],
                sensor_df['y'][df[s+'_anomaly']==1][::sample_step],
                color='red', s=50, label='Anomaly')
    plt.title(f"{s} Monitoring with Anomalies")
    plt.xlabel("Time")
    plt.ylabel(s)
    plt.legend()

    chart_path = os.path.join(output_dir, f"{s}_anomaly_chart.png")
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()

# --- Save updated dataset in place ---
end_time=time.time()
elapsed=end_time-start_time

df.to_csv("envai_full_mock_dataset_v2_1.csv", index=False)
print("Anomaly detection completed and dataset updated in place (original 'anomaly_flag' preserved).")
print(f"All charts saved in folder: {output_dir}")
print(f"Total time elapsed: {elapsed:.2f} seconds")
