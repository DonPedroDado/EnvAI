import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("envai_full_mock_dataset_v2.csv")

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Resample numeric columns to 5-min intervals
numeric_cols = df.select_dtypes(include='number').columns
df = df.set_index('timestamp')[numeric_cols].resample('5min').mean().reset_index()

# Define sensors
sensors = ["temperature_C", "humidity_%", "CO2_ppm", "CO_ppm",
           "NO2_ppb", "VOC_ppb", "CH4_ppm", "C3H8_ppm", "H2S_ppm", "Rn_Bq_m3"]

# Initialize anomaly columns
for s in sensors:
    df[s + "_anomaly"] = 0

# Create folder to save charts
output_dir = "sensor_charts"
os.makedirs(output_dir, exist_ok=True)

# Loop through sensors
for s in sensors:
    print(f"Processing {s}...")

    # Prepare data for Prophet
    sensor_df = df[['timestamp', s]].rename(columns={'timestamp':'ds', s:'y'}).dropna()

    # Fit Prophet model
    m = Prophet(daily_seasonality=True, yearly_seasonality=False, interval_width=0.95, mcmc_samples=0)
    m.fit(sensor_df)

    # Forecast
    future = m.make_future_dataframe(periods=0, freq='5min')
    forecast = m.predict(future)

    # Detect anomalies
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
                color='red', s=50, label='Anomaly')  # bigger dots for anomalies
    plt.title(f"{s} Monitoring with Anomalies")
    plt.xlabel("Time")
    plt.ylabel(s)
    plt.legend()

    chart_path = os.path.join(output_dir, f"{s}_anomaly_chart.png")
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()

# Save results with anomalies
df.to_csv("envai_full_anomalies.csv", index=False)
print(f"Full anomaly detection completed. Saved as envai_full_anomalies.csv")
print(f"All charts saved in folder: {output_dir}")
