import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Parameters ---
hours = 2190            
interval_minutes = 1
num_points = int(hours * 60 / interval_minutes)
start_time = datetime(2023, 1, 1, 0, 0)
timestamps = [start_time + timedelta(minutes=i * interval_minutes) for i in range(num_points)]
t = np.linspace(0, 60 * np.pi, num_points)  # many day/night cycles

# --- Multi-room setup ---
rooms = ["Kitchen", "Living Room", "Garage", "Basement"]
room_assignments = np.random.choice(rooms, size=num_points, p=[0.3, 0.3, 0.2, 0.2])

# --- Base Sensor Drift and Cycles ---
drift = np.linspace(0, 1, num_points) * np.random.uniform(-0.5, 0.5)  # gradual drift
noise = lambda scale: np.random.normal(0, scale, num_points)

temperature = 21 + 5 * np.sin(t / (2 * np.pi)) + drift + noise(0.5)
humidity = 45 + 10 * np.sin(t / (1.5 * np.pi) + 1.5) + drift * 2 + noise(2)
CO2 = 420 + 300 * (np.sin(t / (np.pi * 2) + 0.4) ** 2) + drift * 10 + noise(20)
CO = np.abs(0.4 + 0.6 * np.sin(t + 2.1) + drift * 0.05 + noise(0.05))
NO2 = np.abs(18 + 10 * np.sin(t + 0.3) + noise(3))
VOC = np.abs(120 + 80 * np.sin(t + 1.1) ** 2 + drift * 5 + noise(10))
CH4 = np.abs(1.5 + 0.8 * np.sin(t + 2.5) + noise(0.2))
C3H8 = np.abs(0.8 + 0.5 * np.sin(t + 1.8) + noise(0.1))
H2S = np.abs(0.05 + 0.03 * np.sin(t + 0.7) + noise(0.01))
Rn = np.abs(15 + 6 * np.sin(t / 3 + 0.5) + noise(1))

# --- Inject Random Anomalies (gas leaks, etc.) ---
def add_anomaly(signal, idx, duration, magnitude):
    end = min(idx + duration, len(signal))
    local_noise = np.random.normal(0, 0.05 * magnitude, end - idx)
    signal[idx:end] += magnitude + local_noise

np.random.seed(42)
for _ in range(25):  # spread across dataset
    idx = np.random.randint(0, num_points - 50)
    duration = np.random.randint(3, 15)
    add_anomaly(CO2, idx, duration, np.random.randint(300, 800))
    add_anomaly(VOC, idx, duration, np.random.randint(100, 300))
    add_anomaly(CO, idx, duration, np.random.uniform(3, 6))
    add_anomaly(CH4, idx, duration, np.random.uniform(2, 5))
    add_anomaly(C3H8, idx, duration, np.random.uniform(1, 2))
    add_anomaly(H2S, idx, duration, np.random.uniform(0.5, 1.0))
    add_anomaly(NO2, idx, duration, np.random.uniform(30, 60))

# --- Assemble Dataset (without anomaly_flag) ---
df = pd.DataFrame({
    "timestamp": timestamps,
    "room": room_assignments,
    "temperature_C": temperature.round(2),
    "humidity_%": humidity.round(2),
    "CO2_ppm": CO2.round(2),
    "CO_ppm": CO.round(3),
    "NO2_ppb": NO2.round(2),
    "VOC_ppb": VOC.round(2),
    "CH4_ppm": CH4.round(2),
    "C3H8_ppm": C3H8.round(2),
    "H2S_ppm": H2S.round(3),
    "Rn_Bq_m3": Rn.round(2)
})

# --- Save Efficiently ---
df.to_csv("envai_training_dataset.csv", index=False)
