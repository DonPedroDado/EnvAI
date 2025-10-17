from prometheus_client import start_http_server, Counter, Gauge
import time
import re

MESSAGES_PROCESSED = Counter("messages_processed_total", "Total number of sensor messages processed")
ANOMALIES_DETECTED_TOTAL = Counter("anomalies_detected_total", "Total number of anomalies detected across all sensors")
QUEUE_SIZE = Gauge("broker_queue_size", "Number of messages currently in broker queue")
LAST_TIMESTAMP = Gauge("last_data_timestamp", "Timestamp of last processed sensor data")

SENSOR_VALUES = {}
SENSOR_ANOMALIES = {}

def sanitize_metric_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def ensure_sensor_metrics(columns):
    for col in columns:
        safe_col = sanitize_metric_name(col)
        if safe_col not in SENSOR_VALUES:
            SENSOR_VALUES[safe_col] = Gauge(f"sensor_{safe_col}_value", f"Latest value of sensor '{col}'")
        if safe_col not in SENSOR_ANOMALIES:
            SENSOR_ANOMALIES[safe_col] = Counter(f"sensor_{safe_col}_anomalies_total", f"Total anomalies detected for '{col}'")

def prometheus_metrics(data):
    MESSAGES_PROCESSED.inc()
    LAST_TIMESTAMP.set(time.time())
    for col, value in data.items():
        safe_col = sanitize_metric_name(col)
        try:
            SENSOR_VALUES[safe_col].set(float(value))
        except (ValueError, TypeError):
            continue
    QUEUE_SIZE.set(0)


def prometheus_anomalies(anomalies):
    if anomalies:
        ANOMALIES_DETECTED_TOTAL.inc(sum(anomalies.values()))
        for sensor, count in anomalies.items():
            safe_col = sanitize_metric_name(sensor)
            if safe_col in SENSOR_ANOMALIES:
                SENSOR_ANOMALIES[safe_col].inc(count)

def start_metrics_server(port=8000):
    start_http_server(port)
