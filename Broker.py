import queue
import pandas as pd
from Anomaly_detection import find_anomalies
from database import SQLiteStorage
from backend import start_metrics_server, ensure_sensor_metrics, prometheus_metrics, prometheus_anomalies
import joblib
import os

class Broker:
    def __init__(self):
        self.queue = queue.Queue()
        self.subscribers = []

    def publish(self, data):
        self.queue.put(data)

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def process(self):
        while not self.queue.empty():
            data = self.queue.get()
            for callback in self.subscribers:
                callback(data)
            self.queue.task_done()

models_folder = "prophet_models"
models = {}

for filename in os.listdir(models_folder):
    if filename.endswith(".joblib"):
        model_path = os.path.join(models_folder, filename)
        model_name = os.path.splitext(filename)[0]
        models[model_name] = joblib.load(model_path)

def anomaly_detection(data):
    anomalies = find_anomalies(models, data)
    prometheus_anomalies(anomalies)

storage = SQLiteStorage("envai_data.db")

def database(data):
    storage.store_data(data)

def dataset_streamer(broker, file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)

    ensure_sensor_metrics(df.columns)

    for _, row in df.iterrows():
        broker.publish(row)
        broker.process()


def backend_metrics(data):
    prometheus_metrics(data)

subscribers = [anomaly_detection, database, backend_metrics]

if __name__ == "__main__":
    start_metrics_server()
    broker = Broker()
    for subscriber in subscribers:
        broker.subscribe(subscriber)
    dataset_streamer(broker, "envai_testing_dataset.csv")