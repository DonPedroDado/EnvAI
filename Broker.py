import queue
import pandas as pd
from Anomaly_detection.py import find_anomalies
import os

class Broker:
    def __init__(self):
        self.queue = queue.Queue()
        self.subscribers = []

    def publish(self, data):
        self.queue.put(data)
        print(f"Published data:\n{data}")

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
        model_name = os.path.splitext(filename)[0]  # e.g., "temperature_C_prophet_model"
        models[model_name] = joblib.load(model_path)
        print(f"Loaded model: {model_name}")

def anomaly_detection(data):
    print(f"[Subscriber] Received:\n{data}")
    anomalies=find_anomalies(models, data)

def dataset_streamer(broker, file_path):
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        broker.publish(row)
        broker.process()

subscribers=[anomaly_detection]

if __name__ == "__main__":
    broker = Broker()
    for subscriber in subscribers:
        Broker.subscribe(subscriber)

    dataset_streamer(broker, "envai_testing_dataset.csv")
