import pandas as pd
def find_anomalies(models, data_row):
    anomalies = {}

    for sensor_name in data_row.index:
        model_key = f"{sensor_name}_prophet_model"
        if model_key in models:
            model = models[model_key]

            df = pd.DataFrame({'ds': [data_row.name], 'y': [data_row[sensor_name]]})
            forecast = model.predict(df)

            y = data_row[sensor_name]
            lower = forecast['yhat_lower'].iloc[0]
            upper = forecast['yhat_upper'].iloc[0]

            if y < lower or y > upper:
                anomalies[sensor_name] = y

    return anomalies