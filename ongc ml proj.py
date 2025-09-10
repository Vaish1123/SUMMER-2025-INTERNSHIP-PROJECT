import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

folder_path = r'C:\Users\LENOVO\Downloads\drive-download-20250607T051604Z-1-001'
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

dfs = []
for file in csv_files:
    df = pd.read_csv(os.path.join(folder_path, file), header=None)
    df.columns = ['name', 'date_time', 'value', 'machine_on']
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df['value'] = df['value'].apply(lambda x: max(0, x))
df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
df = df.dropna(subset=['date_time'])

def classify_sensor(name):
    if 'XVT' in name:
        return 'vibration'
    elif 'TE' in name:
        return 'temperature'
    elif 'PT' in name:
        return 'pressure'
    return 'unknown'

df['sensor_type'] = df['name'].apply(classify_sensor)

df_pivot = df.pivot_table(index=['date_time', 'machine_on'], columns='sensor_type', values='value', aggfunc='mean').reset_index()
df_pivot = df_pivot.dropna(subset=['vibration', 'temperature', 'pressure'])

print("\n\U0001F4C1 Pivoted Sensor Data Table:")
print(df_pivot.head(10))

df_pivot['hour'] = df_pivot['date_time'].dt.hour
df_pivot['day'] = df_pivot['date_time'].dt.day
df_pivot['weekday'] = df_pivot['date_time'].dt.weekday

np.random.seed(42)
df_pivot['true_failure'] = np.random.choice([False, True], size=len(df_pivot), p=[0.97, 0.03])

features = ['vibration', 'temperature', 'pressure', 'hour', 'day', 'weekday']
X = df_pivot[features]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n\U0001F50D Running Isolation Forest with contamination = 3%")
iso_model = IsolationForest(contamination=0.03, random_state=42)
iso_model.fit(X_scaled)
iso_scores = iso_model.decision_function(X_scaled)
iso_thresh = np.percentile(iso_scores, 5)
df_pivot['pred_iso_failure'] = iso_scores < iso_thresh

print("\n\U0001F50D Running One-Class SVM")
svm_model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.03)
svm_model.fit(X_scaled)
svm_scores = svm_model.decision_function(X_scaled)
svm_thresh = np.percentile(svm_scores, 5)
df_pivot['pred_svm_failure'] = svm_scores < svm_thresh

df_pivot['predicted_failure'] = df_pivot['pred_iso_failure'] | df_pivot['pred_svm_failure']

print("\n\U0001F4CA Classification Report (simulated true values):")
print(classification_report(df_pivot['true_failure'], df_pivot['predicted_failure']))

print("\n\U0001F50D Top 10 Predicted Failures:")
print(df_pivot[df_pivot['predicted_failure'] == True][['date_time', 'vibration', 'temperature', 'pressure']].head(10))

print("\n\U0001F4CB Next 100 Pump Predictions:")
print(df_pivot[['date_time', 'vibration', 'temperature', 'pressure', 'predicted_failure']].head(100))
