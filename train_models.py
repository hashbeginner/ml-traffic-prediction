import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # To Suppress TensorFlow logs

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input


df = pd.read_csv("data/Datafile.csv").sample(10000, random_state=42)


df['holiday'] = df['holiday'].fillna('No')

df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True)
df['hour'] = df['date_time'].dt.hour
df['day'] = df['date_time'].dt.day
df['month'] = df['date_time'].dt.month
df['weekday'] = df['date_time'].dt.weekday
df.drop('date_time', axis=1, inplace=True)

df['holiday'] = df['holiday'].map({'Yes':1,'No':0})

df = pd.get_dummies(df, columns=['weather_main','weather_description'], drop_first=True)
df.fillna(0, inplace=True)

X = df.drop('traffic_volume', axis=1)
y = df['traffic_volume']


joblib.dump(X.columns.tolist(), "models/model_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10),
    "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVR": SVR(),
    "XGBoost": XGBRegressor(n_estimators=50, max_depth=6, learning_rate=0.1, verbosity=0)
}

best_model = None
best_error = float("inf")
results = {}

for name, model in models.items():
    print(f"\nTraining: {name}")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    error = mean_absolute_error(y_test, preds)
    print(f"{name} MAE: {error:.4f}")
    results[name] = error
    if error < best_error:
        best_error = error
        best_model = model

# ---------------- LSTM Model ---------------- #
print("\nTraining: LSTM")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1,1))

X_lstm = X_scaled.reshape((X_scaled.shape[0],1,X_scaled.shape[1]))
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_lstm, y_scaled, test_size=0.2, random_state=42
)

lstm_model = Sequential([
    Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    LSTM(50),
    Dense(1)
])
lstm_model.compile(optimizer="adam", loss="mse")

lstm_model.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=32, verbose=1)

preds_lstm = lstm_model.predict(X_test_lstm)
error_lstm = mean_absolute_error(y_test_lstm, preds_lstm)
print(f"LSTM MAE: {error_lstm:.4f}")
results["LSTM"] = error_lstm

plt.figure(figsize=(10,5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Comparison (MAE)")
plt.ylabel("MAE")
plt.xlabel("Model")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()


joblib.dump(best_model, "models/traffic_model.pkl")
lstm_model.save("models/lstm_model.h5")

print("\nBest classical ML model saved!")
print("LSTM model saved!")