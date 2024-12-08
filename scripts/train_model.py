# Import pustaka
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Membaca dataset
df = pd.read_csv("property-price-prediction/data/dataset_harga_properti.csv")

# Pra-pemrosesan data
X = df.drop(columns=["Harga_Properti"])
y = df["Harga_Properti"]
X["Lokasi"] = LabelEncoder().fit_transform(X["Lokasi"])
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Membagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Gradient Boosting
model_gb = GradientBoostingRegressor(random_state=42)
model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)
print(f"Gradient Boosting MAE: {mean_absolute_error(y_test, y_pred_gb)}, R²: {r2_score(y_test, y_pred_gb)}")

# Model 2: Neural Network
model_nn = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1)
])
model_nn.compile(optimizer="adam", loss="mse", metrics=["mae"])
model_nn.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
y_pred_nn = model_nn.predict(X_test).flatten()
print(f"Neural Network MAE: {mean_absolute_error(y_test, y_pred_nn)}, R²: {r2_score(y_test, y_pred_nn)}")

# Menyimpan model
joblib.dump(model_gb, "property-price-prediction/results/gradient_boosting_model.pkl")
model_nn.save("property-price-prediction/results/neural_network_model.h5")
print("Model disimpan ke folder 'results/'")
