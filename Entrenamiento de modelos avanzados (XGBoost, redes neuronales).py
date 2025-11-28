import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# 1. Cargar datos
df = pd.read_excel(r"C:\Users\anzol\OneDrive\Desktop\Diego Anzola Programacion-1\Proyecto\DataSupermercado.xlsx")
df.columns = df.columns.str.strip()

# 2. Variables de tiempo
df["Date"] = pd.to_datetime(df["Date"])
df["Mes"] = df["Date"].dt.month
df["DiaSemana"] = df["Date"].dt.dayofweek 

# 3. Selección de variables relevantes
features = ["Mes", "DiaSemana", "Product line", "Unit price", "Quantity", "Payment", "Gender"]
X = pd.get_dummies(df[features], drop_first=True)
y = df["Sales"]

# 4. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Modelo XGBoost
model_xgb = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42)
model_xgb.fit(X_train, y_train)

# 6. Evaluación
y_pred_xgb = model_xgb.predict(X_test)
print("XGBoost:")
print("RMSE:", mean_squared_error(y_test, y_pred_xgb, squared=False))
print("R2:", r2_score(y_test, y_pred_xgb))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score

# 1. Definir el modelo
model_nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # salida para regresión
])

# 2. Compilar
model_nn.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 3. Entrenar
history = model_nn.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# 4. Evaluar
y_pred_nn = model_nn.predict(X_test).flatten()
print("Red Neuronal:")
print("RMSE:", mean_squared_error(y_test, y_pred_nn, squared=False))
print("R2:", r2_score(y_test, y_pred_nn))