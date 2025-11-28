import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

# Mostrar carpeta actual
print("Carpeta actual:", os.getcwd())

# 1. Cargar datos
df = pd.read_excel(r"C:\Users\anzol\OneDrive\Desktop\Diego Anzola Programacion-1\Proyecto\DataSupermercado.xlsx")

# 2. Limpieza de columnas
df.columns = df.columns.str.strip()
print("Columnas disponibles:", df.columns.tolist())

# 3. Crear variables de tiempo
df["Date"] = pd.to_datetime(df["Date"])
df["Mes"] = df["Date"].dt.month
df["DiaSemana"] = df["Date"].dt.dayofweek

# 4. Selección de variables relevantes
features = ["Mes", "DiaSemana", "Product line", "Unit price", "Quantity", "Payment", "Gender"]
X = pd.get_dummies(df[features], drop_first=True)  # convertir categóricas
y = df["Sales"]  # variable objetivo

# 5. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Modelo predictivo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluación
y_pred = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

# 8. Importancia de variables
importances = pd.Series(model.feature_importances_, index=X.columns)
print("Importancia de variables:")
print(importances.sort_values(ascending=False))