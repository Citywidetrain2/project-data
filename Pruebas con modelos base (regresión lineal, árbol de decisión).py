import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import kaggle

# Download latest version
path = kagglehub.dataset_download("faresashraf1001/supermarket-sales")

print("Path to dataset files:", path)

# 0. Datos (lee Excel correctamente)
df = pd.read_excel("DATA_DISFRAZADA_SUPERMERCADO.xlsx")   

# Definición de las variables independientes (X) y dependiente (y)
X = df.drop("target", axis=1)   # Cambia "target" por el nombre real de tu columna objetivo
y = df["target"]

# 1. División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Modelo base: Regresión Lineal
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

print("\n=== Regresión Lineal ===")
print("RMSE:", mean_squared_error(y_test, y_pred_lin, squared=False))
print("MAE:", mean_absolute_error(y_test, y_pred_lin))
print("R²:", r2_score(y_test, y_pred_lin))

# 3. Modelo base: Árbol de Decisión
tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

print("\n=== Árbol de Decisión ===")
print("RMSE:", mean_squared_error(y_test, y_pred_tree, squared=False))
print("MAE:", mean_absolute_error(y_test, y_pred_tree))
print("R²:", r2_score(y_test, y_pred_tree))

