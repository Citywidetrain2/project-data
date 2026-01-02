import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#import kaggle

# Download latest version
#path = kagglehub.dataset_download("faresashraf1001/supermarket-sales")

#print("Path to dataset files:", path)

# 0. Datos
df = pd.read_excel("DATA_DISFRAZADA_SUPERMERCADO.xlsx")
df.columns = df.columns.str.strip() # Limpia espacios en nombres

# 1. Filtramos para quedarnos SOLO con números
df_numeric = df.select_dtypes(include=['number'])

# 2. Definimos X e y usando los nombres reales de tu Excel
columna_objetivo = "VALORES"  # Cambiado de "Ventas" a "VALORES"

X = df_numeric.drop(columna_objetivo, axis=1)
y = df_numeric[columna_objetivo]

print("Columnas usadas para predecir (X):", X.columns.tolist())
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

# --- PRUEBA DE PREDICCIÓN REAL ---
print("\n=== Simulando una Venta Nueva ===")

# Inventamos los datos de una venta: 
# [CODIGOTIENDA, ZONAVENTA, UNDS]
nueva_venta = [[412964, 3, 10]] # Tienda Gema, Zona 3, vende 10 unidades

# Usamos el modelo ganador (Tree Model)
prediccion = tree_model.predict(nueva_venta)

print(f"Para una venta de 10 unidades en la Tienda 412964 (Zona 3):")
print(f"💰 El modelo predice un VALOR de: ${prediccion[0]:.2f}")