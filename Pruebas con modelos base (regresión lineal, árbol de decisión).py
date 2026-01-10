import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
<<<<<<< HEAD
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar datos
df = pd.read_excel(r"C:\Users\anzol\OneDrive\Desktop\Diego Anzola Programacion-1\Proyecto\DataSupermercado.xlsx")

# 2. Limpieza de columnas
df.columns = df.columns.str.strip()

# 3. Crear variables de tiempo
df["Date"] = pd.to_datetime(df["Date"])
df["Mes"] = df["Date"].dt.month
df["DiaSemana"] = df["Date"].dt.dayofweek

# 4. Selección de variables relevantes (ajustadas a tu archivo)
features = ["Mes", "DiaSemana", "Product line", "Unit price", "Quantity", "Payment", "Gender"]
X = pd.get_dummies(df[features], drop_first=True)
y = df["Sales"]  # variable objetivo real en tu archivo

# 5. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
=======
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
>>>>>>> da4a140ec4de4a324bd5909aab1e3a571927abef

# 6. Modelo de Regresión Lineal
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

print("Regresión Lineal:")
print("RMSE:", mean_squared_error(y_test, y_pred_lin, squared=False))
print("R2:", r2_score(y_test, y_pred_lin))

# 7. Modelo de Árbol de Decisión
tree_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_reg.fit(X_train, y_train)
y_pred_tree = tree_reg.predict(X_test)

print("\nÁrbol de Decisión:")
print("RMSE:", mean_squared_error(y_test, y_pred_tree, squared=False))
<<<<<<< HEAD
print("R2:", r2_score(y_test, y_pred_tree))
=======
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
>>>>>>> da4a140ec4de4a324bd5909aab1e3a571927abef
