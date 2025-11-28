import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
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
print("R2:", r2_score(y_test, y_pred_tree))