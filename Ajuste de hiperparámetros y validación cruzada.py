import pandas as pd

# 1. Cargar datos
df = pd.read_excel(
    r"C:\Users\anzol\OneDrive\Desktop\Diego Anzola Programacion-1\Proyecto\DataSupermercado.xlsx"
)

# Mostrar las columnas disponibles en tu dataset
print("Columnas en el dataset:", df.columns.tolist())

target_column = "TotalVentas"   

# Variables predictoras y objetivo
X = df.drop(columns=[target_column])
y = df[target_column]

# 2. Dividir en entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Definir modelo y espacio de hiperparámetros
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10]
}

# 4. Ajuste con validación cruzada
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Mejores hiperparámetros:", grid_search.best_params_)
print("Mejor score de validación:", grid_search.best_score_)

# 5. Evaluar en conjunto de prueba
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)

print("Score en datos de prueba:", test_score)