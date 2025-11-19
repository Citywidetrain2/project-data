# Librerías
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================
# 1. Cargar y limpiar dataset
# ============================
df = pd.read_excel("DATA_DISFRAZADA_SUPERMERCADO.xlsx")
df.columns = df.columns.str.strip()  # Elimina espacios ocultos

# ============================
# 2. Definir variable objetivo
# ============================
columna_objetivo = None
for col in ["Ventas", "VALORES"]:
    if col in df.columns:
        columna_objetivo = col
        break

if columna_objetivo is None:
    raise ValueError("No se encontró la columna 'Ventas' ni 'VALORES' en el dataset.")

# ============================
# 3. Crear variables derivadas
# ============================
if "Fecha" in df.columns:
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df["Mes"] = df["Fecha"].dt.month
    df["DiaSemana"] = df["Fecha"].dt.day_name()

# ============================
# 4. Selección de variables relevantes
# ============================
variables_relevantes = [
    columna_objetivo,   # Objetivo
    "Fecha", "Mes", "DiaSemana", "Producto",   # Principales
    "Precio", "Cantidad", "Descuento", "Categoría", "Región", "Cliente"  # Adicionales
]

# Filtrar solo las columnas que existan en el dataset
variables_finales = [var for var in variables_relevantes if var in df.columns]
df_modelo = df[variables_finales].copy()

print("✅ Variables seleccionadas para análisis/modelado:")
print(df_modelo.head())

# ============================
# 5. Preparación para modelado
# ============================
# Separar objetivo y explicativas
y = df_modelo[columna_objetivo]
X = df_modelo.drop(columns=[columna_objetivo])

# Convertir variables categóricas a dummies (one-hot encoding)
X = pd.get_dummies(X, drop_first=True)

print("\nDimensiones de X (explicativas):", X.shape)
print("Dimensiones de y (objetivo):", y.shape)