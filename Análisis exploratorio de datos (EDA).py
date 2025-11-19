# Librerías
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración estética
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# ============================
# 1. Cargar y limpiar dataset
# ============================
df = pd.read_excel("DATA_DISFRAZADA_SUPERMERCADO.xlsx")
df.columns = df.columns.str.strip()  # Elimina espacios ocultos

# ============================
# 2. Información general
# ============================
print("Dimensiones del dataset:", df.shape)
print("\nTipos de datos:\n", df.dtypes)
print("\nValores nulos por columna:\n", df.isnull().sum())
print("\nResumen estadístico:\n", df.describe())

# ============================
# 3. Distribución de ventas
# ============================
columna_ventas = None
for col in ["VALORES", "Ventas"]:
    if col in df.columns:
        columna_ventas = col
        break

if columna_ventas:
    plt.figure()
    sns.histplot(df[columna_ventas], bins=30, kde=True)
    plt.title(f"Distribución de {columna_ventas}")
    plt.show()

    # ============================
    # 4. Outliers con boxplot
    # ============================
    plt.figure()
    sns.boxplot(x=df[columna_ventas])
    plt.title(f"Detección de outliers en {columna_ventas}")
    plt.show()

    # ============================
    # 5. Tendencia temporal
    # ============================
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        ventas_por_fecha = df.groupby("Fecha")[columna_ventas].sum().reset_index()

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=ventas_por_fecha, x="Fecha", y=columna_ventas)
        plt.title(f"Tendencia de {columna_ventas} por fecha")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # ============================
    # 6. Ventas por producto
    # ============================
    if "Producto" in df.columns:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x="Producto", y=columna_ventas, estimator=sum, ci=None)
        plt.xticks(rotation=45)
        plt.title(f"{columna_ventas} totales por producto")
        plt.tight_layout()
        plt.show()

# ============================
# 7. Correlaciones
# ============================
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlaciones")
plt.show()

# ============================
# 8. Segmentación temporal
# ============================
if "Fecha" in df.columns and columna_ventas:
    df["Mes"] = df["Fecha"].dt.month
    df["DiaSemana"] = df["Fecha"].dt.day_name()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Mes", y=columna_ventas, estimator=sum, ci=None)
    plt.title(f"{columna_ventas} por mes")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="DiaSemana", y=columna_ventas, estimator=sum, ci=None,
                order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    plt.title(f"{columna_ventas} por día de la semana")
    plt.show()



