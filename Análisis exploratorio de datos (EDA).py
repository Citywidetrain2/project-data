# Librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta absoluta al archivo Excel
ruta_excel = r"C:\Users\anzol\OneDrive\Desktop\Diego Anzola Programacion-1\Proyecto\DataSupermercado.xlsx"

# Cargar el archivo Excel
df = pd.read_excel(ruta_excel)

# Verificar que se cargó correctamente
print("Primeras filas del dataset:")
print(df.head())

# 1. Vista inicial de los datos
print("\nInformación general:")
print(df.info())

print("\nEstadísticas descriptivas:")
print(df.describe())

# 2. Distribución de variables categóricas
print("\nDistribución por ciudad:")
print(df['City'].value_counts())

print("\nDistribución por línea de producto:")
print(df['Product line'].value_counts())

# 3. Visualizaciones básicas
# Histograma de ventas
plt.figure(figsize=(8,5))
sns.histplot(df['Sales'], bins=30, kde=True)
plt.title("Distribución de Ventas")
plt.xlabel("Ventas")
plt.ylabel("Frecuencia")
plt.show()

# Ventas por ciudad
plt.figure(figsize=(8,5))
sns.barplot(x="City", y="Sales", data=df, estimator=sum, ci=None)
plt.title("Ventas Totales por Ciudad")
plt.ylabel("Ventas Totales")
plt.show()

# Ventas por línea de producto
plt.figure(figsize=(10,6))
sns.barplot(x="Product line", y="Sales", data=df, estimator=sum, ci=None)
plt.title("Ventas Totales por Línea de Producto")
plt.xticks(rotation=45)
plt.ylabel("Ventas Totales")
plt.show()

# Boxplot de rating por ciudad
plt.figure(figsize=(8,5))
sns.boxplot(x="City", y="Rating", data=df)
plt.title("Distribución de Ratings por Ciudad")
plt.show()

# 4. Correlación entre variables numéricas
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Mapa de Correlaciones")
plt.show()



