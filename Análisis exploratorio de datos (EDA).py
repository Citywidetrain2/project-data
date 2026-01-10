#----------------------------------------------------------------------
# Librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta absoluta al archivo Excel
ruta_excel = r"C:\Users\anzol\OneDrive\Desktop\Diego Anzola Programacion-1\Proyecto\DataSupermercado.xlsx"

# Cargar el archivo Excel
df = pd.read_excel(ruta_excel)

# 1. Vista inicial de los datos
print("Primeras filas del dataset:")
print(df.head())

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

# Histograma de ventas (único y completo)
plt.figure(figsize=(12,7))
sns.histplot(df['Sales'], bins=100, kde=True, color="dodgerblue", edgecolor="black")
plt.axvline(df['Sales'].mean(), color='red', linestyle='--', linewidth=2, label=f"Media: {df['Sales'].mean():.2f}")
plt.axvline(df['Sales'].median(), color='green', linestyle='-', linewidth=2, label=f"Mediana: {df['Sales'].median():.2f}")
plt.xlim(df['Sales'].min(), df['Sales'].max())
plt.title("Distribución Completa de Ventas", fontsize=16, fontweight="bold")
plt.xlabel("Valor de Ventas", fontsize=14)
plt.ylabel("Frecuencia", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
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

# 5. Gráficas adicionales relevantes

# Ventas mensuales (si existe columna 'Date')
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    plt.figure(figsize=(12,6))
    df.groupby('Month')['Sales'].sum().plot(kind='line', marker='o', color="purple")
    plt.title("Ventas Mensuales")
    plt.xlabel("Mes")
    plt.ylabel("Ventas Totales")
    plt.grid(True)
    plt.show()

# Ventas por método de pago
if 'Payment' in df.columns:
    plt.figure(figsize=(8,5))
    sns.barplot(x="Payment", y="Sales", data=df, estimator=sum, ci=None, palette="Set2")
    plt.title("Ventas Totales por Método de Pago")
    plt.ylabel("Ventas Totales")
    plt.show()

# Ventas por género
if 'Gender' in df.columns:
    plt.figure(figsize=(8,5))
    sns.barplot(x="Gender", y="Sales", data=df, estimator=sum, ci=None, palette="pastel")
    plt.title("Ventas Totales por Género de Cliente")
    plt.ylabel("Ventas Totales")
    plt.show()

# Relación cantidad vs ventas
if 'Quantity' in df.columns:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x="Quantity", y="Sales", data=df, alpha=0.6, color="darkorange")
    plt.title("Relación entre Cantidad y Ventas")
    plt.xlabel("Cantidad")
    plt.ylabel("Ventas")
    plt.show()

# Distribución de ratings
plt.figure(figsize=(10,6))
sns.histplot(df['Rating'], bins=20, kde=True, color="teal", edgecolor="black")
plt.title("Distribución de Ratings de Clientes")
plt.xlabel("Rating")
plt.ylabel("Frecuencia")
plt.show()


#-------------------------------
import numpy as np
from scipy.stats import norm

plt.figure(figsize=(12,7))
sns.histplot(df['Sales'], bins=100, kde=True, color="dodgerblue", edgecolor="black")

# Parámetros de la normal teórica
mu, sigma = df['Sales'].mean(), df['Sales'].std()
x = np.linspace(df['Sales'].min(), df['Sales'].max(), 100)
plt.plot(x, norm.pdf(x, mu, sigma) * len(df['Sales']) * (df['Sales'].max()-df['Sales'].min())/100,
         color='red', linewidth=2, label="Distribución Normal Teórica")

plt.title("Distribución de Ventas vs Normal Teórica", fontsize=16, fontweight="bold")
plt.xlabel("Valor de Ventas", fontsize=14)
plt.ylabel("Frecuencia", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

import scipy.stats as stats

plt.figure(figsize=(8,6))
stats.probplot(df['Sales'], dist="norm", plot=plt)
plt.title("Gráfico Q-Q de Ventas vs Normal")
plt.show()