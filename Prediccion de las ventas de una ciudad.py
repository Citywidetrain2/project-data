import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Ruta absoluta al archivo Excel
ruta_excel = r"C:\Users\anzol\OneDrive\Desktop\Diego Anzola Programacion-1\Proyecto\DataSupermercado.xlsx"

# 1. Cargar el archivo Excel
df = pd.read_excel(ruta_excel)

# Mostrar columnas para confirmar
print("Columnas disponibles:", df.columns)

# 2. Filtrar la ciudad Yangon
city = "Yangon"
df_city = df[df["City"] == city]

# 3. Asegurar que la columna Date esté en formato datetime
df_city["Date"] = pd.to_datetime(df_city["Date"])
df_city = df_city.sort_values("Date")

# 4. Detectar automáticamente la columna de ventas
columna_ventas = None
for col in ["Total", "gross income", "cogs", "Sales"]:
    if col in df_city.columns:
        columna_ventas = col
        break

if columna_ventas is None:
    raise ValueError("No se encontró ninguna columna de ventas en el dataset")

ventas = df_city.set_index("Date")[columna_ventas]

# 5. Graficar las ventas históricas
plt.figure(figsize=(10,5))
plt.plot(ventas, label="Ventas históricas")
plt.title(f"Ventas históricas en {city}")
plt.xlabel("Fecha")
plt.ylabel("Ventas")
plt.legend()
plt.show()

# 6. Ajustar un modelo ARIMA sencillo
modelo = ARIMA(ventas, order=(1,1,1))  # puedes ajustar el orden
modelo_fit = modelo.fit()

# 7. Predicción de los próximos 6 períodos
pred = modelo_fit.forecast(steps=6)

# 8. Graficar ventas + predicción
plt.figure(figsize=(10,5))
plt.plot(ventas, label="Ventas históricas")
plt.plot(pred.index, pred, label="Predicción futura", color="red")
plt.title(f"Predicción de ventas en {city}")
plt.xlabel("Fecha")
plt.ylabel("Ventas")
plt.legend()
plt.show()