import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Ruta del archivo
ruta = "DATA_DISFRAZADA_SUPERMERCADO.xlsx"
# Verificar existencia y permisos del archivo
print("¿Existe el archivo?", os.path.exists(ruta))
print("¿Tengo permiso de lectura?", os.access(ruta, os.R_OK))

# Cargar el archivo Excel
df = pd.read_excel(ruta)

# Limpiar nombres de columnas (eliminar espacios y convertir a minúsculas si se desea)
df.columns = df.columns.str.strip()

# Mostrar nombres de columnas disponibles
print("Columnas disponibles:", df.columns.tolist())

# Verificar contenido (muestra aleatoria de 10 filas)
print(df.sample(10))

# Verificar si la columna 'Producto' existe
if "Producto" in df.columns:
    productos = df["Producto"].unique().tolist()
    print("Productos únicos:", productos)
else:
    print(" La columna 'Producto' no se encuentra en el archivo.")

# Verificar si la columna 'Ventas' existe antes de graficar
if "Producto" in df.columns and "Ventas" in df.columns:
    # Visualización de ventas por producto
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Producto", y="Ventas", estimator=sum, ci=None)
    plt.xticks(rotation=45)
    plt.title("Ventas totales por producto")
    plt.tight_layout()
    plt.show()
else:
    print(" No se puede graficar porque faltan las columnas 'Producto' o 'Ventas'.")
