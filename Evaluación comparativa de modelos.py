import matplotlib.pyplot as plt
import numpy as np

# Datos
modelos = ["Regresión Lineal", "Árbol de Decisión", "XGBoost", "Red Neuronal"]
rmse = [79.52, 42.33, 38.75, 45.10]
r2 = [0.90, 0.97, 0.98, 0.96]

x = np.arange(len(modelos))  # posiciones en el eje X
width = 0.35  # ancho de las barras

# Crear gráfico
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, rmse, width, label='RMSE', color='tomato')
bars2 = ax.bar(x + width/2, r2, width, label='R²', color='steelblue')

# Etiquetas y título
ax.set_ylabel('Valor')
ax.set_title('Comparación de Modelos de Regresión')
ax.set_xticks(x)
ax.set_xticklabels(modelos, rotation=15)
ax.legend()

# Mostrar valores sobre las barras
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.show()