import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

# --- Configuración ---
ruta_excel = r"C:\Users\anzol\OneDrive\Desktop\Diego Anzola Programacion-1\Proyecto\DataSupermercado.xlsx"
city = "Yangon"
pred_horizon = 6  # horizonte de predicción

# --- Cargar y filtrar datos ---
df = pd.read_excel(ruta_excel)
df = df[df["City"] == city].copy()

# Asegurar fecha y ordenar
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# Detectar columna de ventas
ventas_col = None
for c in ["Total", "gross income", "cogs", "Sales"]:
    if c in df.columns:
        ventas_col = c
        break
if ventas_col is None:
    raise ValueError("No se encontró columna de ventas (Total, gross income, cogs, Sales)")

# Features de calendario
df["month"] = df["Date"].dt.month
df["dayofweek"] = df["Date"].dt.dayofweek

# time_idx consecutivo (en días desde el inicio)
df = df.reset_index(drop=True)
df["time_idx"] = (df["Date"] - df["Date"].min()).dt.days

# ID de serie
df["series_id"] = city

# Tipos
df[ventas_col] = df[ventas_col].astype(float)

# --- Comprobaciones y ajuste automático de ventanas ---
n_points = len(df)
if n_points < pred_horizon + 5:
    raise ValueError(f"La serie es demasiado corta: {n_points} puntos para un horizonte de {pred_horizon}.")

# Elegir encoder dinámico: al menos 5, como mucho 30 y limitado por datos
max_encoder_length = max(5, min(30, n_points - (pred_horizon + 5)))

# Corte de entrenamiento inicial
max_time_idx = df["time_idx"].max()
training_cutoff = max_time_idx - pred_horizon

# Definir conjuntos de validación: targets (solo el período de validación) y
# valid_df_history (incluye los últimos pasos del entrenamiento para proporcionar
# historial suficiente al encoder)
valid_targets = df[df.time_idx > training_cutoff]
valid_df_history = df[df.time_idx > (training_cutoff - max_encoder_length)]

# Si no hay suficientes puntos de validación, retroceder el corte o reducir encoder
if len(valid_targets) < pred_horizon:
    # Corre el corte hacia atrás para dejar más validación
    training_cutoff = max_time_idx - (pred_horizon + 3)
    valid_targets = df[df.time_idx > training_cutoff]
    valid_df_history = df[df.time_idx > (training_cutoff - max_encoder_length)]
    if len(valid_targets) < pred_horizon:
        # Último intento: encoder más pequeño
        max_encoder_length = max(5, min(max_encoder_length, n_points - (pred_horizon + 2)))
        training_cutoff = max_time_idx - (pred_horizon + 2)
        valid_targets = df[df.time_idx > training_cutoff]
        valid_df_history = df[df.time_idx > (training_cutoff - max_encoder_length)]

if len(valid_targets) == 0:
    raise ValueError("No quedan datos para validación. Reduce pred_horizon o verifica la frecuencia/longitud de la serie.")

print(f"Registros totales: {n_points} | max_encoder_length: {max_encoder_length} | pred_horizon: {pred_horizon}")
print(f"Train hasta time_idx <= {training_cutoff} | Valid desde > {training_cutoff} (valid={len(valid_targets)})")

# --- Datasets ---
training = TimeSeriesDataSet(
    df[df.time_idx <= training_cutoff],
    time_idx="time_idx",
    target=ventas_col,
    group_ids=["series_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=pred_horizon,
    static_categoricals=["series_id"],
    time_varying_known_reals=["month", "dayofweek"],
    time_varying_unknown_reals=[ventas_col],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,  # acepta huecos en la línea temporal
)

validation = TimeSeriesDataSet.from_dataset(training, valid_df_history, predict=True)

train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

# --- Entrenador ---
pl.seed_everything(42)
trainer = pl.Trainer(
    max_epochs=20,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    enable_progress_bar=True,
)

# --- Modelo TFT ---
model = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
    loss=QuantileLoss(),
)

trainer.fit(model, train_loader, val_loader)

# --- Predicción (cuantil 0.5 = mediana) ---
pred_result = model.predict(validation, return_x=True)
print("DEBUG: predict returned type:", type(pred_result))
try:
    print("DEBUG: length:", len(pred_result))
except Exception as e:
    print("DEBUG: len failed", e)

# Inspeccionar objeto Prediction
print("DEBUG: dir(pred_result):", [a for a in dir(pred_result) if not a.startswith("__")][:80])
for attr in ["predictions", "x", "y", "quantiles", "target", "to_numpy", "to_torch"]:
    if hasattr(pred_result, attr):
        try:
            val = getattr(pred_result, attr)
            print(f"DEBUG: attr {attr}: type={type(val)}")
            if hasattr(val, "shape"):
                print(f"DEBUG: {attr} shape: {getattr(val, 'shape')}")
        except Exception as e:
            print(f"DEBUG: reading {attr} failed", e)

# intentar extraer las predicciones reales
if hasattr(pred_result, "predictions"):
    raw_predictions = pred_result.predictions
    x_input = getattr(pred_result, "x", None)
else:
    # fallback a comportamiento anterior
    if isinstance(pred_result, tuple) or isinstance(pred_result, list):
        raw_predictions = pred_result[0]
        x_input = pred_result[1] if len(pred_result) > 1 else None
    else:
        raw_predictions = pred_result
        x_input = None

# Última ventana predicha / extracción de mediana según formato
if hasattr(pred_result, "output"):
    out = pred_result.output
else:
    out = raw_predictions

try:
    last_pred = out[-1]
except Exception:
    last_pred = out

# Si last_pred es 1D (por ejemplo torch.Size([pred_horizon])) usamos directamente esa serie
if getattr(last_pred, "ndim", None) == 1:
    y_future = last_pred.detach().cpu().numpy()
else:
    # 2D: [prediction_length, num_quantiles]
    quantiles = getattr(model, "quantiles", None)
    if quantiles and 0.5 in quantiles:
        median_idx = quantiles.index(0.5)
    else:
        median_idx = last_pred.shape[1] // 2
    y_future = last_pred[:, median_idx].detach().cpu().numpy()

# --- Fechas futuras ---
last_date = df["Date"].max()
freq = pd.infer_freq(df["Date"])
if freq is None:
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=pred_horizon, freq="D")
else:
    future_dates = pd.date_range(last_date, periods=pred_horizon + 1, freq=freq)[1:]

# --- Serie histórica ---
historical = df.set_index("Date")[ventas_col]

# --- Graficar estilo trading ---
plt.figure(figsize=(12,6))
plt.plot(historical.index, historical.values, color="blue", label="Ventas históricas")
plt.plot(future_dates, y_future, color="red", label="Predicción futura")
plt.title(f"Ventas históricas y predicción (PyTorch Forecasting) — {city}")
plt.xlabel("Fecha")
plt.ylabel("Ventas")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# --- Dirección de la predicción ---
last_real = historical.values[-1]
last_forecast = y_future[-1]
direction = "SUBE " if last_forecast > last_real else "BAJA "
print(f"Dirección de la predicción: {direction} (último real: {last_real:.2f} → último predicho: {last_forecast:.2f})")