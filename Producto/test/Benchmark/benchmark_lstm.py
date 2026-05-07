from pathlib import Path
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    max_error,
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# 1. Configuración de Rutas
ROOT_DIR = Path(__file__).resolve().parents[3]
TRAINING_PATH = ROOT_DIR / "NoSeSube" / "Data" / "DATOS_ENTRENAMIENTO.csv"
VALIDATION_PATH = ROOT_DIR / "NoSeSube" / "Data" / "DATOS_VALIDACION.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "benchmark_result"
OUTPUT_DIR.mkdir(exist_ok=True)

# 2. Carga de Datos
ENTRENAMIENTO = pd.read_csv(TRAINING_PATH).head(5000)
VALIDACION = pd.read_csv(VALIDATION_PATH)

features = ['var1','var2','var3','var4','var5','var6','var7','var8','var9','var10','var11','var12','var13']
target = 'INTENSIDAD_4H'

X_train_raw = ENTRENAMIENTO[features]
y_train = ENTRENAMIENTO[target]
X_valid_raw = VALIDACION[features]
y_true = VALIDACION[target]

# 3. Preprocesamiento (Escalado y Reshape a 3D)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_valid_scaled = scaler.transform(X_valid_raw)

# Formato requerido para LSTM: (muestras, pasos_de_tiempo, características)
X_train = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_valid = X_valid_scaled.reshape((X_valid_scaled.shape[0], 1, X_valid_scaled.shape[1]))

# 4. Construcción del Modelo LSTM
# Mantenemos la estructura similar a la GRU para una comparación justa
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 5. Entrenamiento y Predicción (Sin visualización de progreso)
start_time = time.perf_counter()
model.fit(X_train, y_train, epochs=20, batch_size=256, verbose=0)
end_time = time.perf_counter()
execution_time = end_time - start_time

y_pred = model.predict(X_valid, verbose=0).ravel()
VALIDACION['PREDICCION_LSTM'] = y_pred

# 6. Evaluación (Top 5% de mayor probabilidad)
VALIDACION_ORD = VALIDACION.sort_values(by='PREDICCION_LSTM', ascending=False)
top_n = int(len(VALIDACION_ORD) * 0.05)
PC5_MAS_PROB_VALIDACION = VALIDACION_ORD.head(top_n)

# Hit rate de arriendos
promedio_arriendo = PC5_MAS_PROB_VALIDACION['ARRIENDO'].mean()

# Métricas de error
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
medae = median_absolute_error(y_true, y_pred)
m_error = max_error(y_true, y_pred)

log_cosh = np.log(np.cosh(y_pred - y_true)).mean()
y_true_diff = np.diff(y_true)
y_pred_diff = np.diff(y_pred)
da = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))

# 7. Salida de Resultados
print("-" * 50)
print(" BENCHMARK: LONG SHORT-TERM MEMORY (LSTM) ")
print("-" * 50)
print(f"Tiempo de ejecución: {execution_time:.4f} segundos")
print(f"Precisión en grupo de alto riesgo (Top 5%): {promedio_arriendo:.4f}")
print("-" * 25)
print(f"MAE (Error Absoluto Medio): {mae:.4f}")
print(f"RMSE (Raíz Error Cuadrático Medio): {rmse:.4f}")
print(f"MedAE (Error Absoluto Mediano): {medae:.4f}")
print(f"Max Error (Error Máximo): {m_error:.4f}")
print("-" * 25)
print(f"R² (Coef. de Determinación): {r2:.4f}")
print(f"MAPE (Error Porcentual Medio): {mape:.4f}")
print(f"Log-Cosh Loss: {log_cosh:.4f}")
print(f"Directional Accuracy (DA): {da:.4f}")
print("-" * 50)

# 8. Guardar en CSV
OUTPUT_FILE = OUTPUT_DIR / "resultados_lstm.csv"
VALIDACION.to_csv(OUTPUT_FILE, index=False)
print(f"Resultados guardados en: {OUTPUT_FILE}")