from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input

# 1. Configuración de Rutas
ROOT_DIR = Path(__file__).resolve().parents[3]
TRAINING_PATH = ROOT_DIR / "NoSeSube" / "Data" / "DATOS_ENTRENAMIENTO.csv"
VALIDATION_PATH = ROOT_DIR / "NoSeSube" / "Data" / "DATOS_VALIDACION.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "benchmark_result"
OUTPUT_DIR.mkdir(exist_ok=True)

# 2. Carga de Datos
ENTRENAMIENTO = pd.read_csv(TRAINING_PATH)
VALIDACION = pd.read_csv(VALIDATION_PATH)

features = ['var1','var2','var3','var4','var5','var6','var7','var8','var9','var10','var11','var12','var13']
target = 'INTENSIDAD_4H'

X_train_raw = ENTRENAMIENTO[features]
y_train = ENTRENAMIENTO[target]
X_valid_raw = VALIDACION[features]
y_true = VALIDACION[target]

# 3. Preprocesamiento y Reshaping
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_valid_scaled = scaler.transform(X_valid_raw)

# Reformateo a 3D: (samples, time_steps, features) -> Usamos time_steps=1
X_train = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_valid = X_valid_scaled.reshape((X_valid_scaled.shape[0], 1, X_valid_scaled.shape[1]))

# 4. Construcción del Modelo GRU
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    GRU(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 5. Entrenamiento y Predicción (Silencioso)
model.fit(X_train, y_train, epochs=20, batch_size=256, verbose=0)

VALIDACION['PREDICCION_GRU'] = model.predict(X_valid, verbose=0)

# 6. Evaluación del Desempeño (Top 5% Riesgo)
VALIDACION_ORD = VALIDACION.sort_values(by='PREDICCION_GRU', ascending=False)
top_n = int(len(VALIDACION_ORD) * 0.05)
PC5_MAS_PROB_VALIDACION = VALIDACION_ORD.head(top_n)

promedio_arriendo = PC5_MAS_PROB_VALIDACION['ARRIENDO'].mean()

# Métricas Estadísticas
y_pred = VALIDACION['PREDICCION_GRU']
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# 7. Reporte de Resultados
print("-" * 50)
print(" BENCHMARK: GATED RECURRENT UNIT (GRU) ")
print("-" * 50)
print(f"Precisión en grupo de alto riesgo (Top 5%): {promedio_arriendo:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print("-" * 50)

# 8. Guardar resultados
OUTPUT_FILE = OUTPUT_DIR / "resultados_gru.csv"
VALIDACION.to_csv(OUTPUT_FILE, index=False)
print(f"Resultados guardados en: {OUTPUT_FILE}")