from pathlib import Path
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    mean_absolute_percentage_error,
    median_absolute_error,
    max_error
)

# 1. Configuración de Rutas (Estructura ROBUS-PREDICTOR)
ROOT_DIR = Path(__file__).resolve().parents[3]
TRAINING_PATH = ROOT_DIR / "NoSeSube" / "Data" / "DATOS_ENTRENAMIENTO.csv"
VALIDATION_PATH = ROOT_DIR / "NoSeSube" / "Data" / "DATOS_VALIDACION.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "benchmark_result"
OUTPUT_DIR.mkdir(exist_ok=True)

# 2. Carga de Datos
ENTRENAMIENTO = pd.read_csv(TRAINING_PATH)
VALIDACION = pd.read_csv(VALIDATION_PATH)

# 3. Configuración del Modelo Tradicional
features = [
    'var1','var2','var3','var4','var5','var6','var7',
    'var8','var9','var10','var11','var12','var13'
]
target = 'INTENSIDAD_4H'

X_train = ENTRENAMIENTO[features]
y_train = ENTRENAMIENTO[target]
X_valid = VALIDACION[features]
y_true = VALIDACION[target]

# 4. Entrenamiento, Predicción y Medición de Tiempo
start_time = time.perf_counter()  # Inicio del cronómetro

modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_valid)

end_time = time.perf_counter()    # Fin del cronómetro
execution_time = end_time - start_time

VALIDACION['PREDICCION'] = y_pred

# 5. Evaluación del "Hit Rate" (5% Superior)
VALIDACION_ORD = VALIDACION.sort_values(by='PREDICCION', ascending=False)
top_n = int(len(VALIDACION_ORD) * 0.05)
PC5_MAS_PROB_VALIDACION = VALIDACION_ORD.head(top_n)
promedio_arriendo = PC5_MAS_PROB_VALIDACION['ARRIENDO'].mean()

# 6. Cálculo de Métricas Adicionales
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
medae = median_absolute_error(y_true, y_pred)
m_error = max_error(y_true, y_pred)

# Log-Cosh Loss
log_cosh = np.log(np.cosh(y_pred - y_true)).mean()

# Directional Accuracy (DA)
# Compara si el movimiento (subida/bajada) de la predicción coincide con el real
y_true_diff = np.diff(y_true)
y_pred_diff = np.diff(y_pred)
da = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))

# 7. Salida de Resultados
print("-" * 50)
print(" BENCHMARK: REGRESIÓN LINEAL TRADICIONAL ")
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

# Guardar el CSV para análisis posterior
OUTPUT_FILE = OUTPUT_DIR / "resultados_regresion_lineal.csv"
VALIDACION.to_csv(OUTPUT_FILE, index=False)
print(f"Resultados detallados guardados en: {OUTPUT_FILE}")