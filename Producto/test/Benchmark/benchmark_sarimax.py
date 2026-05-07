from pathlib import Path
import pandas as pd
import numpy as np
import time
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    max_error,
)

# Omitir warnings de convergencia para limpieza de consola
warnings.filterwarnings("ignore")

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

# 3. Preparación de Series Temporales
# Al igual que con Prophet, ARIMA requiere un orden temporal.
def prepare_ts(df):
    temp_df = df.copy()
    if 'fecha' not in temp_df.columns:
        temp_df.index = pd.date_range(start='2026-01-01', periods=len(temp_df), freq='H')
    else:
        temp_df['fecha'] = pd.to_datetime(temp_df['fecha'])
        temp_df.set_index('fecha', inplace=True)
    return temp_df

train_ts = prepare_ts(ENTRENAMIENTO)
valid_ts = prepare_ts(VALIDACION)

# 4. Configuración y Entrenamiento de SARIMAX
# order=(p, d, q) -> Autoregresivo, Integrado, Media Móvil
print("Entrenando SARIMAX (esto puede tardar dependiendo del volumen de datos)...")
start_time = time.perf_counter()
modelo_sarima = SARIMAX(
    train_ts[target],
    exog=train_ts[features],
    order=(1, 1, 1), 
    seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=False,
    enforce_invertibility=False
)

resultado = modelo_sarima.fit(disp=False)
end_time = time.perf_counter()
execution_time = end_time - start_time

# 5. Predicción
# Necesitamos pasar las variables exógenas de validación
predicciones = resultado.get_forecast(steps=len(valid_ts), exog=valid_ts[features])
VALIDACION['PREDICCION_SARIMA'] = predicciones.predicted_mean.values

# 6. Evaluación (Top 5% Riesgo)
VALIDACION_ORD = VALIDACION.sort_values(by='PREDICCION_SARIMA', ascending=False)
top_n = int(len(VALIDACION_ORD) * 0.05)
PC5_MAS_PROB_VALIDACION = VALIDACION_ORD.head(top_n)

promedio_arriendo = PC5_MAS_PROB_VALIDACION['ARRIENDO'].mean()
y_true = VALIDACION[target]
y_pred = VALIDACION['PREDICCION_SARIMA']
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

# 7. Reporte
print("-" * 50)
print(" BENCHMARK: SARIMAX (ARIMA con Exógenas) ")
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

# 8. Guardar
VALIDACION.to_csv(OUTPUT_DIR / "resultados_sarima.csv", index=False)
print(f"Resultados guardados en: {OUTPUT_DIR / 'resultados_sarima.csv'}")