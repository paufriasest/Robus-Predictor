from pathlib import Path
import pandas as pd
import numpy as np
import time
from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    max_error,
)

# 1. Rutas del Proyecto
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

# 3. Preparación para Prophet (ds y y)
def prepare_for_prophet(df, is_train=True):
    pdf = df.copy()
    # Si no tienes columna de fecha, creamos una secuencia diaria artificial
    if 'fecha' not in pdf.columns:
        pdf['ds'] = pd.date_range(start='2026-01-01', periods=len(pdf), freq='H')
    else:
        pdf['ds'] = pd.to_datetime(pdf['fecha'])
    
    if is_train:
        pdf['y'] = pdf[target]
    return pdf

train_p = prepare_for_prophet(ENTRENAMIENTO)
valid_p = prepare_for_prophet(VALIDACION, is_train=False)

# 4. Configuración del Modelo Prophet
# En entornos de alta volatilidad, ajustamos la flexibilidad de la tendencia
m = Prophet(
    changepoint_prior_scale=0.05, 
    yearly_seasonality=False, 
    weekly_seasonality=True, 
    daily_seasonality=True
)

# Añadimos las variables predictoras como regresores adicionales
for feature in features:
    m.add_regressor(feature)

print("Entrenando Prophet (esto puede tardar un poco más que los otros modelos)...")
start_time = time.perf_counter()
m.fit(train_p)

# 5. Predicción
forecast = m.predict(valid_p)
end_time = time.perf_counter()
execution_time = end_time - start_time
VALIDACION['PREDICCION_PROPHET'] = forecast['yhat'].values

# 6. Evaluación (Top 5% Riesgo)
VALIDACION_ORD = VALIDACION.sort_values(by='PREDICCION_PROPHET', ascending=False)
top_n = int(len(VALIDACION_ORD) * 0.05)
PC5_MAS_PROB_VALIDACION = VALIDACION_ORD.head(top_n)

promedio_arriendo = PC5_MAS_PROB_VALIDACION['ARRIENDO'].mean()

# Métricas Estadísticas
y_true = VALIDACION[target]
y_pred = VALIDACION['PREDICCION_PROPHET']
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
print(" BENCHMARK: FACEBOOK PROPHET ")
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
OUTPUT_FILE = OUTPUT_DIR / "resultados_prophet.csv"
VALIDACION.to_csv(OUTPUT_FILE, index=False)
print(f"Resultados guardados en: {OUTPUT_FILE}")