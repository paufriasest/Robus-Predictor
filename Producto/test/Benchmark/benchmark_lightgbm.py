from pathlib import Path
import pandas as pd
import numpy as np
import time
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    max_error,
)

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

X_train, y_train = ENTRENAMIENTO[features], ENTRENAMIENTO[target]
X_valid, y_true = VALIDACION[features], VALIDACION[target]

# 3. Entrenamiento LightGBM
start_time = time.perf_counter()
modelo_lgb = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    importance_type='gain',
    verbose=-1 # Omitimos progreso
)

modelo_lgb.fit(X_train, y_train)

# 4. Predicción
y_pred = modelo_lgb.predict(X_valid)
end_time = time.perf_counter()
execution_time = end_time - start_time

VALIDACION['PREDICCION_LGBM'] = y_pred

# 5. Evaluación (Top 5% Riesgo)
VALIDACION_ORD = VALIDACION.sort_values(by='PREDICCION_LGBM', ascending=False)
top_n = int(len(VALIDACION_ORD) * 0.05)
PC5_MAS_PROB_VALIDACION = VALIDACION_ORD.head(top_n)

promedio_arriendo = PC5_MAS_PROB_VALIDACION['ARRIENDO'].mean()

# 6. Cálculo de Métricas Estadísticas
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
medae = median_absolute_error(y_true, y_pred)
m_error = max_error(y_true, y_pred)

# Log-Cosh Loss
log_cosh = np.log(np.cosh(y_pred - y_true)).mean()

# Directional Accuracy (DA)
y_true_diff = np.diff(y_true)
y_pred_diff = np.diff(y_pred)
da = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))

print("-" * 50)
print(" BENCHMARK: LIGHTGBM REGRESSOR ")
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

# 7. Guardar
VALIDACION.to_csv(OUTPUT_DIR / "resultados_lightgbm.csv", index=False)
print(f"Resultados guardados en: {OUTPUT_DIR / 'resultados_lightgbm.csv'}")