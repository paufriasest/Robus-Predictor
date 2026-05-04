from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# 3. Entrenamiento XGBoost
modelo_xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    verbosity=0 # Omitimos progreso
)

modelo_xgb.fit(X_train, y_train)

# 4. Predicción
VALIDACION['PREDICCION_XGB'] = modelo_xgb.predict(X_valid)

# 5. Evaluación (Top 5% Riesgo)
VALIDACION_ORD = VALIDACION.sort_values(by='PREDICCION_XGB', ascending=False)
top_n = int(len(VALIDACION_ORD) * 0.05)
PC5_MAS_PROB_VALIDACION = VALIDACION_ORD.head(top_n)

promedio_arriendo = PC5_MAS_PROB_VALIDACION['ARRIENDO'].mean()
y_pred = VALIDACION['PREDICCION_XGB']
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# 6. Reporte
print("-" * 50)
print(" BENCHMARK: XGBOOST REGRESSOR ")
print("-" * 50)
print(f"Precisión en grupo de alto riesgo (Top 5%): {promedio_arriendo:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print("-" * 50)

# 7. Guardar
VALIDACION.to_csv(OUTPUT_DIR / "resultados_xgboost.csv", index=False)