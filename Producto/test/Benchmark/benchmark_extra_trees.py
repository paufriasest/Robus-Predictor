from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
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

# 3. Entrenamiento de Extra Trees
# n_jobs=-1 para usar todos los núcleos del servidor silenciosamente
modelo_et = ExtraTreesRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

modelo_et.fit(X_train, y_train)

# 4. Predicción
VALIDACION['PREDICCION_ET'] = modelo_et.predict(X_valid)

# 5. Evaluación (Top 5% Riesgo)
VALIDACION_ORD = VALIDACION.sort_values(by='PREDICCION_ET', ascending=False)
top_n = int(len(VALIDACION_ORD) * 0.05)
PC5_MAS_PROB_VALIDACION = VALIDACION_ORD.head(top_n)

promedio_arriendo = PC5_MAS_PROB_VALIDACION['ARRIENDO'].mean()
y_pred = VALIDACION['PREDICCION_ET']
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# 6. Reporte de Resultados
print("-" * 50)
print(" BENCHMARK: EXTRA TREES REGRESSOR ")
print("-" * 50)
print(f"Precisión en grupo de alto riesgo (Top 5%): {promedio_arriendo:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print("-" * 50)

# 7. Guardar resultados
OUTPUT_FILE = OUTPUT_DIR / "resultados_extra_trees.csv"
VALIDACION.to_csv(OUTPUT_FILE, index=False)
print(f"Resultados guardados en: {OUTPUT_FILE}")