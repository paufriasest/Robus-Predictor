from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

features = [
    'var1','var2','var3','var4','var5','var6','var7',
    'var8','var9','var10','var11','var12','var13'
]
target = 'INTENSIDAD_4H'

X_train = ENTRENAMIENTO[features]
y_train = ENTRENAMIENTO[target]
X_valid = VALIDACION[features]

# 3. Entrenamiento del Modelo Random Forest
# Usamos parámetros estándar para un benchmark sólido
modelo_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1  # Usa todos los núcleos disponibles para acelerar el cálculo silenciosamente
)

modelo_rf.fit(X_train, y_train)

# 4. Predicción
VALIDACION['PREDICCION_RF'] = modelo_rf.predict(X_valid)

# 5. Evaluación del Desempeño (Top 5% Riesgo)
VALIDACION_ORD = VALIDACION.sort_values(by='PREDICCION_RF', ascending=False)
top_n = int(len(VALIDACION_ORD) * 0.05)
PC5_MAS_PROB_VALIDACION = VALIDACION_ORD.head(top_n)

# Métrica de negocio: Promedio de arriendos en el grupo de mayor riesgo
promedio_arriendo = PC5_MAS_PROB_VALIDACION['ARRIENDO'].mean()

# Métricas estadísticas
y_true = VALIDACION[target]
y_pred = VALIDACION['PREDICCION_RF']
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# 6. Reporte de Resultados
print("-" * 50)
print(" BENCHMARK: RANDOM FOREST REGRESSOR ")
print("-" * 50)
print(f"Precisión en grupo de alto riesgo (Top 5%): {promedio_arriendo:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print("-" * 50)

# 7. Guardar resultados
OUTPUT_FILE = OUTPUT_DIR / "resultados_random_forest.csv"
VALIDACION.to_csv(OUTPUT_FILE, index=False)
print(f"Resultados guardados en: {OUTPUT_FILE}")