from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Configuración de Rutas (Estructura ROBUS-PREDICTOR)
# parents[3] llega a la raíz desde Producto/test/Benchmark/
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

# 4. Entrenamiento y Predicción
modelo = LinearRegression()
modelo.fit(X_train, y_train)

X_valid = VALIDACION[features]
VALIDACION['PREDICCION'] = modelo.predict(X_valid)

# 5. Evaluación del "Hit Rate" (5% Superior)
# Ordenamos para encontrar los casos donde el modelo predice mayor intensidad
VALIDACION_ORD = VALIDACION.sort_values(by='PREDICCION', ascending=False)
top_n = int(len(VALIDACION_ORD) * 0.05)
PC5_MAS_PROB_VALIDACION = VALIDACION_ORD.head(top_n)

# Métrica clave de negocio/tesis
promedio_arriendo = PC5_MAS_PROB_VALIDACION['ARRIENDO'].mean()

# 6. Métricas Estadísticas Adicionales
y_true = VALIDACION[target]
y_pred = VALIDACION['PREDICCION']
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# 7. Salida de Resultados
print("-" * 50)
print(" BENCHMARK: REGRESIÓN LINEAL TRADICIONAL ")
print("-" * 50)
print(f"Precisión en grupo de alto riesgo (Top 5%): {promedio_arriendo:.4f}")
print(f"MAE (Error Absoluto Medio): {mae:.4f}")
print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.4f}")
print("-" * 50)

# Guardar el CSV para análisis posterior de "arriendos innecesarios"
OUTPUT_FILE = OUTPUT_DIR / "resultados_regresion_lineal.csv"
VALIDACION.to_csv(OUTPUT_FILE, index=False)
print(f"Resultados detallados guardados en: {OUTPUT_FILE}")