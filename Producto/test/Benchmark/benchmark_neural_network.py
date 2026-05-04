from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Configuración de Rutas (Estructura ROBUS-PREDICTOR)
ROOT_DIR = Path(__file__).resolve().parents[3]
TRAINING_PATH = ROOT_DIR / "NoSeSube" / "Data" / "DATOS_ENTRENAMIENTO.csv"
VALIDATION_PATH = ROOT_DIR / "NoSeSube" / "Data" / "DATOS_VALIDACION.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "benchmark_result"
OUTPUT_DIR.mkdir(exist_ok=True)

# 2. Carga de Datos
ENTRENAMIENTO = pd.read_csv(TRAINING_PATH)
VALIDACION = pd.read_csv(VALIDATION_PATH)

# 3. Definición de Variables
features = [
    'var1','var2','var3','var4','var5','var6','var7',
    'var8','var9','var10','var11','var12','var13'
]
target = 'INTENSIDAD_4H'

X_train = ENTRENAMIENTO[features]
y_train = ENTRENAMIENTO[target]
X_valid = VALIDACION[features]

# 4. Configuración y Entrenamiento de la Red Neuronal
modelo_nn = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(16,),      # Red simple para benchmark inicial
        activation='relu',
        solver='adam',
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=50,                   # Entrenamiento rápido
        early_stopping=True,
        n_iter_no_change=5,
        tol=1e-3,
        random_state=42
    ))
])

print("Entrenando Red Neuronal...")
modelo_nn.fit(X_train, y_train)

# 5. Predicción
VALIDACION['PREDICCION_NN'] = modelo_nn.predict(X_valid)

# 6. Evaluación del Desempeño (Top 5% Riesgo)
VALIDACION_ORD = VALIDACION.sort_values(by='PREDICCION_NN', ascending=False)
top_n = int(len(VALIDACION_ORD) * 0.05)
PC5_MAS_PROB_VALIDACION = VALIDACION_ORD.head(top_n)

# Métrica de negocio: Arriendos efectivos en el grupo de mayor riesgo
promedio_arriendo = PC5_MAS_PROB_VALIDACION['ARRIENDO'].mean()

# Métricas estadísticas
y_true = VALIDACION[target]
y_pred = VALIDACION['PREDICCION_NN']
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# 7. Reporte de Resultados
print("-" * 50)
print(" BENCHMARK: RED NEURONAL (MLP) ")
print("-" * 50)
print(f"Precisión en grupo de alto riesgo (Top 5%): {promedio_arriendo:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print("-" * 50)

# 8. Guardar resultados
OUTPUT_FILE = OUTPUT_DIR / "resultados_red_neuronal.csv"
VALIDACION.to_csv(OUTPUT_FILE, index=False)
print(f"Resultados guardados en: {OUTPUT_FILE}")