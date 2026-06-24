import pandas as pd
from robuspredictor import RobusPredictor

ENTRENAMIENTO = pd.read_csv("NoSeSube/Data/ENTRENAMIENTO2.csv")
VALIDACION = pd.read_csv("NoSeSube/Data/VALIDACION2.csv")

features = [
    "var1", "var2", 
]
target = "INTENSIDAD_4H"

X_train = ENTRENAMIENTO[features]
y_train = ENTRENAMIENTO[target]

X_valid = VALIDACION[features]
y_valid = VALIDACION[target]

ARRIENDO_REAL= VALIDACION["ARRIENDO"]

modelo = RobusPredictor(
    n_min= 3, 
    n_max= 4, 
    n_dom= 2,
    mean_min= 0.05,
    mean_max=  16.0,
    std_min=  0.0,
    std_max=  16.0,
    default_value= 0
)

# ── Entrenamiento ─────────────────────────────────────────────────────────────
modelo.fit(X_train, y_train)

# ── Prediccion ────────────────────────────────────────────────────────────────
resultado = X_valid.copy()
resultado["pred"] = modelo.predict(X_valid)

# ── Exportar checkpoint ───────────────────────────────────────────────────────

# Para xlsx se exportara con nombre checkpoint_robuspredictor.xlsx
modelo.export_checkpoint(
    X_valid=X_valid,
    y_valid=y_valid
)

# Exportar el excel que sirve para la trazabilidad de las predicciones efectuadas, se le puede agregar el dato real que es nuestra var de comparacion
scoring_df = modelo.export_prediction_checkpoint(
    X_valid=X_valid,
    y_valid=y_valid,
    dato_real=ARRIENDO_REAL
)

# ── Funciones Utiles ───────────────────────────────────────────────────────

# Funcion Mejor N% 
resultado_top5 = modelo.best_percentage(
    y_target=ARRIENDO_REAL,
    top_pct=0.05
)

print(resultado_top5)
print(f"Scoring Top 5%: {resultado_top5:.2%}")
# En este caso el output es de 1.0 entonces
# El 100% de los registros ubicados en el 5% superior de predicciones realmente tuvo ARRIENDO = 1.

# Lo que nos entrega es solamente el float del scoring
# entonces el 5% es la cantidad total de registros que fueron predichos por el modelo
# 1. Toma todas las predicciones.
# 2. Las ordena de mayor a menor.
# 3. Selecciona el 5% con predicción más alta.
# 4. Revisa cuántos de esos registros tienen y_target = 1.
# 5. Calcula la precisión dentro de ese grupo.

# Funcion predict_cubes, que permite agregar una nueva columna indicando a que cubo corresponde el valor 
cube_ids = modelo.predict_cubes(X_valid)

resultado = X_valid.copy()
resultado["cube_id"] = cube_ids

print(resultado)

# NUEVA FUNCION OMAIGA: Retorna un dataframe que el usuario puede utilizar para trazabilidad contiene:
# - ID del cubo que es legible
# - <variable>_min que es el valor menor que tomó el cubo para esa variable
# - <variable>_max que es el valor mayor que tomó el cubo para esa variable
# - Pred prediccion efectuada a ese cubo en particular

MODELO_ENTRENADO_RP = modelo.export_dataframe_cubes()
print(MODELO_ENTRENADO_RP.head())

MODELO_ENTRENADO_RP.to_csv("scoring_comparacion.csv", index=False)