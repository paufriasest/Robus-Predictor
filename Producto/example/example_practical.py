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
print(f"Scoring Top 5%: {resultado_top5:.2%}")

# Funcion predict_cubes, que permite agregar una nueva columna indicando a que cubo corresponde el valor 
cube_ids = modelo.predict_cubes(X_valid)

resultado = X_valid.copy()
resultado["cube_id"] = cube_ids

print(resultado)

# Funcion export_dataframe_cubes()
cubes_df = modelo.export_dataframe_cubes()
cubes_df.head()

# Funcion que retorna las condiciones en las que se hicieron la grilla donde se cortó en cada variables 
cube_grid = modelo.export_cubes_grid()
cube_grid.to_csv("grilla.csv", index=False)