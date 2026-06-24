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

# Funcion export_dataframe_cubes()
MODELO_ENTRENADO_RP = modelo.export_dataframe_cubes()
print(MODELO_ENTRENADO_RP.head())
# MODELO_ENTRENADO_RP.to_csv("scoring_comparacion.csv", index=False)

VALIDACION = X_valid.copy()

VALIDACION["pred"] = modelo.predict(X_valid)
VALIDACION["cube_id"] = modelo.predict_cubes(X_valid)


# NUEVA FUNCION OMAIGAAA: Me devuelve las condiciones en las que se hicieron la grilla donde se cortó enb cada variables 
# - cube_id: entrega el id del cubo 
# - group_id: el grupo de corte
# - estable: bool si fue estable o no ese cubo en la pred
# - regla_completa: donde fue que se cortó realmente como la condicion que hacia Lipari 

cube_grid = modelo.export_cubes_grid()
cube_grid.to_csv("grilla.csv", index=False)



# conteo_validacion = (
#     VALIDACION
#     .groupby("cube_id")
#     .size()
#     .reset_index(name="n_validacion")
# )


# grid_con_validacion = cube_grid.merge(
#     conteo_validacion,
#     on="cube_id",
#     how="left"
# )

# grid_con_validacion["n_validacion"] = (
#     grid_con_validacion["n_validacion"]
#     .fillna(0)
#     .astype(int)
# )

# grid_con_validacion.to_csv("grillaconcvalid.csv", index=False)