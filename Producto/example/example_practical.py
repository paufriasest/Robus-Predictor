from pathlib import Path
import sys
import pandas as pd
from robuspredictor import RobusPredictor

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "Producto"))

TRAINING_PATH = ROOT_DIR / "NoSeSube" / "Data" / "ENTRENAMIENTO2.csv"
VALIDATION_PATH = ROOT_DIR / "NoSeSube" / "Data" / "VALIDACION2.csv"

ENTRENAMIENTO = pd.read_csv(TRAINING_PATH)
VALIDACION = pd.read_csv(VALIDATION_PATH)

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
    default_value= 0,
    verbose=  False,
    random_state=  42, 
)

# ── Entrenamiento ─────────────────────────────────────────────────────────────
modelo.fit(X_train, y_train)

print("\nResumen de dominios:")
for i, domain in enumerate(modelo.domains, start=1):
    print(f"\nDominio {i}")
    print(f"X shape: {domain['x'].shape}")
    print(f"y shape: {domain['y'].shape}")
    print(f"Cantidad de grupos: {len(domain['groups'])}")

print("\nCubos estables:")
for cube in modelo.stable_cubes:
    print(cube)

print("\nZonas rojas:")
for zone in modelo.red_zones:
    print(zone)


# ── Prediccion ────────────────────────────────────────────────────────────────
resultado = X_valid.copy()
resultado["pred"] = modelo.predict(X_valid)

print("\nPredicciones:")
print(resultado)

# ── Exportar checkpoint ───────────────────────────────────────────────────────
# Se pasan X_valid e y_valid para que el checkpoint incluya las columnas
# n_validacion, prom_target_validacion, std_target_validacion y
# prom_target_consolidado. Si no se pasan, esas columnas apareceran como null.
modelo.export_checkpoint(
    path="checkpoint_robuspredictor.xlsx",
    file_format="xlsx",
    X_valid=X_valid,
    y_valid=y_valid,
)

# Exportar el excel que sirve para la trazabilidad de las predicciones efectuadas, ocupa los mismos parametros que el excel anterior
modelo.export_prediction_checkpoint(
    X=X_valid,
    y=y_valid,
    path="scoring_robuspredictor.xlsx",
    dato_real=ARRIENDO_REAL,
    file_format="xlsx"
)