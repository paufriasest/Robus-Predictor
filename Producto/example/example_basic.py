import pandas as pd
from robuspredictor import RobusPredictor

# Datasets de entrenamientos
# TODO: ARREGLAR ESTA WEA

X_train = pd.DataFrame({
    "var1": [
        52, 10, 13, 51, 11, 50, 12, 53,
        15, 55, 16, 56, 14, 54, 17, 57,
        12.5, 50.5, 11.5, 53.5, 10.5, 52.5, 13.5, 51.5,
        15.5, 55.5, 16.5, 56.5, 14.5, 54.5, 17.5, 57.5,
    ],
    "var2": [
        82, 20, 23, 81, 21, 80, 22, 83,
        25, 85, 26, 86, 24, 84, 27, 87,
        22.5, 80.5, 21.5, 83.5, 20.5, 82.5, 23.5, 81.5,
        25.5, 85.5, 26.5, 86.5, 24.5, 84.5, 27.5, 87.5,
    ],
    "var3": [
        92, 30, 33, 91, 31, 90, 32, 93,
        35, 95, 36, 96, 34, 94, 37, 97,
        32.5, 90.5, 31.5, 93.5, 30.5, 92.5, 33.5, 91.5,
        35.5, 95.5, 36.5, 96.5, 34.5, 94.5, 37.5, 97.5,
    ],
})
#target
y_train = pd.Series([
    2.55, 1.50, 1.65, 2.60, 1.60, 2.50, 1.55, 2.65,
    1.70, 2.70, 1.75, 2.75, 1.68, 2.68, 1.78, 2.78,
    1.57, 2.52, 1.62, 2.67, 1.52, 2.57, 1.67, 2.62,
    1.72, 2.72, 1.77, 2.77, 1.70, 2.70, 1.80, 2.80,
])

modelo = RobusPredictor(
    n_min=2,
    n_max=4,
    n_dom=2,
    mean_max=3.0,
    mean_min=1.0,
    std_min=0.0,
    std_max=0.5,
    default_value=0,
    verbose=True
)

#FITITIFITIT
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


# Data de validación
X_valid = pd.DataFrame({
    "var1": [11, 52, 100],
    "var2": [21, 82, 100],
    "var3": [31, 92, 100],
})


resultado = X_valid.copy()
resultado["pred"] = modelo.predict(X_valid)


print("\nPredicciones:")
print(resultado)