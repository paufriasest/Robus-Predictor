import pandas as pd
from robuspredictor import RobusPredictor

# Dataset de entrenamiento
X_train = pd.DataFrame({
    "edad": [20, 21, 22, 23, 20, 21, 22, 23],
    "ingreso": [500, 510, 520, 530, 505, 515, 525, 535],
})

y_train = pd.Series([1.5, 1.6, 1.55, 1.65, 1.52, 1.62, 1.58, 1.68])


modelo = RobusPredictor(
    n_min=2,
    n_max=4,
    n_dom=2,
    mean_max=2.0,
    mean_min=1.0,
    std=0.20,
    default_value=0,
)

modelo.fit(X_train, y_train)

print("Cubos generados:")
print(modelo.cubes)

print("\nCubos estables:")
print(modelo.stable_cubes)


# Datos nuevos
X_new = pd.DataFrame({
    "edad": [21, 22, 50],
    "ingreso": [512, 527, 2000],
})

df_resultado = X_new.copy()
df_resultado["pred"] = modelo.predict(X_new)

print("\nPredicciones:")
print(df_resultado)