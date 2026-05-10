import pandas as pd
from robuspredictor import RobusPredictor

# Datasets de entrenamientos
# VAR PREDIC
X1 = pd.DataFrame({
    "var1": [10, 11, 12, 50, 51, 52],
    "var2": [20, 21, 22, 80, 81, 82],
    "var3": [30, 31, 32, 90, 91, 92],
})
X2 = pd.DataFrame({
    "var1": [10.5, 11.5, 12.5, 50.5, 51.5, 52.5],
    "var2": [20.5, 21.5, 22.5, 80.5, 81.5, 82.5],
    "var3": [30.5, 31.5, 32.5, 90.5, 91.5, 92.5],
})

# VAR TARGET
y1 = pd.Series([1.5, 1.6, 1.55, 2.5, 2.6, 2.55])
y2 = pd.Series([1.55, 1.65, 1.60, 2.55, 2.65, 2.60])

#  INVOKER
modelo = RobusPredictor(
    n_min=2,
    n_max=4,
    n_dom=2,
    mean_max=2.0,
    mean_min=1.0,
    std_min=0.20,
    default_value=0,
    verbose=True
)

#FITITIFITIT
modelo.fit(X1, y1, X2, y2)

print("\nCubos por dominio:")
print(modelo.domain_cubes)

print("\nCubos estables:")
print(modelo.stable_cubes)

# DATA DE VALIDACION
X_new = pd.DataFrame({
    "var1": [11, 51, 100],
    "var2": [21, 81, 100],
    "var3": [31, 91, 100],
})

df_resultado = X_new.copy()
df_resultado["pred"] = modelo.predict(X_new)

print("\nPredicciones:")
print(df_resultado)