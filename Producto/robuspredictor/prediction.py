import pandas as pd

# Función que verifica si una fila nueva pertenece a un cubo estable.
# Recibe:
# - row: una fila del DataFrame X.
# - cube_bounds: los límites mínimos y máximos del cubo.
def row_belongs_to_cube(row, cube_bounds):
    
    # Recorre cada variable del cubo y sus límites
    for column, limits in cube_bounds.items():
        # Obtiene el valor de la fila nueva para la columna actual.
        value = row[column]
        
        # Verifica si el valor está fuera del rango permitido por el cubo.
        # Si está por debajo del mínimo o por encima del máximo,
        # significa que la fila no pertenece a este cubo.
        if value < limits["min"] or value > limits["max"]:
            return False
    # Si ninguna variable quedó fuera de los límites,
    # entonces la fila pertenece al cubo.
    return True

# Función que genera predicciones usando los cubos estables.
# Recibe:
# - X: DataFrame con los nuevos datos a predecir.
# - stable_cubes: lista de cubos estables encontrados durante fit().
# - default_value: valor por defecto cuando una fila no cae en ningún cubo estable.
def predict_from_stable_cubes(X, stable_cubes, default_value):
    # Lista donde se almacenarán las predicciones finales.
    predictions = []

    # Recorre cada fila del DataFrame X.
    # El guion bajo "_" representa el índice de la fila, pero no lo usamos directamente.
    for _, row in X.iterrows():
        # Se asigna inicialmente el valor por defecto.
        # Si la fila no cae en ningún cubo estable, este será el resultado final.
        prediction = default_value

        # Recorre cada cubo estable guardado por el modelo
        for cube in stable_cubes:
              # Verifica si la fila actual pertenece al cubo estable.
            if row_belongs_to_cube(row, cube["bounds"]):
                 # Si pertenece, se usa el valor predictivo asociado al cubo.
                prediction = cube["prediction_value"]
                    # Se detiene la búsqueda porque ya se encontró un cubo válido.
                break
        # Guarda la predicción obtenida para la fila actual.
        predictions.append(prediction)

    # Retorna las predicciones como una Series de pandas.
    # Se usa el mismo índice de X para que pueda asignarse directamente:
    # df["pred"] = modelo.predict(X)
    return pd.Series(predictions, index=X.index, name="pred")