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
def predict_from_stable_cubes(X, stable_cubes, default_value, verbose=False):
    # Lista donde se almacenarán las predicciones finales.
    predictions = []
    
    if verbose:
        print("\n[Predict] Inicio de predicción")
        print(f"[Predict] Registros a predecir: {len(X)}")
        print(f"[Predict] Cubos estables disponibles: {len(stable_cubes)}")
        
    # Recorre cada fila del DataFrame X.
    # El guion bajo "_" representa el índice de la fila, pero no lo usamos directamente.
    for row_number, (_, row) in enumerate(X.iterrows(), start=1):
        # Se asigna inicialmente el valor por defecto.
        # Si la fila no cae en ningún cubo estable, este será el resultado final.
        prediction = default_value
        matched = False
        
        for cube in stable_cubes:
            if row_belongs_to_cube(row, cube["bounds"]):
                prediction = cube["prediction_value"]
                matched = True
                
                if verbose:
                    print(
                        f"[Predict] Fila {row_number}: coincide con cubo "
                        f"posición {cube['cube_position']} | pred={prediction}"
                    )
                    
                break
            
        if verbose and not matched:
            print(
                f"[Predict] Fila {row_number}: sin coincidencia, "
                f"default={default_value}"
            )
            
        predictions.append(prediction)
        
    return pd.Series(predictions, index=X.index, name="pred")