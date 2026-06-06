import pandas as pd
import json

def get_row_group_id(row, cuts_by_node):
    """
    Recorre el árbol de cortes para una fila individual y retorna
    el group_id final donde cae la fila.

    Ejemplo:
        ROOT -> L -> LR -> LRR

    Parámetros:
    -----------
    row : pd.Series
        Fila individual del DataFrame X.

    cuts_by_node : dict
        Diccionario de cortes indexado por node_id.

    Retorna:
    --------
    str
        group_id final.
    """

    node_id = "ROOT"

    while node_id in cuts_by_node:
        cut = cuts_by_node[node_id]

        variable = cut["variable"]
        left_max = cut["left_max"]

        left_path = cut["left_path"]
        right_path = cut["right_path"]

        value = row[variable]

        if pd.isna(value):
            # Misma lógica usada en sort_values(..., na_position="last"):
            # los nulos quedan al final, por lo tanto se van a la derecha.
            node_id = right_path
        elif value <= left_max:
            node_id = left_path
        else:
            node_id = right_path

    return node_id


def predict_from_stable_cubes(
    X,
    stable_cubes,
    red_zones,
    cuts,
    use_default_value,
    default_value,
    verbose=False
):
    """
    Genera predicciones usando los cubos calculados durante fit().
    
    La predicción se realiza así:
        1. Cada fila se envía por el árbol de cortes aprendido.
        2. Se obtiene el group_id final.
        3. Si el group_id está en stable_cubes, se usa su prediction_value.
        4. Si el group_id está en red_zones:
            - si use_default_value=True, se usa default_value.
            - si use_default_value=False, se usa prediction_value de la zona roja.
        5. Si el group_id no existe ni en stable_cubes ni en red_zones,
        se lanza error por inconsistencia interna.
        
    Importante:
        prediction_value representa el promedio de los promedios del cubo
        entre los dominios de entrenamiento.
    Parámetros:
    -----------
    X : pd.DataFrame
        Nuevos datos a predecir.

    stable_cubes : list[dict]
        Lista de cubos estables generados en fit().
        Cada cubo debe contener al menos:
            - group_id
            - prediction_value

    cuts : list[dict]
        Lista de cortes aprendidos desde el dominio base.

    default_value : float
        Valor por defecto si una fila cae en una zona no estable.

    verbose : bool
        Si True, imprime trazabilidad.

    Retorna:
    --------
    pd.Series
        Serie con las predicciones.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X debe ser un pandas DataFrame.")

    if stable_cubes is None:
        raise ValueError("stable_cubes no puede ser None.")
    
    if red_zones is None:
        raise ValueError("red_zones no puede ser None.")

    if cuts is None:
        raise ValueError("cuts no puede ser None.")
    
    if not isinstance(use_default_value, bool):
        raise TypeError("use_default_value debe ser booleano: True o False.")
    
    if use_default_value and default_value is None:
        raise ValueError(
            "default_value no puede ser None cuando use_default_value=True."
        )

    # Indexar cortes por nodo para recorrer el árbol eficientemente
    cuts_by_node = {
        cut["node_id"]: cut
        for cut in cuts
    }

    # Indexar cubos estables por group_id para búsqueda rápida
    stable_cubes_by_group = {
        cube["group_id"]: cube
        for cube in stable_cubes
    }
    
    red_zones_by_group = {
        cube["group_id"]: cube
        for cube in red_zones
    }

    predictions = []
    assigned_groups = []
    predicted_cubes = {}

    if verbose:
        print("\n[Predict] Inicio de predicción")
        print(f"[Predict] Registros a predecir: {len(X)}")
        print(f"[Predict] Cortes disponibles: {len(cuts)}")
        print(f"[Predict] Cubos estables disponibles: {len(stable_cubes)}")

    for row_number, (idx, row) in enumerate(X.iterrows(), start=1):
        group_id = get_row_group_id(
            row=row,
            cuts_by_node=cuts_by_node
        )

        assigned_groups.append(group_id)

        if group_id in stable_cubes_by_group:
            cube = stable_cubes_by_group[group_id]
            prediction = cube["prediction_value"]
            cube_type = "estable"

            if verbose:
                print(
                    f"[Predict] Fila {row_number} | "
                    f"index={idx} | "
                    f"grupo={group_id} | "
                    f"zona estable | "
                    f"pred_asignada_registro={prediction}"
                )
        elif group_id in red_zones_by_group:
            cube = red_zones_by_group[group_id]
            cube_type = "zona_roja_default"
            
            if use_default_value:
                prediction = default_value
                
                if verbose:
                    print(
                        f"[Predict] Fila {row_number} | "
                        f"index={idx} | "
                        f"grupo={group_id} | "
                        f"zona roja | "
                        f"default={default_value}"
                    )
                    
            else:
                prediction = cube["prediction_value"]
                cube_type = "zona_roja_promedio"

                if verbose:
                    print(
                        f"[Predict] Fila {row_number} | "
                        f"index={idx} | "
                        f"grupo={group_id} | "
                        f"zona roja | "
                        f"pred_zona_roja={prediction}"
                    )
        else:
            raise ValueError(
                f"Inconsistencia interna del modelo: el registro con índice {idx} "
                f"cayó en el cubo '{group_id}', pero ese cubo no existe ni en "
                "stable_cubes ni en red_zones. Esto puede ocurrir si se modificaron "
                "manualmente los cortes, los cubos estables o las zonas rojas "
                "después del entrenamiento."
            )
            
        if group_id not in predicted_cubes:
            predicted_cubes[group_id] = {
                "group_id": group_id,
                "tipo": cube_type,
                "prediccion": prediction,
                "cantidad_registros": 0
            }
        
        predicted_cubes[group_id]["cantidad_registros"] += 1
        predictions.append(prediction)
        
    if verbose:
        print("\n[Predict] Cubos utilizados en predicción:")
        print(f"[Predict] Total registros predichos: {len(X)}")
        print(f"[Predict] Total cubos utilizados: {len(predicted_cubes)}")
        
        for cube_info in predicted_cubes.values():
            print(
                f"- {cube_info['group_id']} | "
                f"{cube_info['tipo']} | "
                f"registros={cube_info['cantidad_registros']} | "
                f"pred={cube_info['prediccion']}"
            )
    
    return pd.Series(predictions, index=X.index, name="pred")

# Funcion que ocuparemos para guardar informacion respecto a los cubos que se les hace la predicción
def build_prediction_detail(
    X,
    stable_cubes,
    red_zones,
    cuts,
    default_value
):
    
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X debe ser un pandas DataFrame.")
    
    if stable_cubes is None:
        raise ValueError("stable_cubes no puede ser None.")
    
    if red_zones is None:
        red_zones = []
        
    if cuts is None:
        raise ValueError("cuts no puede ser None.")
    
    cuts_by_node = {
        cut["node_id"]: cut
        for cut in cuts
    }
    
    stable_cubes_by_group = {
        cube["group_id"]: cube
        for cube in stable_cubes
    }
    
    red_zones_by_group = {
        cube["group_id"]: cube
        for cube in red_zones
    }
    
    rows = []
    
    for idx, row in X.iterrows():
        
        group_id = get_row_group_id(
            row=row,
            cuts_by_node=cuts_by_node
        )
        
        if group_id in stable_cubes_by_group:
            cube = stable_cubes_by_group[group_id]
            stable = 1
            promedio_cubo = cube.get("prediction_value")
            prediccion_aplicada = promedio_cubo
            motivo_rechazo = []
        
        else:
            stable = 0
            prediccion_aplicada = default_value
            
            if group_id in red_zones_by_group:
                cube = red_zones_by_group[group_id]
                promedio_cubo = cube.get("prediction_value")
                motivo_rechazo = cube.get("rejection_reasons", [])
            else:
                promedio_cubo = None
                motivo_rechazo = [
                    "El registro cayó en un cubo no encontrado dentro de stable_cubes ni red_zones."
                ]
        
        rows.append({
            "indice_registro": idx,
            "id_cubo": group_id,
            "estable": stable,
            "promedio_cubo": promedio_cubo,
            "prediccion_aplicada": prediccion_aplicada,
            "motivo_rechazo": json.dumps(motivo_rechazo, ensure_ascii=False),
        })
        
        
    return pd.DataFrame(rows).set_index("indice_registro")