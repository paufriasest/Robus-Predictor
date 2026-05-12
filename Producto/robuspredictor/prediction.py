import pandas as pd


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
    cuts,
    default_value,
    verbose=False
):
    """
    Genera predicciones usando los cubos estables encontrados durante fit().

    La predicción se realiza así:
        1. Cada fila se envía por el árbol de cortes aprendido.
        2. Se obtiene el group_id final.
        3. Si el group_id está en stable_cubes, se usa su prediction_value.
        4. Si no está, se usa default_value.

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

    if cuts is None:
        raise ValueError("cuts no puede ser None.")

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

    predictions = []
    assigned_groups = []

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

            if verbose:
                print(
                    f"[Predict] Fila {row_number} | "
                    f"index={idx} | "
                    f"grupo={group_id} | "
                    f"zona estable | "
                    f"pred={prediction}"
                )

        else:
            prediction = default_value

            if verbose:
                print(
                    f"[Predict] Fila {row_number} | "
                    f"index={idx} | "
                    f"grupo={group_id} | "
                    f"zona no estable | "
                    f"default={default_value}"
                )

        predictions.append(prediction)

    return pd.Series(predictions, index=X.index, name="pred")