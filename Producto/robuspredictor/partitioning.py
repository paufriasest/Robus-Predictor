import pandas as pd
import numpy as np

def median_partition(x, n_min, n_max, verbose=False):
    """
    Divide un dataset usando particiones recursivas locales.

    La lógica es:
    - En cada nodo se recibe un subconjunto de filas.
    - Se selecciona la variable según el nivel del árbol.
    - Se ordena SOLO ese subconjunto por esa variable.
    - Se corta en dos lotes.
    - Cada lote conserva todas sus columnas.
    - La siguiente variable trabaja solamente sobre el lote recibido.
    - Se guardan los cortes para replicarlos sobre otros datasets.


    Parámetros:
    -----------
    x : pd.DataFrame
        Dataset base con filas como elementos y columnas como variables.

    n_min : int
        Tamaño mínimo de grupo final.

    n_max : int
        Tamaño máximo permitido de grupo final.

    verbose : bool
        Si es True, imprime detalle del proceso.

    Retorna:
    --------
    dict:
        {
            "groups": dict[str, pd.DataFrame],
            "cuts": list[dict],
            "variables": list[str]
        }
    """
    if not isinstance(x, pd.DataFrame):
        raise TypeError("x debe ser un DataFrame de pandas.")
    
    if x.empty:
        return {
            "groups": {},
            "cuts": [],
            "variables": []
        }

    if n_min <= 0:
        raise ValueError("n_min debe ser mayor a 0.")

    if n_max < n_min:
        raise ValueError("n_max debe ser mayor o igual a n_min.")

    variables = list(x.columns)

    if len(variables) == 0:
        raise ValueError("El x debe tener al menos una variable.")

    groups = {}
    cuts = []

    def cortar_recursivo(df_subset, nivel=0, path=""):
        """
        Esta función recibe solamente el lote actual.

        Ejemplo:
        - ROOT recibe todo x.
        - L recibe solo las filas que quedaron a la izquierda.
        - LL recibe solo las filas que quedaron a la izquierda de L.
        """
        
        tamaño_actual = len(df_subset)
        node_id = path if path != "" else "ROOT"

        # Caso base
        if tamaño_actual <= n_min:
            groups[node_id] = df_subset.copy()

            if verbose:
                print(
                    f"[FIN] Nodo {node_id} | "
                    f"Nivel {nivel} | "
                    f"Tamaño: {tamaño_actual} | "
                    f"Índices: {list(df_subset.index)}"
                )
                
            return
        
        # Evita cortar si el corte dejaría grupos menores al mínimo
        if tamaño_actual // 2 < n_min:
            groups[node_id] = df_subset.copy()
        
            if verbose:
                print(
                    f"[FIN SIN CORTE] Nodo {node_id} | "
                    f"Nivel {nivel} | "
                    f"Tamaño: {tamaño_actual} | "
                    f"No se puede dividir sin dejar grupos menores a n_min | "
                    f"Índices: {list(df_subset.index)}"
                )
            
            return
        
        variable_corte = variables[nivel % len(variables)]
        
        df_ordenado = df_subset.sort_values(
            by=variable_corte,
            kind="mergesort",
            na_position="last"
        )

        punto_corte = tamaño_actual // 2

        izquierda = df_ordenado.iloc[:punto_corte].copy()
        derecha = df_ordenado.iloc[punto_corte:].copy()

        left_path = path + "L"
        right_path = path + "R"
        
        left_max = izquierda[variable_corte].max()
        right_min = derecha[variable_corte].min()

        cut_info = {
            "node_id": node_id,
            "level": nivel,
            "variable": variable_corte,
            "left_path": left_path,
            "right_path": right_path,
            "left_size": len(izquierda),
            "right_size": len(derecha),
            "left_indices": list(izquierda.index),
            "right_indices": list(derecha.index),
            "left_max": left_max,
            "right_min": right_min,
            "cut_position": punto_corte,
            "rule": (
                f"{variable_corte} <= {left_max} va a izquierda; "
                f"{variable_corte} > {left_max} va a derecha"
            )
        }

        cuts.append(cut_info)

        if verbose:
            print("\n[CORTE]")
            print(f"Nodo: {node_id}")
            print(f"Nivel: {nivel}")
            print(f"Variable usada: {variable_corte}")
            print(f"Tamaño recibido: {tamaño_actual}")
            print(f"Índices recibidos: {list(df_subset.index)}")

            print("\nSubconjunto recibido:")
            print(df_subset)

            print(f"\nOrden local por {variable_corte}:")
            for idx, value in df_ordenado[variable_corte].items():
                print(f"  idx {idx} -> {variable_corte} = {value}")

            print(f"\nPunto de corte: {punto_corte}")

            print(f"\nIzquierda {left_path}:")
            print(izquierda)

            print(f"\nDerecha {right_path}:")
            print(derecha)

            print(
                f"\nRegla replicable: "
                f"{variable_corte} <= {left_max} -> {left_path}; "
                f"{variable_corte} > {left_max} -> {right_path}"
            )

        cortar_recursivo(izquierda, nivel + 1, left_path)
        cortar_recursivo(derecha, nivel + 1, right_path)

    cortar_recursivo(x, nivel=0, path="")

    grupos_sobre_maximo = {
        group_id: grupo
        for group_id, grupo in groups.items()
        if len(grupo) > n_max
    }

    if grupos_sobre_maximo:
        raise ValueError(
            f"Existen {len(grupos_sobre_maximo)} grupos con tamaño mayor "
            f"a n_max={n_max}."
        )

    return {
        "groups": groups,
        "cuts": cuts,
        "variables": variables
    }


def apply_median_cuts(x, cuts, verbose=False):
    """
    Aplica sobre un nuevo dataset los cortes aprendidos desde otro dataset.

    Parámetros:
    -----------
    DATASET : pd.DataFrame
        Nuevo dataset que se quiere particionar usando los mismos cortes.

    cuts : list[dict]
        Cortes generados por median_partition(...).

    verbose : bool
        Si es True, imprime el detalle de aplicación de cortes.

    Retorna:
    --------
    dict[str, pd.DataFrame]
        Grupos finales generados usando los mismos cortes.
    """
    if not isinstance(x, pd.DataFrame):
        raise TypeError("x debe ser un DataFrame de pandas.")

    if cuts is None:
        raise ValueError("cuts no puede ser None.")
    
    # Convertimos la lista de cortes a diccionario por node_id
    cuts_by_node = {
        cut["node_id"]: cut
        for cut in cuts
    }

    groups = {}

    def aplicar_recursivo(df_subset, node_id="ROOT"):
        """
        Aplica el árbol aprendido.

        Cada nodo recibe solamente el subconjunto que llegó a ese nodo,
        igual que en el entrenamiento.
        """
        # Si este nodo no tiene corte, es un grupo final
        if node_id not in cuts_by_node:
            groups[node_id] = df_subset.copy()

            if verbose:
                print(
                    f"[FIN APPLY] Grupo {node_id} | "
                    f"Tamaño: {len(df_subset)} | "
                    f"Índices: {list(df_subset.index)}"
                )

            return

        cut = cuts_by_node[node_id]

        variable = cut["variable"]
        left_max = cut["left_max"]
        left_path = cut["left_path"]
        right_path = cut["right_path"]
        
        if variable not in df_subset.columns:
            raise ValueError(
                f"La variable '{variable}' del corte no existe en el dataset recibido."
            )
            
        mascara_izquierda = df_subset[variable].notna() & (
            df_subset[variable] <= left_max
        )
        
        izquierda = df_subset[mascara_izquierda].copy()
        derecha = df_subset[~mascara_izquierda].copy()

        if verbose:
            print("\n[APLICANDO CORTE]")
            print(f"Nodo: {node_id}")
            print(f"Variable: {variable}")
            print(f"Regla: {variable} <= {left_max}")
            print(f"Tamaño recibido: {len(df_subset)}")
            print(f"Índices recibidos: {list(df_subset.index)}")
            print(f"Izquierda {left_path}: {len(izquierda)} filas | índices {list(izquierda.index)}")
            print(f"Derecha {right_path}: {len(derecha)} filas | índices {list(derecha.index)}")

        aplicar_recursivo(izquierda, left_path)
        aplicar_recursivo(derecha, right_path)

    aplicar_recursivo(x, node_id="ROOT")

    return groups
