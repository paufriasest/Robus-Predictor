import pandas as pd
import numpy as np


def median_partition(x, n_min, n_max, verbose=False, random_state=None):
    """Divide un dataset usando particiones recursivas locales por mediana.
    
    La lógica es:
    - En cada nodo se recibe un subconjunto de filas.
    - Se selecciona una variable según el nivel del árbol.
    - Se calcula la mediana local de esa variable dentro del subconjunto.
    - Se corta usando:
        variable <= mediana -> izquierda
        variable > mediana  -> derecha
    - Si el corte no cumple n_min, se prueba con las siguientes variables.
    - Si ninguna variable permite un corte válido, el nodo queda como grupo final.
    - Se guardan los cortes para replicarlos sobre otros datasets.
    
    Args:
    x : pd.DataFrame
        Dataset base con filas como elementos y columnas como variables.
    
    n_min : int
        Tamaño mínimo de grupo final.
    
    n_max : int
        Tamaño máximo permitido de grupo final.
        
    verbose : bool
        Si es True, imprime detalle del proceso.
        
    Returns:
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

    def intentar_corte_por_mediana(df_subset, nivel):
            """Intenta encontrar un corte válido por mediana.
            
            Primero prueba la variable correspondiente al nivel.
            Si no funciona, prueba las siguientes variables en orden cíclico.
            
            Returns:
                dict | None
            """
            n_variables = len(variables)
            
            for offset in range(n_variables):
                variable_corte = variables[(nivel + offset) % n_variables]
                
                serie = df_subset[variable_corte].dropna()
                
                if serie.empty:
                    continue
                
                threshold = serie.median()
                
                mascara_izquierda = df_subset[variable_corte].notna() & (
                    df_subset[variable_corte] <= threshold
                )
                
                izquierda = df_subset[mascara_izquierda].copy()
                derecha = df_subset[~mascara_izquierda].copy()
                
                left_size = len(izquierda)
                right_size = len(derecha)
                
                corte_valido = left_size >= n_min and right_size >= n_min
                
                if not corte_valido:
                    if verbose:
                        print(
                            f"[CORTE NO VÁLIDO] Variable {variable_corte} | "
                            f"Mediana: {threshold} | "
                            f"Izquierda: {left_size} | Derecha: {right_size} | "
                            f"n_min={n_min}"
                        )
                    continue
                
                return {
                    "variable": variable_corte,
                    "threshold": threshold,
                    "median_value": threshold,
                    "izquierda": izquierda,
                    "derecha": derecha,
                    "left_size": left_size,
                    "right_size": right_size,
                    "fallback_used": offset > 0,
                    "variable_offset": offset,
                }
                
            return None
    
    def cortar_recursivo(df_subset, nivel=0, path=""):
        """Recibe el subconjunto local del nodo actual y lo divide por mediana.
        """
        
        tamaño_actual = len(df_subset)
        node_id = path if path != "" else "ROOT"
        
        # Caso base: si el grupo ya está dentro del tamaño máximo permitido
        if tamaño_actual <= n_max:
            groups[node_id] = df_subset.copy()
            
            if verbose:
                print(
                    f"[FIN] Nodo {node_id} | "
                    f"Nivel {nivel} | "
                    f"Tamaño: {tamaño_actual} | "
                    f"Índices: {list(df_subset.index)}"
                )
            
            return
        
        # Protección: si ni siquiera un corte perfecto podría cumplir n_min
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
        
        resultado_corte = intentar_corte_por_mediana(df_subset, nivel)
        
        if resultado_corte is None:
            groups[node_id] = df_subset.copy()
            
            if verbose:
                print(
                    f"[FIN SIN CORTE VÁLIDO] Nodo {node_id} | "
                    f"Nivel {nivel} | "
                    f"Tamaño: {tamaño_actual} | "
                    f"Ninguna variable permitió cortar por mediana respetando n_min."
                )
            
            return
        
        variable_corte = resultado_corte["variable"]
        threshold = resultado_corte["threshold"]
        median_value = resultado_corte["median_value"]
        izquierda = resultado_corte["izquierda"]
        derecha = resultado_corte["derecha"]
        
        left_path = path + "L"
        right_path = path + "R"
        
        left_max = (
            izquierda[variable_corte].max()
            if not izquierda.empty and variable_corte in izquierda.columns
            else None
        )
        
        right_min = (
            derecha[variable_corte].min()
            if not derecha.empty and variable_corte in derecha.columns
            else None
        )
        
        n_equal_threshold = int((df_subset[variable_corte] == threshold).sum())
        
        cut_info = {
            "node_id": node_id,
            "level": nivel,
            "variable": variable_corte,
            "threshold": threshold,
            "median_value": median_value,
            "left_path": left_path,
            "right_path": right_path,
            "left_size": len(izquierda),
            "right_size": len(derecha),
            "left_indices": list(izquierda.index),
            "right_indices": list(derecha.index),
            "left_max": left_max,
            "right_min": right_min,
            "n_equal_threshold": n_equal_threshold,
            "fallback_used": resultado_corte["fallback_used"],
            "variable_offset": resultado_corte["variable_offset"],
            "rule": (
                f"{variable_corte} <= {threshold} va a izquierda; "
                f"{variable_corte} > {threshold} va a derecha"
            )
        }
        
        cuts.append(cut_info)
        
        if verbose:
            print("\n[CORTE POR MEDIANA]")
            print(f"Nodo: {node_id}")
            print(f"Nivel: {nivel}")
            print(f"Variable usada: {variable_corte}")
            print(f"Mediana / threshold: {threshold}")
            print(f"Tamaño recibido: {tamaño_actual}")
            print(f"Izquierda {left_path}: {len(izquierda)} filas")
            print(f"Derecha {right_path}: {len(derecha)} filas")
            print(f"Regla replicable: {cut_info['rule']}")
        
        cortar_recursivo(izquierda, nivel + 1, left_path)
        cortar_recursivo(derecha, nivel + 1, right_path)
    
    cortar_recursivo(x, nivel=0, path="")
    
    grupos_sobre_maximo = {
        group_id: grupo
        for group_id, grupo in groups.items()
        if len(grupo) > n_max
    }
    
    if grupos_sobre_maximo:
        detalle = {
            group_id: len(grupo)
            for group_id, grupo in grupos_sobre_maximo.items()
        }
        
        raise ValueError(
            f"Existen {len(grupos_sobre_maximo)} grupos con tamaño mayor "
            f"a n_max={n_max}. Detalle: {detalle}. "
            "Esto puede ocurrir si ninguna variable permitió un corte por mediana "
            "que respetara n_min."
        )
    
    return {
        "groups": groups,
        "cuts": cuts,
        "variables": variables
    }


def apply_median_cuts(x, cuts, verbose=False):
    """ Aplica sobre un nuevo dataset los cortes aprendidos desde otro dataset.
    Args:
    DATASET : pd.DataFrame
        Nuevo dataset que se quiere particionar usando los mismos cortes.
    
    cuts : list[dict]
        Cortes generados por median_partition(...).
    
    verbose : bool
        Si es True, imprime el detalle de aplicación de cortes.
    
    Returns:
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
        """Aplica el árbol aprendido.
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
        
        threshold = cut.get("threshold", cut.get("left_max"))
        
        left_path = cut["left_path"]
        right_path = cut["right_path"]
        
        if variable not in df_subset.columns:
            raise ValueError(
                f"La variable '{variable}' del corte no existe en el dataset recibido."
            )
        
        mascara_izquierda = df_subset[variable].notna() & (
            df_subset[variable] <= threshold
        )
        izquierda = df_subset[mascara_izquierda].copy()
        derecha = df_subset[~mascara_izquierda].copy()

        if verbose:
            print("\n[APLICANDO CORTE]")
            print(f"Nodo: {node_id}")
            print(f"Variable: {variable}")
            print(f"Threshold: {threshold}")
            print(f"Regla: {variable} <= {threshold}")
            print(f"Tamaño recibido: {len(df_subset)}")
            print(f"Izquierda {left_path}: {len(izquierda)} filas")
            print(f"Derecha {right_path}: {len(derecha)} filas")

        aplicar_recursivo(izquierda, left_path)
        aplicar_recursivo(derecha, right_path)

    aplicar_recursivo(x, node_id="ROOT")

    return groups
