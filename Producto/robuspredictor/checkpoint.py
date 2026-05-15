import json
import pandas as pd

def _serializar_dataframe(df):
    """
    Convierte un DataFrame a texto JSON para guardarlo en Excel/CSV.
    Mantiene índices y valores completos.
    """
    if df is None or df.empty:
        return json.dumps([], ensure_ascii=False)

    data = []

    for idx, row in df.iterrows():
        item = {"indice": idx}
        item.update(row.to_dict())
        data.append(item)

    return json.dumps(data, ensure_ascii=False)


def _serializar_serie(serie):
    """
    Convierte una Serie a lista JSON.
    """
    if serie is None or len(serie) == 0:
        return json.dumps([], ensure_ascii=False)

    valores = []

    for idx, value in serie.items():
        valores.append({
            "indice": idx,
            "valor": None if pd.isna(value) else value
        })

    return json.dumps(valores, ensure_ascii=False)


def _obtener_info_cubos(cubos):
    """
    Convierte stable_cubes o red_zones a diccionario por id de cubo.
    """
    return {
        cubo["group_id"]: cubo
        for cubo in cubos
    }


def build_cubes_checkpoint(
    domains,
    stable_cubes,
    red_zones,
    feature_names
):
    """
    Construye checkpoint con nombres de columnas en español.

    Columnas principales:
    - id_cubo
    - var1, var2, var3, ... según las variables recibidas
    - datos_entrenamiento
    - datos_validacion
    - promedio_entrenamiento
    - promedio_validacion
    - desviacion_entrenamiento
    - desviacion_validacion
    - estable
    """

    cubos_estables = _obtener_info_cubos(stable_cubes)
    zonas_rojas = _obtener_info_cubos(red_zones)

    ids_cubos = sorted(
        set(cubos_estables.keys()) | set(zonas_rojas.keys())
    )

    filas = []

    for id_cubo in ids_cubos:
        es_estable = id_cubo in cubos_estables

        info_cubo = (
            cubos_estables[id_cubo]
            if es_estable
            else zonas_rojas[id_cubo]
        )

        fila = {
            "id_cubo": id_cubo,
            "estable": 1 if es_estable else 0,
            "valor_prediccion": info_cubo.get("prediction_value"),
            "motivo_rechazo": json.dumps(
                info_cubo.get("rejection_reasons", []),
                ensure_ascii=False
            ),
            "profundidad_particion": len(id_cubo),
        }

        # Columnas dinámicas para las variables:
        # var1, var2, var3, etc. guardando rangos del cubo en entrenamiento.
        dominio_entrenamiento = domains[0]
        grupo_entrenamiento = dominio_entrenamiento["groups"].get(
            id_cubo,
            pd.DataFrame()
        )

        for variable in feature_names:
            if grupo_entrenamiento.empty:
                fila[variable] = None
            else:
                minimo = grupo_entrenamiento[variable].min()
                maximo = grupo_entrenamiento[variable].max()
                fila[variable] = f"{minimo} - {maximo}"

        # Dominio 1 = entrenamiento
        x_entrenamiento = domains[0]["groups"].get(id_cubo, pd.DataFrame())
        y_entrenamiento = domains[0]["y"].loc[x_entrenamiento.index]

        if isinstance(y_entrenamiento, pd.DataFrame):
            y_entrenamiento = y_entrenamiento.iloc[:, 0]

        fila["datos_entrenamiento"] = _serializar_dataframe(x_entrenamiento)
        fila["target_entrenamiento"] = _serializar_serie(y_entrenamiento)
        fila["cantidad_entrenamiento"] = len(x_entrenamiento)

        if len(y_entrenamiento) > 0:
            fila["promedio_entrenamiento"] = y_entrenamiento.mean()

            std_entrenamiento = y_entrenamiento.std()
            fila["desviacion_entrenamiento"] = (
                0.0 if pd.isna(std_entrenamiento) else std_entrenamiento
            )
        else:
            fila["promedio_entrenamiento"] = None
            fila["desviacion_entrenamiento"] = None

        # Dominios 2..N = validaciones
        datos_validacion = []
        targets_validacion = []
        promedios_validacion = []
        desviaciones_validacion = []
        cantidades_validacion = []

        for i in range(1, len(domains)):
            dominio = domains[i]
            x_validacion = dominio["groups"].get(id_cubo, pd.DataFrame())
            y_validacion = dominio["y"].loc[x_validacion.index]

            if isinstance(y_validacion, pd.DataFrame):
                y_validacion = y_validacion.iloc[:, 0]

            datos_validacion.append({
                "dominio": i + 1,
                "datos": json.loads(_serializar_dataframe(x_validacion))
            })

            targets_validacion.append({
                "dominio": i + 1,
                "target": json.loads(_serializar_serie(y_validacion))
            })

            cantidades_validacion.append({
                "dominio": i + 1,
                "cantidad": len(x_validacion)
            })

            if len(y_validacion) > 0:
                promedio = y_validacion.mean()

                desviacion = y_validacion.std()
                if pd.isna(desviacion):
                    desviacion = 0.0
            else:
                promedio = None
                desviacion = None

            promedios_validacion.append({
                "dominio": i + 1,
                "promedio": promedio
            })

            desviaciones_validacion.append({
                "dominio": i + 1,
                "desviacion": desviacion
            })

        fila["datos_validacion"] = json.dumps(datos_validacion, ensure_ascii=False)
        fila["target_validacion"] = json.dumps(targets_validacion, ensure_ascii=False)
        fila["cantidad_validacion"] = json.dumps(cantidades_validacion, ensure_ascii=False)
        fila["promedio_validacion"] = json.dumps(promedios_validacion, ensure_ascii=False)
        fila["desviacion_validacion"] = json.dumps(desviaciones_validacion, ensure_ascii=False)

        filas.append(fila)

    return pd.DataFrame(filas)

def build_cuts_checkpoint(cuts):
    if cuts is None:
        return pd.DataFrame()

    cortes_df = pd.DataFrame(cuts)

    columnas_renombradas = {
        "node_id": "id_nodo",
        "level": "nivel",
        "variable": "variable_corte",
        "left_path": "ruta_izquierda",
        "right_path": "ruta_derecha",
        "left_size": "tamaño_izquierda",
        "right_size": "tamaño_derecha",
        "left_indices": "indices_izquierda",
        "right_indices": "indices_derecha",
        "left_max": "maximo_izquierda",
        "right_min": "minimo_derecha",
        "cut_position": "posicion_corte",
        "rule": "regla"
    }

    cortes_df = cortes_df.rename(columns=columnas_renombradas)

    return cortes_df


def build_summary_checkpoint(
    stable_cubes,
    red_zones,
    cuts,
    domains
):
    total_cubos = len(stable_cubes) + len(red_zones)
    cubos_estables = len(stable_cubes)
    cubos_no_estables = len(red_zones)

    porcentaje_exitosos = (
        cubos_estables / total_cubos
        if total_cubos > 0
        else 0
    )

    porcentaje_no_estables = (
        cubos_no_estables / total_cubos
        if total_cubos > 0
        else 0
    )

    filas = [
        {
            "metrica": "total_cubos",
            "valor": total_cubos
        },
        {
            "metrica": "cubos_estables",
            "valor": cubos_estables
        },
        {
            "metrica": "cubos_no_estables",
            "valor": cubos_no_estables
        },
        {
            "metrica": "porcentaje_cubos_exitosos",
            "valor": porcentaje_exitosos
        },
        {
            "metrica": "porcentaje_cubos_no_estables",
            "valor": porcentaje_no_estables
        },
        {
            "metrica": "cantidad_particiones_realizadas",
            "valor": len(cuts) if cuts is not None else 0
        },
        {
            "metrica": "cantidad_dominios",
            "valor": len(domains)
        }
    ]

    for i, dominio in enumerate(domains, start=1):
        grupos = dominio.get("groups", {})
        tamaños = [len(grupo) for grupo in grupos.values()]

        if not tamaños:
            continue

        filas.extend([
            {
                "metrica": f"dominio_{i}_total_grupos",
                "valor": len(tamaños)
            },
            {
                "metrica": f"dominio_{i}_tamaño_minimo_grupo",
                "valor": min(tamaños)
            },
            {
                "metrica": f"dominio_{i}_tamaño_maximo_grupo",
                "valor": max(tamaños)
            },
            {
                "metrica": f"dominio_{i}_tamaño_promedio_grupo",
                "valor": sum(tamaños) / len(tamaños)
            },
            {
                "metrica": f"dominio_{i}_grupos_vacios",
                "valor": sum(1 for tamaño in tamaños if tamaño == 0)
            }
        ])

    return pd.DataFrame(filas)


def export_checkpoint(
    path,
    domains,
    stable_cubes,
    red_zones,
    cuts,
    feature_names,
    file_format="xlsx"
):
    """
    Exporta checkpoint de trazabilidad.

    Parámetros:
    -----------
    path : str
        Ruta de salida. Ej:
        'checkpoint.xlsx' o 'checkpoint.csv'

    domains : list[dict]
        Dominios usados en el fit.

    stable_cubes : list[dict]
        Cubos estables.

    red_zones : list[dict]
        Zonas rojas.

    cuts : list[dict]
        Cortes aprendidos.

    feature_names : list[str]
        Variables usadas por el modelo.

    file_format : str
        'xlsx' o 'csv'
    """

    cubes_df = build_cubes_checkpoint(
        domains=domains,
        stable_cubes=stable_cubes,
        red_zones=red_zones,
        feature_names=feature_names
    )

    cuts_df = build_cuts_checkpoint(cuts)

    summary_df = build_summary_checkpoint(
        stable_cubes=stable_cubes,
        red_zones=red_zones,
        cuts=cuts,
        domains=domains
    )

    if file_format == "xlsx":
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            cubes_df.to_excel(writer, sheet_name="cubes_checkpoint", index=False)
            cuts_df.to_excel(writer, sheet_name="cuts_checkpoint", index=False)
            summary_df.to_excel(writer, sheet_name="summary", index=False)

    elif file_format == "csv":
        cubes_df.to_csv(path, index=False, sep=",")

    else:
        raise ValueError("file_format debe ser 'xlsx' o 'csv'.")

    return {
        "cubes_checkpoint": cubes_df,
        "cuts_checkpoint": cuts_df,
        "summary": summary_df
    }