import json
import pandas as pd
from .prediction import build_prediction_detail
from pathlib import Path

def build_export_file_path(file_name, file_format, default_name):
    """Construye la ruta final del archivo de exportación.

    Args:
        file_name (str): Nombre del archivo
        file_format (str): Tipo extensión del archivo 'xlsx' o 'csv'
        default_name (_type_): Nombre por default que tendrá la extensión

    Raises:
        ValueError:  Si file_format no es "xlsx" ni "csv"

    Returns:
        pathlib.Path: Ruta final del archivo con la extensión correspondiente.
    """

    if file_format not in ["xlsx", "csv"]:
        raise ValueError("file_format debe ser 'xlsx' o 'csv'.")

    if file_name is None:
        file_name = default_name

    file_path = Path(file_name)

    # Si el usuario escribe una extensión, se reemplaza por file_format.
    file_path = file_path.with_suffix(f".{file_format}")

    return file_path

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _obtener_info_cubos(cubos):
    return {cubo["group_id"]: cubo for cubo in cubos}

def _promedio(serie):
    if serie is None or len(serie) == 0:
        return None
    val = serie.mean()
    return round(float(val), 6) if not pd.isna(val) else None

def _std(serie):
    if serie is None or len(serie) == 0:
        return None
    val = serie.std()
    if pd.isna(val):
        return 0.0
    return round(float(val), 6)

def _promedio_consolidado(*valores):
    """Promedio de los valores no nulos provistos."""
    validos = [v for v in valores if v is not None]
    if not validos:
        return None
    return round(sum(validos) / len(validos), 6)


# ─── Cubes checkpoint ────────────────────────────────────────────────────────

def build_cubes_checkpoint(
    domains,
    stable_cubes,
    red_zones,
    feature_names,
    validacion_groups=None,
    y_validacion=None,
    cube_id_map=None,
):
    """
    Construye el DataFrame de auditoria de cubos.

    Estructura de columnas (sin JSON, todo legible en Excel):

    Identificacion:
        id_cubo | group_id | estable | valor_prediccion
        promedio_prom_dominios_entrenamiento : promedio de los promedios del target por dominio de entrenamiento
        prom_dominios_entrenamiento_detalle  : lista JSON con el promedio de cada dominio de entrenamiento por separado
        motivo_rechazo | profundidad_particion

    Promedios de variables predictoras por dominio:
        prom_<var>_dom1   : promedio en dominio 1 (referencia espacial del cubo)
        prom_<var>_dom2   : promedio en dominio 2 (validacion interna)
        prom_<var>_consol : promedio consolidado dom1 + dom2

    Metricas del target por dominio de entrenamiento:
        n_dom1     | prom_target_dom1 | std_target_dom1
        n_dom2     | prom_target_dom2 | std_target_dom2

    Metricas del target — dataset de validacion (columnas opcionales):
        n_validacion           : registros del dataset de validacion en este cubo
        prom_target_validacion : promedio del target real en validacion para este cubo
        std_target_validacion  : desviacion estandar del target real en validacion

    Consolidado:
        prom_target_consolidado : promedio de todos los promedios disponibles
                                (dom1 + dom2 + validacion)

    Parametros opcionales:
        validacion_groups : dict[str, pd.DataFrame]  — grupos del dataset de validacion,
                            generados con apply_median_cuts(X_valid, cuts)
        y_validacion      : pd.Series                — target real del dataset de validacion
    """
    if cube_id_map is None:
        cube_id_map = {}
    
    cubos_estables  = _obtener_info_cubos(stable_cubes)
    zonas_rojas     = _obtener_info_cubos(red_zones)
    ids_cubos       = sorted(set(cubos_estables.keys()) | set(zonas_rojas.keys()))
    tiene_validacion = validacion_groups is not None and y_validacion is not None

    filas = []

    for group_id  in ids_cubos:
        es_estable = group_id in cubos_estables
        info_cubo  = cubos_estables[group_id] if es_estable else zonas_rojas[group_id]

        cube_id = cube_id_map.get(group_id, group_id)

        # ── Identificacion ───────────────────────────────────────────────────
        fila = {
            "cube_id": cube_id,
            "group_id ": group_id,
            "estable": 1 if es_estable else 0,
            # valor_prediccion = preoedio proedios
            "valor_prediccion": info_cubo.get("prediction_value"),
            # promedio de los promedios de cada dominio de entrenamiento
            "promedio_prom_dominios_entrenamiento": info_cubo.get("mean_of_means"),
            # lista JSON con el promedio individual de cada dominio de entrenamiento
            "prom_dominios_entrenamiento_detalle": json.dumps(
                info_cubo.get("domain_means", []),
                ensure_ascii=False
            ),
            "motivo_rechazo": json.dumps(
                info_cubo.get("rejection_reasons", []),
                ensure_ascii=False
            ),
            "profundidad_particion": len(group_id),
        }

        # ── Promedios de variables predictoras por dominio ───────────────────
        grupo_dom1 = domains[0]["groups"].get(group_id, pd.DataFrame())
        grupo_dom2 = domains[1]["groups"].get(group_id, pd.DataFrame()) \
                    if len(domains) > 1 else pd.DataFrame()

        for variable in feature_names:
            prom_d1 = None if (grupo_dom1.empty or variable not in grupo_dom1.columns) \
                    else _promedio(grupo_dom1[variable])
            prom_d2 = None if (grupo_dom2.empty or variable not in grupo_dom2.columns) \
                    else _promedio(grupo_dom2[variable])

            fila["prom_" + variable + "_dom1"]  = prom_d1
            fila["prom_" + variable + "_dom2"]  = prom_d2
            fila["prom_" + variable + "_consol"] = _promedio_consolidado(prom_d1, prom_d2)

        # ── Metricas del target — Dom1 ────────────────────────────────────────
        x_d1 = domains[0]["groups"].get(group_id, pd.DataFrame())
        if x_d1.empty:
            y_d1 = pd.Series([], dtype=float)
        else:
            y_d1 = domains[0]["y"].loc[x_d1.index]
            if isinstance(y_d1, pd.DataFrame):
                y_d1 = y_d1.iloc[:, 0]

        fila["n_dom1"]           = len(y_d1)
        fila["prom_target_dom1"] = _promedio(y_d1)
        fila["std_target_dom1"]  = _std(y_d1)

        # ── Metricas del target — dominios adicionales ────────────────────────
        prom_adicionales = [fila["prom_target_dom1"]]

        for i in range(1, len(domains)):
            x_di = domains[i]["groups"].get(group_id, pd.DataFrame())
            if x_di.empty:
                y_di = pd.Series([], dtype=float)
            else:
                y_di = domains[i]["y"].loc[x_di.index]
                if isinstance(y_di, pd.DataFrame):
                    y_di = y_di.iloc[:, 0]

            prom = _promedio(y_di)
            fila["n_dom"           + str(i + 1)] = len(y_di)
            fila["prom_target_dom" + str(i + 1)] = prom
            fila["std_target_dom"  + str(i + 1)] = _std(y_di)

            if prom is not None:
                prom_adicionales.append(prom)

        # ── Metricas del target — Dataset de validacion (opcional) ─────────────
        if tiene_validacion:
            x_val = validacion_groups.get(group_id, pd.DataFrame())
            if x_val.empty:
                y_val_cubo = pd.Series([], dtype=float)
            else:
                y_val_cubo = y_validacion.loc[x_val.index]
                if isinstance(y_val_cubo, pd.DataFrame):
                    y_val_cubo = y_val_cubo.iloc[:, 0]

            prom_val = _promedio(y_val_cubo)
            fila["n_validacion"]           = len(y_val_cubo)
            fila["prom_target_validacion"] = prom_val
            fila["std_target_validacion"]  = _std(y_val_cubo)

            if prom_val is not None:
                prom_adicionales.append(prom_val)
        else:
            fila["n_validacion"]           = None
            fila["prom_target_validacion"] = None
            fila["std_target_validacion"]  = None

        # ── Consolidado final ─────────────────────────────────────────────────
        fila["prom_target_consolidado"] = _promedio_consolidado(*prom_adicionales)

        filas.append(fila)

    return pd.DataFrame(filas)


# ─── Cuts checkpoint ─────────────────────────────────────────────────────────

def build_cuts_checkpoint(cuts):
    if cuts is None:
        return pd.DataFrame()

    cortes_df = pd.DataFrame(cuts)

    columnas_renombradas = {
        "node_id":       "id_nodo",
        "level":         "nivel",
        "variable":      "variable_corte",
        "left_path":     "ruta_izquierda",
        "right_path":    "ruta_derecha",
        "left_size":     "tamanio_izquierda",
        "right_size":    "tamanio_derecha",
        "left_indices":  "indices_izquierda",
        "right_indices": "indices_derecha",
        "left_max":      "maximo_izquierda",
        "right_min":     "minimo_derecha",
        "cut_position":  "posicion_corte",
        "rule":          "regla",
    }

    return cortes_df.rename(columns=columnas_renombradas)


# ─── Summary checkpoint ──────────────────────────────────────────────────────

def build_summary_checkpoint(stable_cubes, red_zones, cuts, domains):
    total_cubos       = len(stable_cubes) + len(red_zones)
    cubos_estables    = len(stable_cubes)
    cubos_no_estables = len(red_zones)

    pct_exitosos    = cubos_estables    / total_cubos if total_cubos > 0 else 0
    pct_no_estables = cubos_no_estables / total_cubos if total_cubos > 0 else 0

    filas = [
        {"metrica": "total_cubos",                    "valor": total_cubos},
        {"metrica": "cubos_estables",                 "valor": cubos_estables},
        {"metrica": "cubos_no_estables",              "valor": cubos_no_estables},
        {"metrica": "porcentaje_cubos_exitosos",       "valor": pct_exitosos},
        {"metrica": "porcentaje_cubos_no_estables",    "valor": pct_no_estables},
        {"metrica": "cantidad_particiones_realizadas", "valor": len(cuts) if cuts else 0},
        {"metrica": "cantidad_dominios",               "valor": len(domains)},
    ]

    for i, dominio in enumerate(domains, start=1):
        grupos   = dominio.get("groups", {})
        tamanios = [len(g) for g in grupos.values()]
        if not tamanios:
            continue
        filas.extend([
            {"metrica": "dominio_" + str(i) + "_total_grupos",
             "valor": len(tamanios)},
            {"metrica": "dominio_" + str(i) + "_tamanio_minimo_grupo",
             "valor": min(tamanios)},
            {"metrica": "dominio_" + str(i) + "_tamanio_maximo_grupo",
             "valor": max(tamanios)},
            {"metrica": "dominio_" + str(i) + "_tamanio_promedio_grupo",
             "valor": sum(tamanios) / len(tamanios)},
            {"metrica": "dominio_" + str(i) + "_grupos_vacios",
             "valor": sum(1 for t in tamanios if t == 0)},
        ])

    return pd.DataFrame(filas)


# ─── Export ──────────────────────────────────────────────────────────────────
def export_checkpoint(
    domains,
    stable_cubes,
    red_zones,
    cuts,
    feature_names,
    file_name="checkpoint_robuspredictor",
    file_format="xlsx",
    validacion_groups=None,
    y_validacion=None,
    cube_id_map=None
):
    """Exporta el checkpoint de trazabilidad a Excel o CSV.

    Args:
        domains (list[dict]): Dominios de entrenamiento generados por el modelo.
        stable_cubes (list[dict]): Cubos considerados estables.
        red_zones (list[dict]): Cubos no estables o zonas rojas.
        cuts (list[dict]): Cortes aprendidos durante el entrenamiento.
        feature_names (list[str]): Nombres de las variables predictoras utilizadas por el modelo.
        file_name (str, optional): Nombre base del archivo de salida. No es necesario indicar extensión. Defaults to "checkpoint_robuspredictor".
        file_format (str, optional): Formato de salida. Valores permitidos: "xlsx" o "csv". Defaults to "xlsx".
        validacion_groups (dict[str, pd.DataFrame], optional): Grupos del dataset de validación generados con los cortes del modelo. Defaults to None.
        y_validacion (pd.Series, optional): Target real del dataset de validación. Defaults to None.

    Raises:
        ValueError: Si file_format no es "xlsx" ni "csv".

    Returns:
        dict[str, pd.DataFrame]: Diccionario con los DataFrames generados para el checkpoint.
    """
    if cube_id_map is None:
        cube_id_map = {}
    
    export_path = build_export_file_path(
        file_name=file_name,
        file_format=file_format,
        default_name="checkpoint_robuspredictor"
    )

    cubes_df = build_cubes_checkpoint(
        domains=domains,
        stable_cubes=stable_cubes,
        red_zones=red_zones,
        feature_names=feature_names,
        validacion_groups=validacion_groups,
        y_validacion=y_validacion,
        cube_id_map=cube_id_map,
    )

    cuts_df = build_cuts_checkpoint(cuts)

    summary_df = build_summary_checkpoint(
        stable_cubes=stable_cubes,
        red_zones=red_zones,
        cuts=cuts,
        domains=domains,
    )

    checkpoint = {
        "cubes_checkpoint": cubes_df,
        "cuts_checkpoint": cuts_df,
        "summary": summary_df,
    }

    if file_format == "xlsx":
        with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
            for sheet_name, df in checkpoint.items():
                df.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False
                )

    elif file_format == "csv":
        base_path = export_path.with_suffix("")

        for i, (sheet_name, df) in enumerate(checkpoint.items(), start=1):
            csv_path = base_path.parent / f"{base_path.name}_{i}_{sheet_name}.csv"

            df.to_csv(
                csv_path,
                index=False,
                sep=",",
                encoding="utf-8-sig"
            )

    else:
        raise ValueError("file_format debe ser 'xlsx' o 'csv'.")

    return checkpoint

def build_and_export_prediction_checkpoint(
    X,
    y=None,
    file_name="scoring_robuspredictor",
    dato_real=None,
    stable_cubes=None,
    red_zones=None,
    cuts=None,
    default_value=0,
    feature_names=None,
    file_format="xlsx"
):
    """ Exporta un checkpoint de predicción a nivel de registro

    Args:
        X (pd.DataFrame): Dataset de validación o datos a predecir
        y (pd.Series | pd.DataFrame | None):Target real asociado a X. Se utiliza solo para trazabilidad. Puede ser None
        file_name (str): Nombre base del archivo de salida.
        dato_real (pd.Series | pd.DataFrame | None): Variable real adicional utilizada para evaluación. Puede ser None
        stable_cubes (list[dict]): Cubos estables generados por el modelo
        red_zones (list[dict]): Zonas rojas generadas por el modelo
        cuts (list[dict]): Cortes aprendidos por el modelo
        default_value (float): Valor por defecto utilizado en zonas rojas
        feature_names (list[str]): Variables predictoras utilizadas durante el entrenamiento
        file_format (str, optional): Formato de salida. Valores permitidos: "xlsx" o "csv". Defaults to "xlsx"

    Raises:
        TypeError: Si X no es un DataFrame de pandas
        ValueError: Si y o dato_real no tienen el mismo índice que X
        ValueError: Si file_format no es "xlsx" ni "csv"

    Returns:
        pd.DataFrame: DataFrame con el detalle de predicción por registro
    """

    export_path = build_export_file_path(
        file_name=file_name,
        file_format=file_format,
        default_name="scoring_robuspredictor"
    )

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X debe ser un DataFrame de pandas.")

    X_export = X[feature_names].copy()

    if y is not None:
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("y debe ser una Series o un DataFrame de una sola columna.")
            y_export = y.iloc[:, 0]

        elif isinstance(y, pd.Series):
            y_export = y

        else:
            raise TypeError("y debe ser None, Series o DataFrame.")

        if len(X_export) != len(y_export):
            raise ValueError("X e y deben tener la misma cantidad de filas.")

        if not X_export.index.equals(y_export.index):
            raise ValueError("X e y deben tener el mismo índice.")

        X_export["target"] = y_export

    detail_df = build_prediction_detail(
        X=X_export[feature_names],
        stable_cubes=stable_cubes,
        red_zones=red_zones,
        cuts=cuts,
        default_value=default_value
    )

    prediction_checkpoint = X_export.join(detail_df)

    prediction_checkpoint["arriendo_segun_predict"] = (
        prediction_checkpoint["prediccion_aplicada"] > 1.5
    ).astype(int)

    if dato_real is not None:
        if isinstance(dato_real, pd.DataFrame):
            if dato_real.shape[1] != 1:
                raise ValueError(
                    "dato_real debe ser una Series o un DataFrame de una sola columna."
                )
            dato_real = dato_real.iloc[:, 0]

        elif not isinstance(dato_real, pd.Series):
            raise TypeError("dato_real debe ser None, Series o DataFrame.")

        if len(prediction_checkpoint) != len(dato_real):
            raise ValueError(
                "prediction_checkpoint y dato_real deben tener la misma cantidad de filas."
            )

        if not prediction_checkpoint.index.equals(dato_real.index):
            raise ValueError(
                "prediction_checkpoint y dato_real deben tener el mismo índice."
            )

        prediction_checkpoint["ARRIENDO_REAL"] = dato_real.astype(int)

        prediction_checkpoint["acierto_del_modelo_de_acuerdo_arriendo_predict"] = (
            (prediction_checkpoint["arriendo_segun_predict"] == 1)
            & (prediction_checkpoint["ARRIENDO_REAL"] == 1)
        ).astype(int)

    prediction_checkpoint.index = range(1, len(prediction_checkpoint) + 1)

    if file_format == "xlsx":
        with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
            prediction_checkpoint.to_excel(
                writer,
                sheet_name="predicciones_checkpoint",
                index=True,
                index_label="id_registro"
            )

    elif file_format == "csv":
        prediction_checkpoint.to_csv(
            export_path,
            index=True,
            index_label="id_registro",
            sep=",",
            encoding="utf-8-sig"
        )

    else:
        raise ValueError("file_format debe ser 'xlsx' o 'csv'.")

    return prediction_checkpoint