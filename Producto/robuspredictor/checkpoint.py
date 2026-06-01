import json
import pandas as pd


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
):
    """
    Construye el DataFrame de auditoria de cubos.

    Estructura de columnas (sin JSON, todo legible en Excel):

    Identificacion:
        id_cubo | estable | valor_prediccion
        suma_prom_dominios_entrenamiento   : suma de los promedios del target por dominio de entrenamiento (= valor_prediccion)
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
    cubos_estables  = _obtener_info_cubos(stable_cubes)
    zonas_rojas     = _obtener_info_cubos(red_zones)
    ids_cubos       = sorted(set(cubos_estables.keys()) | set(zonas_rojas.keys()))
    tiene_validacion = validacion_groups is not None and y_validacion is not None

    filas = []

    for id_cubo in ids_cubos:
        es_estable = id_cubo in cubos_estables
        info_cubo  = cubos_estables[id_cubo] if es_estable else zonas_rojas[id_cubo]

        # ── Identificacion ───────────────────────────────────────────────────
        fila = {
            "id_cubo":    id_cubo,
            "estable":    1 if es_estable else 0,
            # valor_prediccion = suma de promedios de dominios de entrenamiento
            "valor_prediccion":                       info_cubo.get("prediction_value"),
            # suma_prom_dominios_entrenamiento: misma cifra que valor_prediccion,
            # expuesta explicitamente para auditoria
            "suma_prom_dominios_entrenamiento":        info_cubo.get("sum_means"),
            # promedio de los promedios de cada dominio de entrenamiento
            "promedio_prom_dominios_entrenamiento":    info_cubo.get("mean_of_means"),
            # lista JSON con el promedio individual de cada dominio de entrenamiento
            "prom_dominios_entrenamiento_detalle":     json.dumps(
                                                           info_cubo.get("domain_means", []),
                                                           ensure_ascii=False
                                                       ),
            "motivo_rechazo":                         json.dumps(
                                                           info_cubo.get("rejection_reasons", []),
                                                           ensure_ascii=False
                                                       ),
            "profundidad_particion":                  len(id_cubo),
        }

        # ── Promedios de variables predictoras por dominio ───────────────────
        grupo_dom1 = domains[0]["groups"].get(id_cubo, pd.DataFrame())
        grupo_dom2 = domains[1]["groups"].get(id_cubo, pd.DataFrame()) \
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
        x_d1 = domains[0]["groups"].get(id_cubo, pd.DataFrame())
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
            x_di = domains[i]["groups"].get(id_cubo, pd.DataFrame())
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
            x_val = validacion_groups.get(id_cubo, pd.DataFrame())
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
    path,
    domains,
    stable_cubes,
    red_zones,
    cuts,
    feature_names,
    file_format="xlsx",
    validacion_groups=None,
    y_validacion=None,
):
    """
    Exporta el checkpoint de trazabilidad a Excel o CSV.

    Parametros:
    -----------
    path               : str                  — ruta de salida
    domains            : list[dict]           — dominios de entrenamiento
    stable_cubes       : list[dict]           — cubos estables
    red_zones          : list[dict]           — zonas rojas
    cuts               : list[dict]           — cortes aprendidos
    feature_names      : list[str]            — variables predictoras
    file_format        : str                  — 'xlsx' o 'csv'
    validacion_groups  : dict[str, DataFrame] — grupos del dataset de validacion,
                         generados con apply_median_cuts(X_valid, cuts). Opcional.
    y_validacion       : pd.Series            — target real del dataset de validacion.
                         Requerido si se provee validacion_groups.
    """
    cubes_df = build_cubes_checkpoint(
        domains=domains,
        stable_cubes=stable_cubes,
        red_zones=red_zones,
        feature_names=feature_names,
        validacion_groups=validacion_groups,
        y_validacion=y_validacion,
    )
    cuts_df    = build_cuts_checkpoint(cuts)
    summary_df = build_summary_checkpoint(
        stable_cubes=stable_cubes,
        red_zones=red_zones,
        cuts=cuts,
        domains=domains,
    )

    if file_format == "xlsx":
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            cubes_df.to_excel(  writer, sheet_name="cubes_checkpoint", index=False)
            cuts_df.to_excel(   writer, sheet_name="cuts_checkpoint",  index=False)
            summary_df.to_excel(writer, sheet_name="summary",          index=False)
    elif file_format == "csv":
        cubes_df.to_csv(path, index=False, sep=",")
    else:
        raise ValueError("file_format debe ser 'xlsx' o 'csv'.")

    return {
        "cubes_checkpoint": cubes_df,
        "cuts_checkpoint":  cuts_df,
        "summary":          summary_df,
    }
