import pandas as pd


# ─── Helpers internos ────────────────────────────────────────────────────────

def _obtener_info_cubos(cubos):
    """Indexa stable_cubes o red_zones por group_id."""
    return {cubo["group_id"]: cubo for cubo in cubos}


def _promedio(serie):
    """Devuelve promedio redondeado a 6 decimales o None si está vacío."""
    if serie is None or len(serie) == 0:
        return None
    val = serie.mean()
    return round(float(val), 6) if not pd.isna(val) else None


def _std(serie):
    """
    Devuelve desviación estándar redondeada a 6 decimales o None si vacío.
    Un grupo de un solo elemento tiene std=NaN -> se trata como 0.0.
    """
    if serie is None or len(serie) == 0:
        return None
    val = serie.std()
    if pd.isna(val):
        return 0.0
    return round(float(val), 6)


# ─── Cubes checkpoint ────────────────────────────────────────────────────────

def build_cubes_checkpoint(domains, stable_cubes, red_zones, feature_names):
    """
    Construye el DataFrame de auditoria de cubos.

    Columnas (sin JSON, todo legible en Excel):

    Identificacion:
        id_cubo | estable | valor_prediccion | motivo_rechazo
        profundidad_particion

    Promedios de variables predictoras (dominio 1 como referencia espacial):
        prom_<var1> | prom_<var2> | ... | prom_<varN>

    Metricas del target por dominio (columnas planas, una por dominio):
        n_dom1 | prom_target_dom1 | std_target_dom1
        n_dom2 | prom_target_dom2 | std_target_dom2

    Sobre null en dominios de validacion:
        n_dom2 = 0 y prom_target_dom2 = null significa que ningun registro
        del dominio 2 cayo en este cubo al aplicar los cortes del dominio 1.
        Es comportamiento esperado con particionamiento granular (n_min bajo).
    """
    import json

    cubos_estables = _obtener_info_cubos(stable_cubes)
    zonas_rojas    = _obtener_info_cubos(red_zones)
    ids_cubos      = sorted(set(cubos_estables.keys()) | set(zonas_rojas.keys()))

    filas = []

    for id_cubo in ids_cubos:
        es_estable = id_cubo in cubos_estables
        info_cubo  = cubos_estables[id_cubo] if es_estable else zonas_rojas[id_cubo]

        # ── Identificacion ───────────────────────────────────────────────────
        fila = {
            "id_cubo":               id_cubo,
            "estable":               1 if es_estable else 0,
            "valor_prediccion":      info_cubo.get("prediction_value"),
            "motivo_rechazo":        json.dumps(
                                         info_cubo.get("rejection_reasons", []),
                                         ensure_ascii=False
                                     ),
            "profundidad_particion": len(id_cubo),
        }

        # ── Promedios de variables predictoras ───────────────────────────────
        grupo_dom1 = domains[0]["groups"].get(id_cubo, pd.DataFrame())

        for variable in feature_names:
            col_name = "prom_" + variable
            if grupo_dom1.empty or variable not in grupo_dom1.columns:
                fila[col_name] = None
            else:
                fila[col_name] = _promedio(grupo_dom1[variable])

        # ── Metricas del target por dominio (columnas planas) ─────────────────
        for i, dominio in enumerate(domains, start=1):
            x_dom = dominio["groups"].get(id_cubo, pd.DataFrame())

            if x_dom.empty:
                y_dom = pd.Series([], dtype=float)
            else:
                y_dom = dominio["y"].loc[x_dom.index]
                if isinstance(y_dom, pd.DataFrame):
                    y_dom = y_dom.iloc[:, 0]

            fila["n_dom"           + str(i)] = len(y_dom)
            fila["prom_target_dom" + str(i)] = _promedio(y_dom)
            fila["std_target_dom"  + str(i)] = _std(y_dom)

        filas.append(fila)

    return pd.DataFrame(filas)


# ─── Cuts checkpoint ─────────────────────────────────────────────────────────

def build_cuts_checkpoint(cuts):
    """
    Construye el DataFrame del arbol de cortes aprendido.
    Una fila por nodo interno (corte).
    """
    if cuts is None:
        return pd.DataFrame()

    cortes_df = pd.DataFrame(cuts)

    columnas_renombradas = {
        "node_id":       "id_nodo",
        "level":         "nivel",
        "variable":      "variable_corte",
        "left_path":     "ruta_izquierda",
        "right_path":    "ruta_derecha",
        "left_size":     "tamaño_izquierda",
        "right_size":    "tamaño_derecha",
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
    """Resumen global de la ejecucion del modelo."""
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
        grupos  = dominio.get("groups", {})
        tamaños = [len(g) for g in grupos.values()]

        if not tamaños:
            continue

        filas.extend([
            {"metrica": "dominio_" + str(i) + "_total_grupos",
             "valor": len(tamaños)},
            {"metrica": "dominio_" + str(i) + "_tamanio_minimo_grupo",
             "valor": min(tamaños)},
            {"metrica": "dominio_" + str(i) + "_tamanio_maximo_grupo",
             "valor": max(tamaños)},
            {"metrica": "dominio_" + str(i) + "_tamanio_promedio_grupo",
             "valor": sum(tamaños) / len(tamaños)},
            {"metrica": "dominio_" + str(i) + "_grupos_vacios",
             "valor": sum(1 for t in tamaños if t == 0)},
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
    file_format="xlsx"
):
    """
    Exporta el checkpoint de trazabilidad a Excel o CSV.

    Parametros:
    -----------
    path         : str  — ruta de salida ('checkpoint.xlsx' o 'checkpoint.csv')
    domains      : list — dominios usados en fit()
    stable_cubes : list — cubos estables
    red_zones    : list — zonas rojas
    cuts         : list — cortes aprendidos
    feature_names: list — variables predictoras del modelo
    file_format  : str  — 'xlsx' o 'csv'

    Hojas generadas (xlsx):
    -----------------------
    cubes_checkpoint : una fila por cubo. Promedios de predictoras y metricas
                       del target por dominio. Sin JSON, legible directo.
    cuts_checkpoint  : una fila por corte del arbol de particion.
    summary          : metricas globales de la ejecucion.
    """
    cubes_df   = build_cubes_checkpoint(
                     domains=domains,
                     stable_cubes=stable_cubes,
                     red_zones=red_zones,
                     feature_names=feature_names,
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
