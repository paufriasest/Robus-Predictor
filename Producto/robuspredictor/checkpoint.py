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
    testing_groups=None,
    y_testing=None,
):
    """
    Construye el DataFrame de auditoria de cubos.

    Estructura de columnas (sin JSON, todo legible en Excel):

    Identificacion:
        id_cubo | estable | valor_prediccion | motivo_rechazo
        profundidad_particion

    Promedios de variables predictoras por dominio:
        prom_<var>_dom1   : promedio en el dominio de entrenamiento (referencia espacial)
        prom_<var>_dom2   : promedio en el dominio de validacion interna
        prom_<var>_consol : promedio consolidado (dom1 + dom2)

    Metricas del target por dominio:
        n_dom1     | prom_target_dom1 | std_target_dom1
        n_dom2     | prom_target_dom2 | std_target_dom2
        n_testing  | prom_target_testing | std_target_testing  (si se proveen datos)
        prom_target_consolidado : promedio de todos los dominios disponibles

    Parametros opcionales:
        testing_groups : dict[str, pd.DataFrame]  — grupos del dataset de testing
                        generados con apply_median_cuts(X_test, cuts)
        y_testing      : pd.Series                — target real del dataset de testing
    """
    cubos_estables = _obtener_info_cubos(stable_cubes)
    zonas_rojas    = _obtener_info_cubos(red_zones)
    ids_cubos      = sorted(set(cubos_estables.keys()) | set(zonas_rojas.keys()))
    tiene_testing  = testing_groups is not None and y_testing is not None

    filas = []

    for id_cubo in ids_cubos:
        es_estable = id_cubo in cubos_estables
        info_cubo  = cubos_estables[id_cubo] if es_estable else zonas_rojas[id_cubo]

        # ── Identificacion ───────────────────────────────────────────────────
        fila = {
            "id_cubo": id_cubo,
            "estable": 1 if es_estable else 0,
            "valor_prediccion": info_cubo.get("prediction_value"),
            "suma_promedios_region": info_cubo.get("sum_means"),
            "promedio_promedios_region": info_cubo.get("mean_of_means"),
            "promedios_region_por_dominio": json.dumps(
                info_cubo.get("domain_means", []),
                ensure_ascii=False
            ),
            "motivo_rechazo": json.dumps(
                                info_cubo.get("rejection_reasons", []),
                                ensure_ascii=False
                            ),
            "profundidad_particion": len(id_cubo),
        }

        # ── Promedios de variables predictoras por dominio ───────────────────
        grupo_dom1 = domains[0]["groups"].get(id_cubo, pd.DataFrame())
        grupo_dom2 = domains[1]["groups"].get(id_cubo, pd.DataFrame()) \
                    if len(domains) > 1 else pd.DataFrame()

        for variable in feature_names:
            # Dom1 — referencia espacial del cubo
            if grupo_dom1.empty or variable not in grupo_dom1.columns:
                prom_d1 = None
            else:
                prom_d1 = _promedio(grupo_dom1[variable])

            # Dom2 — validacion interna
            if grupo_dom2.empty or variable not in grupo_dom2.columns:
                prom_d2 = None
            else:
                prom_d2 = _promedio(grupo_dom2[variable])

            fila["prom_" + variable + "_dom1"]   = prom_d1
            fila["prom_" + variable + "_dom2"]   = prom_d2
            fila["prom_" + variable + "_consol"]  = _promedio_consolidado(prom_d1, prom_d2)

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

        # ── Metricas del target — Dom2 (y dominios adicionales) ──────────────
        prom_adicionales = [fila["prom_target_dom1"]]   # acumula para consolidado

        for i in range(1, len(domains)):
            x_di = domains[i]["groups"].get(id_cubo, pd.DataFrame())
            if x_di.empty:
                y_di = pd.Series([], dtype=float)
            else:
                y_di = domains[i]["y"].loc[x_di.index]
                if isinstance(y_di, pd.DataFrame):
                    y_di = y_di.iloc[:, 0]

            n    = len(y_di)
            prom = _promedio(y_di)
            std  = _std(y_di)

            fila["n_dom"           + str(i + 1)] = n
            fila["prom_target_dom" + str(i + 1)] = prom
            fila["std_target_dom"  + str(i + 1)] = std

            if prom is not None:
                prom_adicionales.append(prom)

        # ── Metricas del target — Testing (opcional) ──────────────────────────
        if tiene_testing:
            x_test = testing_groups.get(id_cubo, pd.DataFrame())
            if x_test.empty:
                y_test_cubo = pd.Series([], dtype=float)
            else:
                y_test_cubo = y_testing.loc[x_test.index]
                if isinstance(y_test_cubo, pd.DataFrame):
                    y_test_cubo = y_test_cubo.iloc[:, 0]

            prom_test = _promedio(y_test_cubo)
            fila["n_testing"]           = len(y_test_cubo)
            fila["prom_target_testing"] = prom_test
            fila["std_target_testing"]  = _std(y_test_cubo)

            if prom_test is not None:
                prom_adicionales.append(prom_test)
        else:
            fila["n_testing"]           = None
            fila["prom_target_testing"] = None
            fila["std_target_testing"]  = None

        # ── Promedio target consolidado (todos los dominios + testing) ────────
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
        grupos  = dominio.get("groups", {})
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
    testing_groups=None,
    y_testing=None,
):
    """
    Exporta el checkpoint de trazabilidad a Excel o CSV.

    Parametros:
    -----------
    path           : str                  — ruta de salida
    domains        : list[dict]           — dominios usados en fit()
    stable_cubes   : list[dict]           — cubos estables
    red_zones      : list[dict]           — zonas rojas
    cuts           : list[dict]           — cortes aprendidos
    feature_names  : list[str]            — variables predictoras
    file_format    : str                  — 'xlsx' o 'csv'
    testing_groups : dict[str, DataFrame] — grupos del dataset de testing,
                    generados con apply_median_cuts(X_test, cuts). Opcional.
    y_testing      : pd.Series            — target real del dataset de testing.
                    Requerido si se provee testing_groups.

    Hojas generadas (xlsx):
    -----------------------
    cubes_checkpoint : una fila por cubo. Incluye promedios de predictoras
                    por dom1/dom2/consolidado, metricas del target por
                    dominio, testing (si se provee) y consolidado global.
    cuts_checkpoint  : arbol de cortes del particionamiento.
    summary          : metricas globales de la ejecucion.
    """
    cubes_df = build_cubes_checkpoint(
        domains=domains,
        stable_cubes=stable_cubes,
        red_zones=red_zones,
        feature_names=feature_names,
        testing_groups=testing_groups,
        y_testing=y_testing,
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
