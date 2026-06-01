import pandas as pd
import numpy as np


def calcular_mae(y_real, y_pred):
    """
    Calcula el MAE (Mean Absolute Error) entre valores reales y predichos.

    Parametros:
    -----------
    y_real : pd.Series — valores reales del target (INTENSIDAD_4H)
    y_pred : pd.Series — predicciones del modelo

    Retorna:
    --------
    float — error absoluto medio
    """
    if len(y_real) == 0 or len(y_pred) == 0:
        return None

    return float(np.mean(np.abs(y_real.values - y_pred.values)))


def calcular_precision_top5(y_pred, y_binaria, top_pct=0.05):
    """
    Calcula la precision en el Top N% de predicciones mas altas.

    Toma los registros con mayor valor de prediccion, y mide que
    proporcion de ellos tiene y_binaria = 1 (ARRIENDO = 1).

    Precision Top 5% = registros_con_arriendo_en_top5 / total_registros_en_top5

    Parametros:
    -----------
    y_pred    : pd.Series — predicciones del modelo (una por registro)
    y_binaria : pd.Series — variable binaria real (ARRIENDO: 0 o 1)
    top_pct   : float     — fraccion del top a considerar (default 0.05 = 5%)

    Retorna:
    --------
    tuple (precision, n_top, n_positivos)
        precision    : float — proporcion de positivos en el top
        n_top        : int   — cantidad de registros en el top
        n_positivos  : int   — registros con y_binaria = 1 en el top
    """
    if len(y_pred) == 0:
        return None, 0, 0

    n_top       = max(1, int(len(y_pred) * top_pct))
    top_indices = y_pred.nlargest(n_top).index

    n_positivos = int(y_binaria.loc[top_indices].sum())
    precision   = n_positivos / n_top if n_top > 0 else 0.0

    return precision, n_top, n_positivos