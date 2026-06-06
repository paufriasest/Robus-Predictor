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


def calculate_precision_top_percentage(y_pred, y_target, top_pct=0.05):
    """
    Calcula la precision en el Top N% de predicciones mas altas.

    Toma los registros con mayor valor de prediccion, y mide que
    proporcion de ellos tiene y_target = 1 (ARRIENDO = 1).

    Precision Top 5% = registros_con_arriendo_en_top5 / total_registros_en_top5

    Parametros:
    -----------
    y_pred    : pd.Series — predicciones del modelo (una por registro)
    y_target : pd.Series — variable binaria real (ARRIENDO: 0 o 1)
    top_pct   : float     — fraccion del top a considerar (default 0.05 = 5%)

    Retorna:
    --------
    tuple (precision, n_top, n_positivos)
        precision    : float — proporcion de positivos en el top
        n_top        : int   — cantidad de registros en el top
        n_positivos  : int   — registros con y_target = 1 en el top
    """
    if not isinstance(y_pred, pd.Series):
        raise TypeError("y_pred debe ser una Series de pandas.")
    
    if not isinstance(y_target, pd.Series):
        raise TypeError("y_target debe ser una Series de pandas.")
    
    if len(y_pred) == 0:
        return {
            "top_pct": top_pct,
            "precision": None,
            "n_total": 0,
            "n_top": 0,
            "n_positivos": 0
        }
        
    if len(y_pred) != len(y_target):
        raise ValueError("y_pred e y_target deben tener la misma cantidad de registros.")
    
    if not y_pred.index.equals(y_target.index):
        raise ValueError("y_pred e y_target deben tener el mismo índice.")
    
    if not isinstance(top_pct, (int, float)):
        raise TypeError("top_pct debe ser numérico.")
    
    if top_pct <= 0 or top_pct > 1:
        raise ValueError("top_pct debe estar entre 0 y 1. Ejemplo: 0.05 para 5%.")
    
    n_total = len(y_pred)
    n_top       = max(1, int(len(y_pred) * top_pct))
    top_indices = y_pred.nlargest(n_top).index
    
    n_positivos = int(y_target.loc[top_indices].sum())
    precision   = n_positivos / n_top if n_top > 0 else 0.0
    
    return {
        "top_pct": top_pct,
        "precision": precision,
        "n_total": n_total,
        "n_top": n_top,
        "n_positivos": n_positivos
    }