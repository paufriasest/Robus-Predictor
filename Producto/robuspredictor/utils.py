import pandas as pd

# Funcion que valida los parametros para robuspredictor
def validate_params(n_min, n_max, n_dom, mean_max, mean_min, std):
    if not isinstance(n_min, int):
        raise TypeError("n_min debe ser un número entero.")

    if not isinstance(n_max, int):
        raise TypeError("n_max debe ser un número entero.")

    if not isinstance(n_dom, int):
        raise TypeError("n_dom debe ser un número entero.")

    if not isinstance(mean_max, (int, float)):
        raise TypeError("mean_max debe ser un número numérico: int o float.")
    
    if not isinstance(mean_min, (int, float)):
        raise TypeError("mean_min debe ser un número numérico: int o float.")

    if not isinstance(std, (int, float)):
        raise TypeError("std debe ser un número numérico: int o float.")

    if n_min <= 0:
        raise ValueError("n_min debe ser mayor a 0.")

    if n_max <= 0:
        raise ValueError("n_max debe ser mayor a 0.")

    if n_dom <= 0:
        raise ValueError("n_dom debe ser mayor a 0.")

    if n_min > n_max:
        raise ValueError("n_min no puede ser mayor que n_max.")

    if std < 0:
        raise ValueError("std no puede ser negativo.")
    
# Funcion que valida los parametros para .fit(), que estos sean dataframes, no est[en vac[ios]] y que la longitud sea igual
def validate_fit_data(x, y):
    if not isinstance(x, pd.DataFrame):
        raise TypeError("X debe ser un DataFrame de pandas.")

    if not isinstance(y, (pd.DataFrame, pd.Series)):
        raise TypeError("y debe ser un DataFrame o Series de pandas.")

    if len(x) != len(y):
        raise ValueError(
            f"La variables predictorias y objetivas deben tener la misma cantidad de registros."
            f"Las variables predictorias tienen {len(x)} filas y las variables objetivas tienen {len(y)} filas."
        )

    if x.empty:
        raise ValueError("La variable predictora no puede estar vacío.")

    if y.empty:
        raise ValueError("La variable objetivo no puede estar vacío.")
    
    
# Funcion que valida el parametro recibido en .predict(), que sea un dataframe y que contenga información
def validate_predict_data(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X debe ser un DataFrame de pandas.")

    if X.empty:
        raise ValueError("X no puede estar vacío.")