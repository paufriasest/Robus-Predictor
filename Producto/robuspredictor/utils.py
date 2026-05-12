import pandas as pd
import numpy as np


def _is_number(value):
    """
    Valida si un valor es numérico, excluyendo booleanos.
    En Python, bool hereda de int, por eso se valida aparte.
    """
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


# Función que valida los parámetros para RobusPredictor
def validate_params(
    n_min,
    n_max,
    n_dom,
    mean_min,
    mean_max,
    std_min,
    std_max
):
    # Tipos enteros
    if not isinstance(n_min, int) or isinstance(n_min, bool):
        raise TypeError("n_min debe ser un número entero.")

    if not isinstance(n_max, int) or isinstance(n_max, bool):
        raise TypeError("n_max debe ser un número entero.")

    if not isinstance(n_dom, int) or isinstance(n_dom, bool):
        raise TypeError("n_dom debe ser un número entero.")

    # Tipos numéricos
    if not _is_number(mean_min):
        raise TypeError("mean_min debe ser un número numérico: int o float.")

    if not _is_number(mean_max):
        raise TypeError("mean_max debe ser un número numérico: int o float.")

    if not _is_number(std_min):
        raise TypeError("std_min debe ser un número numérico: int o float.")

    if not _is_number(std_max):
        raise TypeError("std_max debe ser un número numérico: int o float.")

    # Rangos básicos
    if n_min <= 0:
        raise ValueError("n_min debe ser mayor a 0.")

    if n_max <= 0:
        raise ValueError("n_max debe ser mayor a 0.")

    if n_dom < 2:
        raise ValueError(
            "n_dom debe ser al menos 2, porque se requieren al menos dos dominios "
            "para comparar estabilidad."
        )

    if n_min > n_max:
        raise ValueError("n_min no puede ser mayor que n_max.")

    if mean_min > mean_max:
        raise ValueError("mean_min no puede ser mayor que mean_max.")

    if std_min < 0:
        raise ValueError("std_min no puede ser negativo.")

    if std_max < 0:
        raise ValueError("std_max no puede ser negativo.")

    if std_min > std_max:
        raise ValueError("std_min no puede ser mayor que std_max.")


# Función que valida los parámetros para .fit()
def validate_fit_data(x, y, n_domain):
    if not isinstance(x, pd.DataFrame):
        raise TypeError("X debe ser un DataFrame de pandas.")

    if not isinstance(y, (pd.Series, pd.DataFrame)):
        raise TypeError("y debe ser Series o DataFrame de pandas.")

    if x.empty:
        raise ValueError("X no puede estar vacío.")

    if y.empty:
        raise ValueError("y no puede estar vacío.")

    if len(x) != len(y):
        raise ValueError("X e y deben tener la misma cantidad de filas.")

    if len(x) < n_domain:
        raise ValueError(
            "La cantidad de filas debe ser mayor o igual a n_domain."
        )

    if n_domain < 2:
        raise ValueError(
            "n_domain debe ser al menos 2 para generar dominios comparables."
        )

    # Validar columnas duplicadas
    if x.columns.duplicated().any():
        duplicated = list(x.columns[x.columns.duplicated()])
        raise ValueError(f"X contiene columnas duplicadas: {duplicated}")

    # Validar que todas las columnas de X sean numéricas
    non_numeric_columns = x.select_dtypes(exclude=["number"]).columns.tolist()

    if non_numeric_columns:
        raise TypeError(
            "Todas las columnas de X deben ser numéricas para aplicar cortes "
            f"por mediana. Columnas no numéricas encontradas: {non_numeric_columns}"
        )

    # Validar valores infinitos en X
    if np.isinf(x.to_numpy()).any():
        raise ValueError("X contiene valores infinitos.")

    # Validar y como DataFrame de una sola columna
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(
                "y debe ser una Series o un DataFrame de una sola columna."
            )

        if y.columns.duplicated().any():
            duplicated = list(y.columns[y.columns.duplicated()])
            raise ValueError(f"y contiene columnas duplicadas: {duplicated}")

        y_values = y.iloc[:, 0]

    else:
        y_values = y

    # Validar que y sea numérico
    if not pd.api.types.is_numeric_dtype(y_values):
        raise TypeError("y debe contener valores numéricos.")

    # Validar valores infinitos en y
    if np.isinf(y_values.to_numpy()).any():
        raise ValueError("y contiene valores infinitos.")

    # Importante para poder hacer y.loc[x_group.index]
    if not x.index.equals(y.index):
        raise ValueError(
            "X e y deben tener el mismo índice. "
            "Esto es necesario para asociar correctamente los grupos de X con los valores de y."
        )


# Función que valida el parámetro recibido en .predict()
def validate_predict_data(x):
    if not isinstance(x, pd.DataFrame):
        raise TypeError("x debe ser un DataFrame de pandas.")

    if x.empty:
        raise ValueError("x no puede estar vacío.")

    if x.columns.duplicated().any():
        duplicated = list(x.columns[x.columns.duplicated()])
        raise ValueError(f"x contiene columnas duplicadas: {duplicated}")

    non_numeric_columns = x.select_dtypes(exclude=["number"]).columns.tolist()

    if non_numeric_columns:
        raise TypeError(
            "Todas las columnas de x deben ser numéricas para aplicar los cortes "
            f"aprendidos. Columnas no numéricas encontradas: {non_numeric_columns}"
        )

    if np.isinf(x.to_numpy()).any():
        raise ValueError("x contiene valores infinitos.")