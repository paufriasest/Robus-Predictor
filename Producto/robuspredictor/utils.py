import pandas as pd

# Funcion que valida los parametros para robuspredictor
def validate_params(n_min, n_max, n_dom, mean_max, mean_min, std_min):
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

    if not isinstance(std_min, (int, float)):
        raise TypeError("std_min debe ser un número numérico: int o float.")

    if n_min <= 0:
        raise ValueError("n_min debe ser mayor a 0.")

    if n_max <= 0:
        raise ValueError("n_max debe ser mayor a 0.")

    if n_dom <= 0:
        raise ValueError("n_dom debe ser mayor a 0.")

    if n_min > n_max:
        raise ValueError("n_min no puede ser mayor que n_max.")

    if std_min < 0:
        raise ValueError("std_min no puede ser negativo.")

def validate_domains(domains, n_domain):
    if len(domains) % 2 != 0:
        raise ValueError("fit() debe recibir pares X, y. Ejemplo: fit(X1, y1, X2, y2).")

    domain_pairs = []

    for i in range(0, len(domains), 2):
        X = domains[i]
        y = domains[i + 1]

        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X del dominio {(i // 2) + 1} debe ser DataFrame.")

        if not isinstance(y, (pd.Series, pd.DataFrame)):
            raise TypeError(f"y del dominio {(i // 2) + 1} debe ser Series o DataFrame.")

        if X.empty:
            raise ValueError(f"X del dominio {(i // 2) + 1} no puede estar vacío.")

        if y.empty:
            raise ValueError(f"y del dominio {(i // 2) + 1} no puede estar vacío.")

        if len(X) != len(y):
            raise ValueError(
                f"X e y del dominio {(i // 2) + 1} deben tener la misma cantidad de filas."
            )

        domain_pairs.append((X, y))

    if len(domain_pairs) != n_domain:
        raise ValueError(
            f"Se esperaban {n_domain} dominios, pero se recibieron {len(domain_pairs)}."
        )

    first_columns = list(domain_pairs[0][0].columns)

    for i, (X, _) in enumerate(domain_pairs, start=1):
        if list(X.columns) != first_columns:
            raise ValueError(
                f"Las columnas del dominio {i} no coinciden con las del primer dominio."
            )

    return domain_pairs

# Funcion que valida los parametros para .fit(), que estos sean dataframes, no est[en vac[ios]] y que la longitud sea igual
def validate_fit_data(x):
    if not isinstance(x, pd.DataFrame):
        raise TypeError("X debe ser un DataFrame de pandas.")

    if x.empty:
        raise ValueError("X no puede estar vacío.")


# Funcion que valida el parametro recibido en .predict(), que sea un dataframe y que contenga información
def validate_predict_data(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X debe ser un DataFrame de pandas.")
    
    if X.empty:
        raise ValueError("X no puede estar vacío.")
    