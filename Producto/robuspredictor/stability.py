# Funcion para dividir el grupo de datos en dominios
def split_domains(X, n_dom):
    
    # Divide los datos en n_dom dominios temporales.
    total_rows = len(X)
    # El tama;o del dominio viene determinado por la cantidad de filas en el numero de dominio
    domain_size = total_rows // n_dom

    # aqui se guardaran los dominios, cada dominio estar en un index
    domains = []

    # recorrido por cada dominio
    for i in range(n_dom):
        start = i * domain_size

        # verificar si esta en el ultimo dominio
        if i == n_dom - 1:
            end = total_rows
        else:
            end = (i + 1) * domain_size

        # guarda los dominios 
        domains.append(X.iloc[start:end].index)

    return domains

# Funcion que seleccionar[a los cubos estables]
def select_stable_cubes(X, y, cubes, n_dom, mean, std):
   
    # comenzamos la lista de cubos estables
    stable_cubes = []

    # se dividen los datos segun la funcion anterior
    domains = split_domains(X, n_dom)

    # recorrer cada cubo
    for cube_index in cubes:
        domain_means = []
        domain_counts = []

        # recorrer cada espacio temporal
        for domain_index in domains:
            # Intersección entre el cubo y el dominio temporal
            cube_domain_index = cube_index.intersection(domain_index)

            # si el cubo no tiene datos, se salta al sigueinte dominio
            if len(cube_domain_index) == 0:
                continue
            
            # obtienes los valores de la variable target para las filas que estan ese punto del cubo y dominio
            y_values = y.loc[cube_domain_index]
            # se calcula el promedio de y dentro de se cubo y dominio
            domain_mean = y_values.mean()
            domain_count = len(y_values)

            # guarda el promedio dentro dentro de la lista
            domain_means.append(domain_mean)
            domain_counts.append(domain_count)

        # Si el cubo no tiene informaci[on] en todos los dominios se descarta
        if len(domain_means) < n_dom:
            continue

        min_mean = min(domain_means)
        max_mean = max(domain_means)

        # Evita división por cero
        if max_mean == 0:
            continue

        # Desviación porcentual máxima respecto al promedio del cubo
        variation = (max_mean - min_mean) / abs(max_mean)

        # Criterios de estabilidad
        if min_mean >= mean and variation <= std:
            stable_cubes.append(
                {
                    "index": cube_index,
                    "domain_means": domain_means,
                    "domain_counts": domain_counts,
                    "min_mean": min_mean,
                    "max_mean": max_mean,
                    "variation": variation,
                    "prediction_value": sum(domain_means) / len(domain_means),
                }
            )

    return stable_cubes