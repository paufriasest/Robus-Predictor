# Funcion para dividir el grupo de datos en dominios
def split_domains(X, n_dom):
    
    # Obtiene la cantidad total de registros del DataFrame
    total_rows = len(X)
    
    # El tama;o del dominio viene determinado por la cantidad de filas en el numero de dominio
    # si hay 1000 filas y n_dom = 2, cada dominio tendrá 500 filas
    domain_size = total_rows // n_dom

    # aqui se guardaran los dominios, cada dominio estar en un index
    domains = []

    # Recorre la cantidad de dominios solicitados
    for i in range(n_dom):
        # Calcula la posicioin inicial del dominio actual
        start = i * domain_size

        # verificar si esta en el ultimo dominio, se toma hasta el final del DataFrame
        # para evitar perder filas cuando la división no es exacta
        if i == n_dom - 1:
            end = total_rows
        else:
            end = (i + 1) * domain_size

        # Guarda los indices de las filas que pertenecen al dominio actual.
        domains.append(X.iloc[start:end].index)

    # devuelve las liosta de los dominios
    return domains

# Funcion que seleccionar[a los cubos estables]
def select_stable_cubes(X, y, cubes, n_dom, mean_max, mean_min, std):
    
    #lista que guarda todos los cubos estables
    stable = []
    
    # aqui se usa la funcion anterior
    domains = split_domains(X, n_dom)

    # recorre cada cubo generado en el particionamiento
    for cube in cubes:
        
        #guardamos los indices de las filas que pértecenen los cubos
        idx = cube["index"]

        # lista donde se guarda el promedio de cada dominio 
        means = []

        #recorre cada dominio temporal
        for d in domains:
            
            #obtiene la interseccion filas que son del cubo y filas que pertenecen al dominio actual
            inter = idx.intersection(d)

            # si el cubo no teine datos dentro del dominio se continua con el otro
            if len(inter) == 0:
                continue

            # obtiene los vlores de las variables objetivos correcponditen
            # a las filñas que estan dentro del cubo y del dominio
            y_vals = y.loc[inter]
            
            # calcula el promedio de la variable objetivo en este dominio
            means.append(y_vals.mean())

        # si el cubo no tiene presencia en todos los dominios no s epuede evaluar sue stabilidad
        # se descarta
        if len(means) < n_dom:
            continue

        # menor promedio entre los dominios
        min_mean = min(means)
        
        #mayor promedio entre los domiunios
        max_mean = max(means)

        # evita la 
        if max_mean == 0:
            continue

        # calcula la varion porcentual entre el dominio con mayor y menor promedio, entre menor es mas estable el cubo
        variation = (max_mean - min_mean) / abs(max_mean)
        
        
        # calcula el valor predictivo del cubo 
        prediction_value = sum(means) / len(means)

        # Criterio de selección:
        # - El valor predictivo debe estar dentro del rango permitido
        #    por mean_min y mean_max
        # - La variación entre dominios debe ser menor o igual a std
        if (
            mean_min <= prediction_value <= mean_max
            and variation <= std
        ):
             # Si cumple los criterios, se guarda como cubo estable
            stable.append({
                "bounds": cube["bounds"],
                "prediction_value": prediction_value,
                "variation": variation,
            })

  # Retorna la lista de cubos estables seleccionados
    return stable