# Funcion para particionar recursivamente por medianas para obtener la grilla de n/rectangulos
def recursive_median_partition(X, n_min, n_max):
    
    # Lista donde se guardaran los cubos o regiones finales generadas.
    cubes = []

    # Funcion interna que obtiene los límites min y max de cada variable dentro de un grupo. Estos lim se ocupaoran luego en predict()
    # para saber si un nuevo dato cae dentro de un cubo estable.
    def get_bounds(group):
        bounds = {}

        # Recorre cada columna del DataFrame.
        for column in group.columns:
            # Guarda el valor min y max de esa columna dentro del grupo
            bounds[column] = {
                "min": group[column].min(),
                "max": group[column].max(),
            }
        # Retorna los lim del grupo.
        return bounds
    
    # Funcion interna recursiva que divide un grupo de datos numero_de_corte indica qué variable corresponde usar para el corte actual.
    def split_group(group, numero_de_corte=0):
        
        # calcula cuantos elementos tiene el grupo
        group_size = len(group)

        # se valida que el grupo tenga la cantdidad minima espoecificada por el usuario, se descarta si es menor
        if group_size < n_min:
            return

        # se valida que tenga la cantidad minima y maxima permitida, se guarda cuando se termina de dividir de acuerod al n_max
        if n_min <= group_size <= n_max:
            #se agrega los indices en la lista de cubos y los limites del grupo
            cubes.append({
                # indice de las filas que pertenecen al cubo
                "index": group.index,   
                # lim min y max del cubo
                "bounds": get_bounds(group),
            })
            return

        # obtiene los nombres de cada columna  v1, v2, v3
        columnas = list(X.columns)
        
        # Selecciona la columna que se usara para cortar
        # El operador % permite ir rotando las columnas
        # corte 0 -> columna 0
        # corte 1 -> columna 1
        # corte 2 -> columna 2
        # cuando se acaban las columnas, vuelve a la primera.
        posicion_columna = numero_de_corte % len(columnas)
        
        #obtiene el nombre de la columna seleccionada par el chop chop
        columna_a_cortar = columnas[posicion_columna]

        # calcula la mediana de la columna seleccionada dentro del grupo actual
        valor_mediana = group[columna_a_cortar].median()
        
        # se divide en 2 grupos los de la
        # izq, menores a la mediana 
        # derecha, mayores a la mediana
        grupo_izquierdo = group[group[columna_a_cortar] <= valor_mediana]
        grupo_derecho = group[group[columna_a_cortar] > valor_mediana]


        # si alguno de los 2 grupos antertiores queda vacio significa que no se cortaron bien los grupos entonce 
        # no cumple con el tamaño esperado y no debe ser guardado como cubo estable
        if grupo_izquierdo.empty or grupo_derecho.empty:
            raise ValueError(
                f"No fue posible dividir el grupo usando la columna '{columna_a_cortar}'. "
                "Esto puede ocurrir si todos los valores de la columna son iguales."
            )
        
        # Aplica recursividad a ada grupo
        # Se incrementa numero_de_corte para que el siguiente corte use otra variable
        split_group(grupo_izquierdo, numero_de_corte + 1)
        split_group(grupo_derecho, numero_de_corte + 1)

    # empieza el proceso de particionamiento
    split_group(X)

    # retorna los cubos
    return cubes