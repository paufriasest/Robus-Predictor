import numpy as np
# Funcion para particionar recursivamente por medianas para obtener la grilla de n/rectangulos
def recursive_median_partition(X, n_min, n_max, verbose=False):
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
    
    def add_cube(group):
        cube = {
            "cube_position": len(cubes),
            "index": group.index,
            "bounds": get_bounds(group),
        }
        
        cubes.append(cube)
        
        if verbose:
            print(f"[Partitioning] Cubo creado posición: {cube['cube_position']}")
            print(f"[Partitioning] Registros: {len(group)}")
            print(f"[Partitioning] Bounds: {cube['bounds']}")
    
    # Funcion interna recursiva que divide un grupo de datos numero_de_corte indica qué variable corresponde usar para el corte actual.
    def split_group(group, numero_de_corte=0):
        # calcula cuantos elementos tiene el grupo
        group_size = len(group)
        
        if verbose:
            print(f"\n[Partitioning] Corte #{numero_de_corte}")
            print(f"[Partitioning] Tamaño del grupo actual: {group_size}")
        
        # se valida que el grupo tenga la cantdidad minima espoecificada por el usuario, se descarta si es menor
        if group_size < n_min:
            if verbose:
                print(f"[Partitioning] Grupo descartado: {group_size} < n_min ({n_min})")
            return
        
        # se valida que tenga la cantidad minima y maxima permitida, se guarda cuando se termina de dividir de acuerod al n_max
        if n_min <= group_size <= n_max:
            add_cube(group)
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
        
        if verbose:
            print(f"[Partitioning] Columna usada para corte: {columna_a_cortar}")
            print(f"[Partitioning] Mediana calculada: {valor_mediana}")
        
        # se divide en 2 grupos los de la
        # izq, menores a la mediana 
        # derecha, mayores a la mediana
        grupo_izquierdo = group[group[columna_a_cortar] <= valor_mediana]
        grupo_derecho = group[group[columna_a_cortar] > valor_mediana]
        
        if verbose:
            print(f"[Partitioning] Grupo izquierdo: {len(grupo_izquierdo)} registros")
            print(f"[Partitioning] Grupo derecho: {len(grupo_derecho)} registros")
        
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