# Funcion para particionar recursivamente por medianas para obtener la grilla de n/rectangulos
def recursive_median_partition(X, n_min, n_max):
    cubes = []

    def split_group(group, numero_de_corte=0):
        # calcula cuantos elementos tiene el grupo
        group_size = len(group)

        # se valida que el grupo tenga la cantdidad minima espoecificada por el usuario
        if group_size < n_min:
            return

        # se valida que tenga la cantidad minima y maxima permitida
        if n_min <= group_size <= n_max:
            #se agrega solo los indices en la lista de cubos
            cubes.append(group.index)
            return

        # obtiene los nombres de cada columna  v1, v2, v3
        columnas = list(X.columns)
        
        #elige que columna se utilizara para el corte , de esta forma se van rotando las columnas 
        posicion_columna = numero_de_corte % len(columnas)
        #obtiene el nombre de la columna que se utilizara para cortar
        columna_a_cortar = columnas[posicion_columna]

        # calcula el valor de la mediana dentro de la columna
        valor_mediana = group[columna_a_cortar].median()
        
        # se divide en 2 grupos los que 
        # izq, menores a la mediana 
        # derecha, mayores a la mediana
        grupo_izquierdo = group[group[columna_a_cortar] <= valor_mediana]
        grupo_derecho = group[group[columna_a_cortar] > valor_mediana]

        if grupo_izquierdo.empty or grupo_derecho.empty:
            cubes.append(group.index)
            return
        
        # recursividad por cada subgrupo
        split_group(grupo_izquierdo, numero_de_corte + 1)
        split_group(grupo_derecho, numero_de_corte + 1)

    split_group(X)

    return cubes