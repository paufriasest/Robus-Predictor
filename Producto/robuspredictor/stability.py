def get_common_bounds(cubes_same_position):
    common_bounds = {}
    
    first_bounds = cubes_same_position[0]["bounds"]
    
    for column in first_bounds.keys():
        common_min = max(cube["bounds"][column]["min"] for cube in cubes_same_position)
        common_max = min(cube["bounds"][column]["max"] for cube in cubes_same_position)
        
        if common_min > common_max:
            return None
        
        common_bounds[column] = {
            "min": common_min,
            "max": common_max,
        }
        
    return common_bounds

# Funcion que seleccionar[a los cubos estables]
def select_stable_cubes(
    domain_cubes,
    mean_max, 
    mean_min, 
    std_max, 
    verbose=False
    ):
    
    #lista que guarda todos los cubos estables
    stable_cubes = []
    
    min_cube_count = min(len(domain["cubes"]) for domain in domain_cubes)
    
    if verbose:
        print("\n[Stability] Inicio selección de cubos estables")
        print(f"[Stability] Dominios recibidos: {len(domain_cubes)}")
        print(f"[Stability] Cubos comparables por posición: {min_cube_count}")
    
    
    for cube_position in range(min_cube_count):
        cubes_same_position = []
        domain_means = []
        domain_counts = []
        
        if verbose:
            print(f"\n[Stability] Evaluando posición de cubo #{cube_position}")
            
        for domain in domain_cubes:
            cube = domain["cubes"][cube_position]
            y = domain["y"]
            
            y_values = y.loc[cube["index"]]
            domain_mean = y_values.mean()
            domain_count = len(y_values)
            
            cubes_same_position.append(cube)
            domain_means.append(domain_mean)
            domain_counts.append(domain_count)
            
            if verbose:
                print(
                    f"[Stability] Dominio {domain['domain']} | "
                    f"Cubo posición {cube_position} | "
                    f"N={domain_count} | Prom target={domain_mean}"
                )
        
        min_domain_mean = min(domain_means)
        max_domain_mean = max(domain_means)
        
        if max_domain_mean == 0:
            if verbose:
                print(f"[Stability] Cubo #{cube_position} descartado: max_mean = 0")
            continue
        
        variation = (max_domain_mean - min_domain_mean) / abs(max_domain_mean)
        prediction_value = sum(domain_means) / len(domain_means)
        
        common_bounds = get_common_bounds(cubes_same_position)
        
        if common_bounds is None:
            if verbose:
                print(
                    f"[Stability] Cubo #{cube_position} descartado: "
                    "no existe intersección común entre dominios."
                )
            continue
        
        if verbose:
            print(f"[Stability] Promedios por dominio: {domain_means}")
            print(f"[Stability] prediction_value: {prediction_value}")
            print(f"[Stability] variation: {variation}")
            print(
                f"[Stability] Criterios usuario: "
                f"mean_min={mean_min}, mean_max={mean_max}, std_max={std_max}"
            )
            
        if mean_min <= prediction_value <= mean_max and variation <= std_max:
            stable_cubes.append(
                {
                    "cube_position": cube_position,
                    "bounds": common_bounds,
                    "prediction_value": prediction_value,
                    "variation": variation,
                    "domain_means": domain_means,
                    "domain_counts": domain_counts,
                }
            )
            
            if verbose:
                print(f"[Stability] Cubo #{cube_position} seleccionado como estable.")
        else:
            if verbose:
                print(f"[Stability] Cubo #{cube_position} descartado por criterios.")
        
    return stable_cubes