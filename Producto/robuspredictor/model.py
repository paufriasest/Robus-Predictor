from .utils import validate_params, validate_predict_data, validate_domains
from .partitioning import recursive_median_partition
from .stability import select_stable_cubes
from .prediction import predict_from_stable_cubes


# Clase priuncipal de la libreria, es el modelo de robuspredictor
# NOTE: Definimos 
#          X = var predictorias    
#          Y = var target
class RobusPredictor:
    # Inicio de robuspredictor donde recibe los parametros
    # Ademas, se valida el tipo de dato de cada parametro
    def __init__(self, n_min, n_max, n_dom,  mean_max, mean_min, std_min, default_value=0, verbose=False):
        
        # funcion que valida parametros
        validate_params(n_min, n_max, n_dom, mean_max, mean_min, std_min)

        self.n_min = n_min
        self.n_max = n_max
        self.n_dom = n_dom
        self.mean_max = mean_max
        self.mean_min = mean_min
        self.std_min = std_min
        self.default_value= default_value
        self.verbose = verbose
        
        self.domain_cubes = None
        # Aqui se guardaran solos los cubos que cumplen los criterios de estabilidad
        self.stable_cubes = None
        
        # inicia donde se guardan los nombres de las columnas usadas en el entrenamiento
        # despues se validara que los datos nuevos tengan las mismas variables
        self.feature_names = None
        
        # flag que indica si el modelo se entreno
        self.is_fitted = False


    # Se def el primer metodo de robuspredictor el cual entrena/construye el modelo
    # Recibe de parametros  X= variables predictoras los dominios que deben venir en un dataframe las variables predictorias y en otro df var target
    def fit(self, *domains):
        domain_pairs = validate_domains(domains, self.n_dom)
        
        self.feature_names = list(domain_pairs[0][0].columns)
        self.domain_cubes = []
        
        if self.verbose:
            print("\n[Fit] Inicio entrenamiento RobusPredictor")
            print(f"[Fit] Cantidad de dominios recibidos: {len(domain_pairs)}")
            print(f"[Fit] Variables predictoras: {self.feature_names}")
            print(
                f"[Fit] Parámetros: "
                f"element_cube_min={self.n_min}, "
                f"element_cube_max={self.n_max}, "
                f"n_dom={self.n_dom}, "
                f"mean_cube_min={self.mean_min}, "
                f"mean_cube_max={self.mean_max}, "
                f"desv_cube_min={self.std_min}"
            )
            
        for domain_number, (X_domain, y_domain) in enumerate(domain_pairs, start=1):
            if self.verbose:
                print(f"\n[Fit] Procesando Dominio {domain_number}")
                print(f"[Fit] X={X_domain.shape}, y={y_domain.shape}")
            
            cubes = recursive_median_partition(
                X_domain,
                n_min=self.n_min,
                n_max=self.n_max,
                verbose=self.verbose,
            )
            
            self.domain_cubes.append(
                {
                    "domain": domain_number,
                    "X": X_domain,
                    "y": y_domain,
                    "cubes": cubes,
                }
            )
            
            if self.verbose:
                print(f"[Fit] Cubos generados Dominio {domain_number}: {len(cubes)}")
        
        self.stable_cubes = select_stable_cubes(
            domain_cubes=self.domain_cubes,
            mean_min=self.mean_min,
            mean_max=self.mean_max,
            std_max=self.std_min,
            verbose=self.verbose,
        )
        
        if self.verbose:
            print(f"\n[Fit] Cubos estables finales: {len(self.stable_cubes)}")
        
        self.is_fitted = True
        return self
    
    # Se def el segundo metodo de robuspredictor el cual predice con datos nuevos y el modelo previamente entrenado con los datos de entrenamiento
    # Recibe solo el x=dataframe de la varia predic
    # Se valida que el modelo se haya entrenado previamente para evitar errores.
    def predict(self, x):
        
        # valida que el modelo haya sido entrenado
        if not self.is_fitted:
            raise ValueError("El modelo debe entrenarse con fit() antes de predecir.")

        #valida que x sea un df y no este vacio
        validate_predict_data(x)
        
        # verifica si faltan columnas en los datos nuevos () restamos los elementos que estaban con lso nuevos para validar si corresponden
        # como el cuento de la oveja y las matematicas
        missing_columns = set(self.feature_names) - set(x.columns)

        # si faltan columnas tira el error
        if missing_columns:
            raise ValueError(f"Faltan columnas en X para predecir: {missing_columns}")

        # reordena las columnas de x para que esten en el mismo orden que en el entrenamiento
        x = x[self.feature_names]

        # devuelve el valor de la prediccion, si al fila cae en un cubo estable usa el predict value si no devuelve default value
        return predict_from_stable_cubes(
            X=x,
            stable_cubes=self.stable_cubes,
            default_value=self.default_value,
            verbose=self.verbose
        )