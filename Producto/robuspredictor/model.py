from .utils import validate_params, validate_predict_data, validate_fit_data
from .partitioning import median_partition, apply_median_cuts
from .stability import select_stable_cubes
from .prediction import predict_from_stable_cubes
from .domains import split_training_domains


class RobusPredictor:
    def __init__(
        self, 
        n_min, 
        n_max, 
        n_dom, 
        mean_min, 
        mean_max, 
        std_min,
        std_max,
        default_value=0, 
        verbose=False):
        
        validate_params(
            n_min, 
            n_max, 
            n_dom,
            mean_min, 
            mean_max, 
            std_min,
            std_max
        )
        
        self.n_min = n_min
        self.n_max = n_max
        self.n_dom = n_dom
        self.mean_max = mean_max
        self.mean_min = mean_min
        self.std_min = std_min
        self.std_max = std_max
        self.default_value= default_value
        self.verbose = verbose
        
        self.base_grid = None
        self.cuts = None
        self.domains = None
        self.stable_cubes = []
        self.red_zones = []
        self.feature_names = None
        self.is_fitted = False
    
    
    def fit(self, x, y):
        validate_fit_data(x, y, self.n_dom)
        
        self.feature_names = list(x.columns)
        
        if self.verbose:
            print("\n[Fit] Inicio entrenamiento RobusPredictor")
            print(f"[Fit] Dataset entrenamiento X={x.shape}, y={y.shape}")
            print(f"[Fit] Variables predictoras: {self.feature_names}")
            
        self.domains = split_training_domains(
            x=x,
            y=y,
            n_domain=self.n_dom,
            verbose=self.verbose,
        )
        
        if len(self.domains) < 2:
            raise ValueError(
                "Se necesitan al menos 2 dominios para comparar estabilidad."
            )
        
        dominio_base = self.domains[0]
        x_base = dominio_base["x"]
        
        if self.verbose:
            print("\n[Fit] Generando grilla base desde Dominio 1")
            print(f"[Fit] Dominio base X={x_base.shape}")
        
        self.base_grid = median_partition(
            x=x_base,
            n_min=self.n_min,
            n_max=self.n_max,
            verbose=self.verbose,
        )
        
        if not isinstance(self.base_grid, dict):
            raise TypeError(
                "recursive_median_partition debe retornar un diccionario "
                "con las claves 'groups' y 'cuts'."
            )
            
        base_groups = self.base_grid["groups"]
        self.cuts = self.base_grid["cuts"]
        
        self.domains[0]["groups"] = base_groups
        
        if self.verbose:
            print(f"\n[Fit] Cubos generados en dominio base: {len(base_groups)}")
            print(f"[Fit] Cortes aprendidos: {len(self.cuts)}")
        
        # 6. Aplicar los mismos cortes al resto de dominios
        for i in range(1, len(self.domains)):
            domain_x = self.domains[i]["x"]
            
            if self.verbose:
                print(f"\n[Fit] Aplicando cortes base sobre Dominio {i + 1}")
                print(f"[Fit] Dominio {i + 1} X={domain_x.shape}")
            
            domain_groups = apply_median_cuts(
                x=domain_x,
                cuts=self.cuts,
                verbose=self.verbose,
            )
            
            self.domains[i]["groups"] = domain_groups
            
            if self.verbose:
                print(
                    f"[Fit] Cubos generados en Dominio {i + 1}: "
                    f"{len(domain_groups)}"
                )
        
        # 7. Seleccionar cubos estables usando los mismos cortes en todos los dominios
        self.stable_cubes, self.red_zones = select_stable_cubes(
            domains=self.domains,
            mean_max=self.mean_max,
            mean_min=self.mean_min,
            std_min=self.std_min,
            std_max=self.std_max,
            verbose=self.verbose,
        )
        
        if self.verbose:
            print(f"\n[Fit] Cubos estables finales: {len(self.stable_cubes)}")
            print(f"[Fit] Zonas rojas detectadas: {len(self.red_zones)}")
        
        self.is_fitted = True
        return self


    def predict(self, x):
        if not self.is_fitted:
            raise ValueError("El modelo debe entrenarse con fit() antes de predecir.")
        
        validate_predict_data(x)
        
        missing_columns = set(self.feature_names) - set(x.columns)
        
        if missing_columns:
            raise ValueError(f"Faltan columnas en X para predecir: {missing_columns}")
        
        x = x[self.feature_names]
        
        return predict_from_stable_cubes(
            X=x,
            stable_cubes=self.stable_cubes,
            cuts=self.cuts,
            default_value=self.default_value,
            verbose=self.verbose
        )