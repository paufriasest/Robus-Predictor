from .utils import validate_params, validate_predict_data, validate_fit_data
from .partitioning import median_partition, apply_median_cuts
from .stability import select_stable_cubes
from .prediction import predict_from_stable_cubes
from .domains import split_training_domains
from .checkpoint import export_checkpoint


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
        verbose=False,
    ):
        """
        Inicializa el modelo RobusPredictor.

        Parametros:
        -----------
        n_min         : int   — Tamanio minimo de registros por cubo (condicion de parada
                                del particionamiento). Controla la granularidad: valores
                                bajos generan muchos cubos pequenios; valores altos generan
                                pocos cubos grandes y mas estables.

        n_max         : int   — Tamanio maximo de registros permitido en un cubo final.
                                Si un cubo excede este valor se lanza un error. Sirve
                                como control de calidad del particionamiento.

        n_dom         : int   — Numero de dominios internos (minimo 2). El dataset de
                                entrenamiento se divide en n_dom partes iguales. El
                                dominio 1 construye el arbol de particion; los dominios
                                2..N validan que los patrones sean consistentes. A mayor
                                n_dom, mayor exigencia de estabilidad.

        mean_min      : float — Promedio minimo aceptable del target dentro de un cubo
                                estable. Cubos con promedio menor son marcados como zona
                                roja. Define el piso del rango de interes de negocio.

        mean_max      : float — Promedio maximo aceptable del target dentro de un cubo
                                estable. Cubos con promedio mayor son marcados como zona
                                roja. Define el techo del rango de interes de negocio.

        std_min       : float — Desviacion estandar minima aceptable del target. En la
                                mayoria de los casos se usa 0.0 (sin restriccion inferior).

        std_max       : float — Desviacion estandar maxima aceptable del target. Controla
                                la homogeneidad del cubo: valores bajos exigen que todos
                                los registros del cubo tengan targets similares. Es el
                                filtro de estabilidad mas importante.

        default_value : float — Valor asignado en la prediccion a registros que caen en
                                cubos inestables (zonas rojas) o que no pertenecen a
                                ningun cubo estable. Por convencion se usa 0.

        verbose       : bool  — Si True, imprime trazabilidad detallada del proceso de
                                particionamiento, entrenamiento y prediccion.
        """
        validate_params(n_min, n_max, n_dom, mean_min, mean_max, std_min, std_max)

        self.n_min         = n_min
        self.n_max         = n_max
        self.n_dom         = n_dom
        self.mean_max      = mean_max
        self.mean_min      = mean_min
        self.std_min       = std_min
        self.std_max       = std_max
        self.default_value = default_value
        self.verbose       = verbose

        self.base_grid    = None
        self.cuts         = None
        self.domains      = None
        self.stable_cubes = []
        self.red_zones    = []
        self.feature_names = None
        self.is_fitted    = False
        self.checkpoint   = None

    def fit(self, x, y):
        validate_fit_data(x, y, self.n_dom)

        self.feature_names = list(x.columns)

        if self.verbose:
            print("\n[Fit] Inicio entrenamiento RobusPredictor")
            print(f"[Fit] Dataset entrenamiento X={x.shape}, y={y.shape}")
            print(f"[Fit] Variables predictoras: {self.feature_names}")

        self.domains = split_training_domains(
            x=x, y=y, n_domain=self.n_dom, verbose=self.verbose
        )

        if len(self.domains) < 2:
            raise ValueError("Se necesitan al menos 2 dominios para comparar estabilidad.")

        dominio_base = self.domains[0]
        x_base = dominio_base["x"]

        if self.verbose:
            print("\n[Fit] Generando grilla base desde Dominio 1")
            print(f"[Fit] Dominio base X={x_base.shape}")

        self.base_grid = median_partition(
            x=x_base, n_min=self.n_min, n_max=self.n_max, verbose=self.verbose
        )

        if not isinstance(self.base_grid, dict):
            raise TypeError(
                "median_partition debe retornar un diccionario con 'groups' y 'cuts'."
            )

        base_groups = self.base_grid["groups"]
        self.cuts   = self.base_grid["cuts"]
        self.domains[0]["groups"] = base_groups

        if self.verbose:
            print(f"\n[Fit] Cubos generados en dominio base: {len(base_groups)}")
            print(f"[Fit] Cortes aprendidos: {len(self.cuts)}")

        for i in range(1, len(self.domains)):
            domain_x = self.domains[i]["x"]

            if self.verbose:
                print(f"\n[Fit] Aplicando cortes sobre Dominio {i + 1}")
                print(f"[Fit] Dominio {i + 1} X={domain_x.shape}")

            domain_groups = apply_median_cuts(
                x=domain_x, cuts=self.cuts, verbose=self.verbose
            )
            self.domains[i]["groups"] = domain_groups

            if self.verbose:
                print(f"[Fit] Cubos en Dominio {i + 1}: {len(domain_groups)}")

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

        missing = set(self.feature_names) - set(x.columns)
        if missing:
            raise ValueError(f"Faltan columnas en X para predecir: {missing}")

        x = x[self.feature_names]

        return predict_from_stable_cubes(
            X=x,
            stable_cubes=self.stable_cubes,
            cuts=self.cuts,
            default_value=self.default_value,
            verbose=self.verbose,
        )

    def export_checkpoint(self, path, file_format="xlsx", X_test=None, y_test=None):
        """
        Exporta el checkpoint de trazabilidad a Excel o CSV.

        Parametros:
        -----------
        path        : str          — ruta de salida ('checkpoint.xlsx' o 'checkpoint.csv')
        file_format : str          — 'xlsx' o 'csv'
        X_test      : pd.DataFrame — dataset de testing (features). Opcional.
                      Si se provee, se aplican los cortes aprendidos para asignar
                      cada registro de testing al cubo correspondiente, y se
                      agregan las columnas n_testing, prom_target_testing,
                      std_target_testing y prom_target_consolidado al checkpoint.
        y_test      : pd.Series    — target real del dataset de testing. Requerido
                      si se provee X_test.
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe entrenarse con fit() antes de exportar checkpoint.")

        # Procesar datos de testing si se proveen
        testing_groups = None
        y_testing      = None

        if X_test is not None and y_test is not None:
            if self.verbose:
                print("[Checkpoint] Aplicando cortes a datos de testing...")

            X_test = X_test[self.feature_names]
            testing_groups = apply_median_cuts(
                x=X_test, cuts=self.cuts, verbose=False
            )
            y_testing = y_test

            if self.verbose:
                print(f"[Checkpoint] Grupos de testing generados: {len(testing_groups)}")

        self.checkpoint = export_checkpoint(
            path=path,
            domains=self.domains,
            stable_cubes=self.stable_cubes,
            red_zones=self.red_zones,
            cuts=self.cuts,
            feature_names=self.feature_names,
            file_format=file_format,
            testing_groups=testing_groups,
            y_testing=y_testing,
        )

        if self.verbose:
            print(f"[Checkpoint] Exportado correctamente en: {path}")

        return self.checkpoint
