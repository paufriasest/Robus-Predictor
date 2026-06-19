from .utils import validate_params, validate_predict_data, validate_fit_data
from .partitioning import median_partition, apply_median_cuts
from .stability import select_stable_cubes
from .prediction import predict_from_stable_cubes, predict_cubes_from_cuts
from .domains import split_training_domains
from .checkpoint import export_checkpoint, build_and_export_prediction_checkpoint
from .metrics import calculate_precision_top_percentage
from pandas import DataFrame


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
        use_default_value=True,
        default_value=0,
        verbose=False
    ):
        """Inicializa el modelo RobusPredictor

        Args:
            n_min (int): Tamaño mínimo permitido para un cubo
            n_max (int): Tamaño máximo  nimo permitido para un cubo
            n_dom (int): Cantidad de dominios en que se dividirá el conjunto de entrenamiento. Debe ser mayor o igual a 2
            mean_min (float): Valor mínimo permitido para el promedio del target dentro de un cubo
            mean_max (float): Valor máximo permitido para el promedio del target dentro de un cubo
            std_min (float): Valor mínimo permitido para la desviación estándar del target dentro de un cubo
            std_max (float): Valor máximo permitido para la desviación estándar del target dentro de un cubo
            use_default_value (bool, optional): Define cómo se comporta el modelo cuando una observación cae en una zona roja durante la predicción. Defaults to True
                - Si es True, la predicción será default_value
                - Si es False, la predicción será el prediction_value calculado para esa zona roja
            default_value (float, optional): Valor utilizado como predicción cuando una observación cae en una zona roja. Defaults to 0
            verbose (bool, optional): Si es True, muestra información adicional durante el entrenamiento y la predicción. Defaults to False
        """
        validate_params(n_min, n_max, n_dom, mean_min, mean_max, std_min, std_max, use_default_value, default_value)

        self.n_min         = n_min
        self.n_max         = n_max
        self.n_dom         = n_dom
        self.mean_max      = mean_max
        self.mean_min      = mean_min
        self.std_min       = std_min
        self.std_max       = std_max
        self.use_default_value = use_default_value
        self.default_value = default_value
        self.verbose       = verbose
        self.random_state  = 67

        self.base_grid     = None
        self.cuts          = None
        self.domains       = None
        self.stable_cubes  = []
        self.red_zones     = []
        self.feature_names = None
        self.is_fitted     = False
        self.checkpoint    = None
        self.last_predictions = None
        self.last_prediction_X = None
        self.last_prediction_cubes = None
        self.cube_id_map = None

    def fit(self, x, y):
        """Funcion para entrenar el modelo

        Args:
            x (Dataframe): Variables de entrenamiento 
            y (Dataframe): Variable objetiva de entrenamiento

        Raises:
            ValueError: Se necesitan al menos 2 dominios
            TypeError: La particion debe contener grupos y cortes

        Returns:
            _type_: _description_
        """
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

        x_base = self.domains[0]["x"]

        if self.verbose:
            print("\n[Fit] Generando grilla base desde Dominio 1")
            print(f"[Fit] Dominio base X={x_base.shape}")

        self.base_grid = median_partition(
            x=x_base, n_min=self.n_min, n_max=self.n_max,
            verbose=self.verbose, random_state=self.random_state
        )

        if not isinstance(self.base_grid, dict):
            raise TypeError(
                "median_partition debe retornar un diccionario con 'groups' y 'cuts'."
            )

        self.cuts = self.base_grid["cuts"]
        self.domains[0]["groups"] = self.base_grid["groups"]

        if self.verbose:
            print(f"\n[Fit] Cubos generados en dominio base: {len(self.base_grid['groups'])}")
            print(f"[Fit] Cortes aprendidos: {len(self.cuts)}")

        for i in range(1, len(self.domains)):
            domain_x = self.domains[i]["x"]

            if self.verbose:
                print(f"\n[Fit] Aplicando cortes sobre Dominio {i + 1}")

            self.domains[i]["groups"] = apply_median_cuts(
                x=domain_x, cuts=self.cuts, verbose=self.verbose
            )

            if self.verbose:
                print(f"[Fit] Cubos en Dominio {i + 1}: {len(self.domains[i]['groups'])}")

        self.stable_cubes, self.red_zones = select_stable_cubes(
            domains=self.domains,
            mean_max=self.mean_max,
            mean_min=self.mean_min,
            std_min=self.std_min,
            std_max=self.std_max,
            verbose=self.verbose,
        )
        
        all_cubes = self.stable_cubes + self.red_zones

        sorted_group_ids = sorted(
            set(cube["group_id"] for cube in all_cubes)
        )

        self.cube_id_map = {
            group_id: f"CUBE_{i:03d}"
            for i, group_id in enumerate(sorted_group_ids, start=1)
        }

        if self.verbose:
            print(f"\n[Fit] Cubos estables finales: {len(self.stable_cubes)}")
            print(f"[Fit] Zonas rojas detectadas: {len(self.red_zones)}")

        self.is_fitted = True
        return self

    def predict(self, x: DataFrame):
        """Función para generar predicciones para nuevos datos usando el modelo entrenado

        Args:
            x (pd.Dataframe): Datos de validación o nuevos datos a predecir

        Raises:
            ValueError: El modelo debe haber sido entrenado
            ValueError: Error en la cantidad de columnas de los datos de validación

        Returns:
            pd.Series: Series con todas las predicciones generadas por cada dato.
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe entrenarse con fit() antes de predecir.")

        validate_predict_data(x)

        missing = set(self.feature_names) - set(x.columns)
        if missing:
            raise ValueError(f"Faltan columnas en X para predecir: {missing}")

        X_predict = x[self.feature_names].copy()
        
        predictions = predict_from_stable_cubes(
            X=X_predict,
            stable_cubes=self.stable_cubes,
            red_zones=self.red_zones,
            cuts=self.cuts,
            use_default_value=self.use_default_value,
            default_value=self.default_value,
            verbose=self.verbose,
        )
        
        self.last_prediction_X = X_predict
        self.last_predictions = predictions
        self.last_prediction_cubes = None
        
        return predictions

    def export_checkpoint(
        self,
        X_valid,
        y_valid,
        file_name="checkpoint_robuspredictor",
        file_format="xlsx",
    ):
        """Exporta el checkpoint de trazabilidad a Excel o CSV

        Args:
            X_valid (pd.DataFrame): Dataset de validación. Si se entrega, se aplican los cortes aprendidos para asignar cada registro de validación al cubo correspondiente
            y_valid (pd.Series): Target real del dataset de validación. Se utiliza solo para trazabilidad. Requerido si se entrega X_valid
            file_name (str, optional): Nombre base del archivo de salida. Defaults to "checkpoint_robuspredictor"
            file_format (str, optional): Formato de salida. Valores permitidos: "xlsx" o "csv". Defaults to "xlsx"

        Raises:
        ValueError: Si el modelo no ha sido entrenado previamente
        ValueError: Si faltan columnas en X_valid
        ValueError: Si X_valid e y_valid no tienen la misma cantidad de registros
        ValueError: Si X_valid e y_valid no tienen el mismo índice

        Returns:
            dict[str, pd.DataFrame]: Diccionario con los DataFrames generados para el checkpoint
        """

        if not self.is_fitted:
            raise ValueError("El modelo debe entrenarse con fit() antes de exportar checkpoint.")

        validate_predict_data(X_valid)

        missing = set(self.feature_names) - set(X_valid.columns)
        if missing:
            raise ValueError(f"Faltan columnas en X_valid para aplicar cortes: {missing}")

        if len(X_valid) != len(y_valid):
            raise ValueError("X_valid e y_valid deben tener la misma cantidad de registros.")

        if not X_valid.index.equals(y_valid.index):
            raise ValueError("X_valid e y_valid deben tener el mismo índice.")

        if self.verbose:
            print("[Checkpoint] Aplicando cortes a datos de validación...")

        validacion_groups = apply_median_cuts(
            x=X_valid[self.feature_names].copy(),
            cuts=self.cuts,
            verbose=False
        )

        if self.verbose:
            print(f"[Checkpoint] Grupos de validación generados: {len(validacion_groups)}")

        self.checkpoint = export_checkpoint(
            domains=self.domains,
            stable_cubes=self.stable_cubes,
            red_zones=self.red_zones,
            cuts=self.cuts,
            feature_names=self.feature_names,
            file_name=file_name,
            file_format=file_format,
            validacion_groups=validacion_groups,
            y_validacion=y_valid,
        )

        return self.checkpoint
    
    def export_prediction_checkpoint(
        self,
        X_valid,
        y_valid,
        file_name="scoring_robuspredictor",
        dato_real=None,
        file_format="xlsx"
    ):
        """Exporta el checkpoint de predicciones a un archivo Excel o CSV

        Args:
            X_valid (pd.DataFrame): Dataset de validación o datos a predecir. Debe contener las mismas columnas utilizadas durante el entrenamiento
            y_valid (pd.Series): Data target de validacion asociado a X_valid
            file_name (str, optional): Nombre base del archivo de salida. Defaults to "scoring_robuspredictor"
            dato_real (pd.Series, optional): Variable real adicional utilizada para evaluación o comparación. No se usa para entrenar ni para predecir. Defaults to None
            file_format (str, optional): Formato de salida del archivo. Valores permitidos: "xlsx" o "csv". Defaults to "xlsx"

        Raises:
        ValueError: Si el modelo no ha sido entrenado previamente
        ValueError: Si faltan columnas en X_valid
        ValueError: Si X_valid e y_valid no tienen la misma cantidad de registros
        ValueError: Si X_valid e y_valid no tienen el mismo índice

        Returns:
            pd.DataFrame: DataFrame con el detalle de predicción por registro. Este mismo contenido es exportado al archivo generado
        """
        
        if not self.is_fitted:
            raise ValueError("El modelo debe entrenarse con fit() antes de exportar predicciones.")

        validate_predict_data(X_valid)
        
        missing = set(self.feature_names) - set(X_valid.columns)
        if missing:
            raise ValueError(f"Faltan columnas en X_valid para predecir: {missing}")
        
        X_export = X_valid[self.feature_names].copy()
        
        if len(X_export) != len(y_valid):
            raise ValueError("X_valid e y_valid deben tener la misma cantidad de registros.")
        
        if not X_export.index.equals(y_valid.index):
            raise ValueError("X_valid e y_valid deben tener el mismo índice.")
        
        prediction_checkpoint = build_and_export_prediction_checkpoint(
            X=X_export,
            y=y_valid,
            file_name=file_name,
            dato_real=dato_real,
            stable_cubes=self.stable_cubes,
            red_zones=self.red_zones,
            cuts=self.cuts,
            default_value=self.default_value,
            feature_names=self.feature_names,
            file_format=file_format
        )
        
        return prediction_checkpoint

    def best_percentage(self, y_target, top_pct=0.05):
        """Calcula la precisión del modelo dentro del Top N% de las últimas predicciones generadas.

        Args:
            y_target (pd.Series): Variable real contra la cual se evalúa la precisión del Top N%. Debe tener el mismo índice y la misma cantidad de registros que las últimas predicciones generadas por predict()
            top_pct (float, optional):  Porcentaje superior de predicciones a evaluar, expresado como fracción entre 0 y 1. Defaults to 0.05

        Raises:
            ValueError: Si el modelo no ha sido entrenado antes de ejecutar el método.
            ValueError: Si el modelo no ha sido entregado predicciones antes de ejecutar el método.

        Returns:
            float: Precisión obtenida dentro del Top N% de predicciones más altas.
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe entrenarse con fit() antes de calcular métricas.")
        
        if self.last_predictions is None:
            raise ValueError(
                "No existen predicciones previas. "
                "Debe ejecutar modelo.predict(X) antes de llamar a best_percentage()."
            )
        
        result = calculate_precision_top_percentage(
            y_pred=self.last_predictions,
            y_target=y_target,
            top_pct=top_pct
        )
        
        return result["precision"]

    def predict_cubes(self, x: DataFrame):
        """Asigna cada registro al cubo correspondiente del modelo.

        Args:
            x (pd.DataFrame): Datos a asignar a cubos. Debe contener las mismas columnas utilizadas durante el entrenamiento

        Raises:
            ValueError: Si el modelo no ha sido entrenado previamente
            ValueError: Si faltan columnas requeridas para aplicar los cortes

        Returns:
            pd.Series: Serie de pandas con el ID público del cubo asignado a cada fila. Mantiene el mismo índice de x y tiene nombre "cube_id"
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe entrenarse con fit() antes de asignar cubos.")

        validate_predict_data(x)

        missing = set(self.feature_names) - set(x.columns)
        if missing:
            raise ValueError(f"Faltan columnas en X para asignar cubos: {missing}")

        X_predict = x[self.feature_names].copy()

        cube_ids = predict_cubes_from_cuts(
            X=X_predict,
            stable_cubes=self.stable_cubes,
            red_zones=self.red_zones,
            cuts=self.cuts,
            cube_id_map=self.cube_id_map
        )
        
        self.last_prediction_cubes = cube_ids

        return cube_ids

    def export_dataframe_cubes(self):
        """Retorna un DataFrame resumen de los cubos utilizados en la última predicción
        
        Raises:
            ValueError: Si el modelo no ha sido entrenado previamente
            ValueError: Si el modelo no ha sido entregado predicciones antes de ejecutar el método
            ValueError: Si existe inconsistencia entre los índices internos guardados
        
        Returns:
            pd.DataFrame: Dataframe con informacionn respecto ID del cubo, valores minimos y maximos de cada variables y la prediccion corespondiente al cubo
        """
        if not self.is_fitted:
            raise ValueError(
                "El modelo debe entrenarse con fit() antes de generar el DataFrame de cubos."
            )

        if self.last_prediction_X is None or self.last_predictions is None:
            raise ValueError(
                "No existen predicciones previas. "
                "Debe ejecutar modelo.predict(x) antes de consultar export_dataframe_cubes."
            )

        if self.last_prediction_cubes is None:
            self.last_prediction_cubes = self.predict_cubes(self.last_prediction_X)

        if not self.last_prediction_X.index.equals(self.last_predictions.index):
            raise ValueError(
                "Inconsistencia interna: last_prediction_X y last_predictions "
                "no tienen el mismo índice."
            )

        if not self.last_prediction_X.index.equals(self.last_prediction_cubes.index):
            raise ValueError(
                "Inconsistencia interna: last_prediction_X y last_prediction_cubes "
                "no tienen el mismo índice."
            )

        df = self.last_prediction_X.copy()
        df["_cube_id"] = self.last_prediction_cubes
        df["_pred"] = self.last_predictions

        rows = []

        for cube_id, group in df.groupby("_cube_id", sort=True):
            row = {
                "ID": cube_id
            }

            for feature in self.feature_names:
                row[f"{feature}_min"] = group[feature].min()
                row[f"{feature}_max"] = group[feature].max()

            unique_predictions = group["_pred"].dropna().unique()

            if len(unique_predictions) == 0:
                row["Pred"] = None
            elif len(unique_predictions) == 1:
                row["Pred"] = unique_predictions[0]
            else:
                row["Pred"] = unique_predictions[0]

            rows.append(row)

        return DataFrame(rows)