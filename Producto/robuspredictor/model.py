from .utils import validate_params, validate_fit_data, validate_predict_data
from .partitioning import recursive_median_partition
from .stability import select_stable_cubes
from .prediction import predict_from_stable_cubes


# Clase priuncipal de la libreria, es el modelo de robuspredictor
class RobusPredictor:
    # Inicio de robuspredictor donde recibe los parametros
    # Ademas, se valida el tipo de dato de cada parametro
    def __init__(self, n_min, n_max, n_dom,  mean_max, mean_min, std, default_value=0):
        
        # funcion que valida parametros
        validate_params(n_min, n_max, n_dom, mean_max, mean_min, std)

        self.n_min = n_min
        self.n_max = n_max
        self.n_dom = n_dom
        self.mean_max = mean_max
        self.mean_min = mean_min
        self.std = std
        self.default_value= default_value
        
        
        # Aqui se guardaran todos los cubos generados por el particionamiento.
        self.cubes = None
        # Aqui se guardaran solos los cubos que cumplen los criterios de estabilidad
        self.stable_cubes = None
        
        # inicia donde se guardan los nombres de las columnas usadas en el entrenamiento
        # despues se validara que los datos nuevos tengan las mismas variables
        self.feature_names = None
        
        # flag que indica si el modelo se entreno
        self.is_fitted = False


    # Se def el primer metodo de robuspredictor el cual entrena/construye el modelo
    # Recibe de parametros  X= variables predictoras   Y= varible objetivo (target)
    # NOTA: En esta primera iteracion solo se permitir[a introducir un conjunto de dataFrame
    #       Si el usuario desea trabajar con otro DataFrame, deberá crear una nueva instancia del modelo
    #       El soporte para m[ultiples dataframes podría quedar para una versión futura.
    def fit(self, x, y):
        
        
        # valida que x e y sean correctos, que no esten vacios y tengan misma cantidad de registros
        validate_fit_data(x,y)
        
        # rellena los nombres de las columnas usadas en el entrenamiento
        self.feature_names = list(x.columns)

        #guarda los cubos / regiones del espacio vecgorial usando la particion por medianas
        self.cubes = recursive_median_partition(x, n_min=self.n_min, n_max=self.n_max)
        
        # selecciona solo los cubos estables cantidad de dominios temporales - promedio min y max de la variable objetivo - desviación porcentual permitida
        self.stable_cubes = select_stable_cubes(x, y, self.cubes, self.n_dom, self.mean_max, self.mean_min, self.std)
                
        # marca la flag como que el modelo fue entrenado
        self.is_fitted = True
        
        # retorna el mismo objeto para el concatenamiento de los metodos
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
        )