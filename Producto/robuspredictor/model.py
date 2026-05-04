import numpy as np
from .utils import validate_params, validate_fit_data
from .partitioning import recursive_median_partition
from .stability import select_stable_cubes


class RobusPredictor:
    # Inicio de robuspredictor donde recibe los parametros
    # Ademas, se valida el tipo de dato de cada parametro
    def __init__(self, n_min, n_max, n_dom, mean, std):
        validate_params(n_min, n_max, n_dom, mean, std)

        self.n_min = n_min
        self.n_max = n_max
        self.n_dom = n_dom
        self.mean = mean
        self.std = std
        
        # Definimos la variable interna de is_fitted, para validar que no se pueda predecir sin antes haber entrenado el modelo
        self.is_fitted = False


    # Se def el primer metodo de robuspredictor el cual entrena/construye el modelo
    # Recibe de parametros  X= variables predictoras   Y= varible objetivo (target)
    # NOTA: En esta primera iteracion solo se permitir[a introducir un conjunto de dataFrame
    #       Si el usuario desea trabajar con otro DataFrame, deberá crear una nueva instancia del modelo
    #       El soporte para m[ultiples dataframes podría quedar para una versión futura.
    def fit(self, x, y):
        validate_fit_data(x,y)
        
        self.cubes= recursive_median_partition(x, n_min=self.n_min, n_max=self.n_max)
        self.stable_cubes = select_stable_cubes(x, y, self.cubes, self.n_dom, self.mean, self.std)
                
        self.is_fitted = True
        
        return self

    def predict(self, x):
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado previamente")
        pass