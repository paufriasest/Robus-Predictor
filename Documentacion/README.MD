# Robus-Predictor

![Python](https://img.shields.io/badge/Python-3.8.10-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24.4-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0.3-150458?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.2-F7931E?logo=scikitlearn&logoColor=white)
![Version](https://img.shields.io/badge/Version-1.3.0-green)
![Status](https://img.shields.io/badge/Status-Development-yellow)

## Descripción
RobusPredictor, es una librería experimental de predicción basada en particionamiento recursivo por medianas, construcción de regiones o cubos estables y evaluación de estabilidad entre dominios de entrenamiento.

El modelo busca identificar patrones robustos en datasets con ruido, dividiendo los datos en regiones locales y validando si esas regiones mantienen un comportamiento consistente entre distintos dominios.

## Funcionamiento general
En la versión actual el modelo consta de cinco etapas principales:

- División de dominios: El dataset de entrenamiento se divide en n dominios.
- Particionamiento recursivo: En el dominio base, el modelo ordena los datos por una variable, divide por mediana y continúa recursivamente alternando las variables.
- Aplicación de cortes: Los cortes aprendidos en el dominio base se aplican al resto de los dominios.
- Evaluación de estabilidad: Cada cubo se evalúa según el promedio y desviación estándar del target en cada dominio.
- Predicción: Una nueva observación recorre el árbol de cortes y cae en un cubo final. Si el cubo es estable, se usa su valor aprendido. Si cae en zona roja, se puede usar el valor por defecto o el valor promedio de la zona roja.

## Tecnologías

- **Python** 3.8.10

## Dependencias principales

- **scikit-learn** 1.3.2
- **joblib** 1.4.2
- **numpy** 1.24.4
- **pandas** 2.0.3
- **openpyxl** (requerida para exportar checkpoints a Excel .xlsx)


# Instalación 

Se recomiendo el uso de un entorno virtual.

## Clonar repositorio

```
git clone https://github.com/paufriasest/Robus-Predictor.git
cd Robus-Predictor
```
## Crear entorno virtual

### Linux/MacOs

```
python3 -m venv venv
source venv/bin/activate
```
### Windows

```
python -m venv venv
venv\Scripts\activate
```
### Instalar dependencias
```
pip install -r requirements.txt
```
### Instalar librería localmente
```
pip install .
```

## Estructura proyecto

```
Robus-Predictor/
│
├── Documentación/
├── Gestión/
│
├── Producto/
│   └── example/
│       ├── example_mockup.py
│       ├── example_practical.py
│       └── test_particion.py
│
│   └── robuspredictor/
│       ├── __init__.py
│       ├── checkpoint.py
│       ├── domains.py
│       ├── metrics.py
│       ├── model.py
│       ├── partitioning.py
│       ├── prediction.py
│       ├── stability.py
│       └── utils.py
│
├── README.md
├── requirements.txt
└── setup.py

```
## Parámetros principales

| Parámetro           | Descripción                                                             |
| ------------------- | ------------------------------------------------------------------------|
| n_min               | Cantidad mínima de elementos permitidos por cubo                        |
| n_max               | Cantidad máxima de elementos permitidos por cubo                        |
| n_dom               | Número de dominios temporales                                           |
| mean_min            | Promedio mínimo permitido para cubos estables                           |
| mean_max            | Promedio máximo permitido para cubos estables                           |
| std_min             | Desviación mínima permitida                                             |
| std_max             | Desviación máxima permitida                                             |
| use_default_value   | Booleano que define que hacer cuando la predicción cae en una zona roja |
| default_value       | Valor utilizado cuando un registro no pertenece a ningún cubo estable   |
| verbose             | Habilita mensajes de trazabilidad del algoritmo                         |

## Ejemplo de uso 
```
import pandas as pd
from robuspredictor import RobusPredictor

ENTRENAMIENTO = pd.read_csv(../DATOS_ENTRENAMIENTO.csv)
VALIDACION = pd.read_csv(../DATOS_VALIDACION.csv)

features = [
    "var1", "var2", 
]
target = "var_target"

# Variables entrenamiento del modelo
X_train = ENTRENAMIENTO[features]
y_train = ENTRENAMIENTO[target]

# Variables validación del modelo
X_valid = VALIDACION[features]
y_valid = VALIDACION[target]

VAR_BINARIA_REAL= VALIDACION["var_binaria"]

# Modelo
modelo = RobusPredictor(
    n_min=2,
    n_max=4,
    n_dom=2,
    mean_min=1.0,
    mean_max=3.0,
    std_min=0.0,
    std_max=0.20,
    use_default_value=0,
    default_value=0,
    verbose=True
)

# Entrenamiento
modelo.fit(X_train, y_train)

# Predicción
predicciones = modelo.predict(X_valid)

# Export checkpoint datos entrenamiento
modelo.export_checkpoint(
    X_valid=X_valid,
    y_valid=y_valid,
    file_name="checkpoint_robuspredictor",
    file_format="xlsx",
)

# Export checkpoint datos validación
modelo.export_prediction_checkpoint(
    X_valid=X_valid,
    y_valid=y_valid,
    dato_real=ARRIENDO_REAL,
    file_name="scoring_robuspredictor",
    file_format="xlsx",
)

# Función para obtner el mejor N% 
resultado_top5 = modelo.best_percentage(
    y_target=VAR_BINARIA_REAL,
    top_pct=0.05
)

# Función para asignar cada registro al cubo correspondiente del modelo.
cube_ids = modelo.predict_cubes(X_valid)

# Función de retorna un dataframe con los cubos de la predicción, en conjunto sus valores minimos y maximos por variables
cubes_df = modelo.export_dataframe_cubes()

# Función que retorna la grilla utilziada en entrenamiento
cube_grid = modelo.export_cubes_grid()

```

Para mayor información de uso consultar RobusPredictor.md dentro de la carpeta de Documentación.
## Versionamiento
Versión actual: 
```
v1.3.0
```

## Autores
- Sebastián Valdivia
- Paula Frías
```
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⠶⠶⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⣀⣀⣿⠁⠀⠀⢹⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢀⣀⠀⠀⠀⠀⣀⣤⠴⠖⠛⠛⠋⠉⠉⠉⠙⠋⠀⠀⠀⠘⠁⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⡴⠛⠉⠛⣦⣴⠟⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠙⠛⠳⣦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠸⡇⠀⠀⠀⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣹⡦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣸⠟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⢀⡾⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠰⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⡆⠀⠀⠀⠀⠀⠀⠀⠀
⠀⢠⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠐⠛⠳⠆⠀⠀⠀⠀⠀⠀⠀⢠⣾⠛⢳⣆⠀⠀⠄⠐⡀⣄⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀
⠀⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣄⡀⠀⠀⠀⠀⠀⠀⠸⣿⣷⣻⡟⠈⡆⣸⢸⡇⠃⣁⡀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀
⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣁⣼⣿⠀⠀⠀⠀⠀⠀⠀⠀⠉⠁⠘⡀⠁⢈⣤⡴⠋⠉⠉⠙⢦⣀⣿⠀⠀⠀⠀⠀⠀⠀⠀
⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠠⠄⠀⠻⠷⠾⠋⠀⠀⡀⠀⣶⣀⡶⠀⠀⠀⠀⢀⣰⠏⠀⣠⡴⠶⠶⣦⡈⢿⣿⠳⣆⠀⠀⠀⠀⠀⠀
⠘⣇⠀⠀⠀⠀⠀⠀⡐⡇⣴⢠⡆⡖⠀⠀⠀⣀⡀⠟⠟⢋⢉⡀⠀⠀⠀⢀⡿⠁⢀⡾⠋⠀⠙⢳⣄⠙⢷⡈⠳⠿⢤⣤⡀⠀⠀⠀
⠀⢿⡄⠀⠀⠀⠀⠀⠻⠇⠙⠈⠁⠠⠊⠀⠘⠤⠗⠀⠀⠈⠉⣠⠤⠶⠛⠉⠀⣠⡞⠀⠀⠀⠀⠀⠙⣆⠈⢷⡀⠠⠀⠀⠙⣦⡀⠀
⠀⠘⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡁⠀⣴⠟⠋⠉⠉⠀⠀⠀⠀⠀⠀⠀⢸⡆⠘⠷⢦⣤⡈⠂⠹⣧⡀
⠀⠀⠈⢿⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣧⠀⢿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠓⠒⠳⣦⢹⡆⠀⠈⣷
⠀⠀⠀⠀⠙⢷⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣤⠶⠷⣦⠈⢷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⢏⣼⠃⠀⢠⡟
⠀⠀⠀⠀⠀⠀⠀⠉⢫⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⡟⠀⢨⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡶⠋⣡⡟⠁⠀⢀⣼⠁
⠀⠀⠀⠀⠀⠀⠀⠀⠈⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢸⡇⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⣿⠀⠀⣴⠋⠁⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠸⣆⠀⠀⠀⢀⣠⡤⢦⣀⠀⠀⢠⡿⠀⡟⠀⢠⡏⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠘⣧⠀⢹⣧⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢷⣄⠀⠉⠛⠛⢛⠏⢁⣀⠀⠙⠳⢦⣤⣤⠞⠁⠀⣸⠇⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠀⠙⠳⠦⣤⡄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠒⢒⡖⠚⠚⠋⠉⠛⢦⣤⣀⣀⣀⣀⣤⠾⠋⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢿⡀⠀⢸⡗⠒⠶⠶⠒⢶⡆⢀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⠒⠛⠁⠀⠀⠀⠀⠈⠛⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀

```


