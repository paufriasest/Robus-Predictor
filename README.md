# Robus-Predictor

## Descripción
RobusPredictor, es una libreria en Ptyhon orientada al desarrollo de modelos predictivos, diseñada para identificar patrones consistentes en conjuntos de datos numéricos caracterizados por alta variabilidad, ruido y presencia de valores atípicos.

## Tecnologías

- **Python** 3.8.10

## Dependencias principales

- **scikit-learn** 1.3.2
- **joblib** 1.4.2
- **numpy** 1.24.4
- **pandas** 2.0.3


# Instalación 

Se recomiendo el uso de un entorno virtual.

## Clonar repositorio

```
git clone <repository_url>
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
│   └── robuspredictor/
│       ├── __init__.py
│       ├── model.py
│       ├── partitioning.py
│       ├── stability.py
│       ├── prediction.py
│       └── utils.py
│ 
│   └── example/
│       └── example_basic.py
│
│    └──test/
│       ├── Benchmark/
│       ├── robus_predictor_010.py
│       └── robus_predictor_020.py
│
├── Benchmark/
│
├── requirements.txt
├── setup.py
└── README.md

```
## Parámetros principales

| Parámetro        | Descripción                                                           |
| ---------------- | --------------------------------------------------------------------- |
| element_cube_min | Cantidad mínima de elementos permitidos por cubo                      |
| element_cube_max | Cantidad máxima de elementos permitidos por cubo                      |
| n_domain         | Número de dominios temporales                                         |
| mean_cube_min    | Promedio mínimo permitido para cubos estables                         |
| mean_cube_max    | Promedio máximo permitido para cubos estables                         |
| desv_cube_min    | Desviación máxima permitida entre dominios                            |
| default_value    | Valor utilizado cuando un registro no pertenece a ningún cubo estable |
| verbose          | Habilita mensajes de trazabilidad del algoritmo                       |

## Ejemplo de uso 
```
import pandas as pd
from robuspredictor import RobusPredictor

# Dominio 1
X1 = pd.DataFrame({
    "var1": [10, 11, 12, 50, 51, 52],
    "var2": [20, 21, 22, 80, 81, 82],
    "var3": [30, 31, 32, 90, 91, 92],
})

y1 = pd.Series([1.5, 1.6, 1.55, 2.5, 2.6, 2.55])

# Dominio 2
X2 = pd.DataFrame({
    "var1": [10.5, 11.5, 12.5, 50.5, 51.5, 52.5],
    "var2": [20.5, 21.5, 22.5, 80.5, 81.5, 82.5],
    "var3": [30.5, 31.5, 32.5, 90.5, 91.5, 92.5],
})

y2 = pd.Series([1.55, 1.65, 1.60, 2.55, 2.65, 2.60])

# Modelo
modelo = RobusPredictor(
    element_cube_min=2,
    element_cube_max=4,
    n_domain=2,
    mean_cube_min=1.0,
    mean_cube_max=3.0,
    desv_cube_min=0.20,
    default_value=0,
    verbose=True,
)

# Entrenamiento
modelo.fit(X1, y1, X2, y2)

# Datos de validación
X_new = pd.DataFrame({
    "var1": [11, 51, 100],
    "var2": [21, 81, 100],
    "var3": [31, 91, 100],
})

# Predicción
predicciones = modelo.predict(X_new)

print(predicciones)

```
## Versionamiento
Versión actual: 
```
v0.2.0
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


