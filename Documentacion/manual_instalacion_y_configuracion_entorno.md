# Manual de Instalación de Entorno — Robus-Predictor

> **Versión del proyecto:** v0.2.0  
> **Fecha:** Mayo 2026  
> **Autores:** Sebastián Valdivia · Paula Frías

---

## 1. Requisitos previos

| Requisito        | Versión                 | Notas                                  |
|------------------|-------------------------|----------------------------------------|
| Sistema operativo| Linux / macOS / Windows | Probado en Ubuntu  24.04.4 LTS         |
| Python           | **3.8.10**              | Gestionado con Conda                   |
| Conda            | 24.11.3                 |                                        |
| Git              | 2.43.0                  | Para clonar el repositorio             |

> **Nota:** El proyecto requiere exactamente Python **3.8.10**. Versiones superiores o inferiores pueden generar incompatibilidades con las dependencias fijadas.

---

## 2. Clonar el repositorio

```bash
git clone https://github.com/paufriasest/Robus-Predictor.git
cd Robus-Predictor
```

---

## 3. Crear y activar el entorno Conda

Se utiliza **Conda** como gestor de entornos virtuales para aislar las dependencias del proyecto.

### 3.1 Crear el entorno con Python 3.8.10

```bash
conda create -n robus-predictor python=3.8.10
```

### 3.2 Activar el entorno

```bash
conda activate robus-predictor
```

> Para desactivar el entorno cuando no esté en uso:
> ```bash
> conda deactivate
> ```

---

## 4. Instalación de dependencias

### 4.1 Dependencias del núcleo de la librería

Estas son las bibliotecas requeridas para el funcionamiento de **RobusPredictor** (definidas en `requirements.txt` y `setup.py`):

```bash
pip install -r requirements.txt
```

| Biblioteca     | Versión  | Rol en el proyecto                              |
|----------------|----------|-------------------------------------------------|
| `numpy`        | 1.24.4   | Operaciones numéricas y manejo de arreglos      |
| `pandas`       | 2.0.3    | Carga y manipulación de datos tabulares (CSV)   |
| `scikit-learn` | 1.3.2    | Modelos ML, métricas de evaluación y pipelines  |
| `joblib`       | 1.4.2    | Serialización de modelos y paralelismo          |

### 4.2 Instalar la librería localmente (modo desarrollo)

```bash
pip install .
```

Esto instala el paquete `robuspredictor` (v0.2.0) en el entorno activo, permitiendo importarlo desde cualquier script:

```python
from robuspredictor import RobusPredictor
```

---

## 5. Dependencias adicionales para el módulo de Benchmark

Los scripts ubicados en `Producto/test/Benchmark/` comparan RobusPredictor contra modelos alternativos. Cada uno requiere bibliotecas adicionales que **no están en `requirements.txt`** y deben instalarse manualmente según el benchmark que se desee ejecutar.

### 5.1 Resumen de dependencias por benchmark

| Script                          | Modelo evaluado        | Biblioteca adicional requerida    |
|---------------------------------|------------------------|------------------------------------|
| `benchmark_linear_regresion.py` | Regresión Lineal       | *(solo scikit-learn)*              |
| `benchmark_rf.py`               | Random Forest          | *(solo scikit-learn)*              |
| `benchmark_extra_trees.py`      | Extra Trees            | *(solo scikit-learn)*              |
| `benchmark_neural_network.py`   | Red Neuronal MLP       | *(solo scikit-learn)*              |
| `benchmark_xgboost.py`          | XGBoost                | `xgboost`                         |
| `benchmark_lightgbm.py`         | LightGBM               | `lightgbm`                        |
| `benchmark_catboost.py`         | CatBoost               | `catboost`                        |
| `benchmark_prophet.py`          | Facebook Prophet       | `prophet`                         |
| `benchmark_sarimax.py`          | SARIMAX (ARIMA+exog)   | `statsmodels`                     |
| `benchmark_lstm.py`             | LSTM (Deep Learning)   | `tensorflow`                      |
| `benchmark_gru.py`              | GRU (Deep Learning)    | `tensorflow`                      |

### 5.2 Instalación de dependencias de benchmark

#### Modelos basados en árboles de decisión (gradient boosting)

```bash
pip install xgboost
pip install lightgbm
pip install catboost
```

#### Modelos de series temporales

```bash
pip install prophet
pip install statsmodels
```

> **Nota Prophet:** puede requerir `pystan` como dependencia. En caso de error, instalar primero:
> ```bash
> pip install pystan==2.19.1.1
> pip install prophet
> ```

#### Modelos de Deep Learning (LSTM / GRU)

```bash
pip install tensorflow
```

> **Compatibilidad TensorFlow con Python 3.8:** Se recomienda `tensorflow==2.12.0` o `tensorflow==2.13.0`, las últimas versiones compatibles con Python 3.8.
>
> ```bash
> pip install tensorflow==2.12.0
> ```

---

## 6. Verificación del entorno

Una vez instaladas las dependencias, se puede verificar el entorno ejecutando:

```bash
python -c "
import numpy as np
import pandas as pd
import sklearn
import joblib
from robuspredictor import RobusPredictor
print('numpy:', np.__version__)
print('pandas:', pd.__version__)
print('scikit-learn:', sklearn.__version__)
print('joblib:', joblib.__version__)
print('RobusPredictor: OK')
"
```

**Salida esperada:**
```
numpy: 1.24.4
pandas: 2.0.3
scikit-learn: 1.3.2
joblib: 1.4.2
RobusPredictor: OK
```

---

## 7. Estructura del proyecto (referencia)

```
Robus-Predictor/
│
├── requirements.txt              # Dependencias del núcleo
├── setup.py                      # Configuración de instalación del paquete
├── README.md
│
├── Producto/
│   ├── robuspredictor/           # Paquete principal (v0.2.0)
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── partitioning.py
│   │   ├── stability.py
│   │   ├── prediction.py
│   │   ├── domains.py
│   │   └── utils.py
│   │
│   ├── example/
│   │   └── example_basic.py
│   │
│   └── test/
│       ├── robus_predictor_010.py
│       ├── robus_predictor_020.py
│       └── Benchmark/
│           ├── benchmark_rf.py
│           ├── benchmark_linear_regresion.py
│           ├── benchmark_extra_trees.py
│           ├── benchmark_neural_network.py
│           ├── benchmark_xgboost.py
│           ├── benchmark_lightgbm.py
│           ├── benchmark_catboost.py
│           ├── benchmark_prophet.py
│           ├── benchmark_sarimax.py
│           ├── benchmark_lstm.py
│           ├── benchmark_gru.py
│           └── benchmark_result/  # Salida de resultados CSV
│
├── Documentacion/
└── Gestion/
```

---

## 8. Datos de entrada (NoSeSube)

Los benchmarks requieren archivos de datos que **no se suben al repositorio** por razones de tamaño de la data y limitaciones de GitHub. Deben ubicarse en:

```
Robus-Predictor/
└── NoSeSube/
    └── Data/
        ├── DATOS_ENTRENAMIENTO.csv
        └── DATOS_VALIDACION.csv
```

Las columnas esperadas son:
- **Variables predictoras:** `var1`, `var2`, ..., `var13` (13 variables numéricas)
- **Variable objetivo:** `INTENSIDAD_4H`
- **Variable de negocio:** `ARRIENDO`

---

## 9. Resumen rápido (Quick Start)

```bash
# 1. Clonar
git clone https://github.com/paufriasest/Robus-Predictor.git
cd Robus-Predictor

# 2. Crear y activar entorno Conda con Python 3.8.10
conda create -n robus-predictor python=3.8.10
conda activate robus-predictor

# 3. Instalar dependencias del núcleo
pip install -r requirements.txt

# 4. Instalar librería localmente
pip install .

# 5. (Opcional) Instalar dependencias de benchmark completo
pip install xgboost lightgbm catboost prophet statsmodels tensorflow==2.12.0

# 6. Ejecutar un benchmark
python Producto/test/Benchmark/benchmark_rf.py
```

---
