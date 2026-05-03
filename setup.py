from setuptools import setup, find_packages

setup(
    name='robusPredictor',
    version='0.1.0',
    description='Algoritmo predictivo para Formulisa',
    author='Sebastian y Paula',
    author_email='se.valdivia@duocuc.cl',
    package_dir={"": "Producto"},
    packages=find_packages(where="Producto"),
    install_requires=[
        'numpy==1.24.4',
        'pandas==2.0.3',
        'scikit-learn==1.3.2',
        'joblib==1.4.2'],
    # external packages as dependencies
    python_requires="==3.8.10"
)
