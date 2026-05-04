import numpy as np
from robuspredictor import RobusPredictor

model = RobusPredictor(
    n_min=500,
    n_max=1000,
    n_dom=2,
    mean=1.5,
    std=0.5
)

print(model)