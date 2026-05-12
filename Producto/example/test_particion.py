import pandas as pd
from robuspredictor.partitioning import median_partition

df = pd.DataFrame({
    "var1": [
        52, 10, 13, 51, 11, 50, 12, 53,
        15, 55, 16, 56, 14, 54, 17, 57
    ],
    "var2": [
        82, 20, 23, 81, 21, 80, 22, 83,
        25, 85, 26, 86, 24, 84, 27, 87
    ],
    "var3": [
        92, 30, 33, 91, 31, 90, 32, 93,
        35, 95, 36, 96, 34, 94, 37, 97
    ],
})

resultado = median_partition(
    x=df,
    n_min=2,
    n_max=2,
    verbose=True
)

print("\nCORTES:")
for cut in resultado["cuts"]:
    print(cut)

print("\nGRUPOS:")
for group_id, group in resultado["groups"].items():
    print(f"\nGrupo {group_id}")
    print(group)