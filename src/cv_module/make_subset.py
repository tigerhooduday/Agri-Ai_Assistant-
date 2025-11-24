# src/cv_module/make_subset.py
import pandas as pd
from pathlib import Path
OUT = Path("data/processed/subset")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/processed/metadata.csv")
# Replace this with desired crop/disease choices (strings must match CSV entries)
chosen = [
    ("Tomato","Early_blight"),
    ("Tomato","Late_blight"),
    ("Tomato","Leaf_Mold"),
    ("Potato","Early_blight"),
    ("Potato","Late_blight"),
    ("Potato","Healthy"),
    ("Apple","Apple_scab"),
    ("Apple","Black_rot"),
    ("Apple","Cedar_apple_rust"),
    ("Corn","Common_rust"),
    ("Corn","Northern_Leaf_Blight"),
    ("Corn","Gray_leaf_spot"),
    ("Grape","Black_rot"),
    ("Grape","Esca"),
    ("Grape","Healthy")
]

labels = [f"{c}_{d}".replace(" ", "_").lower() for c,d in chosen]
subset = df[df['label'].isin(labels)]
subset.to_csv(OUT / "metadata_subset.csv", index=False)
print("Subset size:", len(subset))
