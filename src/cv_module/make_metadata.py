# src/cv_module/make_metadata.py
import csv
from pathlib import Path

RAW_ROOT = Path("data/raw/plantvillage")
OUT_META = Path("data/processed/metadata.csv")
OUT_META.parent.mkdir(parents=True, exist_ok=True)

img_exts = {".jpg", ".jpeg", ".png"}

rows = []
for img in RAW_ROOT.rglob("*"):
    if img.suffix.lower() in img_exts and img.is_file():
        # assume folder structure .../<crop>/<disease>/<image>.jpg
        parts = img.relative_to(RAW_ROOT).parts
        if len(parts) >= 2:
            crop = parts[0]
            disease = parts[1]
        elif len(parts) == 1:
            # if images are in single-level class folders like 'Tomato___Early_blight'
            folder = parts[0]
            if "___" in folder:
                crop, disease = folder.split("___", 1)
            else:
                crop = folder
                disease = "unknown"
        else:
            crop, disease = "unknown", "unknown"

        label = f"{crop.strip()}_{disease.strip()}".replace(" ", "_").lower()
        rows.append({
            "filepath": str(img),
            "crop": crop.replace(" ", "_"),
            "disease": disease.replace(" ", "_"),
            "label": label
        })

# write CSV
with open(OUT_META, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filepath","crop","disease","label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT_META}")
