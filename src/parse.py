import os
import pandas as pd

IMG_DIR = "../data/60000ImagesOfCars/"

def parse_filename(fname):
    parts = fname.replace(".jpg", "").split("_")

    year_idx = None
    for i, p in enumerate(parts):
        if p.isdigit() and len(p) == 4:
            year_idx = i
            break

    if year_idx is None:
        return None

    make = parts[0]
    model = "_".join(parts[1:year_idx])
    year = parts[year_idx]

    return make, model, year


data = []

for fname in os.listdir(IMG_DIR):
    if not fname.endswith(".jpg"):
        continue

    parsed = parse_filename(fname)
    if parsed is None:
        continue

    make, model, year = parsed

    data.append({
        "path": os.path.join(IMG_DIR, fname),
        "make": make,
        "model": model,
        "year": year
    })

df = pd.DataFrame(data)

print(df.head())
print("Total images:", len(df))