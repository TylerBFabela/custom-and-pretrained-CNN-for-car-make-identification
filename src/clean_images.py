import os
import torch
from torchvision import models
from PIL import Image
import torchvision.transforms as T
import pandas as pd
from collections import defaultdict
import hashlib

# Load model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Transform
transform = T.Compose([T.ToTensor()])

# Function to get hash
def get_image_hash(image_path):
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# Function to check if full car
def is_full_car(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img)
        with torch.no_grad():
            predictions = model([img_tensor])
        pred = predictions[0]
        # Find car class, COCO has car as 3
        car_indices = (pred['labels'] == 3) & (pred['scores'] > 0.5)
        if not car_indices.any():
            return False
        # Get the bbox with highest score
        best_idx = pred['scores'][car_indices].argmax()
        bbox = pred['boxes'][best_idx]
        # Area of bbox
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        img_area = img.size[0] * img.size[1]
        # If bbox area > 0.6 * img_area, consider full car
        return bbox_area > 0.6 * img_area
    except:
        return False

# Parse function
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

# Main
IMG_DIR = "/Users/semv/SJSU/CS171/CNN-for-car-MMCR/data/60000ImagesOfCars/"

data = []
hashes = defaultdict(list)

# Limit to first 100 for testing
files = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")][:100]

for fname in files:
    path = os.path.join(IMG_DIR, fname)
    parsed = parse_filename(fname)
    if parsed is None:
        continue
    make, model_name, year = parsed
    model_key = f"{make}_{model_name}_{year}"
    h = get_image_hash(path)
    hashes[h].append(path)
    is_full = is_full_car(path)
    data.append({
        "path": path,
        "model": model_key,
        "hash": h,
        "is_full": is_full
    })

df = pd.DataFrame(data)

# Now, for each model, collect good images
to_keep = []
for model_key, group in df.groupby('model'):
    # Get unique hashes, prefer full cars
    good_images = group[group['is_full'] == True]
    if len(good_images) >= 2:
        # Take 2 with different hashes if possible
        unique_hashes = good_images.drop_duplicates('hash')
        if len(unique_hashes) >= 2:
            to_keep.extend(unique_hashes.head(2)['path'].tolist())
        else:
            to_keep.extend(good_images.head(2)['path'].tolist())
    elif len(good_images) == 1:
        to_keep.append(good_images.iloc[0]['path'])
        # Find another not full but different hash
        others = group[group['is_full'] == False]
        unique_others = others.drop_duplicates('hash')
        if not unique_others.empty:
            to_keep.append(unique_others.iloc[0]['path'])
    else:
        # No full, take 2 different hashes
        unique_group = group.drop_duplicates('hash')
        to_keep.extend(unique_group.head(2)['path'].tolist())

# Now, delete the rest
all_paths = set(df['path'])
to_delete = all_paths - set(to_keep)
for path in to_delete:
    os.remove(path)

print(f"Kept {len(to_keep)} images, deleted {len(to_delete)}")