import os
import torch
from torchvision import models
from PIL import Image
import torchvision.transforms as T
import pandas as pd
from collections import defaultdict
import hashlib
import pickle

# ====== CONFIGURATION ======
SAMPLE_MODE = False  # Set to True to test on 1000 images first, False to run on all 60000
IMAGES_PER_CAR_MODEL = 2  # Keep 1-2 images per car model (make + model + year)
CACHE_FILE = "/Users/semv/SJSU/CS171/CNN-for-car-MMCR/inference_cache.pkl"
IMG_DIR = "/Users/semv/SJSU/CS171/CNN-for-car-MMCR/data/60000ImagesOfCars/"
# ===========================

# Load model
print("Loading Faster R-CNN model...")
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
print("Model loaded and ready.\n")

# Transform
transform = T.Compose([T.ToTensor()])

# Function to get hash
def get_image_hash(image_path):
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# Function to check if full car and not interior shot
def is_full_car(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img)
        img_area = img.size[0] * img.size[1]
        
        with torch.no_grad():
            predictions = model([img_tensor])
        pred = predictions[0]
        
        # Find car class, COCO has car as 3
        car_indices = (pred['labels'] == 3) & (pred['scores'] > 0.5)
        if not car_indices.any():
            return False
        
        # Get the bbox with highest score
        best_idx = pred['scores'][car_indices].argmax()
        car_bbox = pred['boxes'][best_idx]
        bbox_area = (car_bbox[2] - car_bbox[0]) * (car_bbox[3] - car_bbox[1])
        
        # Moderate threshold: bbox must cover > 0.5 of image (removes zoomed-in partial views)
        if bbox_area <= 0.5 * img_area:
            return False
        
        # Check for interior shots: detect people (COCO class 1) inside the car bbox
        person_indices = (pred['labels'] == 1) & (pred['scores'] > 0.5)
        if person_indices.any():
            x1, y1, x2, y2 = car_bbox.tolist()
            for person_idx in torch.where(person_indices)[0]:
                person_bbox = pred['boxes'][person_idx]
                px1, py1, px2, py2 = person_bbox.tolist()
                # Check if person is inside or heavily overlaps with car bbox
                overlap_x = max(0, min(x2, px2) - max(x1, px1))
                overlap_y = max(0, min(y2, py2) - max(y1, py1))
                overlap_area = overlap_x * overlap_y
                person_area = (px2 - px1) * (py2 - py1)
                if overlap_area > 0.5 * person_area:  # If >50% of person overlaps with car, likely interior
                    return False
        
        return True
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
    model_name = "_".join(parts[1:year_idx])
    year = parts[year_idx]
    return make, model_name, year

# Load or create cache
def load_cache():
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached inference results from {CACHE_FILE}...")
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return None

def save_cache(data_list):
    print(f"Saving inference results to cache ({CACHE_FILE})...")
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(data_list, f)

# Main
data = []
cached_data = load_cache()

if cached_data is not None:
    print(f"Using cached data: {len(cached_data)} images\n")
    data = cached_data
else:
    print("Running inference on images...")
    
    files = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
    print(f"Found {len(files)} JPG images in {IMG_DIR}")
    
    if SAMPLE_MODE:
        files = files[:1000]
        print(f"SAMPLE MODE: Processing first 1000 images for testing\n")
    else:
        print("Processing all images...\n")
    
    for i, fname in enumerate(files, start=1):
        if i % 100 == 0 or i == len(files):
            print(f"Processing image {i}/{len(files)}")
        
        path = os.path.join(IMG_DIR, fname)
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        
        make, model_name, year = parsed
        model_key = f"{make}_{model_name}_{year}"
        h = get_image_hash(path)
        is_full = is_full_car(path)
        
        data.append({
            "path": path,
            "model_key": model_key,
            "hash": h,
            "is_full": is_full
        })
    
    # Save cache for future runs
    save_cache(data)
    print()

df = pd.DataFrame(data)
print(f"Total images processed: {len(df)}")
print(f"Good images (full car, no interior): {df['is_full'].sum()}")
print(f"Percentage good: {100 * df['is_full'].sum() / len(df):.1f}%")
print(f"Unique car models: {df['model_key'].nunique()}\n")

# For each car model, keep 1-2 unique images (by hash), preferring full cars
to_keep = []
for model_key, group in df.groupby('model_key'):
    # First, try to get full cars
    good_images = group[group['is_full'] == True].copy()
    
    if len(good_images) > 0:
        # Remove duplicates by hash, keep 1-2
        unique_good = good_images.drop_duplicates('hash')
        to_keep.extend(unique_good.head(IMAGES_PER_CAR_MODEL)['path'].tolist())
    else:
        # Fallback: if no full cars, get any images for this model
        unique_all = group.drop_duplicates('hash')
        to_keep.extend(unique_all.head(IMAGES_PER_CAR_MODEL)['path'].tolist())

to_keep = set(to_keep)
print(f"Total images to keep: {len(to_keep)}")
print(f"  ({IMAGES_PER_CAR_MODEL} per car model × {df['model_key'].nunique()} unique models)")

# Delete images not in keep list
all_paths = set(df['path'])
to_delete = all_paths - to_keep
print(f"Deleting {len(to_delete)} images...")
for i, path in enumerate(to_delete, start=1):
    if i % 500 == 0:
        print(f"  Deleted {i}/{len(to_delete)} images...")
    try:
        os.remove(path)
    except Exception as e:
        print(f"Failed to delete {path}: {e}")

print(f"\n✓ Final dataset: {len(to_keep)} images")
print(f"  • {IMAGES_PER_CAR_MODEL} image(s) per car model")
print(f"  • No duplicates (same image hash)")
print(f"  • Full cars only (no interior shots)")
if SAMPLE_MODE:
    print("\n⚠ SAMPLE MODE was ON. To process all 60000 images, set SAMPLE_MODE = False and run again.")