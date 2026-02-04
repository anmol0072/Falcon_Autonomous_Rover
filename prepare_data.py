import cv2
import os
import numpy as np
from tqdm import tqdm  # You might need to install this: pip install tqdm

# --- CONFIGURATION ---
INPUT_DIR = "dataset/train"
OUTPUT_DIR = "dataset_small/train"
TARGET_SIZE = (320, 320)

def process_data():
    # 1. Create New Folders
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True)

    img_dir = os.path.join(INPUT_DIR, "images")
    mask_dir = os.path.join(INPUT_DIR, "masks")
    files = os.listdir(img_dir)

    print(f"📦 Processing {len(files)} images... This happens only ONCE.")

    for f in tqdm(files):
        # Paths
        img_path = os.path.join(img_dir, f)
        mask_path = os.path.join(mask_dir, f) # Assumes PNG name matches

        # Read
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0) # Load as grayscale

        if image is None or mask is None:
            continue

        # --- RESIZE & CLEAN ---
        # 1. Resize Image (Linear looks best for photos)
        image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # 2. Resize Mask (NEAREST is mandatory to keep classes 0,1,2,3)
        mask = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

        # 3. Clean Garbage Values (Fix the bug permanently)
        mask[mask > 3] = 0

        # --- SAVE ---
        cv2.imwrite(os.path.join(OUTPUT_DIR, "images", f), image)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "masks", f), mask)

    print(f"\n✅ Done! Optimized data saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_data()