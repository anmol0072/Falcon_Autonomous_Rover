import cv2
import numpy as np
import os

# Point this to your actual folder
mask_dir = "dataset/train/masks"

# Get the first 5 files
files = os.listdir(mask_dir)[:5]

print("--- MASK INSPECTION ---")
for f in files:
    path = os.path.join(mask_dir, f)
    # Read as grayscale (how the AI reads it)
    mask = cv2.imread(path, 0) 
    
    if mask is None:
        print(f"❌ Error reading {f}")
        continue
        
    unique_values = np.unique(mask)
    print(f"File: {f}")
    print(f"  > Unique Values found: {unique_values}")
    
    if np.max(unique_values) > 3:
        print("  ⚠️ PROBLEM: Values > 3 found. This will crash the model.")
    else:
        print("  ✅ OK: Values are safe.")
    print("-" * 30)