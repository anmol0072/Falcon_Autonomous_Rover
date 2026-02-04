import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
import time

# --- CONFIGURATION ---
BATCH_SIZE = 4           # We can increase this now! (Pre-resized images take less RAM)
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CUSTOM DATASET LOADER (OPTIMIZED) ---
class FalconDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # 1. Load (ALREADY RESIZED & CLEANED)
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0) 

        # 2. Safety Check
        if image is None or mask is None:
            return self.__getitem__((idx + 1) % len(self))

        # 3. Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 4. Transform
        if self.transform:
            image = self.transform(image)
        
        mask = torch.as_tensor(mask, dtype=torch.long)
        
        return image, mask

# --- MODEL SETUP ---
def get_model(num_classes=4):
    model = models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    return model.to(DEVICE)

# --- TRAINING LOOP ---
def train_model():
    print(f"🚀 Device: {DEVICE}")
    
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # POINTING TO NEW DATASET
    train_dataset = FalconDataset(
        image_dir="dataset_small/train/images", 
        mask_dir="dataset_small/train/masks", 
        transform=data_transforms
    )
    
    # Increased workers slightly because files are small now
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    print(f"📂 Optimized Images Found: {len(train_dataset)}")

    model = get_model(num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("\n🔥 STARTING COOL TRAINING... (Watch your temps drop)")
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)['out']
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            if (i+1) % 50 == 0:
                print(f"   Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{i+1}] Loss: {loss.item():.4f}")

        print(f"✅ Epoch {epoch+1} Done. Avg Loss: {epoch_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "best_model.pth")
    print("🎉 Done!")

if __name__ == "__main__":
    train_model()