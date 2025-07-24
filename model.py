#############################################
# 1. Import Required Libraries
#############################################
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import tifffile as tiff  # For reading .tif images
import matplotlib.pyplot as plt
from torchvision import models

#############################################
# 2. Convert Bounding Box Text to Mask
#############################################
def bbox_txt_to_mask(label_path, img_shape):
    """
    Converts a bounding box text file to a binary segmentation mask.
    Supports:
    - YOLO format (class, x_center, y_center, width, height) - normalized
    - Absolute format (x_min, y_min, x_max, y_max)

    Returns a mask (numpy array) of shape (height, width) with 1 inside bounding boxes.
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:  # YOLO format
            _, cx, cy, bw, bh = map(float, parts)
            x_min = int((cx - bw / 2) * w)
            y_min = int((cy - bh / 2) * h)
            x_max = int((cx + bw / 2) * w)
            y_max = int((cy + bh / 2) * h)
        elif len(parts) == 4:  # Absolute format
            x_min, y_min, x_max, y_max = map(int, parts)
        else:
            continue  # Skip invalid lines

        # Ensure coordinates are within bounds
        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)

        mask[y_min:y_max, x_min:x_max] = 1
    return mask

#############################################
# 3. Define U-Net Architecture
#############################################

class ConvBlock(nn.Module):
    """Two conv layers with batch norm and ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResUNet50(nn.Module):
    def __init__(self, out_channels=1, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)

        # Extract encoder layers from ResNet-50
        self.input_layer = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu
        )  # -> 64 channels

        self.encoder1 = resnet.layer1  # -> 256 channels
        self.encoder2 = resnet.layer2  # -> 512 channels
        self.encoder3 = resnet.layer3  # -> 1024 channels
        self.encoder4 = resnet.layer4  # -> 2048 channels

        # Bottleneck
        self.bottleneck = ConvBlock(2048, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024 + 1024, 1024)

        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512 + 512, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256 + 256, 256)

        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64 + 64, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.input_layer(x)         # [B, 64, H/2, W/2]
        x2 = self.encoder1(x1)           # [B, 256, H/4, W/4]
        x3 = self.encoder2(x2)           # [B, 512, H/8, W/8]
        x4 = self.encoder3(x3)           # [B, 1024, H/16, W/16]
        x5 = self.encoder4(x4)           # [B, 2048, H/32, W/32]

        bottleneck = self.bottleneck(x5)

        d4 = self.up4(bottleneck)
        d4 = F.interpolate(d4, size=x4.shape[2:])  # ensure same size
        d4 = self.dec4(torch.cat([d4, x4], dim=1))

        d3 = self.up3(d4)
        d3 = F.interpolate(d3, size=x3.shape[2:])
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=x2.shape[2:])
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=x1.shape[2:])
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        out = self.final_conv(d1)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)  # <- add this
        return out


#############################################
# 4. Define Custom Dataset for TIFF Files
#############################################
class SegmentationDatasetTIF(Dataset):
    def __init__(self, image_dir, label_dir, target_size=(256, 256)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.target_size = target_size

        # Get all image files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])

        # Build a dictionary to match masks with images
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.tif') or f.endswith('.txt')]
        self.label_map = {}

        # Create a mapping from image base name to mask filename
        for label_file in label_files:
            base_name = label_file.replace('_mask.tif', '').replace('.txt', '')
            self.label_map[base_name] = label_file

        # Filter image files that have corresponding labels
        self.image_files = [img for img in self.image_files if img.replace('.tif', '') in self.label_map]

        if len(self.image_files) == 0:
            raise ValueError("No matching images and labels found!")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = tiff.imread(image_path)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize image
        image = cv2.resize(image, self.target_size).astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)

        # Get corresponding label file
        base_name = image_filename.replace('.tif', '')
        label_filename = self.label_map[base_name]
        label_path = os.path.join(self.label_dir, label_filename)

        # Load mask
        if label_filename.endswith('.txt'):
            mask = bbox_txt_to_mask(label_path, image.shape[1:])
        else:
            mask = tiff.imread(label_path)

        # Resize and normalize mask
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0

        return image, mask


#############################################
# 5. Define Training Loop
#############################################
def train_unet(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

#############################################
# 6. Train the Model
#############################################
if __name__ == "__main__":
    train_image_dir = r"D:\projects\radiologist assistant system\TCGA_CS_4941_19960909-20250715T091155Z-1-001\TCGA_CS_4941_19960909\normal"
    train_label_dir = r"D:\projects\radiologist assistant system\TCGA_CS_4941_19960909-20250715T091155Z-1-001\TCGA_CS_4941_19960909\mask"

    train_dataset = SegmentationDatasetTIF(train_image_dir, train_label_dir)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    model = ResUNet50(out_channels=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_unet(model, train_loader, criterion, optimizer, device, num_epochs=10)

    torch.save(model.state_dict(), "resunet50_brain_segmentation.pth")
    print("✅ Model saved after training.")
