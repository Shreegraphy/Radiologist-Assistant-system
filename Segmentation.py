import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import tifffile as tiff
import matplotlib.pyplot as plt

def bbox_txt_to_mask(label_path, img_shape):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            _, cx, cy, bw, bh = map(float, parts)
            x_min = int((cx - bw / 2) * w)
            y_min = int((cy - bh / 2) * h)
            x_max = int((cx + bw / 2) * w)
            y_max = int((cy + bh / 2) * h)
        elif len(parts) == 4:
            x_min, y_min, x_max, y_max = map(int, parts)
        else:
            continue
        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)
        mask[y_min:y_max, x_min:x_max] = 1
    return mask

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        curr_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_channels, feature))
            curr_channels = feature
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
        return self.final_conv(x)

class SegmentationDatasetTIF(Dataset):
    def __init__(self, image_dir, label_dir, target_size=(256, 256)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.target_size = target_size
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.tif') or f.endswith('.txt')]
        self.label_map = {}
        for label_file in label_files:
            base_name = label_file.replace('_mask.tif', '').replace('.txt', '')
            self.label_map[base_name] = label_file
        self.image_files = [img for img in self.image_files if img.replace('.tif', '') in self.label_map]
        if len(self.image_files) == 0:
            raise ValueError("No matching images and labels found!")
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = tiff.imread(image_path)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, self.target_size).astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)
        base_name = image_filename.replace('.tif', '')
        label_filename = self.label_map[base_name]
        label_path = os.path.join(self.label_dir, label_filename)
        if label_filename.endswith('.txt'):
            mask = bbox_txt_to_mask(label_path, image.shape[1:])
        else:
            mask = tiff.imread(label_path)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0
        return image, mask

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

train_image_dir = "/content/drive/MyDrive/TCGA_CS_4941_19960909/normal"
train_label_dir = "/content/drive/MyDrive/TCGA_CS_4941_19960909/mask"
train_dataset = SegmentationDatasetTIF(train_image_dir, train_label_dir)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
model = UNet(in_channels=3, out_channels=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_unet(model, train_loader, criterion, optimizer, device, num_epochs=10)
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F

def predict_and_show_sample(dataset, model, device):
    model.eval()
    idx = random.randint(0, len(dataset) - 1)
    image, true_mask = dataset[idx]
    image_input = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_input)
        output = torch.sigmoid(output)
        pred_mask = (output > 0.5).float()
    
    image_np = image.permute(1, 2, 0).cpu().numpy()
    true_mask_np = true_mask.squeeze().cpu().numpy()
    pred_mask_np = pred_mask.squeeze().cpu().numpy()
    
    image_filename = dataset.image_files[idx]
    print(f"Sample Index: {idx}, Filename: {image_filename}")
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(true_mask_np, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask_np, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
    
    plt.show()

predict_and_show_sample(train_dataset, model, device)
