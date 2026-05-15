import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import os
from PIL import Image
import numpy as np

# Dataset class for loading images and masks
class BroccoliDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        
        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(np.array(mask), dtype=torch.long)
        
        return image, mask

# Data transforms
transform = transforms.Compose([
    transforms.Resize((output_size, output_size)),
    transforms.ToTensor()
])

# Load dataset
image_dir = "path_to_images/"
mask_dir = "path_to_masks/"
dataset = BroccoliDataset(image_dir, mask_dir, transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load DeepLabV3 model
model = deeplabv3_resnet50(pretrained=True)
model.classifier[4] = torch.nn.Conv2d(256, 4, kernel_size=1)  # 3 classes + background
model = model.cuda()

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for images, masks in dataloader:
        images, masks = images.cuda(), masks.cuda()
        
        # Forward pass
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        epoch_loss += loss.item()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "deeplabv3_broccoli.pth")
