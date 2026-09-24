#Python file where ill train a model on the data we have!
#Looking at using ResNet

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import torch
from torch.utils.data import DataLoader, TensorDataset, dataloader
from PIL import Image



# 1. Define image preprocessing (ResNet standard requirements)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), # ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Load pre-trained ResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 3. Freeze earlier layers (optional, stops them from updating during training)
for param in model.parameters():
    param.requires_grad = False

# 4. Replace the final fully-connected layer for your specific task
num_classes = 2  # Example: Cats vs Dogs
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load the dataset from folder
batch_size = 32
dataset = datasets.ImageFolder(root='data/spectrograms/', transform=transform)
# 1. Instantiate the DataLoader with your dataset and parameters
train_loader = DataLoader(
    dataset=dataset,  # Replace with your actual dataset object
    batch_size=32,         # Set your desired batch size
    shuffle=True           # Shuffle data every epoch
)

# Access classes and sample targets
print(dataset.classes)  # List of class folder names
print(dataset.class_to_idx)  # Mapping of class to index integer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
for epoch in range(5):
    total_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Your training loop code here
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    total_loss /= len(train_loader)
    print(epoch,total_loss)