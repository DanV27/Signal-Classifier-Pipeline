#Python file where ill train a model on the data we have!
#Looking at using ResNet

import torch
import torch.nn as nn
from torchvision import models, transforms
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