#Python file where ill train a model on the data we have!
#Looking at using ResNet
import torch.nn as nn
from torchvision import models, transforms, datasets
import torch
from torch.utils.data import DataLoader

# Define image preprocessing (ResNet standard requirements)
transform = transforms.Compose([
    transforms.Resize((224, 224)),# ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load pre-trained ResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze body layers
for param in model.parameters():
    param.requires_grad = False

#a new 2-class head is the only part that trains
num_classes = 2
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Load the dataset from folder
batch_size = 32
dataset = datasets.ImageFolder(root='data/spectrograms/', transform=transform)
#split data
train_set, test_set = torch.utils.data.random_split(dataset, [.8,.2])
# Instantiate the DataLoader with your dataset and parameters
train_loader = DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True           # Shuffle data every epoch
)
test_loader = DataLoader(
    dataset=test_set,
    batch_size=batch_size
)
#the grader takes the 2 logits and converts them to percentages that
#add up to 100%, then checks how much went to the correct answer.
criterion = nn.CrossEntropyLoss()
#lr=learning rate, Too big and it overshoots, too small and it takes forever.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train() #training mode
for epoch in range(5): #epoch = one full pas through all training images
    total_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader): #grab batch of 32

        optimizer.zero_grad()
        output = model(data) #shape [32, 2] shape/logits
        loss = criterion(output, targets) #grade guess
        total_loss += loss.item()
        loss.backward() #calculates how to get better
        optimizer.step() #changes that
    total_loss /= len(train_loader)
    print(epoch,total_loss)

model.eval() #evaluation mode
correct = 0
total = 0
with torch.no_grad(): #cause it'll still be recording every calculation
    for batch_idx, (data, targets) in enumerate(test_loader):
        output = model(data)
        correct += (torch.argmax(output, dim=1) == targets).sum().item()
        total += targets.size(0)

# Calculate accuracy percentage
batch_accuracy = correct / total
print(batch_accuracy)