#Python file where ill train a model on the data we have!
#Looking at using ResNet
import torch.nn as nn
from torchvision import models, transforms, datasets
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# Define image preprocessing (ResNet standard requirements)
transform = transforms.Compose([
    transforms.Resize((224, 224)),# ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")   # Mac GPU
else:
    device = torch.device("cpu")
print(device)

# Load pre-trained ResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# param.requires_grad = False are frozen body weights
# True is unfrozen
for param in model.parameters():
    param.requires_grad = False          # freeze everything
for param in model.layer4.parameters():
    param.requires_grad = True           # unfreeze last block only

#a new 2-class head is the only part that trains
num_classes = 2
model.fc = nn.Linear(model.fc.in_features, num_classes)
#move to GPU
model = model.to(device)
# Load the dataset from folder
batch_size = 32
dataset = datasets.ImageFolder(root='data/spectrograms/', transform=transform)
#split data
train_set, test_set = torch.utils.data.random_split(dataset, [.8,.2], generator=torch.Generator().manual_seed(42))
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
#Criterion is now dynamic, wights are determined on ratio on about vs not about files
counts = torch.bincount(torch.tensor(dataset.targets))
class_weights = counts.max() / counts   # ≈ [4.6, 1.0] for you
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
#lr=learning rate, Too big and it overshoots, too small and it takes forever.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
model = model.to(device)
best_loss = float('inf')   # set ONCE, before training

for epoch in range(10):
    # train
    model.train()
    total_loss = 0.0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(train_loader)

    # test
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            test_loss += criterion(model(data), targets).item()
    test_loss /= len(test_loader)

    print(epoch, total_loss, test_loss)

    # save only on a new test-loss low
    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), 'best_model.pt')
        print("  saved")

#293 about files
#532 not about files
