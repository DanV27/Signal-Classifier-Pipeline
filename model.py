# Train an "about" vs "not_about" keyword classifier on mel spectrograms
# using transfer learning from a pretrained ResNet-50.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from sklearn.metrics import confusion_matrix, classification_report
import torchaudio.transforms as T
from torch.utils.data import Subset

# --- Device ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Mac GPU
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Preprocessing ---
# Resize the whole spectrogram (no crop) so the word never gets cut off.
# Normalize with ImageNet stats because ResNet was pretrained on them.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.FrequencyMasking(freq_mask_param=30),  # blank up to 30 rows (frequencies)
    T.TimeMasking(time_mask_param=40),       # blank up to 40 columns (time)
])
eval_transform = transform  # your existing clean one

train_data = datasets.ImageFolder("data/spectrograms/", transform=train_transform)
eval_data = datasets.ImageFolder("data/spectrograms/", transform=eval_transform)
# --- Data ---
# Folder names become the labels, sorted alphabetically (about/ = 0)
batch_size = 32


# Fixed seed so every run is graded on the same test clips
# Split indices once, same seed, so no clip lands in two sets
train_idx, val_idx, test_idx = torch.utils.data.random_split(
    range(len(train_data)), [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(42)
)
train_set = Subset(train_data, train_idx.indices)  # augmented
val_set = Subset(eval_data, val_idx.indices)       # clean
test_set = Subset(eval_data, test_idx.indices)     # clean

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)


# --- Model ---
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze everything, then unfreeze the last block so it can adapt to spectrograms
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True

# New 2-class head (trainable by default). Must come BEFORE .to(device).
num_classes = 2
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# --- Loss + optimizer ---
# Class weights from the real file counts, so missing a rarer "about" costs more
counts = torch.bincount(torch.tensor(train_data.targets))
class_weights = counts.max() / counts  # ~[1.8, 1.0] with 293 about / 532 not_about
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # low lr protects pretrained layer4

# --- Train with early stopping ---
num_epochs = 10
best_loss = float("inf")

for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss = 0.0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Test loss (no learning happens here)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            val_loss += criterion(model(data), targets).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f}")

    # Keep only the epoch with the lowest test loss (before memorizing kicks in)
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        print("  saved best model")

# --- Final evaluation on the BEST model, not the last epoch ---
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

all_predictions = []
all_targets = []
with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        output = model(data)  # fresh predictions every batch
        all_predictions.extend(output.argmax(dim=1).tolist())
        all_targets.extend(targets.tolist())

print(confusion_matrix(all_targets, all_predictions))
print(classification_report(all_targets, all_predictions, target_names=train_data.classes))
#293 about files
#532 not about files

