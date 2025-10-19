# file: train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# --- 1. DATA LOADING AND PREPARATION ---
# Define transformations for the images
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# IMPORTANT: Update this path to your dataset folder
DATASET_PATH = "/home/harshan/Documents/edl/biological_tuned_ai/dataset/flowers"

# Create a dataset object from your folders
image_dataset = datasets.ImageFolder(DATASET_PATH, transform=data_transforms)

# This is the line that defines the missing variable!
num_classes = len(image_dataset.classes)
print(f"Found {len(image_dataset)} images in {num_classes} classes/species.")

# Create a data loader to feed images to the model in batches
dataloader = DataLoader(image_dataset, batch_size=32, shuffle=True)


# --- 2. MODEL FINE-TUNING ---
# Load a pre-trained ResNet-50 model
model = models.resnet50(weights='IMAGENET1K_V1')

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False

# Get the number of input features for the final layer
num_ftrs = model.fc.in_features

# Replace the final layer with a new one for our specific number of classes
# This line will now work because 'num_classes' is defined above
model.fc = nn.Linear(num_ftrs, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)


# --- 3. TRAINING LOOP ---
print("Starting model training...")
num_epochs = 10 # You can adjust the number of epochs

for epoch in range(num_epochs):
    # In a real scenario, you'd track the loss more carefully
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(image_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


# --- 4. SAVE THE TRAINED MODEL ---
torch.save(model.state_dict(), "biologically_tuned_model.pth")
print("\n🚀 Model fine-tuning complete and saved as 'biologically_tuned_model.pth'")