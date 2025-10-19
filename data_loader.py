# file: data_loader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations for the images
# We resize them to 224x224, which is a standard size for many pre-trained models.
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path to your dataset root folder
DATASET_PATH = "/home/harshan/Documents/edl/biological_tuned_ai/dataset/flowers"

# Create a dataset object
image_dataset = datasets.ImageFolder(DATASET_PATH, transform=data_transforms)

# Create a data loader to feed images to the model in batches
dataloader = DataLoader(image_dataset, batch_size=32, shuffle=True)

# This gives us the class (species) names
class_names = image_dataset.classes
num_classes = len(class_names)

print(f"Found {len(image_dataset)} images in {num_classes} classes.")