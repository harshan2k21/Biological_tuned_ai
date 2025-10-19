# import os
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import torch
# import torch.nn as nn
# from torchvision import models, datasets, transforms
# import numpy as np
# from PIL import Image
# from sklearn.metrics.pairwise import cosine_similarity
# import io

# # --- 1. SETUP & CONFIG (WITH ROBUST PATHING) ---
# # Get the directory where the script is located
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # Define paths relative to the script's location
# DATASET_PATH = os.path.join(script_dir, "dataset", "flowers")
# TAXONOMY_FILE = os.path.join(script_dir, "taxonomy.json")
# MODEL_FILE = os.path.join(script_dir, "biologically_tuned_model.pth")
# FEATURES_DB_FILE = os.path.join(script_dir, "features_db.npy")
# LABELS_DB_FILE = os.path.join(script_dir, "labels_db.npy")

# print(f"Attempting to load dataset from: {DATASET_PATH}")

# # Define transformations (should be the same as during training)
# data_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Load dataset info to get class names and count
# try:
#     image_dataset = datasets.ImageFolder(DATASET_PATH, transform=data_transforms)
#     class_names = image_dataset.classes
#     num_classes = len(class_names)
#     print(f"Flask app loading. Found {num_classes} classes.")
# except FileNotFoundError:
#     # --- ENHANCED ERROR HANDLING ---
#     print(f"\n--- ERROR ---")
#     print(f"Dataset not found at the calculated path: {DATASET_PATH}")
#     print(f"\n--- DEBUGGING INFO ---")
#     try:
#         print(f"Checking contents of the project directory ('{script_dir}'):")
#         print(f"   >>> {os.listdir(script_dir)}")
        
#         dataset_dir = os.path.join(script_dir, "dataset")
#         if "dataset" in os.listdir(script_dir):
#             print(f"\nChecking contents of the 'dataset' directory ('{dataset_dir}'):")
#             print(f"   >>> {os.listdir(dataset_dir)}")
#         else:
#             print("\n>>> CRITICAL: A folder named 'dataset' was not found in your project directory.")
            
#     except Exception as e:
#         print(f"Could not list directories for debugging. Error: {e}")
    
#     print("\n--- ACTION REQUIRED ---")
#     print("Please compare the file listing above with your actual folder structure.")
#     print("Ensure the path 'dataset/Flowers' containing all your species folders exists.")
#     exit()


# # Load the taxonomy data from the JSON file
# with open(TAXONOMY_FILE, 'r') as f:
#     taxonomy_data = json.load(f)

# # --- 2. LOAD THE TRAINED MODEL & DATABASE ---
# app = Flask(__name__)
# CORS(app)

# # Load the model structure
# model = models.resnet50()
# model.fc = nn.Linear(model.fc.in_features, num_classes)

# # Load the saved weights (use map_location for CPU compatibility)
# model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))

# # Create the feature extractor by removing the last layer
# feature_extractor = nn.Sequential(*list(model.children())[:-1])
# feature_extractor.eval()

# # Load the feature database
# features_db = np.load(FEATURES_DB_FILE)
# labels_db = np.load(LABELS_DB_FILE)


# # --- 3. HELPER FUNCTIONS ---
# def get_prediction(image_bytes):
#     """Identifies an image and returns only the species name."""
#     img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     img_t = data_transforms(img)
#     batch_t = torch.unsqueeze(img_t, 0)
    
#     with torch.no_grad():
#         features = feature_extractor(batch_t)
#         features_flat = torch.flatten(features, 1)
    
#     similarities = cosine_similarity(features_flat, features_db)
#     best_match_idx = np.argmax(similarities)
    
#     predicted_species_idx = labels_db[best_match_idx]
#     predicted_species = class_names[predicted_species_idx]
    
#     return predicted_species

# def compare_taxonomies(taxa1, taxa2):
#     """Compares two taxonomies and finds the divergence point."""
#     ranks = ["domain", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
#     for rank in ranks:
#         if taxa1.get(rank) != taxa2.get(rank):
#             return f"Organisms diverge at the {rank.capitalize()} level."
#     return "Organisms are the Same Species."


# # --- 4. API ENDPOINT ---
# @app.route('/compare', methods=['POST'])
# def compare():
#     if 'file1' not in request.files or 'file2' not in request.files:
#         return jsonify({'error': 'Two files are required'}), 400
    
#     file1 = request.files['file1'].read()
#     file2 = request.files['file2'].read()
    
#     # Get predictions for both images
#     species1 = get_prediction(file1)
#     species2 = get_prediction(file2)
    
#     # Look up the taxonomy for each predicted species
#     taxa1 = taxonomy_data.get(species1, {})
#     taxa2 = taxonomy_data.get(species2, {})
    
#     # Compare the taxonomies to find the divergence point
#     divergence_point = compare_taxonomies(taxa1, taxa2)

#     # Return the results
#     return jsonify({
#         'image1_species': species1,
#         'image2_species': species2,
#         'image1_taxa': taxa1,
#         'image2_taxa': taxa2,
#         'divergence_point': divergence_point,
#     })

# # --- 5. RUN THE APP ---
# if __name__ == '__main__':
#     app.run(debug=True)
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import io

# --- 1. SETUP & CONFIG ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Note: Using 'dataset/flowers' to match the successful log output
DATASET_PATH = os.path.join(script_dir, "dataset", "flowers") 
TAXONOMY_FILE = os.path.join(script_dir, "taxonomy.json")
MODEL_FILE = os.path.join(script_dir, "biologically_tuned_model.pth")
FEATURES_DB_FILE = os.path.join(script_dir, "features_db.npy")
LABELS_DB_FILE = os.path.join(script_dir, "labels_db.npy")

print(f"Attempting to load dataset from: {DATASET_PATH}")

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    image_dataset = datasets.ImageFolder(DATASET_PATH, transform=data_transforms)
    class_names = image_dataset.classes
    num_classes = len(class_names)
    print(f"Flask app loading. Found {num_classes} classes.")
except FileNotFoundError:
    print(f"\n--- ERROR ---")
    print(f"Dataset not found at the calculated path: {DATASET_PATH}")
    print(f"\nPlease ensure the path is correct and the folder exists.")
    exit()

with open(TAXONOMY_FILE, 'r') as f:
    taxonomy_data = json.load(f)

# --- 2. LOAD THE TRAINED MODEL & DATABASE ---
app = Flask(__name__)
CORS(app)

model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval()

features_db = np.load(FEATURES_DB_FILE)
labels_db = np.load(LABELS_DB_FILE)

# --- 3. HELPER FUNCTIONS ---
def get_prediction(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_t = data_transforms(img)
    batch_t = torch.unsqueeze(img_t, 0)
    
    with torch.no_grad():
        features = feature_extractor(batch_t)
        features_flat = torch.flatten(features, 1)
    
    similarities = cosine_similarity(features_flat, features_db)
    best_match_idx = np.argmax(similarities)
    
    predicted_species_idx = labels_db[best_match_idx]
    predicted_species = class_names[predicted_species_idx]
    confidence_score = similarities[0, best_match_idx]
    
    # THIS IS THE FIX: Convert the numpy.float32 to a standard Python float
    return predicted_species, float(confidence_score)

def compare_taxonomies(taxa1, taxa2):
    ranks = ["domain", "kingdom", "phylum", "class", "order", "family", "genus", "species"]
    for i, rank in enumerate(ranks):
        if taxa1.get(rank) != taxa2.get(rank):
            return f"Organisms diverge at the {rank.capitalize()} level.", i
    return "Organisms are the Same Species.", len(ranks)

def calculate_similarity_score(divergence_index, confidence1, confidence2):
    base_scores = {0: 10, 1: 20, 2: 35, 3: 50, 4: 65, 5: 75, 6: 85, 7: 95, 8: 100}
    score = base_scores.get(divergence_index, 0)
    avg_confidence = (confidence1 + confidence2) / 2
    score = min(100, score + (avg_confidence * 5))
    return round(score, 2)

# --- 4. API ENDPOINT ---
@app.route('/compare', methods=['POST'])
def compare():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Two files are required'}), 400
    
    file1 = request.files['file1'].read()
    file2 = request.files['file2'].read()
    
    species1, confidence1 = get_prediction(file1)
    species2, confidence2 = get_prediction(file2)
    
    taxa1 = taxonomy_data.get(species1, {})
    taxa2 = taxonomy_data.get(species2, {})
    
    divergence_point, divergence_index = compare_taxonomies(taxa1, taxa2)
    similarity_score = calculate_similarity_score(divergence_index, confidence1, confidence2)

    return jsonify({
        'image1_species': species1,
        'image2_species': species2,
        'image1_taxa': taxa1,
        'image2_taxa': taxa2,
        'divergence_point': divergence_point,
        'similarity_score': similarity_score,
        'confidence_score1': round(confidence1 * 100, 2),
        'confidence_score2': round(confidence2 * 100, 2)
    })

# --- 5. RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True)
