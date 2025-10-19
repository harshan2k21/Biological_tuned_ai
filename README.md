# Biological Taxonomic Comparison Tool

A deep learning-powered web application that enables users to upload two images of organisms (specifically flowers, in this implementation) and receive a detailed taxonomic comparison, a similarity score, and rich descriptive information for each.

This tool goes beyond simple identification by visualizing the exact point where the two organisms diverge in their biological classification, providing an educational and intuitive user experience.

✨ ## Features

-   **Dual Image Upload**: A clean interface to upload two separate images for comparison.
-   **Deep Learning Identification**: Uses a fine-tuned ResNet-50 model to accurately identify the species in each image.
-   **Taxonomic Comparison Table**: Displays a side-by-side comparison of the full taxonomic tree (from Domain to Species) for both organisms.
-   **Visual Divergence Highlight**: The table automatically highlights the first rank where the classifications differ.
-   **Similarity Score**: Calculates and displays a quantitative similarity score based on the taxonomic relationship and model confidence.
-   **Confidence Visualization**: A bar chart shows the AI's confidence in each identification.
-   **Rich Information Cards**: Provides detailed descriptions, common names, blooming seasons, and fun facts for each identified species.

🎥 ## Live Demo

<!-- [A screenshot or GIF of the application in action] -->
<img width="651" height="926" alt="Screenshot from 2025-10-19 10-47-28" src="https://github.com/user-attachments/assets/c0b94321-eceb-444c-b94d-c266ac91482e" />


<!-- You can try out the live application here: `[Link to your deployed application on Hugging Face Spaces or other platform]` -->

🤔 ## How It Works

This project follows a complete machine learning pipeline from data preparation to a final web-based user interface.

1.  **Model Fine-Tuning**: A pre-trained ResNet-50 Convolutional Neural Network (CNN) is fine-tuned on a custom dataset of flower images. This retrains the final layer of the model to become an expert at identifying the specific species in the dataset.

2.  **Feature Extraction**: The fine-tuned model is used as a feature extractor. The final classification layer is removed, and every image in the reference dataset is passed through the network to generate a high-dimensional feature vector—a numerical "fingerprint" that represents the image's key visual characteristics.

3.  **Database Creation**: These feature vectors and their corresponding labels are stored in NumPy (`.npy`) files, creating a fast and efficient database for similarity searches.

4.  **Backend API**: A Flask server provides a `/compare` API endpoint. When a user uploads two images, the server:
    -   Generates a feature vector for each uploaded image.
    -   Uses Cosine Similarity to compare these new vectors against the entire feature database to find the closest match for each.
    -   Looks up the full taxonomic details and descriptive information from a `taxonomy.json` file.
    -   Compares the two taxonomies to find the divergence point and calculate a similarity score.

5.  **Frontend Interface**: A simple HTML, CSS, and JavaScript frontend allows users to upload images. It uses the `fetch` API to send the images to the Flask backend and then dynamically renders the received JSON data into the comparison table, charts, and info cards.

🛠️ ## Tech Stack

-   **Backend**: Python, Flask
-   **Machine Learning**: PyTorch (Torch & Torchvision)
-   **Data Science**: NumPy, Scikit-learn
-   **Frontend**: HTML, CSS, JavaScript
-   **Visualization**: Chart.js

🚀 ## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

-   Python 3.8 or higher
-   `pip` and `venv` for package management

### 1. Clone the Repository

First, clone the project from GitHub to your local machine:


git clone https://github.com/harshan2k21/Biological_tuned_ai
cd your-repository-name





### 2. Set Up the Python Environment
Create and activate a virtual environment to manage the project's dependencies.

Bash

#### Create the virtual environment
python3 -m venv .venv

#### Activate it (on Linux/macOS)
source .venv/bin/activate

#### On Windows, use:
#### .venv\Scripts\activate
### 3. Install Dependencies
Install all the required Python modules using the requirements.txt file.

Bash

pip install -r requirements.txt
#### 4. Prepare the Dataset
This model was trained on a specific dataset structure.

Ensure you have a directory named dataset/flowers.

Inside dataset/flowers, each subdirectory should be named after a species (e.g., rose, tulip, dandelion) and should contain the corresponding training images.

### 5. Run the Full Pipeline
You must run the Python scripts in the following order to train the model, create the necessary database files, and then launch the web server.

Step A: Train the Model This will fine-tune the ResNet model on your dataset and create the biologically_tuned_model.pth file. This can take a long time, especially without a GPU.

Bash

python3 train_model.py
Step B: Create the Feature Database This script uses the trained model to generate feature vectors for all images, creating features_db.npy and labels_db.npy.

Bash

python3 create_feature_db.py
Step C: Launch the Web Server This starts the Flask server. It will be accessible at http://127.0.0.1:5000.

Bash

python3 app.py
#### 6. Use the Application
Open the index.html file in your web browser. You can now upload two images and click "Compare Organisms" to see the results.
```bash

📁 ## File Structure

.
├── dataset/
│   └── flowers/            # Root folder for training images
│       ├── rose/
│       └── ...
├── .venv/                  # Python virtual environment
├── app.py                  # The main Flask web server
├── create_feature_db.py    # Script to generate the feature database
├── train_model.py          # Script to train the AI model
├── index.html              # The frontend user interface
├── taxonomy.json           # The database with descriptive information
├── requirements.txt        # List of Python dependencies
├── biologically_tuned_model.pth  # (Generated by train_model.py)
├── features_db.npy         # (Generated by create_feature_db.py)
└── labels_db.npy           # (Generated by create_feature_db.py)
