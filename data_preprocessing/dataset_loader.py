pip install medmnist

import os
import medmnist
from medmnist import INFO
from google.colab import drive

print("=== STEP 1: MOUNTING GOOGLE DRIVE ===")
# This connects Colab to your actual personal Google Drive
drive.mount('/content/drive')

# Define the permanent directory path inside your main Google Drive
drive_data_dir = "/content/drive/MyDrive/Medical_Project_Data"
os.makedirs(drive_data_dir, exist_ok=True)

# CORRECTED: 'organmnist' has been changed to 'organamnist'
target_datasets = ["chestmnist", "organamnist", "pneumoniamnist"]

print("\n=== INITIATING DIRECT DATASET DOWNLOADS TO DRIVE ===")
print(f"Destination Folder: {os.path.abspath(drive_data_dir)}\n")

for name in target_datasets:
    print(f"🔄 Fetching metadata and downloading: {name.upper()}...")

    # Retrieve structural specifications from the library
    data_info = INFO[name]
    DatasetClass = getattr(medmnist, data_info['python_class'])

    # download=True pulls files from Zenodo if they aren't already in root
    # split='train' forces the initial raw archive verification
    raw_archive = DatasetClass(root=drive_data_dir, split='train', download=True)

    print(f"Success: {name.upper()} is verified and locked in Google Drive!")

print("\n=== ALL DATASETS ARE PERMANENTLY SAVED! ===")
