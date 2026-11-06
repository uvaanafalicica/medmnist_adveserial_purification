import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class MedMNISTPipelineDataset(Dataset):
    """
    Standardized dataset wrapper to handle on-the-fly 256x256 upscaling,
    RGB channel expansion, and tensor normalization conformant to SOTA standards.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # Enforce 3-channel layout if raw array is grayscale (H, W -> H, W, 3)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32).squeeze()

def initialize_spatial_pipeline(data_directory, batch_size=32):
    """
    Loads official splits directly from storage and binds them to optimized PyTorch DataLoaders.
    Strictly enforces shuffle=True only on the training partition to prevent data leakage.
    """
    target_files = {
        "pneumonia": "pneumoniamnist.npz",
        "organ": "organamnist.npz",
        "chest": "chestmnist.npz"
    }

    # Standard spatial transformation pipeline configuration
    pipeline_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()  # Automatically scales pixels from [0-255] to [0.0 - 1.0]
    ])

    loaders = {}

    print("=== 🚀 INITIALIZING SPATIAL PIPELINE LAYERS (256x256) ===\n")

    for key, file_name in target_files.items():
        file_path = os.path.join(data_directory, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target missing from directory path: {file_path}")

        archive = np.load(file_path)

        # Binding official partitions uniformly to the pipeline wrapper
        train_set = MedMNISTPipelineDataset(archive['train_images'], archive['train_labels'], transform=pipeline_transforms)
        val_set   = MedMNISTPipelineDataset(archive['val_images'], archive['val_labels'], transform=pipeline_transforms)
        test_set  = MedMNISTPipelineDataset(archive['test_images'], archive['test_labels'], transform=pipeline_transforms)

        # Construct optimized DataLoaders with strict architectural separation
        loaders[key] = {
            "train": DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True),
            "val":   DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
            "test":  DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        }

        # CORRECTION: Indexing [0] ensures safe unpack of a singular matrix sample
        sample_img, _ = train_set[0]
        print(f"[{key.upper()} PIPELINE ONLINE]")
        print(f"  -> Tensor Layout Shape : {list(sample_img.shape)} (Verified 3-Channel RGB)")
        print(f"  -> Normalized Bounds   : Min={sample_img.min():.1f} | Max={sample_img.max():.1f}")
        print("-" * 75)

    return loaders

if __name__ == "__main__":
    # Standard runtime configuration mapping
    DRIVE_DATA_DIR = "/content/drive/MyDrive/Medical_Project_Data"
    pipeline_loaders = initialize_spatial_pipeline(data_directory=DRIVE_DATA_DIR, batch_size=32)
