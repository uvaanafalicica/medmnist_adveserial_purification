import os
import numpy as np
import torch
import torch.nn as nn

def compute_real_weights_and_verify(data_directory, classifier_suite, pipeline_loaders):
    """
    Dynamically extracts actual class frequencies from your Drive files,
    calculates the precise inverse class weights, and runs the Result 2 audit.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_files = {
        "pneumonia": "pneumoniamnist.npz",
        "organ": "organamnist.npz",
        "chest": "chestmnist.npz"
    }

    real_weights = {}

    print("=== COMPUTING GENUINE INVERSE FREQUENCY WEIGHTS ===")

    for key, file_name in target_files.items():
        file_path = os.path.join(data_directory, file_name)
        archive = np.load(file_path)
        labels = archive['train_labels']

        if key == "pneumonia":
            # Binary classification handles imbalance via pos_weight inside BCE
            labels_flat = labels.ravel()
            pos_cases = np.sum(labels_flat == 1)
            neg_cases = np.sum(labels_flat == 0)
            # pos_weight = total_negative / total_positive
            weight_val = neg_cases / pos_cases if pos_cases > 0 else 1.0
            real_weights[key] = torch.tensor([weight_val], dtype=torch.float32)

        elif key == "chest":
            # Multi-label classification (14 classes independent pos_weight)
            pos_weights = []
            for i in range(14):
                pos = np.sum(labels[:, i])
                neg = len(labels) - pos
                w = neg / pos if pos > 0 else 1.0
                pos_weights.append(w)
            real_weights[key] = torch.tensor(pos_weights, dtype=torch.float32)

        else:
            # Multi-class classification (11 classes categorical weights)
            labels_flat = labels.ravel()
            classes, counts = np.unique(labels_flat, return_counts=True)
            total_samples = len(labels_flat)
            # Inverse frequency formula: total_samples / (num_classes * class_samples)
            inv_weights = total_samples / (len(classes) * counts)
            real_weights[key] = torch.tensor(inv_weights, dtype=torch.float32)

        print(f" Calculated real weight vector for {key.upper()} (Shape: {list(real_weights[key].shape)})")

    print("\n==========================================================================================")
    print("         REAL EXPERIMENTAL ANALYSIS: LOSS CONFIGURATION & IMBALANCE SANITY CHECK          ")
    print("==========================================================================================\n")
    print(f"{'Dataset Target':<16} | {'Unweighted Loss':<16} | {'Weighted Loss':<16} | {'Penalty Inflation Factor'}")
    print("-" * 80)

    for name, components in classifier_suite.items():
        model = components["model"]
        loader_key = "pneumonia" if name == "pneumonia" else ("organ" if name == "organ" else "chest")

        if loader_key in pipeline_loaders:
            train_loader = pipeline_loaders[loader_key]["train"]
            images, labels = next(iter(train_loader))

            images = images.to(device)
            if loader_key == "organ":
                labels = labels.to(device).long()
            else:
                labels = labels.to(device).float()

            weights_tensor = real_weights[loader_key].to(device)

            if loader_key == "pneumonia":
                criterion_raw = nn.BCEWithLogitsLoss()
                criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=weights_tensor)
            elif loader_key == "chest":
                criterion_raw = nn.BCEWithLogitsLoss()
                criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=weights_tensor)
            else:
                criterion_raw = nn.CrossEntropyLoss()
                criterion_weighted = nn.CrossEntropyLoss(weight=weights_tensor)

            model.eval()
            with torch.no_grad():
                outputs = model(images)
                if loader_key == "pneumonia":
                    outputs = outputs.squeeze()
                    labels = labels.squeeze()

                loss_raw = criterion_raw(outputs, labels).item()
                loss_weighted = criterion_weighted(outputs, labels).item()

            inflation_factor = loss_weighted / loss_raw if loss_raw != 0 else 1.0
            print(f"{name.upper() + 'MNIST':<16} | {loss_raw:<16.4f} | {loss_weighted:<16.4f} | {inflation_factor:.3f}x Shift")

    print("-" * 80)
    return real_weights

if __name__ == "__main__":
    DRIVE_DATA_DIR = "/content/drive/MyDrive/Medical_Project_Data"
    # Execute and overwrite the placeholder weight dictionary with real clinical weights
    actual_loss_weights = compute_real_weights_and_verify(DRIVE_DATA_DIR, base_classifiers, pipeline_loaders)
