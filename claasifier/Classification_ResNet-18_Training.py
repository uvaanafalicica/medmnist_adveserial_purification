import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import os

def generate_and_save_confusion_matrices(classifier_suite, pipeline_loaders, device, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. PNEUMONIAMNIST CONFUSION MATRIX (2x2) ---
    print("Generating Confusion Matrix for PneumoniaMNIST...")
    p_model = classifier_suite["pneumonia"]["model"].to(device)
    p_loader = pipeline_loaders["pneumonia"]["test"]
    p_model.eval()

    p_targets, p_preds = [], []
    with torch.no_grad():
        for images, labels in p_loader:
            images = images.to(device)
            outputs = p_model(images)
            preds = (torch.sigmoid(outputs).squeeze(-1) >= 0.5).int()
            p_targets.append(labels.numpy().ravel())
            p_preds.append(preds.cpu().numpy())

    cm_p = confusion_matrix(np.concatenate(p_targets), np.concatenate(p_preds))

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_p, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.title('PneumoniaMNIST Confusion Matrix', fontsize=12, fontweight='bold')
    plt.ylabel('True Diagnosis')
    plt.xlabel('Predicted Diagnosis')
    plt.savefig(os.path.join(save_dir, 'pneumoniamnist_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # --- 2. ORGANAMNIST CONFUSION MATRIX (11x11) ---
    print(" Generating Confusion Matrix for OrganAMNIST...")
    o_model = classifier_suite["organ"]["model"].to(device)
    o_loader = pipeline_loaders["organ"]["test"]
    o_model.eval()

    o_targets, o_preds = [], []
    with torch.no_grad():
        for images, labels in o_loader:
            images = images.to(device)
            outputs = o_model(images)
            preds = torch.argmax(outputs, dim=1)
            o_targets.append(labels.numpy().ravel())
            o_preds.append(preds.cpu().numpy())

    cm_o = confusion_matrix(np.concatenate(o_targets), np.concatenate(o_preds))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_o, annot=True, fmt='d', cmap='Greens')
    plt.title('OrganAMNIST Confusion Matrix (11-Classes)', fontsize=12, fontweight='bold')
    plt.ylabel('True Structural Class')
    plt.xlabel('Predicted Structural Class')
    plt.savefig(os.path.join(save_dir, 'organamnist_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("Both Confusion Matrices saved to Drive perfectly!")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate_and_save_confusion_matrices(
        classifier_suite=base_classifiers,
        pipeline_loaders=pipeline_loaders,
        device=device,
        save_dir="/content/drive/MyDrive/Medical_Project_Data/Plots"
    )
