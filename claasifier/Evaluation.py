import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import torchvision.models as models

# 1. Base Classifier Architecture Setup (As implemented in previous cells)
class MedMNISTClassifier(nn.Module):
    def __init__(self, task_type, num_classes):
        super(MedMNISTClassifier, self).__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        if task_type == "binary":
            self.backbone.fc = nn.Linear(in_features, 1)
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# 2. Universal Matrix Evaluation Evaluator Function
def evaluate_dataset_metrics(model, loader, task_type, device):
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)

            if task_type in ["binary", "multi-label"]:
                probs = torch.sigmoid(outputs)
            else:
                probs = torch.softmax(outputs, dim=1)

            all_targets.append(labels.numpy())
            all_predictions.append(probs.cpu().numpy())

    # Format and aggregate predictions arrays cleanly
    if task_type == "binary":
        all_targets = np.concatenate([t.ravel() for t in all_targets], axis=0)
        all_predictions = np.concatenate([p.ravel() for p in all_predictions], axis=0)
        acc = accuracy_score(all_targets, (all_predictions >= 0.5).astype(int))
        try: auc = roc_auc_score(all_targets, all_predictions)
        except ValueError: auc = 0.5

    elif task_type == "multi-label":
        all_targets = np.vstack(all_targets)
        all_predictions = np.vstack(all_predictions)
        acc = accuracy_score(all_targets, (all_predictions >= 0.5).astype(int))
        try: auc = roc_auc_score(all_targets, all_predictions, average="macro")
        except ValueError: auc = 0.5

    else: # multi-class classification execution boundaries
        all_targets = np.concatenate([t.ravel() for t in all_targets], axis=0)
        all_predictions = np.vstack(all_predictions)
        acc = accuracy_score(all_targets, np.argmax(all_predictions, axis=1))
        try: auc = roc_auc_score(all_targets, all_predictions, multi_class="ovr", average="macro")
        except ValueError: auc = 0.5

    return acc, auc

# 3. Automatic Matrix Generator Function for Table 2
def generate_manuscript_table_2(pipeline_loaders, checkpoint_directory):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Static Configuration Map matching your exact model framework parameters
    dataset_profiles = {
        "pneumonia": {"task": "binary", "classes": 1, "sota_auc": 0.958},
        "organ": {"task": "multi-class", "classes": 11, "sota_auc": 0.932},
        "chest": {"task": "multi-label", "classes": 14, "sota_auc": 0.768}
    }

    table_rows = []

    print("================================================================================")
    print("      📊 GENERATING TABLE 2: DEFINITIVE BASELINE EVALUATION MATRIX (CLEAN)      ")
    print("================================================================================\n")

    for name, profile in dataset_profiles.items():
        checkpoint_name = f"best_clean_resnet18_{name}.pth"
        checkpoint_path = os.path.join(checkpoint_directory, checkpoint_name)

        if not os.path.exists(checkpoint_path):
            print(f"⚠️ Verification Skipped: Weights for '{name.upper()}' not found at {checkpoint_path}")
            continue

        print(f"🔄 Processing '{name.upper()}MNIST' -> Evaluating target validation & test profiles...")

        # Instantiate model layout framework and inject weights parameters smoothly
        model = MedMNISTClassifier(task_type=profile["task"], num_classes=profile["classes"]).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        # Compute baseline target criteria across active validation parameters
        val_loader = pipeline_loaders[name]["val"]
        test_loader = pipeline_loaders[name]["test"]

        val_acc, val_auc = evaluate_dataset_metrics(model, val_loader, profile["task"], device)
        test_acc, test_auc = evaluate_dataset_metrics(model, test_loader, profile["task"], device)

        # Append data parameters directly to build structured manuscript row configurations
        table_rows.append({
            "Dataset Target": f"{name.upper()}MNIST",
            "Validation ACC": f"{val_acc * 100:.2f}%",
            "Validation ROC AUC": f"{val_auc:.4f}",
            "Test Blind ACC": f"{test_acc * 100:.2f}%",
            "Test Blind ROC AUC": f"{test_auc:.4f}",
            "MedMNIST v2 SOTA Baseline": f"{profile['sota_auc']:.3f}"
        })
        print(f"  Finished '{name.upper()}MNIST' Evaluation. Test AUC: {test_auc:.4f}")

    # Convert dictionary configuration records directly into production-grade Pandas DataFrames
    df_table_2 = pd.DataFrame(table_rows)

    print("\n" + "="*80)
    print("                        FINAL MANUSCRIPT MATRIX PRINT                        ")
    print("="*80)
    display(df_table_2) # Pretty print output inside Google Colab workflows environment
    print("="*80 + "\n")

    return df_table_2

# 4. Trigger Evaluation Code Execution
if __name__ == "__main__":
    CHECKPOINT_DIR = "/content/drive/MyDrive/Medical_Project_Data/Checkpoints"
    # Execute matrix compilation routines smoothly
    table_2_dataframe = generate_manuscript_table_2(pipeline_loaders, checkpoint_directory=CHECKPOINT_DIR)
