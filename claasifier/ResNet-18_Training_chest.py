import os
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

def evaluate_recovery_performance_optimized(model, data_loader, criterion, task_type, device):
    """
    Production-grade evaluation loop optimized with native PyTorch 2.x amp context layers
    and strict matrix structural validations.
    """
    model.eval()
    epoch_loss = 0.0
    all_targets = []      # Fixed: Initialized empty lists properly to prevent SyntaxError
    all_predictions = []  # Fixed: Initialized empty lists properly to prevent SyntaxError

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)

            if task_type == "multi-class":
                labels = labels.to(device, non_blocking=True).long()
            else:
                labels = labels.to(device, non_blocking=True).float()

            with autocast(device_type='cuda'):
                outputs = model(images)
                if task_type == "binary":
                    outputs = outputs.squeeze(-1)
                    labels = labels.squeeze(-1)
                loss = criterion(outputs, labels)

            epoch_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs) if task_type in ["binary", "multi-label"] else torch.softmax(outputs, dim=1)

            all_targets.append(labels.cpu().numpy())
            all_predictions.append(probs.cpu().numpy())

    total_samples = len(data_loader.dataset)
    avg_loss = epoch_loss / total_samples

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

    else: # multi-class
        all_targets = np.concatenate([t.ravel() for t in all_targets], axis=0)
        all_predictions = np.vstack(all_predictions)
        acc = accuracy_score(all_targets, np.argmax(all_predictions, axis=1))
        try: auc = roc_auc_score(all_targets, all_predictions, multi_class="ovr", average="macro")
        except ValueError: auc = 0.5

    return avg_loss, acc, auc

def execute_accelerated_hybrid_recovery(classifier_suite, pipeline_loaders, save_directory, epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler(device='cuda')

    print("================================================================================")
    print("      🚀 ULTRASONIC HYBRID RUNTIME: LOADER OPTIMIZED NATIVE AMP PIPELINE        ")
    print("================================================================================\n")

    for name in ["pneumonia", "organ"]:
        checkpoint_path = os.path.join(save_directory, f"best_clean_resnet18_{name}.pth")
        if os.path.exists(checkpoint_path):
            classifier_suite[name]["model"].load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f" INSTANT RESTORE: Pre-trained '{name.upper()}MNIST' weights linked from Drive.")
        else:
            print(f" WARNING: Pre-trained file for {name.upper()} missing from Drive pathways.")

    print("\n STARTING 15-MINUTE HIGH-VELOCITY OPTIMIZATION FOR: CHESTMNIST")
    chest_comp = classifier_suite["chest"]
    model, criterion, optimizer = chest_comp["model"], chest_comp["criterion"], chest_comp["optimizer"]

    best_val_auc = 0.0
    chest_checkpoint = os.path.join(save_directory, "best_clean_resnet18_chest.pth")

    train_loader = pipeline_loaders["chest"]["train"]
    val_loader = pipeline_loaders["chest"]["val"]
    test_loader = pipeline_loaders["chest"]["test"]

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item() * images.size(0)

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        val_loss, val_acc, val_auc = evaluate_recovery_performance_optimized(model, val_loader, criterion, "multi-label", device)
        print(f"  ↳ Epoch [{epoch:02d}/{epochs:02d}] -> Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val ACC: {val_acc*100:.2f}% | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), chest_checkpoint)
            print("      Peak validation reached. Saved accelerated checkpoint weights.")

    print("\n Optimization complete. Running blind final exam on ChestMNIST...")
    model.load_state_dict(torch.load(chest_checkpoint))
    test_loss, test_acc, test_auc = evaluate_recovery_performance_optimized(model, test_loader, criterion, "multi-label", device)

    print("\n [FINAL BLIND TEST MATRIX: CHEST]")
    print(f"   -> Accuracy Score : {test_acc*100:.2f}%")
    print(f"   -> ROC AUC Score   : {test_auc:.4f}")
    print("-" * 80)

if __name__ == "__main__":
    SAVE_DIR = "/content/drive/MyDrive/Medical_Project_Data/Checkpoints"
    execute_accelerated_hybrid_recovery(base_classifiers, pipeline_loaders, save_directory=SAVE_DIR, epochs=15)
