import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

def evaluate_classifier_performance(model, data_loader, criterion, task_type, device):
    """
    Computes precise evaluation metrics (Loss, Accuracy, and AUC)
    conforming to standard medical classification benchmarks.
    """
    model.eval()
    epoch_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)

            if task_type == "multi-class":
                labels = labels.to(device).long()
            else:
                labels = labels.to(device).float()

            outputs = model(images)

            if task_type == "binary":
                outputs = outputs.squeeze()
                labels = labels.squeeze()

            loss = criterion(outputs, labels)
            epoch_loss += loss.item() * images.size(0)

            # Extract probability vectors based on architectural layout
            if task_type == "binary":
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                targets = labels.cpu().numpy()
            elif task_type == "multi-label":
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                targets = labels.cpu().numpy()
            else: # multi-class
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                targets = labels.cpu().numpy()

            all_targets.append(targets)
            all_predictions.append(probs if task_type != "multi-class" and len(probs.shape) == 1 else probs)

    total_samples = len(data_loader.dataset)
    avg_loss = epoch_loss / total_samples

    # Concatenate batched records into uniform arrays
    all_targets = np.concatenate(all_targets, axis=0) if task_type != "binary" else np.concatenate([t.ravel() if len(t.shape)>0 else np.array([t]) for t in all_targets])
    all_predictions = np.concatenate(all_predictions, axis=0) if task_type != "binary" else np.concatenate([p.ravel() if len(p.shape)>0 else np.array([p]) for p in all_predictions])

    # Generate binary decision arrays for accuracy math mappings
    if task_type == "binary":
        all_preds_binary = (all_predictions >= 0.5).astype(int)
        acc = accuracy_score(all_targets, all_preds_binary)
        try:
            auc = roc_auc_score(all_targets, all_predictions)
        except:
            auc = 0.5
    elif task_type == "multi-label":
        all_preds_binary = (all_predictions >= 0.5).astype(int)
        acc = accuracy_score(all_targets, all_preds_binary)
        auc = roc_auc_score(all_targets, all_predictions, average="macro")
    else: # multi-class
        all_preds_idx = np.argmax(all_predictions, axis=1)
        acc = accuracy_score(all_targets, all_preds_idx)
        auc = roc_auc_score(all_targets, all_predictions, multi_class="ovr", average="macro")

    return avg_loss, acc, auc

def execute_baseline_training_pipeline(classifier_suite, pipeline_loaders, save_directory, epochs=15):
    """
    Executes Step 1.5: Loops through tasks, saves best checkpoints,
    and outputs final peer-review verification metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_directory, exist_ok=True)

    # Task metadata tracking profiles
    task_map = {
        "pneumonia": "binary",
        "organ": "multi-class",
        "chest": "multi-label"
    }

    print("================================================================================")
    print(f"      STARTING STEP 1.5: BASELINE TRAINING LOOP ({epochs} TOTAL EPOCHS)")
    print("================================================================================\n")

    for name, components in classifier_suite.items():
        model = components["model"]
        criterion = components["criterion"]
        optimizer = components["optimizer"]
        task_type = task_map[name]

        print(f" INITIALIZING RUNTIME OPTIMIZATION FOR: {name.upper()}MNIST")
        best_val_auc = 0.0
        checkpoint_path = os.path.join(save_directory, f"best_clean_resnet18_{name}.pth")

        train_loader = pipeline_loaders[name]["train"]
        val_loader = pipeline_loaders[name]["val"]
        test_loader = pipeline_loaders[name]["test"]

        for epoch in range(1, epochs + 1):
            model.train()
            running_train_loss = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                if task_type == "multi-class":
                    labels = labels.to(device).long()
                else:
                    labels = labels.to(device).float()

                optimizer.zero_grad()
                outputs = model(images)

                if task_type == "binary":
                    outputs = outputs.squeeze()
                    labels = labels.squeeze()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * images.size(0)

            avg_train_loss = running_train_loss / len(train_loader.dataset)
            val_loss, val_acc, val_auc = evaluate_classifier_performance(model, val_loader, criterion, task_type, device)

            print(f"  ↳ Epoch [{epoch:02d}/{epochs:02d}] -> Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val ACC: {val_acc*100:.2f}% | Val AUC: {val_auc:.4f}")

            # Step 1.5.3 Checkpoint Save Guardrail
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), checkpoint_path)
                print(f"     💾 Peak validation layer reached. Saved checkpoint weights.")

        # Final Test evaluation using best weights
        print(f" Training finished for {name.upper()}. Loading peak validation state for final exam...")
        model.load_state_dict(torch.load(checkpoint_path))
        test_loss, test_acc, test_auc = evaluate_classifier_performance(model, test_loader, criterion, task_type, device)

        print(f" [FINAL BLIND TEST MATRIX: {name.upper()}]")
        print(f"   -> Accuracy Score : {test_acc*100:.2f}%")
        print(f"   -> ROC AUC Score   : {test_auc:.4f}")
        print("-" * 80)

if __name__ == "__main__":
    SAVE_DIR = "/content/drive/MyDrive/Medical_Project_Data/Checkpoints"
    # Execute the baseline optimization process for 15 rounds
    execute_baseline_training_pipeline(base_classifiers, pipeline_loaders, save_directory=SAVE_DIR, epochs=15)
