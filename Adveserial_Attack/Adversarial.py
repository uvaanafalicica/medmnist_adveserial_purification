"""
Defensive Shield Pipeline Phase 3 - Cell 2: Automated Validation Execution Loop
Production Status: Active | Pure Weight Checkpoint Mapping Sequence
"""

def run_integrated_vulnerability_pipeline(pipeline_loaders, base_directory, sample_budget=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📡 High-Yield Workspace Active -> Execution Engine: [{device.type.upper()}]")

    epsilon_matrix = [1/255, 2/255, 4/255, 8/255]
    manifest_records = []

    task_files = {
        "pneumonia": "best_clean_resnet18_pneumonia.pth",
        "organ": "best_clean_resnet18_organ.pth",
        "chest": "best_clean_resnet18_chest.pth"
    }

    print("\n" + "="*85)
    print("      ⚔️  INITIATING HIGH-YIELD CROSS-DOMAIN EXPERIMENTAL VALIDATION LOOPS       ")
    print("="*85)

    for task, weight_file in task_files.items():
        if task not in pipeline_loaders:
            print(f"⚠️  Scope Skip: Task Key '{task}' missing from active environment parameters.")
            continue

        weight_path = os.path.join(base_directory, weight_file)
        if not os.path.exists(weight_path):
            print(f"🛑 Critical Structural Flaw: Checkpoint missing at -> {weight_path}")
            continue

        checkpoint_state = torch.load(weight_path, map_location=device)
        if "backbone.fc.weight" in checkpoint_state:
            detected_classes = checkpoint_state["backbone.fc.weight"].shape[0]
        else:
            detected_classes = 1

        print(f"\n⚡ Processing Task Space Modality -> [{task.upper()}MNIST]")
        print(f" 🔍 Introspection Scan: Automatically Detected Trained Output Classes = {detected_classes}")

        if detected_classes == 11:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()

        classifier = TargetMedicalClassifier(num_classes=detected_classes).to(device)
        classifier.load_state_dict(checkpoint_state)
        classifier.eval()
        print(f" 💾 Checkpoint Synced -> Successfully Mounted State Dict: [{weight_file}]")

        task_data = pipeline_loaders[task]
        if isinstance(task_data, dict) and "test" in task_data:
            target_source = task_data["test"].dataset if hasattr(task_data["test"], 'dataset') else task_data["test"]
        else:
            target_source = task_data.dataset if hasattr(task_data, 'dataset') else task_data

        actual_budget = min(sample_budget, len(target_source))
        indices = list(range(actual_budget))
        optimized_subset = Subset(target_source, indices)
        sub_loader = DataLoader(optimized_subset, batch_size=32, shuffle=False)

        clean_accum, label_accum, onepixel_accum = [], [], []
        fgsm_accum = {eps: [] for eps in epsilon_matrix}
        pgd_accum = {eps: [] for eps in epsilon_matrix}
        mifgsm_accum = {eps: [] for eps in epsilon_matrix}

        for batch_images, batch_labels in sub_loader:
            batch_images = batch_images.to(device, non_blocking=True)

            if detected_classes == 11:
                batch_labels = batch_labels.to(device, non_blocking=True).long().view(-1)
            else:
                batch_labels = batch_labels.to(device, non_blocking=True).float()

            clean_accum.append(batch_images.cpu().detach())
            label_accum.append(batch_labels.cpu().detach())

            op_adv = execute_one_pixel_vectorized(batch_images, pixel_count=1)
            onepixel_accum.append(op_adv.cpu().detach())

            for eps in epsilon_matrix:
                fgsm_adv = execute_fgsm_vectorized(batch_images, batch_labels, classifier, criterion, eps)
                fgsm_accum[eps].append(fgsm_adv.cpu().detach())

                pgd_adv = execute_pgd_vectorized(batch_images, batch_labels, classifier, criterion, eps, steps=5)
                pgd_accum[eps].append(pgd_adv.cpu().detach())

                mifgsm_adv = execute_mifgsm_vectorized(batch_images, batch_labels, classifier, criterion, eps, steps=5)
                mifgsm_accum[eps].append(mifgsm_adv.cpu().detach())

        tensors_manifest = {
            "clean": torch.cat(clean_accum, dim=0),
            "labels": torch.cat(label_accum, dim=0),
            "one_pixel": torch.cat(onepixel_accum, dim=0),
            "fgsm": {eps: torch.cat(fgsm_accum[eps], dim=0) for eps in epsilon_matrix},
            "pgd": {eps: torch.cat(pgd_accum[eps], dim=0) for eps in epsilon_matrix},
            "mifgsm": {eps: torch.cat(mifgsm_accum[eps], dim=0) for eps in epsilon_matrix}
        }

        os.makedirs(base_directory, exist_ok=True)
        target_save_path = os.path.join(base_directory, f"compromised_{task}_tensors.pt")
        torch.save(tensors_manifest, target_save_path)
        print(f"  Array Serialization: Cached Attacked Array Slices at -> {target_save_path}")

        clean_acc = calculate_metric_accuracy(tensors_manifest["clean"], tensors_manifest["labels"], classifier, detected_classes)

        for attack_name in ["fgsm", "pgd", "mifgsm"]:
            for eps in epsilon_matrix:
                attack_images = tensors_manifest[attack_name][eps]
                classification_acc = calculate_metric_accuracy(attack_images, tensors_manifest["labels"], classifier, detected_classes)

                manifest_records.append({
                    "Dataset Modality": task.upper() + "MNIST",
                    "Attack Method": attack_name.upper(),
                    "Epsilon Bounds": f"{eps*255:.1f}/255",
                    "Baseline Clean Acc": f"{clean_acc:.1f}%",
                    "Degraded Attack Acc": f"{classification_acc:.1f}%"
                })

        op_acc = calculate_metric_accuracy(tensors_manifest["one_pixel"], tensors_manifest["labels"], classifier, detected_classes)
        manifest_records.append({
            "Dataset Modality": task.upper() + "MNIST",
            "Attack Method": "ONE-PIXEL",
            "Epsilon Bounds": "Localized",
            "Baseline Clean Acc": f"{clean_acc:.1f}%",
            "Degraded Attack Acc": f"{op_acc:.1f}%"
        })

        del tensors_manifest, classifier, clean_accum, label_accum, fgsm_accum, pgd_accum, mifgsm_accum, onepixel_accum
        gc.collect()
        torch.cuda.empty_cache()

    if manifest_records:
        df_manifest = pd.DataFrame(manifest_records)
        csv_out_path = os.path.join(base_directory, "phase3_vulnerability_manifest.csv")
        df_manifest.to_csv(csv_out_path, index=False)

        print("\n" + "="*85)
        print(" OFFICIAL PROOF LOG: DEFENSIVE RESNET-18 ARCHITECTURE VULNERABILITY MATRIX")
        print("="*85)
        print(df_manifest.to_string(index=False))
        print("="*85)
    else:
        print(" Critical Failure: Metric processing failed.")

if __name__ == "__main__":
    PROJECT_DIR = "/content/drive/MyDrive/Medical_Project_Data/Checkpoints"
    if 'pipeline_loaders' in globals() or 'pipeline_loaders' in locals():
        run_integrated_vulnerability_pipeline(pipeline_loaders=pipeline_loaders, base_directory=PROJECT_DIR, sample_budget=100)
