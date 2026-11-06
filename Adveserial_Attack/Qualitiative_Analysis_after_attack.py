"""
Defensive Shield Pipeline Phase 3 - Cell 3: Publication-Grade Accuracy Curve Generator
Author: Senior AI Systems Research Engineer
Output: Journal-ready high-contrast vectorized plots exported directly to workspace
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_academic_epsilon_curves(csv_path, output_directory):
    """
    Parses your vulnerability manifest file to generate distinct Accuracy vs Epsilon
    line charts matching strict IEEE/MDPI journal formatting constraints.
    """
    if not os.path.exists(csv_path):
        print(f" Visualizer Error: Target manifest file missing at -> {csv_path}")
        return

    # 1. Load and parse matrix fields
    df = pd.read_csv(csv_path)

    # Clean string punctuation flags out of the metric fields to convert back to float
    df['Baseline Clean Acc'] = df['Baseline Clean Acc'].str.rstrip('%').astype('float')
    df['Degraded Attack Acc'] = df['Degraded Attack Acc'].str.rstrip('%').astype('float')

    # Map epsilon boundary string fractions to continuous numeric values for correct X-axis scaling
    eps_mapping = {'0.0/255': 0.0, '1.0/255': 1.0, '4.0/255': 4.0, '8.0/255': 8.0}
    df['Eps_Val'] = df['Epsilon Bounds'].map(eps_mapping)

    # Filter out localized single-point control variants (One-Pixel) to focus purely on continuous gradient attacks
    gradient_df = df[df['Attack Method'] != 'ONE-PIXEL'].copy()

    # 2. Establish uniform academic plot styling rules
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 14
    })

    unique_modalities = gradient_df['Dataset Modality'].unique()

    # Process each medical dataset domain into a distinct manuscript-ready figure array
    for modality in unique_modalities:
        mod_data = gradient_df[gradient_df['Dataset Modality'] == modality].copy()

        # Inject standard baseline anchor points at Epsilon = 0 to show true clean starting accuracy
        clean_row_val = mod_data['Baseline Clean Acc'].iloc[0]
        attack_types = mod_data['Attack Method'].unique()

        anchor_records = []
        for attack in attack_types:
            anchor_records.append({
                'Dataset Modality': modality,
                'Attack Method': attack,
                'Epsilon Bounds': '0.0/255',
                'Baseline Clean Acc': clean_row_val,
                'Degraded Attack Acc': clean_row_val,
                'Eps_Val': 0.0
            })

        # Combine structural anchors with current logs and sort accurately by expanding epsilon values
        plot_data = pd.concat([pd.DataFrame(anchor_records), mod_data], ignore_index=True)
        plot_data = plot_data.sort_values(by='Eps_Val')

        # Initialize high-DPI figure footprint bounding frames matching standard column dimensions
        plt.figure(figsize=(6.5, 4.8), dpi=300)

        # Use high-contrast color arrangements and distinct geometric shape marks for clear readability in print
        style_mapping = {
            'FGSM': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '--'},   # Blue Dash Dot
            'PGD': {'color': '#d62728', 'marker': 's', 'linestyle': '-'},    # Red Solid Line
            'MIFGSM': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'}  # Green Long Dash
        }

        for attack in attack_types:
            attack_slice = plot_data[plot_data['Attack Method'] == attack]
            cfg = style_mapping.get(attack, {'color': '#7f7f7f', 'marker': 'x', 'linestyle': ':'})

            plt.plot(
                attack_slice['Eps_Val'],
                attack_slice['Degraded Attack Acc'],
                label=attack,
                color=cfg['color'],
                marker=cfg['marker'],
                linestyle=cfg['linestyle'],
                linewidth=2.2,
                markersize=7,
                alpha=0.95
            )

        # 3. Apply plot labels and annotations
        plt.title(f"Adversarial Vulnerability Profile: {modality}", weight='bold', pad=12)
        plt.xlabel(r"Perturbation Magnitude Epsilon Bound ($\epsilon \times 255$)")
        plt.ylabel("Classification Accuracy (%)")

        plt.xlim(-0.3, 8.5)
        plt.ylim(-2.0, 102.0)

        plt.xticks([0, 1, 4, 8], labels=['0 (Clean)', '1', '4', '8'])
        plt.grid(True, linestyle=':', alpha=0.6)

        # Position the legend box cleanly inside the open region of the graph
        plt.legend(title="Attack Vector Configuration", loc="lower left", frameon=True, shadow=False)

        # Optimize boundary space margins to remove white bleed zones
        plt.tight_layout()

        # Export the figure to your storage subfolder using cross-platform safe lower-case characters
        safe_name = modality.lower().replace("mnist", "_epsilon_curve.png")
        save_path = os.path.join(output_directory, safe_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"  Vector Graphics Complete -> Exported high-contrast figure matrix asset to: [{save_path}]")

    print("\n [VISUALIZATION PIPELINE SUCCESS]: All adversarial curve figures match publication guidelines.")

if __name__ == "__main__":
    # Point directly to your active checkpoints data path location fields
    PROJECT_DIR = "/content/drive/MyDrive/Medical_Project_Data/Checkpoints"
    MANIFEST_CSV = os.path.join(PROJECT_DIR, "phase3_vulnerability_manifest.csv")

    generate_academic_epsilon_curves(csv_path=MANIFEST_CSV, output_directory=PROJECT_DIR)
