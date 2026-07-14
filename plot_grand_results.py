import json
import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality formatting
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300
})

def generate_multi_dataset_plot():
    datasets = []
    gin_accs = []
    static_cw_accs = []
    dyn_cw_accs = []
    
    # 1. Load NCI1 from ablation_statistics.json
    try:
        with open('ablation_statistics.json', 'r') as f:
            nci_data = json.load(f)
            datasets.append('NCI1\n(4,110 Graphs)')
            gin_accs.append(nci_data['GINBaseline']['Accuracy']['Mean'])
            static_cw_accs.append(nci_data['StaticCWNetwork']['Accuracy']['Mean'])
            dyn_cw_accs.append(nci_data['DynamicCWNetwork']['Accuracy']['Mean'])
    except Exception as e:
        print(f"Warning: Could not load NCI1 data: {e}")

    # 2. Load MUTAG, PROTEINS, ENZYMES from grand_statistics.json
    try:
        with open('grand_statistics.json', 'r') as f:
            grand_data = json.load(f)
            for ds in ['MUTAG', 'PROTEINS', 'ENZYMES']:
                if ds in grand_data:
                    datasets.append(ds)
                    gin_accs.append(grand_data[ds]['GINBaseline']['Mean_Accuracy'])
                    static_cw_accs.append(grand_data[ds]['StaticCWNetwork']['Mean_Accuracy'])
                    dyn_cw_accs.append(grand_data[ds]['DynamicCWNetwork']['Mean_Accuracy'])
    except Exception as e:
        print(f"Warning: Could not load grand data: {e}")

    if not datasets:
        print("Error: No data available to plot.")
        return

    x = np.arange(len(datasets))
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Colors matching the Pareto plot
    color_gin = '#2980b9'   # Blue
    color_stat = '#f39c12'  # Orange
    color_dyn = '#c0392b'   # Crimson
    
    rects1 = ax.bar(x - width, gin_accs, width, label='GIN Baseline (1-WL)', color=color_gin, edgecolor='black')
    rects2 = ax.bar(x, static_cw_accs, width, label='StaticCW-Net (Ablation)', color=color_stat, edgecolor='black')
    rects3 = ax.bar(x + width, dyn_cw_accs, width, label='DynamicCW-Net (Ours)', color=color_dyn, edgecolor='black')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Generalization Across Biochemical Datasets (5-Fold Mean)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper right', shadow=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Function to attach a text label above each bar
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Set Y-axis limits dynamically
    min_acc = min(gin_accs + static_cw_accs + dyn_cw_accs)
    max_acc = max(gin_accs + static_cw_accs + dyn_cw_accs)
    ax.set_ylim([max(0, min_acc - 5), min(100, max_acc + 10)])

    plt.tight_layout()
    plt.savefig('Multi_Dataset_Generalization.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('Multi_Dataset_Generalization.png', format='png', dpi=300, bbox_inches='tight')
    print("Successfully generated Multi_Dataset_Generalization.pdf and .png")

if __name__ == "__main__":
    generate_multi_dataset_plot()
