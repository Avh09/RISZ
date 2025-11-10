import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
# Smart device (Ui) costs from the paper
PAPER_COSTS = {
    'Ayub et al.': 5.9089,
    'Irshad et al.': 27.2349,
    'Mishra et al.': 9.7285,
    'Rewal et al.': 7.9417,
    'Huang et al.': 13.5767,
    'Hu et al.': 8.2681,
    'HPostQCA-VSS (Paper)': 4.1741,
}

# --- YOUR measured Smart Device cost ---
MY_UI_COST = 10.9028  # <-- replace with your value

# --- Output folder ---
PLOT_DIR = "plots"
PLOT_FILENAME = os.path.join(PLOT_DIR, "smart_device_relative_costs.png")

def generate_relative_plot():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Compute relative values (paper value / your value)
    schemes = list(PAPER_COSTS.keys())
    relative_values = [PAPER_COSTS[name] / MY_UI_COST for name in schemes]

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(schemes, relative_values, color='skyblue', edgecolor='black')

    # Highlight your reference line (1.0)
    plt.axhline(1.0, color='r', linestyle='--', linewidth=2, label="HPostQCA-VSS (My Code)")

    # Annotate each bar with the relative ratio
    for bar, val in zip(bars, relative_values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}",
                 ha='center', va='bottom', fontsize=13, fontweight='medium')

    # Titles and labels
    plt.title("Relative Smart Device Computational Costs (vs. Implementation)", fontsize=16, fontweight='bold')
    plt.ylabel("Relative Cost (Paper Value / Obtained Value)", fontsize=15)
    plt.xticks(rotation=30, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Adjust top limit to ensure labels fit ---
    y_max = max(relative_values) * 1.15
    plt.ylim(0, y_max)

    plt.tight_layout()
    plt.savefig(PLOT_FILENAME, dpi=300)
    print(f"✅ Plot saved to {PLOT_FILENAME}")
    plt.show()

if __name__ == "__main__":
    generate_relative_plot()
