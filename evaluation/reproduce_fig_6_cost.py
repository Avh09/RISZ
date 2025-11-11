import matplotlib.pyplot as plt
import numpy as np
import os
import json

# --- Configuration ---
RESULTS_FILE = "benchmark_results.json"
PLOT_DIR = "plots"
PLOT_FILENAME = os.path.join(PLOT_DIR, "fig06_relative_Ui_cost.png")

plt.style.use('seaborn-v0_8-whitegrid')

def calculate_costs(primitives):
    """Calculate per-scheme computational costs based on Table II formulas."""
    T = primitives
    T.setdefault('Th', 0); T.setdefault('Tecm', 0); T.setdefault('Teca', 0)
    T.setdefault('Tsenc', 0); T.setdefault('Tsdec', 0)
    T.setdefault('Tg', 0); T.setdefault('Tsm', 0)
    T.setdefault('Tpm', 0); T.setdefault('Tpa', 0); T.setdefault('Tcha', 0)

    costs = {
        "Ayub et al.":   5*T['Th'] + 2*T['Tecm'],
        "Irshad et al.": 20*T['Th'] + 9*T['Tecm'] + 3*T['Teca'],
        "Mishra et al.": 8*T['Th'] + 4*T['Tg'] + 2*T['Tsm'] + 3*T['Tpm'] + 2*T['Tpa'] + 2*T['Tcha'],
        "Rewal et al.":  8*T['Th'] + 4*T['Tg'] + 2*T['Tsm'] + 4*T['Tpm'] + 2*T['Tpa'] + T['Tcha'],
        "Huang et al.":  5*T['Th'] + 5*T['Tecm'] + T['Tsdec'],
        "Hu et al.":     4*T['Th'] + 3*T['Tecm'],
        "HPostQCA-VSS (Implementation)": 6*T['Th'] + 2*T['Tg'] + T['Tsm'] + 2*T['Tpm'] + T['Tpa']
    }
    return costs


def generate_relative_plot():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- Load benchmark results ---
    try:
        with open(RESULTS_FILE, 'r') as f:
            primitives = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: '{RESULTS_FILE}' not found. Run 'python3 run_benchmark.py' first.")
        return

    # --- Compute costs ---
    all_ui_costs = calculate_costs(primitives)
    my_cost = all_ui_costs["HPostQCA-VSS (Implementation)"]

    # --- Compute relative ratios ---
    schemes = list(all_ui_costs.keys())
    relative_values = [all_ui_costs[name] / my_cost for name in schemes]
    abs_values = [all_ui_costs[name] for name in schemes]

    # --- Sort by relative cost ---
    sorted_idx = np.argsort(relative_values)
    schemes = [schemes[i] for i in sorted_idx]
    relative_values = [relative_values[i] for i in sorted_idx]
    abs_values = [abs_values[i] for i in sorted_idx]

    # --- Create plot ---
    plt.figure(figsize=(12, 7))
    colors = ['#87CEEB' if 'Implementation' not in s else '#E74C3C' for s in schemes]

    bars = plt.barh(schemes, relative_values, color=colors, edgecolor='black', height=0.55, zorder=3)
    plt.axvline(1.0, color='r', linestyle='--', linewidth=2, label='Implemented Scheme (Baseline = 1.0)', zorder=2)

    # --- Add annotations ---
    for bar, rel, abs_val in zip(bars, relative_values, abs_values):
        plt.text(rel + 0.05, bar.get_y() + bar.get_height()/2,
                 f"{rel:.2f}×  ({abs_val*1000:.2f} µs)",
                 va='center', fontsize=15, fontweight='semibold')

    # --- Labels and aesthetics ---
    plt.title("Relative Computational Cost Comparison\n(All Schemes vs. HPostQCA-VSS)", 
              fontsize=20, fontweight='bold', pad=25)
    plt.xlabel("Relative Cost (Scheme Cost / Implemented Cost)", fontsize=18, labelpad=14)
    plt.ylabel("Authentication Scheme", fontsize=18, labelpad=14)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='lower right', fontsize=15)

    # Make grid lines extend fully and sit below bars
    plt.grid(True, axis='x', linestyle='--', alpha=0.7, zorder=0)

    # Extend x-axis slightly more to show full grid lines and annotations
    plt.xlim(0, max(relative_values) * 1.35)

    plt.tight_layout()

    # --- Save and show ---
    plt.savefig(PLOT_FILENAME, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to {PLOT_FILENAME}")
    plt.show()


if __name__ == "__main__":
    generate_relative_plot()
