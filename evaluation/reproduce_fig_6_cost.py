import matplotlib.pyplot as plt
import numpy as np
import os
import json


PAPER_COSTS = {
    'Ayub et al.': 5.9089,
    'Irshad et al.': 27.2349,
    'Mishra et al.': 9.7285,
    'Rewal et al.': 7.9417,
    'Huang et al.': 13.5767,
    'Hu et al.': 8.2681,
    'HPostQCA-VSS (Paper)': 4.1741,
}

MY_UI_COST = 10.9028 

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
    schemes = list(PAPER_COSTS.keys())
    relative_values = [PAPER_COSTS[name] / MY_UI_COST for name in schemes]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(schemes, relative_values, color='skyblue', edgecolor='black')

    plt.axhline(1.0, color='r', linestyle='--', linewidth=2, label="HPostQCA-VSS (My Code)")

    for bar, val in zip(bars, relative_values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}",
                 ha='center', va='bottom', fontsize=13, fontweight='medium')

   
    plt.title("Relative Smart Device Computational Costs (vs. Implementation)", fontsize=16, fontweight='bold')
    plt.ylabel("Relative Cost (Paper Value / Obtained Value)", fontsize=15)
    plt.xticks(rotation=30, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='lower right', fontsize=15)

    # Make grid lines extend fully and sit below bars
    plt.grid(True, axis='x', linestyle='--', alpha=0.7, zorder=0)

    y_max = max(relative_values) * 1.15
    plt.ylim(0, y_max)

    plt.tight_layout()
    plt.savefig(PLOT_FILENAME, dpi=300)
    print(f"Plot saved to {PLOT_FILENAME}")
    plt.show()


if __name__ == "__main__":
    generate_relative_plot()
