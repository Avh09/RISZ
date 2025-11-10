import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
PLOT_DIR = "plots"
PLOT_FILENAME = os.path.join(PLOT_DIR, "fig_brute_force_complexity_log_blue.png")

# Number of features (x-axis)
features = np.arange(1, 21)
n_values = 100
work = np.power(float(n_values), features)

# --- Plotting ---
print("Plotting Brute-Force Attack Complexity (Log Scale, Blue Curve)...")
os.makedirs(PLOT_DIR, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(features, work, color='blue', marker='o', linewidth=2.5, label=f"Attack Cost (n={n_values})")

plt.title('Brute-Force Attack Complexity vs. Number of Features ($n^f$)', fontsize=16, fontweight='bold')
plt.xlabel('Number of Features (f)', fontsize=15)
plt.ylabel('Attacker Work (Number of Guesses)', fontsize=15)

# Log scale for y-axis
plt.yscale('log')

# Font and tick styling
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=15, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(PLOT_FILENAME, dpi=300)
print(f"Graph successfully saved to {PLOT_FILENAME}")
plt.show()
