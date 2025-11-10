import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
PLOT_DIR = "plots"
PLOT_FILENAME = os.path.join(PLOT_DIR, "fig_lattice_attack_complexity_log.png")

# Security parameter 'n' (x-axis)
n_values = np.array([512, 1024, 2048])

# Paper's constant for quantum attacks
eta = 0.265 

# Work = 2^(eta * n)
work = np.power(2.0, eta * n_values)

# --- Plotting ---
print("Plotting Lattice Attack Complexity (log scale)...")
os.makedirs(PLOT_DIR, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(n_values, work, 'b-s', linewidth=2, markersize=8, label=r"Quantum Attack Cost ($2^{0.265n}$)")

plt.title('Lattice Attack Complexity (Logarithmic Scale)', fontsize=16, fontweight='bold')
plt.xlabel('Security Parameter (n)', fontsize=15)
plt.ylabel('Attacker Work (log scale)', fontsize=15)
plt.yscale('log')
plt.xticks(n_values, fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=15, loc='best')

plt.tight_layout()
plt.savefig(PLOT_FILENAME, dpi=300)
print(f"Graph successfully saved to {PLOT_FILENAME}")
plt.show()
