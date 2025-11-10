import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)
PLOT_FILENAME = os.path.join(PLOT_DIR, "fig_exponential_visible.png")

# Parameters
n_values = np.linspace(100, 5000, 100)

# Scale down exponent growth — visually still exponential, just not astronomically large
eta = 0.005  
work = np.power(2.0, eta * n_values)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(n_values, work, 'r-', linewidth=2.5, label=r"Visible Exponential ($2^{0.005n}$)")
plt.fill_between(n_values, work, color='red', alpha=0.15)

plt.title('Visible Exponential Growth (Scaled)', fontsize=16, fontweight='bold')
plt.xlabel('Security Parameter (n)', fontsize=15)
plt.ylabel('Relative Attacker Work (arbitrary units)', fontsize=15)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=15)
plt.tight_layout()

plt.savefig(PLOT_FILENAME, dpi=300)
plt.show()
