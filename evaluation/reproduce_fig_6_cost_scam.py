import matplotlib.pyplot as plt
import numpy as np
import os

PAPER_COSTS = {
    'Ayub et al.': (5.9089, 1.1567),
    'Irshad et al.': (27.2349, 1.9132),
    'Mishra et al.': (9.7285, 0.2716),
    'Rewal et al.': (7.9417, 0.1327),
    'Huang et al.': (13.5767, 7.2146),
    'Hu et al.': (8.2681, 1.6577),
    'HPostQCA-VSS (Paper)': (4.1741, 0.4682),
}

MY_UI_COST = 10.9028  
MY_SERVER_COST = 10.9288 

# --- Output folder ---
PLOT_DIR = "plots"
PLOT_FILENAME = os.path.join(PLOT_DIR, "fig06_computation_cost_reproduction.png")
# -----------------------

def generate_plot():
    os.makedirs(PLOT_DIR, exist_ok=True)
    user_counts = np.array([10, 100, 200, 300, 400, 500])

    plt.figure(figsize=(10, 6.5))
    styles = ['k--*', 'g--X', 'y-.P', 'C1--v', 'c-.D', 'b--^', 'r-o']
    for (name, (ui_cost, server_cost)), style in zip(PAPER_COSTS.items(), styles):
        total_cost_per_user = ui_cost + server_cost
        total_costs = user_counts * total_cost_per_user
        plt.plot(user_counts, total_costs, style, label=name)

    my_total_cost_per_user = MY_UI_COST + MY_SERVER_COST
    my_total_costs = user_counts * my_total_cost_per_user

    plt.plot(user_counts, my_total_costs, 'm-*', label="HPostQCA-VSS (My Code)", linewidth=2.5, markersize=8)

    plt.title('Computational Costs vs. Number of Smart Devices')
    plt.xlabel('Number of Us/Smart devices')
    plt.ylabel('Computational costs (in ms)')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.xlim(0, 510)
    plt.ylim(0, 14000)
    plt.xticks([10, 100, 200, 300, 400, 500])

    plt.savefig(PLOT_FILENAME)
    print(f"\nGraph successfully saved to {PLOT_FILENAME}")
    plt.show()

if __name__ == "__main__":

    generate_plot()