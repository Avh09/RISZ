import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
# These are the final costs for a single user,
# taken directly from the paper's Table II 
# (Scheme: (Ui_Cost, Server_Cost))
PAPER_COSTS = {
    'Ayub et al.': (5.9089, 1.1567),
    'Irshad et al.': (27.2349, 1.9132),
    'Mishra et al.': (9.7285, 0.2716),
    'Rewal et al.': (7.9417, 0.1327),
    'Huang et al.': (13.5767, 7.2146),
    'Hu et al.': (8.2681, 1.6577),
    'HPostQCA-VSS (Paper)': (4.1741, 0.4682),
}

# --- YOUR BENCHMARK RESULTS ---
# !! Manually enter your results from run_benchmark.py here !!
# Example: MY_UI_COST = 3.1292
# Example: MY_SERVER_COST = 3.1368
MY_UI_COST = 10.9028  # <-- REPLACE with your value
MY_SERVER_COST = 10.9288 # <-- REPLACE with your value

# --- Output folder ---
PLOT_DIR = "plots"
PLOT_FILENAME = os.path.join(PLOT_DIR, "fig06_computation_cost_reproduction.png")
# -----------------------

def generate_plot():
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # --- X-axis (Number of Users) ---
    # We use the same x-axis as the paper's graph [cite: 571-575, 624-625]
    user_counts = np.array([10, 100, 200, 300, 400, 500])

    # --- Plotting ---
    plt.figure(figsize=(10, 6.5))

    # Plot the lines from the paper
    styles = ['k--*', 'g--X', 'y-.P', 'C1--v', 'c-.D', 'b--^', 'r-o']
    for (name, (ui_cost, server_cost)), style in zip(PAPER_COSTS.items(), styles):
        total_cost_per_user = ui_cost + server_cost
        total_costs = user_counts * total_cost_per_user
        plt.plot(user_counts, total_costs, style, label=name)

    # Plot your implementation's line
    my_total_cost_per_user = MY_UI_COST + MY_SERVER_COST
    my_total_costs = user_counts * my_total_cost_per_user
    # Use a thick, solid, starred line to make it stand out
    plt.plot(user_counts, my_total_costs, 'm-*', label="HPostQCA-VSS (My Code)", linewidth=2.5, markersize=8)


    # --- Formatting ---
    plt.title('Computational Costs vs. Number of Smart Devices')
    plt.xlabel('Number of Us/Smart devices')
    plt.ylabel('Computational costs (in ms)')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.xlim(0, 510)
    plt.ylim(0, 14000) # Set Y-limit to match paper's graph
    plt.xticks([10, 100, 200, 300, 400, 500])
    
    # --- Save the plot ---
    plt.savefig(PLOT_FILENAME)
    print(f"\nGraph successfully saved to {PLOT_FILENAME}")
    plt.show()

if __name__ == "__main__":
    # print("Generating Figure 6 reproduction...")
    # print("NOTE: Please update 'MY_UI_COST' and 'MY_SERVER_COST' in this script")
    # print("      with the values from your 'run_benchmark.py' output.")
    generate_plot()