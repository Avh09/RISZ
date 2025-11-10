import timeit
import os
import numpy as np
from numpy.polynomial import polynomial as p
import matplotlib.pyplot as plt

# --- Import only the hash/AES functions ---
# We will define the RLWE functions locally to change 'N'
from src.otaka_protocol.helper import (
    h, encrypt_data, decrypt_data, Q,
    get_timestamp, check_timestamp, # Needed for IV
    str_to_hex, xor_data # Needed for registration
)

# --- Benchmark Setup ---
N_RUNS = 1000 # 1000 is good, but use 100 if it's too slow
N_VALUES = [512, 1024, 2048] # The security parameters to test

# We must re-define the core RLWE functions here
# so they can use a variable 'N'

def gen_poly(N, POLY_MOD):
    poly = np.floor(np.random.normal(0, 2, size=(N))).astype(int)
    remainder = p.polydiv(poly, POLY_MOD)[1] % Q
    if len(remainder) < N:
        remainder = np.pad(remainder, (0, N - len(remainder)), 'constant')
    return remainder.astype(int)

def rlwe_compute_shared_secret(poly1, poly2, N, POLY_MOD):
    c = p.polymul(poly1, poly2) % Q
    c = p.polydiv(c, POLY_MOD)[1] % Q
    if len(c) < N:
        c = np.pad(c, (0, N - len(c)), 'constant')
    return c.astype(int)

def Cha(poly, N, Q_val):
    Q_4 = Q_val / 4
    Q_2 = Q_val / 2
    Q_3_4 = 3 * Q_val / 4
    cond1 = (poly >= Q_4) & (poly < Q_2)
    cond2 = (poly >= Q_3_4) & (poly <= Q_val)
    signal_bits = np.zeros(N, dtype=int)
    signal_bits[cond1] = 1
    signal_bits[cond2] = 1
    return signal_bits

def run_full_benchmark(N):
    """Runs the entire benchmark suite for a given N."""
    
    # Define N-dependent polynomial
    POLY_MOD = [1] + [0] * (N - 1) + [1] 
    
    # --- Setup data for benchmarks ---
    key = os.urandom(32).hex()
    iv = os.urandom(16).hex()[:32]
    plaintext = "test"
    ciphertext = encrypt_data(key, iv, plaintext)

    poly1 = gen_poly(N, POLY_MOD)
    poly2 = gen_poly(N, POLY_MOD)
    shared_secret_poly = rlwe_compute_shared_secret(poly1, poly2, N, POLY_MOD)
    scalar = 12345

    # --- Run Benchmarks ---
    results_ms = {}

    results_ms["Th"] = timeit.timeit(lambda: h("benchmark"), number=N_RUNS) / N_RUNS * 1000
    results_ms["Tsenc"] = timeit.timeit(lambda: encrypt_data(key, iv, plaintext), number=N_RUNS) / N_RUNS * 1000
    results_ms["Tsdec"] = timeit.timeit(lambda: decrypt_data(key, iv, ciphertext), number=N_RUNS) / N_RUNS * 1000
    
    # --- N-dependent benchmarks ---
    results_ms["Tg"] = timeit.timeit(lambda: gen_poly(N, POLY_MOD), number=N_RUNS) / N_RUNS * 1000
    results_ms["Tsm"] = timeit.timeit(lambda: (poly1 * scalar) % Q, number=N_RUNS) / N_RUNS * 1000
    results_ms["Tpm"] = timeit.timeit(lambda: rlwe_compute_shared_secret(poly1, poly2, N, POLY_MOD), number=N_RUNS) / N_RUNS * 1000
    results_ms["Tpa"] = timeit.timeit(lambda: p.polyadd(poly1, poly2) % Q, number=N_RUNS) / N_RUNS * 1000
    results_ms["Tcha"] = timeit.timeit(lambda: Cha(shared_secret_poly, N, Q), number=N_RUNS) / N_RUNS * 1000
    
    # --- Calculate Costs (from Table II) ---
    cost_ui = (6 * results_ms["Th"]) + (2 * results_ms["Tg"]) + results_ms["Tsm"] + (2 * results_ms["Tpm"]) + results_ms["Tpa"]
    cost_server = cost_ui + results_ms["Tcha"]
    
    # --- Theoretical Communication Cost (from Table I) ---
    # The paper states 4096 bits for n=1024 [cite: 507]. So, poly_size_bits = N * 4.
    poly_size_bits = N * 4 
    
    # M1 = {X1(256), X2(256), TIDi(160), ai(poly), s2(256), TS1(32)}
    M1_bits = 256 + 256 + 160 + poly_size_bits + 256 + 32
    # M2 = {SKVji(256), TS2(32), bj(poly), dj(N bits), TIDn*(256)}
    # Note: dj is a signal vector of N bits, not 1 bit as the paper's 
    # table calculation mistakenly implies [cite: 508].
    M2_bits = 256 + 32 + poly_size_bits + N + 256 
    # M3 = {ACK(256), TS3(32)}
    M3_bits = 256 + 32
    
    comm_cost = M1_bits + M2_bits + M3_bits

    return cost_ui, cost_server, comm_cost

# --- Main Execution ---
print("--- Starting Security Parameter (n) Analysis (Phase 4) ---")
print(f"Testing N values: {N_VALUES}. This will take several minutes...")

results_by_n = {
    "n": [],
    "ui_cost": [],
    "server_cost": [],
    "comm_cost": []
}

for n_val in N_VALUES:
    print(f"\nBenchmarking with N = {n_val}...")
    ui_cost, server_cost, comm_cost = run_full_benchmark(n_val)
    
    results_by_n["n"].append(n_val)
    results_by_n["ui_cost"].append(ui_cost)
    results_by_n["server_cost"].append(server_cost)
    results_by_n["comm_cost"].append(comm_cost)
    
    print(f"  N={n_val} | Ui Cost: {ui_cost:.4f} ms | Server Cost: {server_cost:.4f} ms | Comm. Cost: {comm_cost} bits")

print("\n--- Analysis Complete ---")

# --- Plotting Results ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
fig.suptitle('Security Parameter (n) vs. Performance Trade-off', fontsize=16)

# Plot 1: Computation Cost
ax1.plot(results_by_n["n"], results_by_n["ui_cost"], 'o-', label="Ui (Client) Cost")
ax1.plot(results_by_n["n"], results_by_n["server_cost"], 's-', label="MS (Server) Cost")
ax1.set_title("Computation Cost")
ax1.set_xlabel("Security Parameter (n)")
ax1.set_ylabel("Time (ms)")
ax1.set_xticks(N_VALUES)
ax1.legend()
ax1.grid(True)

# Plot 2: Communication Cost
ax2.plot(results_by_n["n"], results_by_n["comm_cost"], 'd-r', label="Total Communication Cost")
ax2.set_title("Communication Cost")
ax2.set_xlabel("Security Parameter (n)")
ax2.set_ylabel("Total Size (bits)")
ax2.set_xticks(N_VALUES)
ax2.legend()
ax2.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- Save the plot ---
PLOT_DIR = "plots"
PLOT_FILENAME = os.path.join(PLOT_DIR, "fig_n_vs_performance.png")
os.makedirs(PLOT_DIR, exist_ok=True)
plt.savefig(PLOT_FILENAME)
print(f"\nGraph successfully saved to {PLOT_FILENAME}")
plt.show()

