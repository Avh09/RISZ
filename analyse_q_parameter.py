import timeit
import os
import numpy as np
from numpy.polynomial import polynomial as p
import matplotlib.pyplot as plt

# --- Import only the hash/AES functions ---
from src.otaka_protocol.helper import (
    h, encrypt_data, decrypt_data,
    gen_poly, rlwe_compute_shared_secret, Cha, POLY_MOD
)

# --- Benchmark Setup ---
N_RUNS = 1000
N = 1024 # We keep N fixed at 1024, as per the paper

# --- NEW: Define Q values to test ---
# We will test a 28-bit, 30-bit (paper's), and 32-bit prime
Q_VALUES = {
    "28-bit": 268435399,
    "30-bit (Paper)": 1073479681,
    "32-bit": 4294967291,
}

# We must re-define the core RLWE functions here
# so they can use a variable 'Q'

def run_full_benchmark(Q):
    """Runs the entire benchmark suite for a given Q."""
    
    # --- Setup data for benchmarks ---
    key = os.urandom(32).hex()
    iv = os.urandom(16).hex()[:32]
    plaintext = "test"
    ciphertext = encrypt_data(key, iv, plaintext)

    poly1 = gen_poly()
    poly2 = gen_poly()
    # We must redefine poly-math functions to use the new Q
    
    def poly_add(a, b, mod_q):
        return p.polyadd(a, b) % mod_q

    def poly_mul_scalar(a, s, mod_q):
        return (a * s) % mod_q
        
    def poly_mul(a, b, mod_q):
        c = p.polymul(a, b) % mod_q
        c = p.polydiv(c, POLY_MOD)[1] % mod_q
        if len(c) < N:
            c = np.pad(c, (0, N - len(c)), 'constant')
        return c.astype(int)
        
    def char_func(poly, mod_q):
        Q_4 = mod_q / 4
        Q_2 = mod_q / 2
        Q_3_4 = 3 * mod_q / 4
        cond1 = (poly >= Q_4) & (poly < Q_2)
        cond2 = (poly >= Q_3_4) & (poly <= mod_q)
        signal_bits = np.zeros(N, dtype=int)
        signal_bits[cond1] = 1
        signal_bits[cond2] = 1
        return signal_bits

    shared_secret_poly = poly_mul(poly1, poly2, Q)
    scalar = 12345

    # --- Run Benchmarks ---
    results_ms = {}
    
    # Q-independent benchmarks
    results_ms["Th"] = timeit.timeit(lambda: h("benchmark"), number=N_RUNS) / N_RUNS * 1000
    results_ms["Tg"] = timeit.timeit(lambda: gen_poly(), number=N_RUNS) / N_RUNS * 1000
    results_ms["Tsenc"] = timeit.timeit(lambda: encrypt_data(key, iv, plaintext), number=N_RUNS) / N_RUNS * 1000
    results_ms["Tsdec"] = timeit.timeit(lambda: decrypt_data(key, iv, ciphertext), number=N_RUNS) / N_RUNS * 1000
    
    # --- Q-dependent benchmarks ---
    results_ms["Tsm"] = timeit.timeit(lambda: poly_mul_scalar(poly1, scalar, Q), number=N_RUNS) / N_RUNS * 1000
    results_ms["Tpm"] = timeit.timeit(lambda: poly_mul(poly1, poly2, Q), number=N_RUNS) / N_RUNS * 1000
    results_ms["Tpa"] = timeit.timeit(lambda: poly_add(poly1, poly2, Q), number=N_RUNS) / N_RUNS * 1000
    results_ms["Tcha"] = timeit.timeit(lambda: char_func(shared_secret_poly, Q), number=N_RUNS) / N_RUNS * 1000
    
    # --- Calculate Costs ---
    cost_ui = (6 * results_ms["Th"]) + (2 * results_ms["Tg"]) + results_ms["Tsm"] + (2 * results_ms["Tpm"]) + results_ms["Tpa"]
    cost_server = cost_ui + results_ms["Tcha"]
    
    return cost_ui, cost_server

# --- Main Execution ---
print("--- Starting Security Parameter Analysis (Phase 4) ---")
print(f"Testing Q values: {Q_VALUES.keys()}. This will take several minutes...")

results_by_q = {
    "q_label": [],
    "ui_cost": [],
    "server_cost": [],
}

for q_label, q_val in Q_VALUES.items():
    print(f"\nBenchmarking with Q = {q_label}...")
    ui_cost, server_cost = run_full_benchmark(q_val)
    
    results_by_q["q_label"].append(q_label)
    results_by_q["ui_cost"].append(ui_cost)
    results_by_q["server_cost"].append(server_cost)
    
    print(f"  Q={q_label} | Ui Cost: {ui_cost:.4f} ms | Server Cost: {server_cost:.4f} ms")

print("\n--- Analysis Complete ---")

# --- Plotting Results ---
# --- Plotting Results ---
plt.figure(figsize=(10, 6))

# Bar chart (stacked)
plt.bar(results_by_q["q_label"], results_by_q["ui_cost"], label="Ui (Client) Cost")
plt.bar(results_by_q["q_label"], results_by_q["server_cost"],
        bottom=results_by_q["ui_cost"], label="MS (Server) Cost (Stacked)")

# Title and labels with larger font sizes
plt.title('Security Parameter (q) vs. Computation Cost (N=1024)', fontsize=16, fontweight='bold')
plt.xlabel("Security Parameter (q) - Prime Modulus Bit-Size", fontsize=14)
plt.ylabel("Total Time (ms)", fontsize=14)

# Increase tick label font sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(fontsize=15)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("plots/security_parameter_vs_computation_cost.png", dpi=300, bbox_inches='tight')
print("✅ Plot saved to plots/security_parameter_vs_computation_cost.png")

plt.show()
