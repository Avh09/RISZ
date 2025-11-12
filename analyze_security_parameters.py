import timeit
import os
import numpy as np
from numpy.polynomial import polynomial as p
import matplotlib.pyplot as plt

from src.otaka_protocol.helper import h, encrypt_data, decrypt_data, Q

N_RUNS = 1000 
N_VALUES = [512, 1024, 2048]

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

def Cha(poly, N):
    Q_4 = Q / 4
    Q_2 = Q / 2
    Q_3_4 = 3 * Q / 4
    cond1 = (poly >= Q_4) & (poly < Q_2)
    cond2 = (poly >= Q_3_4) & (poly <= Q)
    signal_bits = np.zeros(N, dtype=int)
    signal_bits[cond1] = 1
    signal_bits[cond2] = 1
    return signal_bits

def run_full_benchmark(N):
    """Runs the entire benchmark suite for a given N."""

    POLY_MOD = [1] + [0] * (N - 1) + [1] 
    key = os.urandom(32).hex()
    iv = os.urandom(16).hex()[:32]
    plaintext = "test"
    ciphertext = encrypt_data(key, iv, plaintext)

    poly1 = gen_poly(N, POLY_MOD)
    poly2 = gen_poly(N, POLY_MOD)
    shared_secret_poly = rlwe_compute_shared_secret(poly1, poly2, N, POLY_MOD)
    scalar = 12345

    results_ms = {}

    results_ms["Th"] = timeit.timeit(lambda: h("benchmark"), number=N_RUNS) / N_RUNS * 1000
    results_ms["Tsenc"] = timeit.timeit(lambda: encrypt_data(key, iv, plaintext), number=N_RUNS) / N_RUNS * 1000
    results_ms["Tsdec"] = timeit.timeit(lambda: decrypt_data(key, iv, ciphertext), number=N_RUNS) / N_RUNS * 1000
    

    results_ms["Tg"] = timeit.timeit(lambda: gen_poly(N, POLY_MOD), number=N_RUNS) / N_RUNS * 1000
    results_ms["Tsm"] = timeit.timeit(lambda: (poly1 * scalar) % Q, number=N_RUNS) / N_RUNS * 1000
    results_ms["Tpm"] = timeit.timeit(lambda: rlwe_compute_shared_secret(poly1, poly2, N, POLY_MOD), number=N_RUNS) / N_RUNS * 1000
    results_ms["Tpa"] = timeit.timeit(lambda: p.polyadd(poly1, poly2) % Q, number=N_RUNS) / N_RUNS * 1000
    results_ms["Tcha"] = timeit.timeit(lambda: Cha(shared_secret_poly, N), number=N_RUNS) / N_RUNS * 1000

    cost_ui = (6 * results_ms["Th"]) + (2 * results_ms["Tg"]) + results_ms["Tsm"] + (2 * results_ms["Tpm"]) + results_ms["Tpa"]
    cost_server = cost_ui + results_ms["Tcha"]
    

    poly_size_bits = N * 4 
    
    M1_bits = 256 + 256 + 160 + poly_size_bits + 256 + 32
    M2_bits = 256 + 32 + poly_size_bits + N + 256 # d_j is N bits, not 1
    M3_bits = 256 + 32
    comm_cost = M1_bits + M2_bits + M3_bits

    return cost_ui, cost_server, comm_cost

print("--- Starting Security Parameter Analysis (Phase 4) ---")
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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
fig.suptitle('Security Parameter (n) vs. Performance Trade-off', fontsize=16)

ax1.plot(results_by_n["n"], results_by_n["ui_cost"], 'o-', label="Ui (Client) Cost")
ax1.plot(results_by_n["n"], results_by_n["server_cost"], 's-', label="MS (Server) Cost")
ax1.set_title("Computation Cost")
ax1.set_xlabel("Security Parameter (n)")
ax1.set_ylabel("Time (ms)")
ax1.set_xticks(N_VALUES)
ax1.legend()
ax1.grid(True)
ax2.plot(results_by_n["n"], results_by_n["comm_cost"], 'd-r', label="Total Communication Cost")
ax2.set_title("Communication Cost")
ax2.set_xlabel("Security Parameter (n)")
ax2.set_ylabel("Total Size (bits)")
ax2.set_xticks(N_VALUES)
ax2.legend()
ax2.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()