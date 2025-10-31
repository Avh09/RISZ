import timeit
import os
import numpy as np
from numpy.polynomial import polynomial as p

# --- Import all the crypto functions from your helper module ---
from src.otaka_protocol.helper import (
    h, encrypt_data, decrypt_data,
    gen_poly, rlwe_generate_keypair, 
    rlwe_compute_shared_secret, Cha, Q, POLY_MOD
)

# --- Benchmark Setup ---
N_RUNS = 1000
print(f"--- Benchmarking Primitives ({N_RUNS} iterations each) ---")
print("This may take a moment...")

# --- Setup data for benchmarks ---
key = os.urandom(32).hex()
iv = os.urandom(16).hex()[:32] # Use 16-byte/32-hex IV
plaintext = "This is a test message for AES."
ciphertext = encrypt_data(key, iv, plaintext)

poly1 = gen_poly()
poly2 = gen_poly()
(priv_key, _), pub_key = rlwe_generate_keypair()
shared_secret_poly = rlwe_compute_shared_secret(priv_key, pub_key)
signal_bits = Cha(shared_secret_poly)
scalar = 12345

# --- Run Benchmarks ---
results_ms = {}

results_ms["Th"] = timeit.timeit(
    lambda: h("benchmark", "data"), 
    number=N_RUNS
) / N_RUNS * 1000 # Convert to ms

results_ms["Tsenc"] = timeit.timeit(
    lambda: encrypt_data(key, iv, plaintext), 
    number=N_RUNS
) / N_RUNS * 1000

results_ms["Tsdec"] = timeit.timeit(
    lambda: decrypt_data(key, iv, ciphertext), 
    number=N_RUNS
) / N_RUNS * 1000

results_ms["Tg"] = timeit.timeit(
    lambda: gen_poly(), 
    number=N_RUNS
) / N_RUNS * 1000

results_ms["Tsm"] = timeit.timeit(
    lambda: (poly1 * scalar) % Q, 
    number=N_RUNS
) / N_RUNS * 1000

results_ms["Tpm"] = timeit.timeit(
    lambda: rlwe_compute_shared_secret(poly1, poly2), 
    number=N_RUNS
) / N_RUNS * 1000

results_ms["Tpa"] = timeit.timeit(
    lambda: p.polyadd(poly1, poly2) % Q, 
    number=N_RUNS
) / N_RUNS * 1000

results_ms["Tcha"] = timeit.timeit(
    lambda: Cha(shared_secret_poly), 
    number=N_RUNS
) / N_RUNS * 1000


# --- Format and Print Results ---
print("\n--- Benchmark Results (Absolute Time) ---")
print("+-----------+---------------+--")
print("| Operation | Time (ms)     |")
print("+-----------+---------------+--")

for op, time_ms in results_ms.items():
    print(f"| {op:<9} | {time_ms:<13.6f} |")
print("+-----------+---------------+--")

# --- NEW: Normalize by Tpm ---
print("\n--- Normalized Cost (Relative to Tpm) ---")
print("This shows the true relative cost of each operation.")
print("+-----------+-------------------+--")
print("| Operation | Cost (Tpm = 1.0)  |")
print("+-----------+-------------------+--")

# Get the bottleneck time
bottleneck = results_ms["Tpm"]
results_ratio = {}

for op, time_ms in results_ms.items():
    ratio = time_ms / bottleneck
    results_ratio[op] = ratio
    print(f"| {op:<9} | {ratio:<17.6f} |")
print("+-----------+-------------------+--")


# --- Cost Calculation (from Table II) ---
print("\n--- Estimated Protocol Cost (Absolute Time) ---")
cost_ui = (6 * results_ms["Th"]) + (2 * results_ms["Tg"]) + results_ms["Tsm"] + (2 * results_ms["Tpm"]) + results_ms["Tpa"]
cost_server = cost_ui + results_ms["Tcha"]
print(f"Estimated Ui (Client) computational cost: {cost_ui:.4f} ms")
print(f"Estimated MS (Server) computational cost: {cost_server:.4f} ms")