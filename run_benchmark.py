import timeit
import os
import numpy as np
from numpy.polynomial import polynomial as p
import json

# --- Import all the crypto functions from your helper module ---
from src.otaka_protocol.helper import (
    h, encrypt_data, decrypt_data,
    gen_poly, rlwe_generate_keypair, 
    rlwe_compute_shared_secret, Cha, Q, POLY_MOD
)
# --- NEW: Import cryptography for ECC benchmarks ---
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes

# --- Benchmark Setup ---
N_RUNS = 1000
print(f"--- Running Full Benchmark ({N_RUNS} iterations each) ---")
print("This may take several minutes (ECC is slow)...")

# --- Setup data for benchmarks ---
key = os.urandom(32).hex()
iv = os.urandom(16).hex()[:32] # Use 16-byte/32-hex IV
plaintext = "This is a test message for AES."
ciphertext = encrypt_data(key, iv, plaintext)

poly1 = gen_poly()
poly2 = gen_poly()
(priv_key, _), pub_key = rlwe_generate_keypair()
shared_secret_poly = rlwe_compute_shared_secret(priv_key, pub_key)
scalar = 12345

# --- NEW: Setup for ECC benchmarks ---
# We use secp256r1 as mentioned in the paper
ecc_private_key = ec.generate_private_key(ec.SECP256R1())
ecc_public_key = ecc_private_key.public_key()
ecc_peer_public_key = ec.generate_private_key(ec.SECP256R1()).public_key()
ecc_data_to_sign = b"test data for signing"

# --- Run Benchmarks ---
results_ms = {}

# 1. Hash
results_ms["Th"] = timeit.timeit(
    lambda: h("benchmark", "data"), 
    number=N_RUNS
) / N_RUNS * 1000 # Convert to ms

# 2. AES
results_ms["Tsenc"] = timeit.timeit(
    lambda: encrypt_data(key, iv, plaintext), 
    number=N_RUNS
) / N_RUNS * 1000
results_ms["Tsdec"] = timeit.timeit(
    lambda: decrypt_data(key, iv, ciphertext), 
    number=N_RUNS
) / N_RUNS * 1000

# 3. RLWE (Lattice)
results_ms["Tg"] = timeit.timeit(lambda: gen_poly(), number=N_RUNS) / N_RUNS * 1000
results_ms["Tsm"] = timeit.timeit(lambda: (poly1 * scalar) % Q, number=N_RUNS) / N_RUNS * 1000
results_ms["Tpm"] = timeit.timeit(lambda: rlwe_compute_shared_secret(poly1, poly2), number=N_RUNS) / N_RUNS * 1000
results_ms["Tpa"] = timeit.timeit(lambda: p.polyadd(poly1, poly2) % Q, number=N_RUNS) / N_RUNS * 1000
results_ms["Tcha"] = timeit.timeit(lambda: Cha(shared_secret_poly), number=N_RUNS) / N_RUNS * 1000

# 4. ECC (Elliptic Curve)
# Tecm: Elliptic Curve Multiplication (Key Exchange is a good proxy)
results_ms["Tecm"] = timeit.timeit(
    lambda: ecc_private_key.exchange(ec.ECDH(), ecc_peer_public_key),
    number=N_RUNS
) / N_RUNS * 1000

# Teca: Elliptic Curve "Addition" (Signing is a good proxy for a lighter ECC op)
results_ms["Teca"] = timeit.timeit(
    lambda: ecc_private_key.sign(ecc_data_to_sign, ec.ECDSA(hashes.SHA256())),
    number=N_RUNS
) / N_RUNS * 1000

# --- Save results to JSON file ---
RESULTS_FILE = "benchmark_results.json"
with open(RESULTS_FILE, 'w') as f:
    json.dump(results_ms, f, indent=4)

print(f"\nBenchmark complete. Results saved to {RESULTS_FILE}")

# --- Format and Print Results ---
print("\n--- Benchmark Results (Absolute Time) ---")
print("+-----------+---------------+--")
print("| Operation | Time (ms)     |")
print("+-----------+---------------+--")
for op, time_ms in results_ms.items():
    print(f"| {op:<9} | {time_ms:<13.6f} |")
print("+-----------+---------------+--")

# --- Cost Calculation (from Table II) ---
print("\n--- Estimated Protocol Cost (Absolute Time) ---")
cost_ui = (6 * results_ms["Th"]) + (2 * results_ms["Tg"]) + results_ms["Tsm"] + (2 * results_ms["Tpm"]) + results_ms["Tpa"]
cost_server = cost_ui + results_ms["Tcha"]
print(f"Estimated Ui (Client) computational cost: {cost_ui:.4f} ms")
print(f"Estimated MS (Server) computational cost: {cost_server:.4f} ms")