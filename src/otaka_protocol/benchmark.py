# benchmark_primitives.py
import time
import numpy as np
from .helper import h, rlwe_generate_keypair, rlwe_compute_shared_secret, Cha, Mod2, gen_poly,canonical_hash

# Number of iterations
N_RUNS = 1000
results = {}

# --- One-way hash ---
start = time.perf_counter()
for _ in range(N_RUNS):
    canonical_hash("benchmark", _)
results["Th"] = (time.perf_counter() - start) / N_RUNS * 1000  # ms

# --- RLWE sampling from χδ (Tg) ---
start = time.perf_counter()
for _ in range(N_RUNS):
    gen_poly()
results["Tg"] = (time.perf_counter() - start) / N_RUNS * 1000

# --- Polynomial multiplication with scalar (Tsm) ---
a = gen_poly(); s = gen_poly()
start = time.perf_counter()
for _ in range(N_RUNS):
    _ = (a * 3) % 1073479681
results["Tsm"] = (time.perf_counter() - start) / N_RUNS * 1000

# --- Polynomial multiplication in Rq (Tpm) ---
a, b = gen_poly(), gen_poly()
start = time.perf_counter()
for _ in range(N_RUNS):
    rlwe_compute_shared_secret(a, b)
results["Tpm"] = (time.perf_counter() - start) / N_RUNS * 1000

# --- Polynomial addition (Tpa) ---
a, b = gen_poly(), gen_poly()
start = time.perf_counter()
for _ in range(N_RUNS):
    _ = (a + b) % 1073479681
results["Tpa"] = (time.perf_counter() - start) / N_RUNS * 1000

# --- Characteristic function (Tcha) ---
c = rlwe_compute_shared_secret(a, b)
start = time.perf_counter()
for _ in range(N_RUNS):
    Cha(c)
results["Tcha"] = (time.perf_counter() - start) / N_RUNS * 1000

results = {k: v / results["Th"] for k, v in results.items()}
print(results)


# --- Print results ---
print("\n=== Average Primitive Timings (ms) ===")
for k, v in results.items():
    print(f"{k:6s}: {v:.6f} ms")
    
# --- Compute total authentication cost ---
Ui_cost = 6*results["Th"] + 2*results["Tg"] + results["Tsm"] + 2*results["Tpm"] + results["Tpa"]
MS_cost = 6*results["Th"] + 2*results["Tg"] + results["Tsm"] + 2*results["Tpm"] + results["Tpa"] + results["Tcha"]

print(f"\nEstimated Ui computational cost: {Ui_cost:.4f} ms")
print(f"Estimated MS computational cost: {MS_cost:.4f} ms")
