import timeit
# We must import the helper functions from the src package
from src.otaka_protocol.helper import (
    h, rlwe_sample_from_chi_delta, rlwe_generate_public_key,
    rlwe_compute_shared_values, Mod2, Cha
)

# [cite_start]Number of times to run each operation for averaging [cite: 493]
N_RUNS = 1000

def benchmark_all():
    print(f"Reproducing Table II: Computation Cost Analysis (Running {N_RUNS} iterations each)...")
    
    # --- Setup for operations ---
    # We need realistic inputs for the functions
    f1 = rlwe_sample_from_chi_delta()
    e1 = rlwe_sample_from_chi_delta()
    ai = rlwe_generate_public_key(f1, e1)
    
    f2 = rlwe_sample_from_chi_delta()
    bj = rlwe_generate_public_key(f2, e1) # Re-use e1 for simplicity
    
    cj = rlwe_compute_shared_values(f2, ai)
    dj = Cha(cj)
    
    # --- 1. Benchmark Hash (Th) ---
    t_h = timeit.timeit(
        lambda: h("ID", "timestamp", "TID", "key"), 
        number=N_RUNS
    ) / N_RUNS * 1000 # Convert to ms
    
    # --- 2. Benchmark RLWE: Sample (Tg) ---
    t_g = timeit.timeit(
        lambda: rlwe_sample_from_chi_delta(), 
        number=N_RUNS
    ) / N_RUNS * 1000 # Convert to ms
    
    # --- 3. Benchmark RLWE: KeyGen (Tpa) ---
    # This is "polynomial multiplication addition"
    # ai = α*f1 + 2*e1
    t_pa = timeit.timeit(
        lambda: rlwe_generate_public_key(f1, e1), 
        number=N_RUNS
    ) / N_RUNS * 1000 # Convert to ms
    
    # --- 4. Benchmark RLWE: Shared (Tpm) ---
    # This is "component-wise polynomial multiplication"
    # cj = ai * f2
    t_pm = timeit.timeit(
        lambda: rlwe_compute_shared_values(f2, ai),
        number=N_RUNS
    ) / N_RUNS * 1000 # Convert to ms
    
    # --- 5. Benchmark RLWE: Cha (Tcha) ---
    t_cha = timeit.timeit(
        lambda: Cha(cj),
        number=N_RUNS
    ) / N_RUNS * 1000 # Convert to ms
    
    # --- 6. Benchmark RLWE: Mod2 (Tsm?) ---
    # The paper uses Tsm for "multiplication with scalar"
    # and Mod2 is not explicitly benchmarked, but let's time it.
    # We'll use t_sm for Mod2 as it's the other main key-gen part.
    t_sm = timeit.timeit(
        lambda: Mod2(cj, dj),
        number=N_RUNS
    ) / N_RUNS * 1000 # Convert to ms

    print("\n--- Benchmark Results (avg in ms) ---")
    print(f"Th   (Hash)           : {t_h:.6f} ms")
    print(f"Tg   (Sample)         : {t_g:.6f} ms")
    print(f"Tpa  (KeyGen/PolyAdd) : {t_pa:.6f} ms")
    print(f"Tpm  (PolyMultiply)   : {t_pm:.6f} ms")
    print(f"Tcha (Characteristic) : {t_cha:.6f} ms")
    print(f"Tsm  (Mod2/Scalar?)   : {t_sm:.6f} ms")
    
    print("\n--- HPostQCA-VSS Cost Calculation (from Table II) ---")
    
    # Ui / Smart device cost = 6Th + 2Tg + Tsm + 2Tpm + Tpa
    cost_ui = (6 * t_h) + (2 * t_g) + t_sm + (2 * t_pm) + t_pa
    
    # Server cost = 6Th + 2Tg + 2Tpm + Tpa + Tsm + Tcha
    cost_server = (6 * t_h) + (2 * t_g) + (2 * t_pm) + t_pa + t_sm + t_cha
    
    print(f"Ui / Smart Device Cost: {cost_ui:.6f} ms")
    print(f"Server Cost           : {cost_server:.6f} ms")
    
    print("\nNote: Your times will differ from the paper based on your CPU.")
    print("The RLWE dummy functions are also likely much faster than a real implementation.")

if __name__ == "__main__":
    benchmark_all()