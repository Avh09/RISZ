import timeit
from src.otaka_protocol.helper import (
    h, rlwe_sample_from_chi_delta, rlwe_generate_public_key,
    rlwe_compute_shared_values, Mod2, Cha
)
N_RUNS = 1000

def benchmark_all():
    print(f"Reproducing Table II: Computation Cost Analysis (Running {N_RUNS} iterations each)...")

    f1 = rlwe_sample_from_chi_delta()
    e1 = rlwe_sample_from_chi_delta()
    ai = rlwe_generate_public_key(f1, e1)
    
    f2 = rlwe_sample_from_chi_delta()
    bj = rlwe_generate_public_key(f2, e1) 
    
    cj = rlwe_compute_shared_values(f2, ai)
    dj = Cha(cj)
    
    t_h = timeit.timeit(
        lambda: h("ID", "timestamp", "TID", "key"), 
        number=N_RUNS
    ) / N_RUNS * 1000 
    t_g = timeit.timeit(
        lambda: rlwe_sample_from_chi_delta(), 
        number=N_RUNS
    ) / N_RUNS * 1000 

    t_pa = timeit.timeit(
        lambda: rlwe_generate_public_key(f1, e1), 
        number=N_RUNS
    ) / N_RUNS * 1000 
    t_pm = timeit.timeit(
        lambda: rlwe_compute_shared_values(f2, ai),
        number=N_RUNS
    ) / N_RUNS * 1000
    
    t_cha = timeit.timeit(
        lambda: Cha(cj),
        number=N_RUNS
    ) / N_RUNS * 1000 

    t_sm = timeit.timeit(
        lambda: Mod2(cj, dj),
        number=N_RUNS
    ) / N_RUNS * 1000 

    print("\n--- Benchmark Results (avg in ms) ---")
    print(f"Th   (Hash)           : {t_h:.6f} ms")
    print(f"Tg   (Sample)         : {t_g:.6f} ms")
    print(f"Tpa  (KeyGen/PolyAdd) : {t_pa:.6f} ms")
    print(f"Tpm  (PolyMultiply)   : {t_pm:.6f} ms")
    print(f"Tcha (Characteristic) : {t_cha:.6f} ms")
    print(f"Tsm  (Mod2/Scalar?)   : {t_sm:.6f} ms")
    
    print("\n--- HPostQCA-VSS Cost Calculation (from Table II) ---")

    cost_ui = (6 * t_h) + (2 * t_g) + t_sm + (2 * t_pm) + t_pa
 
    cost_server = (6 * t_h) + (2 * t_g) + (2 * t_pm) + t_pa + t_sm + t_cha
    
    print(f"Ui / Smart Device Cost: {cost_ui:.6f} ms")
    print(f"Server Cost           : {cost_server:.6f} ms")
   

if __name__ == "__main__":
    benchmark_all()