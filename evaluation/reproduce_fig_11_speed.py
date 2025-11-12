import requests
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random

# --- Configuration ---
VSS_API_URL = "http://127.0.0.1:8000/check_similarity"
BATCH_SIZES = [10, 100, 500, 1000, 1500, 2000] 
TOTAL_QUERIES = 14316
DATASET_PATH = "features_extracted.csv"
FEATURE_COLUMNS = [
    'strokeDuration', 'startX', 'startY', 'stopX', 'stopY',
    'directEndToEndDistance', 'meanResultantLength', 'upDownLeftRightFlag',
    'directionOfEndToEndLine', 'largestDeviationFromEndToEndLine',
    'averageDirection', 'lengthOfTrajectory', 'averageVelocity',
    'midStrokePressure', 'midStrokeArea'
]
MAX_BATCH = max(BATCH_SIZES)
N_REPEATS = 3 

PLOT_DIR = "plots"
PLOT_FILENAME = os.path.join(PLOT_DIR, "fig11_speed_reproduction.png")

def load_original_vectors():
    """Loads a sample of 'original' vectors from the dataset."""
    print(f"Loading {MAX_BATCH} original vectors from dataset...")
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"FATAL: {DATASET_PATH} not found.")
        return []
    
    df['upDownLeftRightFlag'], _ = pd.factorize(df['upDownLeftRightFlag'])

    return df[FEATURE_COLUMNS].sample(MAX_BATCH, replace=True).values.tolist()

def generate_shuffled_vectors():
    """Generates 'shuffled' (fake) vectors."""
    print(f"Generating {MAX_BATCH} random shuffled vectors...")
    return (np.random.rand(MAX_BATCH, 15) * 1000).tolist()

def run_speed_test():
    print("Reproducing Figure 11: VSS Query Speed Analysis...")
    os.makedirs(PLOT_DIR, exist_ok=True)
    original_vectors = load_original_vectors()
    shuffled_vectors = generate_shuffled_vectors()

    if not original_vectors:
        return 
    
    print("\n--- Warming up the VSS server (10 queries)... ---")
    for vector in original_vectors[:10]:
        try:
            requests.post(VSS_API_URL, json={"vector": vector})
        except Exception:
            pass 
    print("Server is warm.")

    original_times = []
    shuffled_times = []

    for query_type, vectors_to_test, times_list in [
        ("Original", original_vectors, original_times),
        ("Shuffled", shuffled_vectors, shuffled_times)
    ]:
        print(f"\n--- Testing '{query_type}' vectors ---")
        
        for batch_size in BATCH_SIZES:
            print(f"  Testing batch size: {batch_size} (running {N_REPEATS} times to average)...")
            batch_to_test = vectors_to_test[:batch_size]
            batch_times = []
            for i in range(N_REPEATS):
                start_time = time.time()
                for vector in batch_to_test:
                    try:
                        requests.post(VSS_API_URL, json={"vector": vector})
                    except Exception as e:
                        pass
                end_time = time.time()
                batch_times.append(end_time - start_time)

            time_per_batch = np.mean(batch_times)
            
            avg_time_per_query = time_per_batch / batch_size
            total_time_for_all = avg_time_per_query * TOTAL_QUERIES
            times_list.append(total_time_for_all)
            
            print(f"    Avg time per query: {avg_time_per_query*1000:.4f} ms")
            print(f"    Extrapolated total time: {total_time_for_all:.2f} s")

    plt.figure(figsize=(10, 6))
    plt.plot(BATCH_SIZES, original_times, marker='o', label="Original Vector")
    plt.plot(BATCH_SIZES, shuffled_times, marker='s', label="Shuffle Vector")

    plt.title("Reproduction of Figure 11: VSS Total Search Time", fontsize=16, fontweight='bold')
    plt.xlabel("Number of users per query search", fontsize=15)
    plt.ylabel(f"Total Search Time (Sec) for {TOTAL_QUERIES} users", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best', frameon=True)
    
    # 4. Save the plot
    plt.tight_layout()
    plt.savefig(PLOT_FILENAME, dpi=300)
    print(f"\nGraph successfully saved to {PLOT_FILENAME}")
    plt.show()

if __name__ == "__main__":
    print("Ensure 'src/vss_backend/vss_server.py' (the 80/20 server) is running.")
    run_speed_test()