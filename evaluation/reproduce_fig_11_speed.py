import requests
import time
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
VSS_API_URL = "http://127.0.0.1:8000/check_similarity"
[cite_start]BATCH_SIZES = [10, 100, 500, 1000, 1500, 2000] # [cite: 772]
[cite_start]TOTAL_QUERIES = 14316 # Total users in dataset [cite: 773]
# -----------------------

def run_speed_test():
    print("Reproducing Figure 11: VSS Query Speed Analysis...")
    
    # Generate random query vectors
    # We simulate queries without loading the whole dataset
    query_vectors = np.random.rand(max(BATCH_SIZES), 15).tolist()
    
    total_times = []
    
    for batch_size in BATCH_SIZES:
        print(f"Testing batch size: {batch_size} users per query")
        
        # We simulate the *total* time it would take to query
        # all users, by timing one batch and extrapolating.
        
        batch_to_test = query_vectors[:batch_size]
        
        start_time = time.time()
        for vector in batch_to_test:
            try:
                requests.post(VSS_API_URL, json={"vector": vector})
            except Exception as e:
                print(f"API request failed: {e}")
                pass
        end_time = time.time()
        
        time_per_batch = end_time - start_time
        avg_time_per_query = time_per_batch / batch_size
        
        # Extrapolate to find total time for all 14,316 users
        total_time_for_all = avg_time_per_query * TOTAL_QUERIES
        total_times.append(total_time_for_all)
        
        print(f"  Avg time per query: {avg_time_per_query*1000:.4f} ms")
        print(f"  Extrapolated total time: {total_time_for_all:.2f} s")

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(BATCH_SIZES, total_times, marker='o')
    plt.title("Reproduction of Figure 11: VSS Total Search Time")
    plt.xlabel("Number of users per query search")
    plt.ylabel(f"Total Search Time (Sec) for {TOTAL_QUERIES} users")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_speed_test()