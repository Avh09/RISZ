import requests
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# --- Configuration ---
DATASET_PATH = "features_extracted.csv"
USER_ID_COLUMN = "user_id" 
FEATURE_COLUMNS = [
    'strokeDuration', 'startX', 'startY', 'stopX', 'stopY',
    'directEndToEndDistance', 'meanResultantLength', 'upDownLeftRightFlag',
    'directionOfEndToEndLine', 'largestDeviationFromEndToEndLine',
    'averageDirection', 'lengthOfTrajectory', 'averageVelocity',
    'midStrokePressure', 'midStrokeArea'
]
# Make sure this points to your 100% server
VSS_API_URL = "http://127.0.0.1:8000/check_similarity"

# --- Output folder ---
PLOT_DIR = "plots"
PLOT_FILENAME = os.path.join(PLOT_DIR, "fig10_paper_method_reproduction.png")
# -----------------------

def get_vectors_for_test():
    """
    Loads two sets of vectors:
    1. original_vectors: 100% of the dataset.
    2. shuffled_vectors: Randomly generated fake vectors.
    """
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        return None, None
    
    # Factorize categorical column
    df['upDownLeftRightFlag'], _ = pd.factorize(df['upDownLeftRightFlag'])

    # 1. Get 100% of "Original" vectors
    original_vectors = df[FEATURE_COLUMNS].values.tolist()
    print(f"Found {len(original_vectors)} original vectors.")

    # 2. Get "Shuffled" (fake) vectors
    # We create fake vectors by shuffling the features of real vectors
    shuffled_vectors = []
    df_shuffled = df[FEATURE_COLUMNS].copy()
    for col in df_shuffled.columns:
        np.random.shuffle(df_shuffled[col].values)
    
    shuffled_vectors = df_shuffled.values.tolist()
    print(f"Created {len(shuffled_vectors)} shuffled (fake) vectors.")
        
    return original_vectors, shuffled_vectors

def test_similarity(test_vectors):
    """
    Queries the VSS server with a list of vectors and returns the distances.
    """
    distances = []
    # To speed this up, we'll only test a sample
    if len(test_vectors) > 2000:
        test_vectors = random.sample(test_vectors, 2000)
    print(f"Testing {len(test_vectors)} sample vectors...")

    for vector in test_vectors:
        try:
            res = requests.post(VSS_API_URL, json={"vector": vector}).json()
            if res.get("status") == "match_found":
                distances.append(res.get("distance"))
        except Exception as e:
            print(f"API request failed: {e}")
            
    return distances

def main():
    print("Reproducing Figure 10 (Paper's 100% Method)...")
    os.makedirs(PLOT_DIR, exist_ok=True)

    original_vectors, shuffled_vectors = get_vectors_for_test()
    if original_vectors is None:
        return
        
    print(f"\nTesting Original vectors...")
    original_distances = test_similarity(original_vectors)
    
    print(f"\n--- Original Vector Test ---")
    print(f"Average Euclidean distance: {np.mean(original_distances):.4f}")
    if np.mean(original_distances) == 0.0:
        print("SUCCESS: Got 0.0 distance, matching the paper's vertical line.")

    print(f"\nTesting Shuffled (fake) vectors...")
    shuffled_distances = test_similarity(shuffled_vectors)
    
    print(f"\n--- Shuffled (Fake) Test ---")
    print(f"Average Euclidean distance: {np.mean(shuffled_distances):.4f}")
    
    # Plotting the histogram
     # Plotting the histogram
    plt.figure(figsize=(10, 6))

    # Plot the "Shuffled Vectors" as a bell curve
    plt.hist(shuffled_distances, bins=50, alpha=0.7, label="Shuffled Vectors")

    # Plot the "Original Vectors" as a single vertical line at 0.0
    plt.axvline(x=0.0, color='k', linestyle='-', linewidth=2, label='Original Vectors (dist = 0.0)')

    # --- Title and labels with larger fonts ---
    plt.title("VSS Similarity Distribution", fontsize=18, fontweight='bold')
    plt.xlabel("Euclidean Distance (Similarity)", fontsize=16)
    plt.ylabel("Number of Query Users", fontsize=16)

    # --- Legend font size ---
    plt.legend(fontsize=14)

    # --- Make tick labels larger ---
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)

    # --- Expand Y-axis to show more top values ---
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, ymax * 1.5)  # Increase top by 20%

    
    # --- Force integer Y-axis ticks only ---
    from matplotlib.ticker import MaxNLocator
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    # --- Thicken spines for clarity ---
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.3)

    # --- Save and show ---
    plt.tight_layout()
    plt.savefig(PLOT_FILENAME, dpi=300)
    print(f"\nGraph successfully saved to {PLOT_FILENAME}")
    plt.show()



if __name__ == "__main__":
    main()