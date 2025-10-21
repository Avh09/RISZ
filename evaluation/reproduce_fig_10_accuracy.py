import requests
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Configuration ---
# This script assumes 'evaluation/load_data.py' has been run.
DATASET_PATH = "data/bioident_dataset.csv" # Update this
USER_ID_COLUMN = "user_id"
VECTOR_COLUMNS = [f"feature_{i}" for i in range(15)]
VSS_API_URL = "http://127.0.0.1:8000/check_similarity"
NUM_SAMPLES = 500 # Number of users to test for the graph

# -----------------------

def get_real_and_fake_vectors():
    """
    Loads samples from the dataset to create "original" and "shuffled"
    [cite_start]vectors as described in the paper [cite: 765-766].
    """
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        return None, None, None

    # Get N unique users
    unique_users = df[USER_ID_COLUMN].unique()
    if len(unique_users) < NUM_SAMPLES:
        print(f"Warning: Not enough unique users. Using {len(unique_users)} samples.")
        sample_users = unique_users
    else:
        sample_users = random.sample(list(unique_users), NUM_SAMPLES)

    original_vectors = [] # List of (user_id, vector)
    shuffled_vectors = [] # List of (user_id_of_vector, vector)
    
    # Create original vectors (test for 100% match)
    for user in sample_users:
        user_vector = df[df[USER_ID_COLUMN] == user].iloc[0][VECTOR_COLUMNS].tolist()
        original_vectors.append((str(user), user_vector))
    
    # Create "shuffled" (fake) vectors
    all_vectors = df[VECTOR_COLUMNS].values.tolist()
    for user_id, _ in original_vectors:
        # Pick a random vector from the entire dataset
        shuffled_vec = random.choice(all_vectors)
        shuffled_vectors.append((str(user_id), shuffled_vec))
        
    return original_vectors, shuffled_vectors

def test_similarity(test_vectors, correct_user_id):
    """
    Queries the VSS server and returns the distance.
    If 'correct_user_id' is provided, it checks for a match.
    """
    distances = []
    matches = 0
    for user_id, vector in test_vectors:
        try:
            res = requests.post(VSS_API_URL, json={"vector": vector}).json()
            if res.get("status") == "match_found":
                distances.append(res.get("distance"))
                if correct_user_id and res.get("matched_user_id") == user_id:
                    matches += 1
        except Exception as e:
            print(f"API request failed: {e}")
            
    return distances, matches

def main():
    print("Reproducing Figure 10: VSS Accuracy Analysis...")
    original, shuffled = get_real_and_fake_vectors()
    if original is None:
        return
        
    print(f"Testing {len(original)} original vectors...")
    original_distances, matches = test_similarity(original, correct_user_id=True)
    
    print(f"\n--- Original Vector Test ---")
    print(f"Accuracy (User ID Match): {matches / len(original) * 100:.2f}%")
    [cite_start]print(f"This should be 100% as per the paper[cite: 767].")
    print(f"Average Euclidean distance: {np.mean(original_distances):.4f} (should be near 0.0)")

    print(f"\nTesting {len(shuffled)} shuffled (fake) vectors...")
    shuffled_distances, _ = test_similarity(shuffled, correct_user_id=False)
    
    print(f"\n--- Shuffled Vector Test ---")
    print(f"Average Euclidean distance: {np.mean(shuffled_distances):.4f}")
    
    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(shuffled_distances, bins=50, alpha=0.7, label="Shuffled Vectors (Fake Users)")
    plt.axvline(x=np.mean(original_distances), color='r', linestyle='dashed', linewidth=2, label=f'Original Vectors (Mean dist = {np.mean(original_distances):.2f})')
    plt.title("Reproduction of Figure 10: VSS Similarity Distribution")
    plt.xlabel("Euclidean Distance (Similarity)")
    plt.ylabel("Number of Query Users")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()