import requests
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

DATASET_PATH = "features_extracted.csv"
USER_ID_COLUMN = "user_id" 
FEATURE_COLUMNS = [
    'strokeDuration', 'startX', 'startY', 'stopX', 'stopY',
    'directEndToEndDistance', 'meanResultantLength', 'upDownLeftRightFlag',
    'directionOfEndToEndLine', 'largestDeviationFromEndToEndLine',
    'averageDirection', 'lengthOfTrajectory', 'averageVelocity',
    'midStrokePressure', 'midStrokeArea'
]
VSS_API_URL = "http://127.0.0.1:8000/check_similarity"
NUM_SAMPLES = 500 
TARGET_USER = 1.0 
PLOT_DIR = "plots"
PLOT_FILENAME = os.path.join(PLOT_DIR, "fig10_accuracy_reproduction.png")

def get_vectors_for_test():

    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        return None, None
    
    df['upDownLeftRightFlag'], _ = pd.factorize(df['upDownLeftRightFlag'])

    valid_user_vectors = []
    all_imposter_live_vectors = []
    
    all_user_ids = df[USER_ID_COLUMN].unique()
    
    print("Splitting data for all users (80% train / 20% test)...")
    
    for user_id in all_user_ids:
        user_df = df[df[USER_ID_COLUMN] == user_id]
        if len(user_df) < 2: 
            continue
            
        _, test_data = train_test_split(
            user_df, 
            test_size=0.20, 
            shuffle=False
        )
        
        live_vectors_list = test_data[FEATURE_COLUMNS].values.tolist()
        
        if user_id == TARGET_USER:
            valid_user_vectors.extend(live_vectors_list)
        else:
            all_imposter_live_vectors.extend(live_vectors_list)

    if not valid_user_vectors:
        print(f"Error: No data found for target user {TARGET_USER}")
        return None, None
        
    print(f"Found {len(valid_user_vectors)} valid 'live' vectors for user {TARGET_USER}.")

    if len(all_imposter_live_vectors) < NUM_SAMPLES:
        print(f"Warning: Not enough imposter live vectors. Using {len(all_imposter_live_vectors)}.")
        imposter_vectors = all_imposter_live_vectors
    else:
        imposter_vectors = random.sample(all_imposter_live_vectors, NUM_SAMPLES)
    print(f"Found {len(imposter_vectors)} 'imposter' (live) vectors.")
        
    return valid_user_vectors, imposter_vectors

def test_similarity(test_vectors):

    distances = []
    for vector in test_vectors:
        try:
            res = requests.post(VSS_API_URL, json={"vector": vector}).json()
            if res.get("status") == "match_found":
                distances.append(res.get("distance"))
        except Exception as e:
            print(f"API request failed: {e}")
            
    return distances

def main():
    print("Reproducing Figure 10: VSS Accuracy Analysis (Corrected Method)...")
    os.makedirs(PLOT_DIR, exist_ok=True)

    valid_vectors, imposter_vectors = get_vectors_for_test()
    if valid_vectors is None:
        return
        
    print(f"Testing {len(valid_vectors)} valid user (live) vectors...")
    valid_distances = test_similarity(valid_vectors)
    
    print(f"\n--- Valid User (Live) Test ---")
    print(f"Average Euclidean distance: {np.mean(valid_distances):.4f}")

    print(f"\nTesting {len(imposter_vectors)} imposter (live) vectors...")
    imposter_distances = test_similarity(imposter_vectors)
    
    print(f"\n--- Imposter (Live) Test ---")
    print(f"Average Euclidean distance: {np.mean(imposter_distances):.4f}")

    plt.figure(figsize=(10, 6))
    plt.hist(imposter_distances, bins=50, alpha=0.7, label="Imposter Vectors (Fake Users' Live Data)")
    plt.hist(valid_distances, bins=20, alpha=1.0, color='red', label=f'Valid User {TARGET_USER} (Live Vectors)')

    plt.title("VSS Similarity Distribution", fontsize=16, fontweight='bold')
    plt.xlabel("Euclidean Distance (Similarity)", fontsize=16)
    plt.ylabel("Number of Query Users", fontsize=16)

    plt.tick_params(axis='both', which='major', labelsize=16, width=1.5)
    plt.tick_params(axis='both', which='minor', labelsize=16, width=1.2)
    
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, ymax * 1.1) 

    plt.legend(fontsize=14)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(PLOT_FILENAME, dpi=300)
    print(f"\nGraph successfully saved to {PLOT_FILENAME}")
    plt.show() 

if __name__ == "__main__":
    main()