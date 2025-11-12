import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- Constants ---
FEATURE_COLUMNS = [
    'strokeDuration', 'startX', 'startY', 'stopX', 'stopY',
    'directEndToEndDistance', 'meanResultantLength', 'upDownLeftRightFlag',
    'directionOfEndToEndLine', 'largestDeviationFromEndToEndLine',
    'averageDirection', 'lengthOfTrajectory', 'averageVelocity',
    'midStrokePressure', 'midStrokeArea'
]
ADAPTIVE_THRESHOLD_STD_DEVS = 2.0 

df = pd.read_csv('../../data/features_extracted.csv')
df['upDownLeftRightFlag'], _ = pd.factorize(df['upDownLeftRightFlag'])

print("Building the VSS Database...")
db_vectors = []
db_labels = []
live_sim_data = {}

all_user_ids = df['user_id'].unique()

for user_id in all_user_ids:
    user_data = df[df['user_id'] == user_id][FEATURE_COLUMNS].values
    if len(user_data) < 2:
        continue
    
    split_index = len(user_data) // 2
    registration_vectors = user_data[:split_index]
    live_vectors = user_data[split_index:]
    
    db_vectors.extend(registration_vectors)
    db_labels.extend([user_id] * len(registration_vectors))
    live_sim_data[user_id] = live_vectors

# --- 3. Scale the Features ---
print("Fitting StandardScaler on registration data...")
scaler = StandardScaler()
scaler.fit(db_vectors)
db_vectors_scaled = scaler.transform(db_vectors)
print("Registration data has been scaled.")
print("Indexing the SCALED database... (Fitting NearestNeighbors model)")
vss_model = NearestNeighbors(n_neighbors=1, algorithm='auto')
vss_model.fit(db_vectors_scaled)
print(f"Database is ready. Total registered vectors: {len(db_vectors_scaled)}")

db_labels = np.array(db_labels)

print("Calculating initial adaptive thresholds for all users...")
user_thresholds = {}
user_seed_distances = {} 

for user_id in all_user_ids:
    user_reg_indices = np.where(db_labels == user_id)[0]
    
    if len(user_reg_indices) < 2: 
        user_thresholds[user_id] = np.inf
        user_seed_distances[user_id] = []
        continue
        
    user_reg_vectors_scaled = db_vectors_scaled[user_reg_indices]
    distances, indices = vss_model.kneighbors(user_reg_vectors_scaled)
    retrieved_ids = db_labels[indices.flatten()]
    correct_distances = distances.flatten()[retrieved_ids == user_id]
    
    if len(correct_distances) > 1:
        mean_dist = np.mean(correct_distances)
        std_dist = np.std(correct_distances)
        threshold = mean_dist + (ADAPTIVE_THRESHOLD_STD_DEVS * std_dist)
        user_thresholds[user_id] = threshold
        user_seed_distances[user_id] = list(correct_distances)
    elif len(correct_distances) == 1:
        user_thresholds[user_id] = correct_distances[0] * 1.5 # 50% buffer
        user_seed_distances[user_id] = list(correct_distances)
    else:
        user_thresholds[user_id] = np.inf 
        user_seed_distances[user_id] = []

print("Initial thresholds calculated.")

def query_vss_database(query_vector):
    query_vector_2d = np.array(query_vector).reshape(1, -1)
    query_vector_scaled = scaler.transform(query_vector_2d)
    distances, indices = vss_model.kneighbors(query_vector_scaled)
    retrieved_id = db_labels[indices[0][0]]
    distance = distances[0][0]
    return retrieved_id, distance

# --- 6. Run the Simulations ---
TARGET_USER_ID = 1.0
IMPOSTER_USER_ID = next(id for id in all_user_ids if id != TARGET_USER_ID)

print(f"\n--- SIMULATION 1: Valid User Session (User {TARGET_USER_ID}) ---")
print(f"Processing all 'live' vectors for User {TARGET_USER_ID}...")

SESSION_USER_ID = TARGET_USER_ID
valid_live_vectors = live_sim_data[TARGET_USER_ID]

current_threshold = user_thresholds[SESSION_USER_ID]
session_valid_distances = user_seed_distances[SESSION_USER_ID].copy() 

if current_threshold == np.inf:
    print("[WARNING] User has no initial threshold. System will be in 'open' mode (high FAR).")
else:
    print(f"Initial session threshold: {current_threshold:.4f} (based on {len(session_valid_distances)} seed vectors)")

success_count = 0
failure_count = 0

for i, live_vector in enumerate(valid_live_vectors):
    retrieved_id, distance = query_vss_database(live_vector)
    if retrieved_id == SESSION_USER_ID and distance < current_threshold:
        success_count += 1
        
        session_valid_distances.append(distance)
        if len(session_valid_distances) > 1:
            mean_dist = np.mean(session_valid_distances)
            std_dist = np.std(session_valid_distances)
            current_threshold = mean_dist + (ADAPTIVE_THRESHOLD_STD_DEVS * std_dist)
        
    else:
        failure_count += 1
        

print(f"\nFinal session threshold: {current_threshold:.4f}")
total_vectors = len(valid_live_vectors)
success_perc = (success_count / total_vectors) * 100
failure_perc = (failure_count / total_vectors) * 100

print("\n[Simulation 1 Report: Valid User with Adaptive Threshold]")
print(f"  Total Vectors Processed: {total_vectors}")
print(f"  Correctly Accepted (True Positives): {success_count} ({success_perc:.2f}%)")
print(f"  Incorrectly Rejected (False Negatives): {failure_count} ({failure_perc:.2f}%)")


print(f"\n--- SIMULATION 2: Imposter Attack (Imposter {IMPOSTER_USER_ID}) ---")
print(f"Processing all 'live' vectors from Imposter {IMPOSTER_USER_ID} against session of User {TARGET_USER_ID}...")

SESSION_USER_ID = TARGET_USER_ID # Session is still for User 1.0
imposter_live_vectors = live_sim_data[IMPOSTER_USER_ID]


imposter_threshold = user_thresholds[TARGET_USER_ID]
print(f"Using static threshold from User {TARGET_USER_ID}: {imposter_threshold:.4f}")


mismatch_detected_count = 0
system_fooled_count = 0

for imposter_vector in imposter_live_vectors:
    retrieved_id, distance = query_vss_database(imposter_vector)

    if retrieved_id == SESSION_USER_ID and distance < imposter_threshold:

        system_fooled_count += 1

        mismatch_detected_count += 1

total_vectors = len(imposter_live_vectors)
detection_perc = (mismatch_detected_count / total_vectors) * 100
fooled_perc = (system_fooled_count / total_vectors) * 100

print("\n[Simulation 2 Report: Imposter with Static Threshold]")
print(f"  Total Vectors Processed: {total_vectors}")
print(f"  Mismatch Correctly Detected (True Negatives): {mismatch_detected_count} ({detection_perc:.2f}%)")
print(f"  System Fooled (False Positives): {system_fooled_count} ({fooled_perc:.2f}%)")
