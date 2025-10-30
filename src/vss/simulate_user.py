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

# --- 1. Load and Prepare Data ---
df = pd.read_csv('../../data/features_extracted.csv')
df['upDownLeftRightFlag'], _ = pd.factorize(df['upDownLeftRightFlag'])

# --- 2. Build the "Feature Vector Database" (FVDB) ---
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

# --- 4. "Index" the SCALED Database ---
print("Indexing the SCALED database... (Fitting NearestNeighbors model)")
vss_model = NearestNeighbors(n_neighbors=1, algorithm='auto')
vss_model.fit(db_vectors_scaled)
print(f"Database is ready. Total registered vectors: {len(db_vectors_scaled)}")

db_labels = np.array(db_labels)

# --- 5. Create the VSS Query Function (Now with Scaling) ---
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

# --- SIMULATION 1: Valid User Session ---
print(f"\n--- SIMULATION 1: Valid User Session (User {TARGET_USER_ID}) ---")
print(f"Processing all 'live' vectors for User {TARGET_USER_ID}...")

SESSION_USER_ID = TARGET_USER_ID
valid_live_vectors = live_sim_data[TARGET_USER_ID]

# --- MODIFICATION ---
success_count = 0
failure_count = 0
# --- END MODIFICATION ---

for live_vector in valid_live_vectors:
    retrieved_id, distance = query_vss_database(live_vector)
    
    # --- MODIFICATION ---
    if retrieved_id == SESSION_USER_ID:
        success_count += 1
    else:
        failure_count += 1
        # Optional: print(f"  [MISMATCH] Expected {SESSION_USER_ID}, got {retrieved_id}")
    # --- END MODIFICATION ---

# --- MODIFICATION: Report Results ---
total_vectors = len(valid_live_vectors)
success_perc = (success_count / total_vectors) * 100
failure_perc = (failure_count / total_vectors) * 100

print("\n[Simulation 1 Report]")
print(f"  Total Vectors Processed: {total_vectors}")
print(f"  Correctly Identified: {success_count} ({success_perc:.2f}%)")
print(f"  Incorrectly Identified: {failure_count} ({failure_perc:.2f}%)")
# --- END MODIFICATION ---


# --- SIMULATION 2: Imposter Attack ---
print(f"\n--- SIMULATION 2: Imposter Attack (Imposter {IMPOSTER_USER_ID}) ---")
print(f"Processing all 'live' vectors from Imposter {IMPOSTER_USER_ID} against session of User {TARGET_USER_ID}...")

SESSION_USER_ID = TARGET_USER_ID # Session is still for User 1.0
imposter_live_vectors = live_sim_data[IMPOSTER_USER_ID]

# --- MODIFICATION ---
# "Success" means we DETECTED the mismatch.
# "Failure" means the imposter FOOLED the system.
mismatch_detected_count = 0
system_fooled_count = 0
# --- END MODIFICATION ---

for imposter_vector in imposter_live_vectors:
    retrieved_id, distance = query_vss_database(imposter_vector)
    
    # --- MODIFICATION ---
    if retrieved_id == SESSION_USER_ID:
        # This is BAD. The imposter was identified as the valid user.
        system_fooled_count += 1
    else:
        # This is GOOD. The imposter was identified as someone else.
        mismatch_detected_count += 1
    # --- END MODIFICATION ---

# --- MODIFICATION: Report Results ---
total_vectors = len(imposter_live_vectors)
detection_perc = (mismatch_detected_count / total_vectors) * 100
fooled_perc = (system_fooled_count / total_vectors) * 100

print("\n[Simulation 2 Report]")
print(f"  Total Vectors Processed: {total_vectors}")
print(f"  Mismatch Correctly Detected (Success): {mismatch_detected_count} ({detection_perc:.2f}%)")
print(f"  System Fooled (Failure): {system_fooled_count} ({fooled_perc:.2f}%)")
# --- END MODIFICATION ---