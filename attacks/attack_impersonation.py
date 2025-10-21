import requests
import pandas as pd
import random

# --- Configuration ---
# This script assumes 'evaluation/load_data.py' has been run.
DATASET_PATH = "data/bioident_dataset.csv" # Update this
USER_ID_COLUMN = "user_id"
VECTOR_COLUMNS = [f"feature_{i}" for i in range(15)]
VSS_API_URL = "http://127.0.0.1:8000/check_similarity"
# -----------------------

def get_vectors_for_attack():
    """Gets a vector for Alice (legit) and Bob (impersonator)"""
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        return None, None

    unique_users = df[USER_ID_COLUMN].unique()
    if len(unique_users) < 2:
        print("Error: Not enough unique users in dataset for this test.")
        return None, None
        
    alice_id, bob_id = random.sample(list(unique_users), 2)
    
    alice_vec = df[df[USER_ID_COLUMN] == alice_id].iloc[0][VECTOR_COLUMNS].tolist()
    bob_vec = df[df[USER_ID_COLUMN] == bob_id].iloc[0][VECTOR_COLUMNS].tolist()
    
    return str(alice_id), alice_vec, str(bob_id), bob_vec

def run_impersonation_attack():
    print("--- Impersonation (Stolen Device) Attack Simulation (Section V-B4) ---")
    
    alice_id, alice_vec, bob_id, bob_vec = get_vectors_for_attack()
    if alice_id is None:
        return

    print(f"  Legitimate User: '{alice_id}'")
    print(f"  Impersonator:    '{bob_id}'")
    
    # 1. Scenario: Alice is using her device.
    # This simulates the system running in the background.
    print(f"\n[+] '{alice_id}' is typing (sending her real vector)...")
    try:
        res = requests.post(VSS_API_URL, json={"vector": alice_vec}).json()
        print(f"  VSS Response: {res}")
        if res.get("matched_user_id") == alice_id:
            print("  Result: Correctly authenticated Alice. Session continues.")
        else:
            print("  Result: ❌ FAILED to authenticate Alice.")
    except Exception as e:
        print(f"API request failed: {e}")

    # 2. Scenario: Attacker "Bob" uses Alice's unlocked device.
    print(f"\n[+] ATTACK: '{bob_id}' starts typing (sending *his* vector)...")
    try:
        res = requests.post(VSS_API_URL, json={"vector": bob_vec}).json()
        print(f"  VSS Response: {res}")
        
        if res.get("matched_user_id") != alice_id:
            print(f"  Result: ✅ SUCCESSFUL DEFENSE. Matched '{res.get('matched_user_id')}', not '{alice_id}'.")
            [cite_start]print("  The system would now terminate the session[cite: 291].")
        else:
            print(f"  Result: ❌ FAILED DEFENSE. Attacker was incorrectly matched as '{alice_id}'.")
    except Exception as e:
        print(f"API request failed: {e}")

if __name__ == "__main__":
    print("Ensure 'src/vss_backend/vss_server.py' is running in another terminal.")
    run_impersonation_attack()