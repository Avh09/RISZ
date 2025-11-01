import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
# Assuming helper.py is in the same directory or accessible
from helper import canonical_hash, h, xor_data, str_to_hex

# --- Configuration ---
DATASET_PATH = 'features_extracted.csv'
VSS_DATA_PATH = 'vss_registration_data.csv'
ATTACKER_DATA_PATH = 'attacker_stolen_data.csv'
SERVER_STORAGE_FILE = "server_storage.json"
CLIENT_STORAGE_FILE = "client_storage.json"

# --- User Credentials (from paper's simulation) ---
USER_ID = "22.0"
PASSWORD = "password123"
BIOMETRIC = "biometric_data_scan_1"

# --- Server's Long-Term Secret ---
# This is 'k' from the paper
SERVER_MASTER_KEY = canonical_hash("server_long_term_secret_k_12345")

def create_protocol_storage():
    """
    Simulates the User Registration Phase (Section IV-C).
    """
    print("--- Starting User Registration Phase (IV-C) ---")
    
    # --- Step 1: User (Ui) ---
    IDi = USER_ID
    PW_i = PASSWORD
    BM_i = BIOMETRIC
    
    # Simulate fuzzy extractor Gen(BMi)
    sigma_i = canonical_hash(BM_i) 
    
    # User's random secret 'x'
    x = canonical_hash("user_random_secret_x_abcde")
    
    t1 = canonical_hash(IDi, PW_i, sigma_i)
    t2 = xor_data(x, t1)
    
    # --- Step 2: Server (MS) ---
    # User sends {IDi, t2, Data}
    # (We are skipping 'Data' as it's handled by the CSV split)
    k = SERVER_MASTER_KEY
    TIDi = "TID_user_1.0_initial" # Pick a temporary ID
    TIDn = "TIDn_user_1.0_next"  # Pre-calculate the *next* TID
    
    t3 = xor_data(t2, canonical_hash(k, IDi))
    
    # MS stores user data, keyed by *current* TID
    server_db = {
        TIDi: {
            "IDi": IDi,
            "t3": t3,
            "TIDn": TIDn # Store the *next* TID for the user
        }
    }
    with open(SERVER_STORAGE_FILE, "w") as f:
        json.dump(server_db, f, indent=2)
    print(f"Server storage created at '{SERVER_STORAGE_FILE}'")
    
    # --- Step 3: User (Ui) ---
    # User receives {TIDi, t3}
    t3_star = xor_data(t3, canonical_hash(sigma_i, IDi))
    x_star = xor_data(x, canonical_hash(PW_i, sigma_i, t3))
    
    client_storage = {
        "IDi": IDi,
        "TIDi": TIDi,
        "t2": t2,
        "t3_star": t3_star,
        "x_star": x_star
    }
    with open(CLIENT_STORAGE_FILE, "w") as f:
        json.dump(client_storage, f, indent=2)
    print(f"Client storage created at '{CLIENT_STORAGE_FILE}'")

def split_dataset():
    """
    Splits the main dataset into 80% for VSS (registration)
    and 20% for the attacker (stolen "live" data).
    """
    print("\n--- Splitting Dataset (80/20) ---")
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset '{DATASET_PATH}' not found.")
        return

    # Get data for the user we are registering
    user_df = df[df['user_id'] == float(USER_ID)]
    if user_df.empty:
        print(f"Error: No data found for user {USER_ID} in dataset.")
        return

    # Split this user's data 80/20
    # We set shuffle=False to be consistent with the paper's concept
    reg_df, attack_df = train_test_split(
        user_df,
        test_size=0.80,
        shuffle=False 
    )
    
    # Save the 80% for the VSS server
    reg_df.to_csv(VSS_DATA_PATH, index=False)
    print(f"VSS Server data (80%) saved to '{VSS_DATA_PATH}' ({len(reg_df)} vectors)")
    
    # Save the 20% for the attacker
    attack_df.to_csv(ATTACKER_DATA_PATH, index=False)
    print(f"Attacker's 'stolen' data (20%) saved to '{ATTACKER_DATA_PATH}' ({len(attack_df)} vectors)")

if __name__ == "__main__":
    if os.path.exists(CLIENT_STORAGE_FILE) or os.path.exists(SERVER_STORAGE_FILE):
        print("Storage files already exist. Skipping registration.")
        if os.path.exists(CLIENT_STORAGE_FILE):
            print(f"- Found Client storage file: '{CLIENT_STORAGE_FILE}'")
            os.remove(CLIENT_STORAGE_FILE)
        if os.path.exists(SERVER_STORAGE_FILE):
            print(f"- Found Server storage file: '{SERVER_STORAGE_FILE}'")
            os.remove(SERVER_STORAGE_FILE)
        
        create_protocol_storage()
    else:
        create_protocol_storage()
        
    if os.path.exists(VSS_DATA_PATH) or os.path.exists(ATTACKER_DATA_PATH):
        print("Data split files already exist. Skipping split.")
        if os.path.exists(VSS_DATA_PATH):
            print(f"- Found VSS data file: '{VSS_DATA_PATH}'")
            os.remove(VSS_DATA_PATH)
        if os.path.exists(ATTACKER_DATA_PATH):
            print(f"- Found Attacker data file: '{ATTACKER_DATA_PATH}'")
            os.remove(ATTACKER_DATA_PATH)
        split_dataset()
    else:
        split_dataset()
    
    print("\nSetup complete. You can now run the servers.")