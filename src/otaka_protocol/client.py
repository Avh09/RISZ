import socket
import json
import numpy as np
import time
import pandas as pd
import requests
from sklearn.model_selection import train_test_split # <-- NEW
from .helper import (
    canonical_hash, h, hex_to_str, str_to_hex, xor_data, get_timestamp, 
    check_timestamp, send_message, recv_message,
    rlwe_generate_keypair, rlwe_compute_shared_secret, Mod2,
    encrypt_data
)

# --- Client (Ui) Setup ---
HOST = '127.0.0.1'
PORT = 65432
DELTA_T = 10
CLIENT_STORAGE_FILE = "client_storage.json"
DATASET_PATH = 'features_extracted.csv'
FEATURE_COLUMNS = [
    'strokeDuration', 'startX', 'startY', 'stopX', 'stopY',
    'directEndToEndDistance', 'meanResultantLength', 'upDownLeftRightFlag',
    'directionOfEndToEndLine', 'largestDeviationFromEndToEndLine',
    'averageDirection', 'lengthOfTrajectory', 'averageVelocity',
    'midStrokePressure', 'midStrokeArea'
]

def simulate_user_login():
    """
    Simulates the User Login Phase (Section IV-D).
    """
    print("[Ui] --- Starting User Login Phase (IV-D) ---")
    try:
        with open(CLIENT_STORAGE_FILE, "r") as f:
            client_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{CLIENT_STORAGE_FILE}' not found.")
        print("Please run 'python -m src.otaka_protocol.registration' first.")
        return None

    # 1. Simulate user inputting credentials
    IDi_input = "1.0" 
    PWi_star_input = "password123"
    BMi_star_input = "biometric_data_scan_1" 
    
    # 2. Simulate fuzzy extractor Rep(BMi*, τi)
    sigma_i_star = h(BMi_star_input)
    
    # 3. Derive secrets
    t3_rec = xor_data(client_data['t3_star'], h(sigma_i_star, IDi_input))
    x_rec_hex = xor_data(client_data['x_star'], h(PWi_star_input, sigma_i_star, t3_rec))
    
    # 4. Verify t'2 with stored t2
    t2_prime = xor_data(x_rec_hex, h(IDi_input, PWi_star_input, sigma_i_star))
    
    if t2_prime == client_data['t2']:
        print("[Ui] Login Successful. Secrets reconstructed.")
        return {
            "IDi": client_data['IDi'], "TIDi": client_data['TIDi'],
            "t3": t3_rec, "x_hex": x_rec_hex
        }
    else:
        print("[Ui] Login Failed. Credentials do not match.")
        return None

def load_live_vectors(user_id): # <-- MODIFIED
    """
    Loads the 'live' (test) vectors for the authenticated user.
    This is now the last 20% of their data.
    """
    print(f"[Ui] Loading 'live' (20% test) simulation vectors for user {user_id}...")
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset '{DATASET_PATH}' not found.")
        return []
    
    # Convert categorical feature
    df['upDownLeftRightFlag'], _ = pd.factorize(df['upDownLeftRightFlag'])
    
    # Get all vectors for our user
    user_df = df[df['user_id'] == float(user_id)]
    if user_df.empty:
        print(f"Error: No vectors found in dataset for user_id {user_id}")
        return []
        
    # Split this user's data and get the 20% test set
    _, test_data = train_test_split(
        user_df,
        test_size=0.20,
        shuffle=False
    )
    
    print(f"[Ui] Found {len(test_data)} live vectors to send.")
    return test_data[FEATURE_COLUMNS].values.tolist()

def run_client():
    login_secrets = simulate_user_login()
    if not login_secrets:
        return

    my_IDi = login_secrets['IDi']
    my_TIDi = login_secrets['TIDi']
    my_t3 = login_secrets['t3']
    my_x_hex = login_secrets['x_hex']
    
    print("\n[Ui] --- Starting OTAKA Phase (IV-E) ---")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            print(f"[Ui] Connected to server {HOST}:{PORT}")

            # --- OTAKA Step 1: Send M1 ---
            (f1, e1), ai = rlwe_generate_keypair()
            TS1 = get_timestamp()
            X1 = xor_data(str_to_hex(my_IDi), h(my_t3, TS1, my_TIDi))
            s1 = h(my_x_hex, TS1)
            s2 = xor_data(s1, h(my_t3, TS1, my_TIDi))
            X2 = h(np.array_str(ai), X1, TS1, my_TIDi, s2)
            M1 = {
                "X1": X1, "X2": X2, "TIDi": my_TIDi, "ai": ai, "s2": s2, "TS1": TS1
            }
            send_message(s, M1)
            print("\n[Ui] Sent M1.")

            # --- OTAKA Step 3: Receive M2 and Send M3 ---
            M2 = recv_message(s)
            if not M2:
                raise ConnectionError("Server disconnected.")
            print("[Ui] Received M2.")
            
            if not check_timestamp(M2['TS2'], DELTA_T):
                raise ValueError("M2 timestamp check failed.")
            
            bj = np.array(M2['bj'])
            dj = np.array(M2['dj'])
            c_prime_j = rlwe_compute_shared_secret(f1, bj)
            w_prime_j = Mod2(c_prime_j, dj)
            SK_ij = h(my_IDi, w_prime_j, M2['TS2'], TS1, s1, my_t3, my_TIDi)
            print(f"[Ui] Client session key computed: {SK_ij[:10]}...")

            hash_hex_cli = canonical_hash(SK_ij, M2['TS2'], my_t3, my_TIDi)
            TIDn_hex = xor_data(M2['TIDn_star'], hash_hex_cli)
            TIDn = hex_to_str(TIDn_hex)       
            
            SKV_ij = canonical_hash(M2['TIDn_star'], SK_ij, M2['TS2'],
                        json.dumps(M2['bj'], separators=(',',':')),
                        json.dumps(M2['dj'], separators=(',',':')),
                        my_t3, TS1)
            
            if SKV_ij != M2['SKVji']:
                raise ValueError("SKV check failed. Server is not authentic.")
            
            print("[Ui] Server SKV verified. Mutual authentication successful.")
            
            my_TIDi_next = TIDn 
            TS3 = get_timestamp()
            ACK = canonical_hash(my_TIDi_next, SK_ij, TS3)
            M3 = {"ACK": ACK, "TS3": TS3}
            send_message(s, M3)
            print("[Ui] Sent M3. OTAKA Handshake Complete.")
            print(f"[Ui] Final Session Key: {SK_ij}")
            print("\n[Ui] --- Starting Continuous Authentication (IV-F) ---")
            
            # # 1. Load vectors to simulate
            # live_vectors = load_live_vectors(my_TIDi) # Use the *original* TIDi to load vectors
            # if not live_vectors:
            #     # Fallback to IDi if TIDi doesn't work (depends on registration logic)
            #     live_vectors = load_live_vectors(my_IDi)
            #     if not live_vectors:
            #         raise ValueError("No live vectors found to simulate.")
            # 1. Load vectors to simulate
            # The VSS database (CSV file) is keyed by the real user ID (my_IDi),
            # not the temporary ID (my_TIDi).
            live_vectors = load_live_vectors(my_IDi)
            if not live_vectors:
                raise ValueError(f"No live vectors found to simulate for user {my_IDi}.")
            
            # 2. Calculate the IV as specified in the paper
            # IV is h(t3, TS3), truncated to 16 bytes (32 hex)
            full_hash_iv = h(my_t3, TS3)
            iv = full_hash_iv[:32]
            
            # 3. Start sending encrypted data
            for vector in live_vectors:
                # Serialize vector to JSON string
                vector_json = json.dumps(vector)
                
                # Encrypt the JSON string
                encrypted_hex = encrypt_data(SK_ij, iv, vector_json)
                
                # Send as a JSON message
                ca_message = {
                    "type": "CA_DATA",
                    "payload": encrypted_hex
                }
                send_message(s, ca_message)
                print(f"[Ui] Sent encrypted vector. Waiting for server ACK...")
                
                # Wait for server's response
                response = recv_message(s)
                if not response or response.get("status") == "TERMINATE":
                    print("[Ui] !!! Server terminated session! (Imposter detected?)")
                    break
                elif response.get("status") == "OK":
                    print("[Ui] Server ACK OK. Session continues.")
                
                time.sleep(1) # Simulate time between actions
            
            print("[Ui] Simulation complete.")

    except (ValueError, ConnectionError, socket.error) as e:
        print(f"\n[Ui] Error: {e}")
    finally:
        print("[Ui] Closing connection.")

if __name__ == "__main__":
    run_client()