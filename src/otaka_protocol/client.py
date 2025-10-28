import socket
import json
import numpy as np
from .helper import (
    h, xor_data, get_timestamp, check_timestamp, send_message, recv_message,
    rlwe_generate_keypair, rlwe_compute_shared_secret, Mod2
)

# --- Client (Ui) Setup ---
HOST = '127.0.0.1'
PORT = 65432
DELTA_T = 10
CLIENT_STORAGE_FILE = "client_storage.json"

def simulate_user_login():
    """
    Simulates the User Login Phase (Section IV-D).
    Reads stored credentials and reconstructs secrets.
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
    IDi_input = "user_001"
    PWi_star_input = "password123"
    BMi_star_input = "biometric_data_scan_1" # Correct scan
    
    # 2. Simulate fuzzy extractor Rep(BMi*, τi)
    sigma_i_star = h(BMi_star_input)
    
    # 3. Derive secrets
    # t3 = t*3 ⊕ h(σ*i || IDi)
    t3_rec = xor_data(client_data['t3_star'], h(sigma_i_star, IDi_input))
    
    # x = x* ⊕ h(PW*i || σ*i || t3)
    x_rec_hex = xor_data(client_data['x_star'], h(PWi_star_input, sigma_i_star, t3_rec))
    
    # 4. Verify t'2 with stored t2
    # t'2 = x ⊕ h(IDi || PW*i || σ*i)
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
            
            # 1. Generate client's keys
            (f1, e1), ai = rlwe_generate_keypair() # (private), public
            
            TS1 = get_timestamp()
            
            # 2. Calculate values for M1
            X1 = xor_data(my_IDi.encode().hex(), h(my_t3, TS1, my_TIDi))
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
                raise ValueError("M2 timestamp check failed. Possible replay attack.")
            
            # 2. Compute session key
            bj = np.array(M2['bj'])
            dj = np.array(M2['dj'])
            
            # c'j = bj * f1
            c_prime_j = rlwe_compute_shared_secret(f1, bj)
            # w'j = Mod2(c'j, dj)
            w_prime_j = Mod2(c_prime_j, dj)
            
            # SKij = h(IDi || w'j || TS2 || TS1 || s1 || t3 || TIDi)
            SK_ij = h(my_IDi, w_prime_j, M2['TS2'], TS1, s1, my_t3, my_TIDi)
            print(f"[Ui] Client session key computed: {SK_ij[:10]}...")

            # 3. Derive new TID and verify server's key
            # TIDn = TIDn* ⊕ h(SKij || TS2 || t3 || TIDi)
            TIDn = xor_data(M2['TIDn_star'], h(SK_ij, M2['TS2'], my_t3, my_TIDi))
            
            # SKVij = h(TIDn* || SKij || TS2 || bj || dj || t3 || TS1)
            SKV_ij = h(M2['TIDn_star'], SK_ij, M2['TS2'], M2['bj'], M2['dj'], my_t3, TS1)
            
            if SKV_ij != M2['SKVji']:
                raise ValueError("SKV check failed. Server is not authentic.")
            
            print("[Ui] Server SKV verified. Mutual authentication successful.")
            
            # 4. Update TID and send ACK (M3)
            my_TIDi = TIDn 
            TS3 = get_timestamp()
            ACK = h(my_TIDi, SK_ij, TS3)
            
            M3 = {"ACK": ACK, "TS3": TS3}
            
            send_message(s, M3)
            print("[Ui] Sent M3. Session fully established.")
            print(f"[Ui] Final Session Key: {SK_ij}")

    except (ValueError, ConnectionError, socket.error) as e:
        print(f"\n[Ui] Error: {e}")
    finally:
        print("[Ui] Closing connection.")

if __name__ == "__main__":
    run_client()