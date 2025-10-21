import socket
from .helper import (
    h, get_timestamp, check_timestamp, send_message, recv_message,
    rlwe_sample_from_chi_delta, rlwe_generate_public_key, 
    rlwe_compute_shared_values, Mod2
)

# --- Client (Ui) Setup ---
HOST = '127.0.0.1'
PORT = 65432
DELTA_T = 10  # 10 seconds for timestamp verification

# --- Dummy Stored Credentials ---
# [cite_start]This data is stored on the device after Registration (Section IV-C) [cite: 220]
# [cite_start]We assume the user has successfully logged in (Phase D) [cite: 231]
my_IDi = "ID_user_001"
my_TIDi = "TID_user_001"
my_t3 = "dummy_t3_for_user_001" # Simulating successful login
my_x = "dummy_user_secret_x"   # Simulating successful login

def run_client():
    print("Client (Ui) is starting...")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            print(f"Client connected to {HOST}:{PORT}")

            # [cite_start]--- OTAKA Step 1: Send M1 [cite: 235-238] ---
            
            # 1. Generate client's keys
            f1 = rlwe_sample_from_chi_delta()
            e1 = rlwe_sample_from_chi_delta()
            ai = rlwe_generate_public_key(f1, e1) # ai = α*f1 + 2*e1
            
            TS1 = get_timestamp()
            
            # 2. Calculate values for M1
            X1 = my_IDi # Dummy, real is IDi XOR h(t3 || TS1 || TIDi)
            s1 = h(my_x, TS1)
            s2 = s1 # Dummy, real is s1 XOR h(t3 || TS1 || TIDi)
            X2 = h(ai, X1, TS1, my_TIDi, s2)
            
            M1 = {
                "X1": X1, "X2": X2, "TIDi": my_TIDi, "ai": ai, "s2": s2, "TS1": TS1
            }
            
            send_message(s, M1)
            print("\n[Ui] Sent M1.")

            # [cite_start]--- OTAKA Step 3: Receive M2 and Send M3 [cite: 244-249] ---
            M2 = recv_message(s)
            if not M2:
                raise ConnectionError("Server disconnected.")
                
            print("[Ui] Received M2.")
            
            # 1. Verify freshness
            if not check_timestamp(M2['TS2'], DELTA_T):
                raise ValueError("M2 timestamp check failed. Possible replay attack.")
            
            # 2. Compute session key
            bj = M2['bj']
            dj = M2['dj']
            
            c_prime_j = rlwe_compute_shared_values(f1, bj) # c'j = bj * f1
            w_prime_j = Mod2(c_prime_j, dj)
            
            # [cite_start]SKij = h(IDi || w'j || TS2 || TS1 || s1 || t3 || TIDi) [cite: 244]
            SK_ij = h(my_IDi, w_prime_j, M2['TS2'], TS1, s1, my_t3, my_TIDi)
            print(f"[Ui] Client session key computed: {SK_ij[:10]}...")

            # [cite_start]3. Derive new TID and verify server's key [cite: 244-246]
            # TIDn = TIDn* XOR h(SKij || TS2 || t3 || TIDi)
            TIDn = M2['TIDn_star'] # Dummy
            SKV_ij = h(M2['TIDn_star'], SK_ij, M2['TS2'], M2['bj'], M2['dj'], my_t3, TS1)
            
            if SKV_ij != M2['SKVji']:
                raise ValueError("SKV check failed. Server is not authentic.")
            
            print("[Ui] Server SKV verified. Mutual authentication successful.")
            
            # [cite_start]4. Update TID and send ACK (M3) [cite: 247-249]
            my_TIDi = TIDn # Update TID for next session
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