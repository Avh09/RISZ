import socket
from .helper import (
    h, get_timestamp, check_timestamp, send_message, recv_message,
    rlwe_sample_from_chi_delta, rlwe_generate_public_key, 
    rlwe_compute_shared_values, Mod2, Cha
)

# --- Server (MS) Setup ---
HOST = '127.0.0.1'
PORT = 65432
DELTA_T = 10  # 10 seconds for timestamp verification

# --- Dummy Database for Registered Users ---
# This simulates the data stored during Registration Phase (Section IV-C)
# [cite_start]We store what the MS needs: {TIDi: (IDi, t3)} [cite: 219]
user_db = {
    "TID_user_001": ("ID_user_001", "dummy_t3_for_user_001") 
}
# We also store the new TIDn, which starts as None
user_tids = {"TID_user_001": "TIDn_user_001_initial"}

def run_server():
    print("Medical Server (MS) is starting...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            
            try:
                # [cite_start]--- OTAKA Step 2: Receive M1 and Send M2 [cite: 239-243] ---
                M1 = recv_message(conn)
                if not M1:
                    raise ConnectionError("Client disconnected.")

                print("\n[MS] Received M1")
                
                # 1. Verify freshness
                if not check_timestamp(M1['TS1'], DELTA_T):
                    raise ValueError("M1 timestamp check failed. Possible replay attack.")
                
                # 2. Fetch user data and authenticate
                TIDi = M1['TIDi']
                if TIDi not in user_db:
                    raise ValueError(f"Unknown TIDi: {TIDi}")
                
                IDi_stored, t3 = user_db[TIDi]
                
                # [cite_start]3. Derive and verify IDi [cite: 240]
                # IDi = X1 XOR h(t3 || TS1 || TIDi)
                IDi_derived = M1['X1'] # This is a dummy derivation
                
                if IDi_derived != IDi_stored:
                     raise ValueError("IDi verification failed.")
                print(f"[MS] Authenticated user: {IDi_stored}")

                # [cite_start]4. Derive s1 and verify X2 [cite: 241]
                # s1 = s2 XOR h(t3 || TS1 || TIDi)
                s1_derived = M1['s2'] # Dummy derivation
                X2_prime = h(M1['ai'], M1['X1'], M1['TS1'], M1['TIDi'], M1['s2'])
                
                if X2_prime != M1['X2']:
                    raise ValueError("X2 verification failed.")
                print("[MS] M1 integrity confirmed.")

                # [cite_start]5. Generate server's keys and compute session key [cite: 242]
                f2 = rlwe_sample_from_chi_delta()
                e2 = rlwe_sample_from_chi_delta()
                bj = rlwe_generate_public_key(f2, e2) # bj = α*f2 + 2*e2
                
                ai = M1['ai']
                cj = rlwe_compute_shared_values(f2, ai) # cj = ai * f2
                dj = Cha(cj)
                wj = Mod2(cj, dj)
                
                TS2 = get_timestamp()
                
                # [cite_start]SKji = h(IDi || wj || TS2 || TS1 || s1 || t3 || TIDi) [cite: 243]
                SK_ji = h(IDi_stored, wj, TS2, M1['TS1'], s1_derived, t3, TIDi)
                print(f"[MS] Server session key computed: {SK_ji[:10]}...")

                # [cite_start]6. Prepare M2 [cite: 243]
                TIDn = user_tids[TIDi] 
                # TIDn* = TIDn XOR h(SKji || TS2 || t3 || TIDi)
                TIDn_star = TIDn # Dummy
                SKV_ji = h(TIDn_star, SK_ji, TS2, bj, dj, t3, M1['TS1'])
                
                M2 = {
                    "SKVji": SKV_ji, "TS2": TS2, "bj": bj, "dj": dj, "TIDn_star": TIDn_star
                }
                
                send_message(conn, M2)
                print("[MS] Sent M2.")

                # [cite_start]--- OTAKA Step 4: Receive M3 [cite: 250-251] ---
                M3 = recv_message(conn)
                if not M3:
                    raise ConnectionError("Client disconnected during M3.")

                print("\n[MS] Received M3:")
                
                if not check_timestamp(M3['TS3'], DELTA_T):
                    raise ValueError("M3 timestamp check failed.")
                
                # Verify ACK' = ACK
                ACK_prime = h(TIDn, SK_ji, M3['TS3'])
                
                if ACK_prime != M3['ACK']:
                    raise ValueError("ACK verification failed.")
                    
                print(f"[MS] ACK verified. Session with {IDi_stored} is fully established.")
                
                # [cite_start]Update TID for next session [cite: 251]
                user_tids[TIDi] = "TIDn_user_001_new" 
                print(f"[MS] Updated TID for user {IDi_stored}.")

            except (ValueError, ConnectionError) as e:
                print(f"\n[MS] Error: {e}")
            finally:
                print("[MS] Closing connection.")

if __name__ == "__main__":
    run_server()