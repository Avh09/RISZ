import socket
import json
import numpy as np
from .helper import (
    canonical_hash, h, str_to_hex, xor_data, get_timestamp, check_timestamp, send_message, recv_message,
    rlwe_generate_keypair, rlwe_compute_shared_secret, Mod2, Cha
)

# --- Server (MS) Setup ---
HOST = '127.0.0.1'
PORT = 65432
DELTA_T = 10
SERVER_STORAGE_FILE = "server_storage.json"

def load_server_data():
    """Loads the server's database from the JSON file."""
    try:
        with open(SERVER_STORAGE_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: '{SERVER_STORAGE_FILE}' not found.")
        print("Please run 'python -m src.otaka_protocol.registration' first.")
        return None

def run_server():
    server_db = load_server_data()
    if not server_db:
        return
        
    print("Medical Server (MS) is starting...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            
            try:
                # --- OTAKA Step 2: Receive M1 and Send M2 [cite: 239-243] ---
                M1 = recv_message(conn)
                if not M1:
                    raise ConnectionError("Client disconnected.")

                print("\n[MS] Received M1")
                
                # 1. Verify freshness [cite: 239]
                if not check_timestamp(M1['TS1'], DELTA_T):
                    raise ValueError("M1 timestamp check failed. Possible replay attack.")
                
                # 2. Fetch user data
                TIDi = M1['TIDi']
                if TIDi not in server_db:
                    raise ValueError(f"Unknown TIDi: {TIDi}")
                
                user_data = server_db[TIDi]
                IDi_stored = user_data['IDi']
                t3 = user_data['t3']
                
                # 3. Derive and verify IDi [cite: 240]
                # IDi = X1 ⊕ h(t3 || TS1 || TIDi)
                IDi_derived_hex = xor_data(M1['X1'], h(t3, M1['TS1'], TIDi))
                IDi_derived = bytes.fromhex(IDi_derived_hex).decode().rstrip('\x00')
                
                if IDi_derived != IDi_stored:
                     raise ValueError("IDi verification failed.")
                print(f"[MS] Authenticated user: {IDi_stored}")

                # 4. Derive s1 and verify X2 [cite: 241-242]
                # s1 = s2 ⊕ h(t3 || TS1 || TIDi)
                s1_derived = xor_data(M1['s2'], h(t3, M1['TS1'], TIDi))
                
                # Need to convert 'ai' back to numpy array
                ai = np.array(M1['ai'])
                X2_prime = h(np.array_str(ai), M1['X1'], M1['TS1'], M1['TIDi'], M1['s2'])
                
                if X2_prime != M1['X2']:
                    raise ValueError("X2 verification failed.")
                print("[MS] M1 integrity confirmed.")

                # 5. Generate server's keys and compute session key [cite: 242-243]
                (f2, e2), bj = rlwe_generate_keypair() # (private), public
                
                # cj = ai * f2 [cite: 242]
                cj = rlwe_compute_shared_secret(f2, ai)
                # dj = Cha(cj) [cite: 242]
                dj = Cha(cj)
                # wj = Mod2(cj, dj) [cite: 242]
                wj = Mod2(cj, dj)
                
                TS2 = get_timestamp()
                
                # SKji = h(IDi || wj || TS2 || TS1 || s1 || t3 || TIDi) [cite: 243]
                print("[MS] Key derivation components:")
                print("IDi:", IDi_stored)
                print("wj:", wj[:10], "...")
                print("TS2:", TS2)
                print("TS1:", M1['TS1'])
                print("s1:", s1_derived[:10], "...")
                print("t3:", t3[:10], "...")
                print("TIDi:", TIDi)

                SK_ji = h(IDi_stored, wj, TS2, M1['TS1'], s1_derived, t3, TIDi)
                print(f"[MS] Server session key computed: {SK_ji[:10]}...")

                # 6. Prepare M2
                TIDn = user_data['TIDn']
                # TIDn* = TIDn ⊕ h(SKji || TS2 || t3 || TIDi) [cite: 243]
                # TIDn_star = xor_data(TIDn.encode().hex(), h(SK_ji, TS2, t3, TIDi))
                TIDn_hex = str_to_hex(TIDn)                               # canonical hex of TIDn
                hash_hex_for_xor = canonical_hash(SK_ji, TS2, t3, TIDi)   # hex string
                TIDn_star = xor_data(TIDn_hex, hash_hex_for_xor)         # hex string result
                                
                # SKVji = h(TIDn* || SKji || TS2 || bj || dj || t3 || TS1) [cite: 243]
                print("Debug: Deriving SKVji for client verification.")
                print("TIDn*:", TIDn_star)
                print("SKij:", SK_ji)
                print("TS2:", TS2)
                print("bj:", bj)
                print("dj:", dj)
                print("t3:", t3)
                print("TS1:", M1['TS1'])
                
                SKV_ji = canonical_hash(TIDn_star, SK_ji, TS2,
                        json.dumps(bj.tolist(), separators=(',',':')),
                        json.dumps(dj.tolist(), separators=(',',':')),
                        t3, M1['TS1'])
                
                M2 = {
                    "SKVji": SKV_ji, "TS2": TS2, "bj": bj, "dj": dj, "TIDn_star": TIDn_star
                }
                
                send_message(conn, M2)
                print("[MS] Sent M2.")

                # --- OTAKA Step 4: Receive M3 [cite: 250-251] ---
                M3 = recv_message(conn)
                if not M3:
                    raise ConnectionError("Client disconnected during M3.")

                print("\n[MS] Received M3:")
                
                if not check_timestamp(M3['TS3'], DELTA_T):
                    raise ValueError("M3 timestamp check failed.")
                
                # Verify ACK' = ACK [cite: 250]
                print("DEBUG (server) TIDn used for ACK:", TIDn)
                print("DEBUG (server) SK_ji used for ACK:", SK_ji)
                print("DEBUG (server) TS3 from client:", M3['TS3'])
                # print("DEBUG (server) ACK computed:", ACK_prime)
                print("DEBUG (server) ACK received :", M3['ACK'])
                TS3_f = M3['TS3']
                TS3_str = str(TS3_f).replace(" ", "T")
                ACK_prime = canonical_hash(TIDn, SK_ji, TS3_str)
                print("DEBUG (server) ACK computed:", ACK_prime)
                
                if ACK_prime != M3['ACK']:
                    raise ValueError("ACK verification failed.")
                    
                print(f"[MS] ACK verified. Session with {IDi_stored} is fully established.")
                print(f"[MS] Final Session Key: {SK_ji}")
                
                # Update TID for next session
                server_db[TIDi]['TIDn'] = "TIDn_user_001_new"
                print(f"[MS] Updated TID for user {IDi_stored}.")

            except (ValueError, ConnectionError) as e:
                print(f"\n[MS] Error: {e}")
            finally:
                print("[MS] Closing connection.")

if __name__ == "__main__":
    run_server()