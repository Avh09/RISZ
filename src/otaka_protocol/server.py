import socket
import json
import numpy as np
import requests 
from .helper import (
    canonical_hash, h, str_to_hex, xor_data, get_timestamp, 
    check_timestamp, send_message, recv_message,
    rlwe_generate_keypair, rlwe_compute_shared_secret, Mod2, Cha,
    decrypt_data 
)

# --- Server (MS) Setup ---
HOST = '127.0.0.1'
PORT = 65432
DELTA_T = 10
SERVER_STORAGE_FILE = "server_storage.json"
VSS_SERVER_URL = "http://127.0.0.1:8000/check_similarity" 

def load_server_data():
    """Loads the server's database from the JSON file."""
    try:
        with open(SERVER_STORAGE_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: '{SERVER_STORAGE_FILE}' not found.")
        print("Please run 'python -m src.otaka_protocol.registration' first.")
        return None

def query_vss_server(vector): 
    """Queries the VSS server to get the user ID."""
    try:
        response = requests.post(VSS_SERVER_URL, json={"vector": vector})
        response.raise_for_status() # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.ConnectionError:
        print("[MS] FATAL ERROR: Cannot connect to VSS Server at", VSS_SERVER_URL)
        print("       Please run 'python -m src.vss_backend.vss_server' in another terminal.")
        return None
    except Exception as e:
        print(f"[MS] VSS Query Error: {e}")
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
        
        # --- NEW: Main server loop to accept multiple connections ---
        while True:
            print("\n[MS] Waiting for a new connection...")
            
            try:
                # --- NEW: All logic is now inside this try block ---
                conn, addr = s.accept()
                with conn:
                    print(f"Connected by {addr}")
                    
                    session_key = None
                    session_iv = None
                    session_user_id = None
                    
                    try:
                        # --- OTAKA Step 2: Receive M1 and Send M2 ---
                        M1 = recv_message(conn)
                        if not M1:
                            raise ConnectionError("Client disconnected.")
        
                        print("\n[MS] Received M1")
                        
                        # --- NEW: Improved error message for replay demo ---
                        if not check_timestamp(M1['TS1'], DELTA_T):
                            raise ValueError("M1 timestamp check failed. (Possible Replay Attack)")
                        
                        TIDi = M1['TIDi']
                        if TIDi not in server_db:
                            raise ValueError(f"Unknown TIDi: {TIDi}")
                        
                        user_data = server_db[TIDi]
                        IDi_stored = user_data['IDi']
                        t3 = user_data['t3']
                        
                        IDi_derived_hex = xor_data(M1['X1'], h(t3, M1['TS1'], TIDi))
                        IDi_derived = bytes.fromhex(IDi_derived_hex).decode().rstrip('\x00')
                        
                        if IDi_derived != IDi_stored:
                             raise ValueError("IDi verification failed.")
                        print(f"[MS] Authenticated user: {IDi_stored}")
                        session_user_id = IDi_stored 
        
                        s1_derived = xor_data(M1['s2'], h(t3, M1['TS1'], TIDi))
                        ai = np.array(M1['ai'])
                        X2_prime = h(np.array_str(ai), M1['X1'], M1['TS1'], M1['TIDi'], M1['s2'])
                        
                        if X2_prime != M1['X2']:
                            raise ValueError("X2 verification failed.")
                        print("[MS] M1 integrity confirmed.")
        
                        # 5. Generate server's keys and compute session key
                        (f2, e2), bj = rlwe_generate_keypair()
                        cj = rlwe_compute_shared_secret(f2, ai)
                        dj = Cha(cj)
                        wj = Mod2(cj, dj)
                        TS2 = get_timestamp()
                        SK_ji = h(IDi_stored, wj, TS2, M1['TS1'], s1_derived, t3, TIDi)
                        session_key = SK_ji 
                        print(f"[MS] Server session key computed: {SK_ji[:10]}...")
        
                        # 6. Prepare M2
                        TIDn = user_data['TIDn']
                        TIDn_hex = str_to_hex(TIDn)
                        hash_hex_for_xor = canonical_hash(SK_ji, TS2, t3, TIDi)
                        TIDn_star = xor_data(TIDn_hex, hash_hex_for_xor)
                        
                        SKV_ji = canonical_hash(TIDn_star, SK_ji, TS2,
                                json.dumps(bj.tolist(), separators=(',',':')),
                                json.dumps(dj.tolist(), separators=(',',':')),
                                t3, M1['TS1'])
                        
                        M2 = {
                            "SKVji": SKV_ji, "TS2": TS2, "bj": bj, "dj": dj, "TIDn_star": TIDn_star
                        }
                        send_message(conn, M2)
                        print("[MS] Sent M2.")
        
                        # --- OTAKA Step 4: Receive M3 ---
                        M3 = recv_message(conn)
                        if not M3:
                            raise ConnectionError("Client disconnected during M3.")
        
                        print("\n[MS] Received M3:")
                        
                        if not check_timestamp(M3['TS3'], DELTA_T):
                            raise ValueError("M3 timestamp check failed.")
                        
                        ACK_prime = canonical_hash(TIDn, SK_ji, M3['TS3'])
                        
                        if ACK_prime != M3['ACK']:
                            raise ValueError("ACK verification failed.")
                            
                        print(f"[MS] ACK verified. Session with {IDi_stored} is fully established.")
                        print(f"[MS] Final Session Key: {SK_ji}")
                        
                        full_hash = h(t3, M3['TS3'])
                        session_iv = full_hash[:32]
                        
                        server_db[TIDi]['TIDn'] = "TIDn_user_1.0_new" 
                        print(f"[MS] Updated TID for user {IDi_stored}.")
                        
                        # ==========================================================
                        # --- CONTINUOUS AUTHENTICATION PHASE (IV-F) ---
                        # ==========================================================
                        print("\n[MS] --- Starting Continuous Authentication (IV-F) ---")
                        print(f"       Monitoring session for user: {session_user_id}")
                        
                        while True:
                            print("\n[MS] Waiting for encrypted CA data...")
                            ca_message = recv_message(conn)
                            if not ca_message or ca_message.get("type") != "CA_DATA":
                                print("[MS] Client disconnected.")
                                break
                            
                            encrypted_hex = ca_message["payload"]
                            vector_json = decrypt_data(session_key, session_iv, encrypted_hex)
                            vector = json.loads(vector_json)
                            print(f"[MS] Received and decrypted vector.")
        
                            vss_result = query_vss_server(vector)
                            if not vss_result or vss_result.get("status") != "match_found":
                                print("[MS] VSS query failed.")
                                continue 
                            
                            retrieved_id = vss_result["matched_user_id"]
                            distance = vss_result["distance"]
                            print(f"[MS] VSS query result: ID={retrieved_id}, Dist={distance:.4f}")
                            
                            if str(retrieved_id) == str(session_user_id):
                                print("[MS] User is valid. Sending OK.")
                                send_message(conn, {"status": "OK"})
                            else:
                                print(f"[MS] !!! MISMATCH: Expected '{session_user_id}' but VSS returned '{retrieved_id}'")
                                print("[MS] Session terminated.")
                                send_message(conn, {"status": "TERMINATE"})
                                break 
        
                    except (ValueError, ConnectionError) as e:
                        print(f"\n[MS] Session Error: {e}")
                    except Exception as e:
                        print(f"\n[MS] A fatal error occurred: {e}")
                    finally:
                        print("[MS] Closing connection.")

            # --- NEW: Add this block to handle Ctrl+C and keep the server alive ---
            except KeyboardInterrupt:
                print("\n[MS] Server shutting down (Ctrl+C).")
                break # Breaks the 'while True' loop
            except Exception as e:
                # This catches errors like a client disconnecting early
                print(f"[MS] An error occurred in the main accept loop: {e}")
                continue # Log error but keep server running

if __name__ == "__main__":
    run_server()