# --- Colors ---
class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"

import socket
import json
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split

# --- Import all the necessary functions from your helper file ---
from src.otaka_protocol.helper import (
    canonical_hash, h, hex_to_str, str_to_hex, xor_data, get_timestamp, 
    send_message, recv_message,
    rlwe_generate_keypair, rlwe_compute_shared_secret, Mod2, Cha,
    encrypt_data
)

# --- Configuration ---
HOST = '127.0.0.1'
PORT = 65432
CLIENT_STORAGE_FILE = "client_storage.json"
DATASET_PATH = 'features_extracted.csv'
FEATURE_COLUMNS = [
    'strokeDuration', 'startX', 'startY', 'stopX', 'stopY',
    'directEndToEndDistance', 'meanResultantLength', 'upDownLeftRightFlag',
    'directionOfEndToEndLine', 'largestDeviationFromEndToEndLine',
    'averageDirection', 'lengthOfTrajectory', 'averageVelocity',
    'midStrokePressure', 'midStrokeArea'
]
# --- End Configuration ---

# --- Global truncation options ---
np.set_printoptions(threshold=10, edgeitems=3, linewidth=100, suppress=True)

def truncate(x, n=25):
    """Return a short preview of x (string/list/np.array)."""
    if x is None:
        return "None"
    if isinstance(x, (list, np.ndarray)):
        x = np.array2string(np.array(x).flatten(), threshold=5)
    s = str(x)
    return s if len(s) <= n else s[:n] + "..."

def simulate_user_login():
    """Simulates the legitimate user login phase."""
    try:
        with open(CLIENT_STORAGE_FILE, "r") as f:
            client_data = json.load(f)
    except FileNotFoundError:
        print(f"{C.RED}Error: '{CLIENT_STORAGE_FILE}' not found.{C.END}")
        print(f"{C.YELLOW}Please run 'python -m src.otaka_protocol.registration' first.{C.END}")
        return None

    # --- Use the same test credentials as registration ---
    IDi_input = "1.0"
    PWi_star_input = "password123"
    BMi_star_input = "biometric_data_scan_1" 
    
    sigma_i_star = h(BMi_star_input)
    t3_rec = xor_data(client_data['t3_star'], h(sigma_i_star, IDi_input))
    x_rec_hex = xor_data(client_data['x_star'], h(PWi_star_input, sigma_i_star, t3_rec))
    t2_prime = xor_data(x_rec_hex, h(IDi_input, PWi_star_input, sigma_i_star))
    
    if t2_prime == client_data['t2']:
        return {
            "IDi": client_data['IDi'], "TIDi": client_data['TIDi'],
            "t3": t3_rec, "x_hex": x_rec_hex
        }
    else:
        return None

def load_impostor_vector(legitimate_user_id):
    """
    Loads one 'live' test vector from a DIFFERENT user (an impostor).
    """
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"{C.RED}Error: Dataset '{DATASET_PATH}' not found.{C.END}")
        return None
    
    df['upDownLeftRightFlag'], _ = pd.factorize(df['upDownLeftRightFlag'])
    
    # Find the first available user_id that is NOT the legitimate one
    impostor_id = None
    for uid in df['user_id'].unique():
        if str(uid) != str(legitimate_user_id):
            impostor_id = uid
            break
            
    if impostor_id is None:
        print(f"{C.RED}Error: Could not find any other users in the dataset to test.{C.END}")
        return None

    impostor_df = df[df['user_id'] == float(impostor_id)]
    
    # Get the 20% "test" data for the impostor
    _, test_data = train_test_split(impostor_df, test_size=0.20, shuffle=False)
    if test_data.empty:
        print(f"{C.RED}Error: No live test vectors found for impostor {impostor_id}{C.END}")
        return None
        
    print(f"{C.GREEN}  Loaded one 'live' vector from impostor user: {impostor_id}{C.END}")
    return test_data[FEATURE_COLUMNS].values.tolist()[0]

def run_attack():
    print(f"="*60)
    print(f"  Demo: Impersonation Attack (Forged CA Data)")
    print(f"  Tests if VSS can reject forged data in a valid session.")
    print(f"="*60)
    
    login_secrets = simulate_user_login()
    if not login_secrets:
        return

    my_IDi = login_secrets['IDi']
    my_TIDi = login_secrets['TIDi']
    my_t3 = login_secrets['t3']
    my_x_hex = login_secrets['x_hex']
    
    print(f"{C.CYAN}\n[Phase 1: Impersonate User '{my_IDi}' via Handshake]{C.END}")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            print(f"  Attacker connection established → {HOST}:{PORT}")

            # --- OTAKA handshake (Simulating a logged-in attacker) ---
            (f1, e1), ai = rlwe_generate_keypair()
            TS1 = get_timestamp()
            X1 = xor_data(str_to_hex(my_IDi), h(my_t3, TS1, my_TIDi))
            s1 = h(my_x_hex, TS1)
            s2 = xor_data(s1, h(my_t3, TS1, my_TIDi))
            X2 = h(np.array_str(ai), X1, TS1, my_TIDi, s2)
            M1 = {"X1": X1, "X2": X2, "TIDi": my_TIDi, "ai": ai, "s2": s2, "TS1": TS1}
            
            send_message(s, M1)
            M2 = recv_message(s)
            bj = np.array(M2['bj']); dj = np.array(M2['dj'])
            
            c_prime_j = rlwe_compute_shared_secret(f1, bj)
            w_prime_j = Mod2(c_prime_j, dj)
            SK_ij = h(my_IDi, w_prime_j, M2['TS2'], TS1, s1, my_t3, my_TIDi)
            hash_hex_cli = canonical_hash(SK_ij, M2['TS2'], my_t3, my_TIDi)
            TIDn_hex = xor_data(M2['TIDn_star'], hash_hex_cli)
            TIDn = hex_to_str(TIDn_hex)       
            
            my_TIDi_next = TIDn 
            TS3 = get_timestamp()
            ACK = canonical_hash(my_TIDi_next, SK_ij, TS3)
            M3 = {"ACK": ACK, "TS3": TS3}
            send_message(s, M3)
            print(f"{C.GREEN}  Handshake complete. Attacker is authenticated as User '{my_IDi}'.{C.END}")
            
            # --- Establish encryption keys ---
            session_key = SK_ij
            session_iv = h(my_t3, TS3)[:32] # The static IV
            print(f"{C.YELLOW}  Session Key established: {truncate(session_key)}{C.END}")

            # --- FORGED DATA PHASE ---
            print(f"{C.CYAN}\n[Phase 2: Load Forged (Impostor) Data]{C.END}")
            forged_vector = load_impostor_vector(my_IDi)
            if not forged_vector:
                raise ValueError("Could not load forged vector.")
            
            print(f"{C.YELLOW}  Vector preview: {truncate(forged_vector, 80)}{C.END}")

            # --- INJECTION PHASE ---
            print(f"{C.CYAN}\n[Phase 3: Inject Forged Data into Valid Session]{C.END}")
            vector_json = json.dumps(forged_vector)
            encrypted_hex = encrypt_data(session_key, session_iv, vector_json)
            print(f"{C.YELLOW}  Encrypted impostor vector: {truncate(encrypted_hex)}{C.END}")

            P_impostor_message = {"type": "CA_DATA", "payload": encrypted_hex}
            send_message(s, P_impostor_message)
            response = recv_message(s)
            
            # --- ANALYZE RESULT ---
            print(f"{C.CYAN}\n[Phase 4: Analyze Server Response]{C.END}")
            if response and response.get("status") == "TERMINATE":
                print(f"  Server Response: 'TERMINATE'")
                print(f"{C.GREEN}\n--- TEST SUCCESSFUL ---{C.END}")
                print(f"{C.GREEN}VSS defense WORKED.{C.END}")
                print(f"{C.GREEN}Server detected the mismatch and terminated the session.{C.END}")
            elif response and response.get("status") == "OK":
                print(f"  Server Response: 'OK'")
                print(f"{C.RED}\n--- TEST FAILED ---{C.END}")
                print(f"{C.RED}❌ VULNERABILITY CONFIRMED: Server accepted impostor data!{C.END}")
            else:
                print(f"  Server Response: {response}")
                print(f"{C.YELLOW}\n--- TEST INCONCLUSIVE ---{C.END}")
                print(f"{C.YELLOW}Server did not send a clear OK or TERMINATE.{C.END}")

    except (ValueError, ConnectionError, socket.error) as e:
        print(f"\n[Attacker] Error: {truncate(e, 120)}")
    finally:
        print(f"{C.BLUE}[Attacker] Connection closed.{C.END}")

if __name__ == "__main__":
    run_attack()
