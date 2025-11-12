import socket
import json
import numpy as np
import time
import sys
from src.otaka_protocol.helper import (
    canonical_hash, h, hex_to_str, str_to_hex, xor_data, get_timestamp, 
    check_timestamp, send_message, recv_message,
    rlwe_generate_keypair, rlwe_compute_shared_secret, Mod2
)
HOST = '127.0.0.1'
PORT = 65432
DELTA_T = 10  
CLIENT_STORAGE_FILE = "client_storage.json"

class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"

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
        print(f"{C.RED}Error: Local login failed.{C.END}")
        return None

def run_attack():
    print(f"\n{'='*60}")
    print(f"  Demo: Handshake Replay Attack (M1)")
    print(f"  Tests if the server rejects stale timestamps.")
    print(f"{'='*60}")
    
    login_secrets = simulate_user_login()
    if not login_secrets:
        return

    my_IDi = login_secrets['IDi']
    my_TIDi = login_secrets['TIDi']
    my_t3 = login_secrets['t3']
    my_x_hex = login_secrets['x_hex']
    
    captured_M1 = None
    print(f"\n{C.BLUE}[Phase 1: Perform legitimate session to capture M1]{C.END}")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            print(f"{C.CYAN}  Connection 1 established → {HOST}:{PORT}{C.END}")
            (f1, e1), ai = rlwe_generate_keypair()
            TS1 = get_timestamp()
            X1 = xor_data(str_to_hex(my_IDi), h(my_t3, TS1, my_TIDi))
            s1 = h(my_x_hex, TS1)
            s2 = xor_data(s1, h(my_t3, TS1, my_TIDi))
            X2 = h(np.array_str(ai), X1, TS1, my_TIDi, s2)
            M1 = {"X1": X1, "X2": X2, "TIDi": my_TIDi, "ai": ai, "s2": s2, "TS1": TS1}
            
            captured_M1 = M1.copy()
            print(f"{C.GREEN}  Captured M1 with fresh TS1: {truncate(TS1, 40)}{C.END}")

            send_message(s, M1)
            M2 = recv_message(s)
            if not M2:
                raise ValueError("Server closed connection, M2 not received.")
            
            print(f"{C.CYAN}  Received M2, completing handshake...{C.END}")
            bj = np.array(M2['bj']); dj = np.array(M2['dj'])
            c_prime_j = rlwe_compute_shared_secret(f1, bj)
            w_prime_j = Mod2(c_prime_j, dj)
            SK_ij = h(my_IDi, w_prime_j, M2['TS2'], TS1, s1, my_t3, my_TIDi)
            hash_hex_cli = canonical_hash(SK_ij, M2['TS2'], my_t3, my_TIDi)
            TIDn_hex = xor_data(M2['TIDn_star'], hash_hex_cli)
            
            my_TIDi_next = hex_to_str(TIDn_hex)       
            TS3 = get_timestamp()
            ACK = canonical_hash(my_TIDi_next, SK_ij, TS3)
            M3 = {"ACK": ACK, "TS3": TS3}
            send_message(s, M3)
            print(f"{C.GREEN}  Session 1 handshake complete. Closing socket.{C.END}")

    except Exception as e:
        print(f"{C.RED}  Error in Phase 1: {e}{C.END}")
        return
    
    if not captured_M1:
        print(f"{C.RED}  Failed to capture M1. Aborting.{C.END}")
        return
    wait_time = DELTA_T + 2
    print(f"\n{C.BLUE}[Phase 2: Waiting {wait_time}s for TS1 to become stale...]{C.END}")
    time.sleep(wait_time)
    print(f"{C.CYAN}  Done waiting. Now replaying stale M1.{C.END}")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_attacker:
            s_attacker.connect((HOST, PORT))
            print(f"{C.CYAN}  Connection 2 (Attacker) established → {HOST}:{PORT}{C.END}")

            print(f"{C.YELLOW}  Sending STALE M1 (TS1: {truncate(captured_M1['TS1'], 40)})...{C.END}")
            send_message(s_attacker, captured_M1)

            print(f"{C.CYAN}  Waiting for M2 response...{C.END}")
            s_attacker.settimeout(3.0)
            M2_response = recv_message(s_attacker)

            if not M2_response:
                print(f"\n{C.GREEN}{C.BOLD}--- TEST SUCCESSFUL ---{C.END}")
                print(f"{C.GREEN} Server did not send M2 and closed the connection.{C.END}")
                print(f"{C.GREEN} Replay attack failed, as expected.{C.END}")
                print(f"{C.GREEN} The paper's claim is VERIFIED.{C.END}")
            else:
                print(f"\n{C.RED}{C.BOLD}--- TEST FAILED ---{C.END}")
                print(f"{C.RED} VULNERABILITY CONFIRMED: Server sent M2!{C.END}")
                print(f"{C.RED} Server is NOT checking timestamps.{C.END}")

    except (socket.timeout, ConnectionResetError, BrokenPipeError):
        print(f"\n{C.GREEN}{C.BOLD}--- TEST SUCCESSFUL ---{C.END}")
        print(f"{C.GREEN} Server forcefully closed connection (timeout/reset).{C.END}")
        print(f"{C.GREEN} Replay attack failed, as expected.{C.END}")
        print(f"{C.GREEN} The paper's claim is VERIFIED.{C.END}")
    except Exception as e:
        print(f"{C.RED}  Attack Error: {e}{C.END}")

if __name__ == "__main__":
    run_attack()
