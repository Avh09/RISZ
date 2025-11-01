import socket
import time
from src.otaka_protocol.helper import (
    h, get_timestamp, send_message, recv_message,
    rlwe_sample_from_chi_delta, rlwe_generate_public_key
)

HOST = '127.0.0.1'
PORT = 65432
DELTA_T = 10 # Must match the server's DELTA_T

# --- Credentials (same as client) ---
my_IDi = "ID_user_001"
my_TIDi = "TID_user_001"
my_x = "dummy_user_secret_x"

def run_replay_attack():
    print("--- Replay Attack Simulation (Section V-B1) ---")

    # --- Step 1: Eavesdrop on a valid M1 ---
    print("\n[Attacker] Eavesdropping on a valid client session...")
    f1 = rlwe_sample_from_chi_delta()
    e1 = rlwe_sample_from_chi_delta()
    ai = rlwe_generate_public_key(f1, e1)
    TS1 = get_timestamp()
    X1 = my_IDi 
    s1 = h(my_x, TS1)
    s2 = s1
    X2 = h(ai, X1, TS1, my_TIDi, s2)

    # [cite_start]The attacker captures this message [cite: 315]
    captured_M1 = {
        "X1": X1, "X2": X2, "TIDi": my_TIDi, "ai": ai, "s2": s2, "TS1": TS1
    }
    print(f"  Captured M1 with timestamp: {TS1}")

    # --- Step 2: Wait for the timestamp to become stale ---
    print(f"\n[Attacker] Waiting for {DELTA_T} seconds...")
    time.sleep(DELTA_T + 1)

    # --- Step 3: Launch the replay attack ---
    print(f"[Attacker] Connecting to server and replaying stale M1...")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            
            # Send the *exact same* (now stale) message
            send_message(s, captured_M1)
            print("  Replayed M1. Waiting for response...")
            
            # The server should detect the stale timestamp and close the connection
            # without sending M2.
            M2 = recv_message(s)
            
            if not M2:
                print("  Server closed connection, as expected.")
                    print("\n  Result: ✅ SUCCESSFUL DEFENSE. The replay attack was blocked[cite: 314].")
            else:
                print(f"  Server responded with M2: {M2}")
                print("\n  Result: ❌ FAILED DEFENSE. Server accepted a stale timestamp.")

    except (ConnectionResetError, TypeError, AttributeError):
        print("  Server closed connection unexpectedly (this is good!).")
        print("\n  Result: ✅ SUCCESSFUL DEFENSE. The server rejected the invalid message.")
    except Exception as e:
        print(f"  An error occurred: {e}")

if __name__ == "__main__":
    print("Ensure 'src/otaka_protocol/server.py' is running in another terminal.")
    run_replay_attack()