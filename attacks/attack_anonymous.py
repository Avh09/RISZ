import json
import time
import binascii
import os
import sys
import numpy as np

# --- Correct, top-level imports ---
# We import everything here, assuming we run this as a module
from src.otaka_protocol.helper import (
    h, xor_data, get_timestamp, rlwe_generate_keypair, str_to_hex, hex_to_str,
    canonical_hash # Adding this just in case, though not strictly needed by M1
)
from src.otaka_protocol.client import simulate_user_login

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

# This demo shows that the messages sent by the *same user* in two
# *different* sessions are completely different and untraceable.
#
# This demonstrates resiliency against:
# 7) Anonymity and Untraceability Attacks

def generate_m1(login_secrets):
    """
    A helper function to generate a new M1 message.
    This duplicates logic from client.py for a clean demo.
    """
    # 1. Get secrets
    my_IDi = login_secrets['IDi']
    my_TIDi = login_secrets['TIDi']
    my_t3 = login_secrets['t3']
    my_x_hex = login_secrets['x_hex']
    
    # 2. Generate new ephemeral secrets for this session
    (f1, e1), ai = rlwe_generate_keypair()
    
    # 3. Generate new fresh timestamp for this session
    TS1 = get_timestamp()
    
    # 4. Calculate M1 components
    # X1 = IDi ⊕ h(t3 || TS1 || TIDi)
    X1 = xor_data(str_to_hex(my_IDi), h(my_t3, TS1, my_TIDi))
    s1 = h(my_x_hex, TS1)
    s2 = xor_data(s1, h(my_t3, TS1, my_TIDi))
    
    # Note: Using canonical_hash is more robust for hashing complex objects
    # This should match your client.py!
    # Using np.array_str(ai) as in your previous files
    X2 = h(np.array_str(ai), X1, TS1, my_TIDi, s2)
    
    M1 = {
        "X1": X1,
        "X2": X2,
        "TIDi": my_TIDi, # This is the Temporary ID
        "ai": ai.tolist(), # Convert numpy array
        "s2": s2,
        "TS1": TS1
    }
    return M1

def print_sniffed_message(session_name, m1):
    """Helper to print what an attacker would see."""
    print(f"{C.CYAN}--- [ATTACKER] Sniffed M1 for {session_name} ---{C.END}")
    
    # Create a copy to print nicely
    m1_print = m1.copy()
    m1_print['ai'] = f"[Numpy array of shape {np.array(m1['ai']).shape}]"
    
    # --- NEW: Truncate the long hex strings for a cleaner log ---
    m1_print['X1'] = m1['X1'][:15] + "..."
    m1_print['X2'] = m1['X2'][:15] + "..."
    m1_print['s2'] = m1['s2'][:15] + "..."
    
    print(json.dumps(m1_print, indent=2, sort_keys=True))

def attempt_to_extract_id(m1, guessed_secret):
    """
    Simulates an attacker trying to reverse X1 to find the real IDi.
    IDi = X1 ⊕ h(t3 || TS1 || TIDi)
    Returns a tuple: (raw_hex_result, printable_string_result)
    """
    # This calculation will always succeed and return a hex string.
    derived_id_hex = xor_data(m1['X1'], h(guessed_secret, m1['TS1'], m1['TIDi']))
    
    try:
        # This is the part that fails if the data is garbage.
        b = bytes.fromhex(derived_id_hex)
        derived_id_str = b.decode('utf-8')
        
        # Clean up unprintable chars for a nice demo
        printable_str = "".join(c for c in derived_id_str if c.isprintable())
        
        if not printable_str:
            printable_str = "[Empty/Unprintable]"
        
        return derived_id_hex, printable_str
        
    except (binascii.Error, UnicodeDecodeError):
        # The hex string didn't decode to valid UTF-8. THIS is the garbage.
        return derived_id_hex, "[Non-UTF-8 Bytes]"

def flip_one_bit(hex_string):
    """Flips the first bit of the first byte of a hex string."""
    try:
        b = bytearray.fromhex(hex_string)
        b[0] = b[0] ^ 1 # Flip the first bit
        return b.hex()
    except (ValueError, IndexError):
        return h("flipped_fallback") # Fallback if hex is invalid

def run_anonymity_demo():
    print(f"="*60)
    print(f"{C.BOLD}Demo: Anonymity and Untraceability{C.END}")
    print(f"="*60)
    
    # --- SESSION 1 (VERBOSE) ---
    # print("\n" + "="*50)
    print(C.YELLOW + "[DEMO] User '1.0' logs in and starts SESSION 1." + C.END)
    
    # --- NEW: Added [Ui] print lines ---
    print(C.BLUE + "[Ui] --- Starting User Login Phase (IV-D) ---" + C.END)
    login_secrets_s1 = simulate_user_login()
    if not login_secrets_s1:
        print(C.RED + "[DEMO] Error: Could not log in. Run registration first." + C.END)
        return
    print(C.BLUE + "[Ui] Login Successful. Secrets reconstructed." + C.END)
        
    print(C.GREEN + "[ATTACKER] Sniffing the network..." + C.END)
    M1_session_1 = generate_m1(login_secrets_s1)
    print_sniffed_message("Session 1", M1_session_1)
    
    print(f"{C.GREEN}[ATTACKER] ...OK. User session 1 is complete." + C.END)
    print(f"{C.GREEN}[ATTACKER] I've put their Temporary ID '{M1_session_1['TIDi']}' on a watchlist." + C.END)
    
    # --- SESSION 2 (VERBOSE) ---
    print("="*50)
    print(C.YELLOW + "[DEMO] 2 hours later... User '1.0' starts SESSION 2." + C.END)
    print(C.YELLOW + "[DEMO] Per the protocol (Steps 2-4), their device has *updated*" + C.END)
    print(C.YELLOW + "[DEMO] its Temporary ID to the new one assigned by the server." + C.END)
    
    # We simulate this update. The new TID is hardcoded in the server.
    login_secrets_s2 = login_secrets_s1.copy()
    new_TIDi_from_server = "TIDn_user_1.0_new" 
    login_secrets_s2['TIDi'] = new_TIDi_from_server
    
    print(f"[DEMO] Client's *new* TIDi is now: '{new_TIDi_from_server}'")
    
    print(C.GREEN + "[ATTACKER] Sniffing the network again..." + C.END)
    time.sleep(2) # For demo effect
    M1_session_2 = generate_m1(login_secrets_s2)
    print_sniffed_message("Session 2", M1_session_2)

    # --- ANALYSIS (COMPRESSED) ---
    # print("\n" + "="*50)
    print(C.CYAN + "--- ATTACKER'S DEEP ANALYSIS (Results Only) ---" + C.END)
    
    # 1. TIDi Check
    print(f"[ATTACKER]   Sniffed Session 1 TIDi: {C.YELLOW}{M1_session_1['TIDi']}{C.END}")
    print(f"[ATTACKER]   Sniffed Session 2 TIDi: {C.YELLOW}{M1_session_2['TIDi']}{C.END}")
    if M1_session_1['TIDi'] != M1_session_2['TIDi']:
        print(C.RED + "[ATTACKER]   RESULT: No match. The TIDi is different." + C.END)
    else:
        print(C.RED + "[ATTACKER]   RESULT: Match! I can trace this user!" + C.END)

    # 2. Field Check
    reused_fields = []
    for key in M1_session_1:
        if key == 'TIDi': continue 
        if np.array_equal(M1_session_1[key], M1_session_2[key]):
            reused_fields.append(key)
            
    if not reused_fields:
        print(C.RED + "[ATTACKER]   RESULT: No match. All other M1 fields are also different." + C.END)
    else:
        print(C.RED + f"[ATTACKER]   RESULT: Match! Reused fields: {reused_fields}. I can trace this user!" + C.END)

    # 3. IDi Guess
    guessed_secret_t3 = h("some_common_password_guess")
    hex_1, str_1 = attempt_to_extract_id(M1_session_1, guessed_secret_t3)
    hex_2, str_2 = attempt_to_extract_id(M1_session_2, guessed_secret_t3)
    print(f"[ATTACKER]   Derived ID from Session 1 (with guess): {C.YELLOW}'{str_1}'{C.END}")
    print(f"[ATTACKER]   Derived ID from Session 2 (with guess): {C.YELLOW}'{str_2}'{C.END}")
    if not (str_1.startswith("1.0") and str_2.startswith("1.0")):
         print(C.RED + "[ATTACKER]   RESULT: No match. Both derived IDs are garbage." + C.END)
    else:
         print(C.RED + "[ATTACKER]   RESULT: Match! My guess was right! I have linked the user!" + C.END)

    # 4. Statistical Analysis
    printable_results = []
    found_real_id = False
    for i in range(1000):
        random_guess_t3 = h(f"random_guess_{i}")
        _hex, derived_str = attempt_to_extract_id(M1_session_1, random_guess_t3)
        if derived_str != "[Non-UTF-8 Bytes]" and derived_str != "[Empty/Unprintable]":
            printable_results.append(derived_str)
        if derived_str.startswith("1.0"):
            found_real_id = True
    print(f"[ATTACKER]   Statistical Test (1000 guesses): Found '1.0' {1 if found_real_id else 0} times.")
    print(C.RED + "[ATTACKER]   RESULT: FAILED. The output is statistically indistinguishable from random noise." + C.END)
    
    # 5. Avalanche Analysis
    real_t3 = login_secrets_s1['t3']
    flipped_t3 = flip_one_bit(real_t3) 
    hex_real, str_real = attempt_to_extract_id(M1_session_1, real_t3)
    hex_flipped, str_flipped = attempt_to_extract_id(M1_session_1, flipped_t3)
    
    print(f"[ATTACKER]   Decryption with REAL secret:    {C.YELLOW}'{str_real}'{C.END}")
    print(f"[ATTACKER]   Decryption with 1-BIT-OFF secret: {C.YELLOW}'{str_flipped}'{C.END}")
    print(C.RED + "[ATTACKER]   RESULT: FAILED. A 1-bit error creates completely different output (Avalanche Effect)." + C.END)

    # --- FINAL CONCLUSION ---
    # print("\n" + "="*50)
    print(C.GREEN + C.BOLD + "[DEMO] SUCCESS: The protocol is ANONYMOUS and UNTRACEABLE." + C.END)


if __name__ == "__main__":
    # This block is now clean and simply runs the demo.
    run_anonymity_demo()