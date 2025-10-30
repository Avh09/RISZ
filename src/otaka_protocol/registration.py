import json
import os
import hashlib
from .helper import h, str_to_hex, xor_data

def run_registration():
    """
    Simulates the User Registration and FVDB Creation Phase (Section IV-C).
    """
    print("--- Running Registration Phase (Section IV-C) ---")

    # --- Step 1: Ui picks credentials ---
    # CRITICAL: This IDi MUST match a user_id in your 'features_extracted.csv'
    IDi = "1.0" 
    PWi = "password123"
    BMi = "biometric_data_scan_1"
    
    # Simulate Fuzzy Extractor Gen(BMi)
    sigma_i = h(BMi) # Biometric secret
    tau_i = h(BMi, "helper") # Public reproduction parameter
    print(f"User '{IDi}' secrets generated.")
    
    # --- Step 2: Ui calculates t1, t2 and sends to MS ---
    x_secret = os.urandom(32).hex() # User's random secret x
    t1 = h(IDi, PWi, sigma_i)
    t2 = xor_data(x_secret, t1)
    
    # --- Step 2 (Server Side): MS receives request and computes t3 ---
    k_master_secret = os.urandom(32).hex() # Server's long-term secret key k
    TIDi = "TID_user_1.0" # Server picks a temporary ID
    t3 = xor_data(t2, h(k_master_secret, IDi))
    print("MS generated t3 and TIDi.")
    
    # --- Step 3: Ui receives {TIDi, t3} and stores credentials ---
    t3_star = xor_data(t3, h(sigma_i, IDi))
    x_star = xor_data(x_secret, h(PWi, sigma_i, t3))
    
    # Store credentials on the client's device
    client_storage = {
        "IDi": IDi,
        "TIDi": TIDi,
        "t2": t2,
        "t3_star": t3_star,
        "x_star": x_star,
        "tau_i": tau_i
    }
    with open("client_storage.json", "w") as f:
        json.dump(client_storage, f, indent=2)
    print("Client credentials stored in 'client_storage.json'")

    # Store credentials on the server's database
    server_storage = {
        TIDi: {
            "IDi": IDi,
            "t3": t3,
            "TIDn": "TIDn_user_1.0_initial" # New TID for next session
        },
        "MASTER_SECRET_K": k_master_secret
    }
    with open("server_storage.json", "w") as f:
        json.dump(server_storage, f, indent=2)
    print("Server credentials stored in 'server_storage.json'")
    print("--- Registration Complete ---")

if __name__ == "__main__":
    run_registration()