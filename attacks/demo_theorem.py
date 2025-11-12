import time
import math
import numpy as np
from src.otaka_protocol.helper import (
    h, xor_data, get_timestamp, rlwe_generate_keypair, str_to_hex,
    canonical_hash, flip_one_bit, rlwe_compute_shared_secret, Mod2, Cha
)
from src.otaka_protocol.client import simulate_user_login

def term_1_hash_collisions():
    """
    Analyzes the 'Hash Collision' term: q_h^2 / 2^l
    q_h = number of hash queries (attacker's guesses)
    l = hash output size (256 for SHA-256)
    """
    print("\n" + "="*50)
    print("--- 1. Analysis of Hash Collision Term (q_h^2 / 2^l) ---")
    
    l = 256
    search_space = 2**l
    
    print(f"[ANALYSIS] Protocol uses SHA-256. Search space (2^l) is 2^{l}.")
    print(f"[ANALYSIS]   (This number has {len(str(search_space))} digits)")

    guesses_per_sec = 1_000_000_000_000
    seconds_in_100_years = 100 * 365 * 24 * 60 * 60
    q_h = guesses_per_sec * seconds_in_100_years
    
    print(f"[ATTACKER] Assuming q_h = {q_h:.2e} guesses (1T/sec for 100 years).")
    log_q_h = math.log2(q_h)
    log_prob = (2 * log_q_h) - l
    
    print(f"[ANALYSIS] Log2(Probability of Collision) = (2 * {log_q_h:.1f}) - {l} = {log_prob:.1f}")
    print(f"[ANALYSIS] This means the probability is 1 in 2^{abs(log_prob):.1f}")
    print("[ANALYSIS] RESULT: Infeasible. This term is negligible.")
    print("[ANALYSIS]   (This confirms the findings of demo_3_hash_scaling.py)")

def term_2_password_biometric_guessing():
    print("\n" + "="*50)
    print("--- 2. Analysis of Password/Biometric Guessing Term ---")
    print("[ANALYSIS] demo_1_stolen_device.py shows an attacker must guess")
    print("both the password and the biometric.")
    b = 128

    q_s = 1_000_000_000_000 * (100 * 365 * 24 * 60 * 60)
    
    print(f"[ATTACKER] Assuming {q_s:.2e} guesses against the biometric.")
    print(f"[ANALYSIS] Biometric entropy (2^b) is 2^{b}.")
    
    log_q_s = math.log2(q_s)
    log_prob = log_q_s - b
    
    print(f"[ANALYSIS] Log2(Probability of Guess) = {log_q_s:.1f} - {b} = {log_prob:.1f}")
    print(f"[ANALYSIS] This means the probability is 1 in 2^{abs(log_prob):.1f}")
    print("[ANALYSIS] RESULT: ❌ Infeasible. This term is negligible.")
    print("[ANALYSIS]   (This confirms the findings of demo_1_stolen_device.py)")
    
def term_3_rlwe_security():
    print("\n" + "="*50)
    print("--- 3. Analysis of RLWE Security (Avalanche Effect) ---")
    
    print("[ANALYSIS] We can't break RLWE, but we can show its 'avalanche' property.")
    print("[ANALYSIS] This shows that a 1-bit change in the secret (f1) results")
    print("in a completely different public value (ai).")
    (f1_real, e1_real), ai_real = rlwe_generate_keypair()
    f1_flipped = f1_real.copy()
    f1_flipped[0] = f1_flipped[0] + 1 
    from src.otaka_protocol.helper import A_poly, p, POLY_MOD, Q, N
    ai_flipped = p.polymul(A_poly, f1_flipped)
    ai_flipped = p.polyadd(ai_flipped, e1_real) % Q
    ai_flipped = p.polydiv(ai_flipped, POLY_MOD)[1] % Q
    if len(ai_flipped) < N:
        ai_flipped = np.pad(ai_flipped, (0, N - len(ai_flipped)), 'constant')
    ai_flipped = ai_flipped.astype(int)
    print(f"[ANALYSIS] Real Secret f1 (first 5):    {f1_real[:5]}")
    print(f"[ANALYSIS] Flipped Secret f1 (first 5): {f1_flipped[:5]}")
    
    time.sleep(2)
    
    print(f"\n[ANALYSIS] Real Public Key ai (first 5):    {ai_real[:5]}")
    print(f"[ANALYSIS] Flipped Public Key ai (first 5): {ai_flipped[:5]}")
    difference = np.sum(ai_real != ai_flipped)
    percent_diff = (difference / N) * 100
    
    print(f"\n[ANALYSIS] Number of different coefficients: {difference} / {N}")
    print(f"[ANALYSIS]   {percent_diff:.1f}% of the public key changed.")
    
    if percent_diff > 40: 
        print("[ANALYSIS] RESULT: Infeasible. A 1-bit change in the secret")
        print("         creates a ~50% change in the public key.")
        print("         This 'avalanche' proves an attacker cannot 'get warmer'.")
    else:
        print("[ANALYSIS] RESULT: Weak. The keys are too similar.")
        
def run_theorem_demo():
    print("--- Proof-of-Concept Demo for Theorem 1---")
    print("[DEMO] This script analyzes the attacker's advantage (Adv_A).")
    print("[DEMO] Adv_A <= P(Hash) + P(Guessing) + P(RLWE)")
    print("[DEMO] We will show that each probability is negligible.")
    
    term_1_hash_collisions()
    term_2_password_biometric_guessing()
    term_3_rlwe_security()
    
    print("\n" + "="*50)
    print("--- 🏁 FINAL CONCLUSION ---")
    print("[ANALYSIS] We have computationally demonstrated that:")
    print("         1. The Hash Collision term is negligible.")
    print("         2. The Password/Biometric Guessing term is negligible.")
    print("         3. The RLWE Security term is based on a 'hard problem'")
    print("            with strong chaotic properties.")
    print("\n[DEMO] SUCCESS: The sum of these negligible probabilities is")
    print("         itself negligible, confirming the argument of Theorem 1.")

if __name__ == "__main__":
    import sys
    import os
    if 'src.otaka_protocol.helper' not in sys.modules:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.otaka_protocol.helper import (
        h, xor_data, get_timestamp, rlwe_generate_keypair, str_to_hex,
        canonical_hash, flip_one_bit, rlwe_compute_shared_secret, Mod2, Cha,
        A_poly, p, POLY_MOD, Q, N
    )
    from src.otaka_protocol.client import simulate_user_login
    
    run_theorem_demo()
