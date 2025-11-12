import hashlib
import time
import itertools
import string
import math


TARGET_HASH = "2b1f7f98b13c706b4d371101b07289d0689b69d4c7b80c35467380a96f1c4e72"
CHARSET = string.ascii_lowercase + string.digits 

def find_prefix_match(prefix_length, charset, max_len):
    target_prefix = TARGET_HASH[:prefix_length]
    print(f"\n[ATTACK] Starting search for prefix: '{target_prefix}' ({prefix_length} chars)")
    
    counter = 0
    start_time = time.time()
    for length in range(1, max_len + 1):
        for guess_tuple in itertools.product(charset, repeat=length):
            guess = "".join(guess_tuple)
            guess_hash = hashlib.sha256(guess.encode('utf-8')).hexdigest()
            counter += 1

            if counter % 1_000_000 == 0:
                elapsed = time.time() - start_time
                rate = counter / elapsed
                print(f"[ATTACK] ... {counter:,} hashes checked ({rate:,.0f} h/s)")

            if guess_hash.startswith(target_prefix):
                elapsed = time.time() - start_time
                print(f"[ATTACK]    SUCCESS! Found a match.")
                print(f"[ATTACK]   Input:   '{guess}'")
                print(f"[ATTACK]   Hash:    {guess_hash}")
                print(f"[ATTACK]   Time:    {elapsed:.4f} seconds")
                print(f"[ATTACK]   Guesses: {counter:,}")
                return elapsed, counter, guess_hash
                
    elapsed = time.time() - start_time
    print(f"[ATTACK] ❌ FAILED. No match found in search space (up to len {max_len}).")
    return elapsed, counter, None

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    if seconds < 3600:
        return f"{seconds/60:.2f} minutes"
    if seconds < 86400:
        return f"{seconds/3600:.2f} hours"
    if seconds < 31536000:
        return f"{seconds/86400:.2f} days"
    if seconds < 31536000 * 10**9:
        return f"{seconds/31536000:.2f} years"
    # For truly massive numbers
    return f"{seconds / 31536000:.2e} years (astronomical)"


def run_scaling_demo():
    print("--- Demo: The Exponential Scaling of Hash Unbreakability ---")
    print(f"[DEMO] We will attack SHA-256 by trying to match its output prefix.")
    print(f"[DEMO] The search space is {len(CHARSET)} characters, up to length {MAX_GUESS_LENGTH}.")

    print("-" * 60)
    print("[DEMO] STEP 1: TRIVIAL (4-char prefix)")
    time.sleep(1)
    find_prefix_match(4, CHARSET, MAX_GUESS_LENGTH)

    print("-" * 60)
    print("[DEMO] STEP 2: EASY (6-char prefix)")
    time.sleep(1)
    find_prefix_match(6, CHARSET, MAX_GUESS_LENGTH)

    print("-" * 60)
    print("[DEMO] STEP 3: HARD (8-char prefix)")
    time.sleep(1)
    elapsed_8, count_8, _ = find_prefix_match(8, CHARSET, MAX_GUESS_LENGTH)
    
    if elapsed_8 < 0.1: 
        print("\n[DEMO] 8-char search was too fast to measure. Stopping demo.")
        return

    print("-" * 60)
    print("[DEMO] STEP 4: THE EXTRAPOLATION")
    hashes_per_second = count_8 / elapsed_8
    print(f"[DEMO] Machine's measured speed: {hashes_per_second:,.0f} hashes/second")
    print("[DEMO] Now, let's calculate the time to crack longer prefixes...")
    time.sleep(3)
    guesses_10 = 16**10
    time_10 = guesses_10 / hashes_per_second
    print(f"\n[CALC] Time for 10-char prefix (16^10 guesses):")
    print(f"[CALC]  {format_time(time_10)}")
    guesses_12 = 16**12
    time_12 = guesses_12 / hashes_per_second
    print(f"\n[CALC] Time for 12-char prefix (16^12 guesses):")
    print(f"[CALC]  {format_time(time_12)}")
    guesses_32 = 16**32
    time_32 = guesses_32 / hashes_per_second
    print(f"\n[CALC] Time for 32-char prefix (half the hash):")
    print(f"[CALC]  {format_time(time_32)}")
    guesses_full = 16**64 
    time_full = guesses_full / hashes_per_second
    print(f"\n[CALC] Time for FULL 64-char hash (16^64 guesses):")
    print(f"[CALC]  {time_full:e} seconds")
    print(f"[CALC]  (This number is ~{time_full / 31536000:.2e} years)")
    


if __name__ == "__main__":
    run_scaling_demo()
