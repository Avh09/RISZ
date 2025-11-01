import hashlib
import os
import json
from datetime import datetime, timezone
import numpy as np
from numpy.polynomial import polynomial as p

# ==============================================================================
# CRYPTOGRAPHIC PRIMITIVES (Section VI)
# ==============================================================================
import json
import hashlib

def canonical_hash(*args) -> str:
    """
    Deterministic hex-string SHA-256 of the canonical serialization
    of each argument. Always returns a hex string.
    Cleans null bytes and whitespace to ensure consistent hashing across devices.
    """
    parts = []
    for a in args:
        # --- Type normalization ---
        if isinstance(a, bytes):
            a_str = a.decode('utf-8', errors='ignore')
        elif isinstance(a, str):
            a_str = a
        elif hasattr(a, 'tolist'):  # numpy arrays, etc.
            a_str = json.dumps(a.tolist(), separators=(',',':'))
        else:
            a_str = str(a)

        # --- Clean up ---
        a_str = a_str.replace('\x00', '').strip()
        parts.append(a_str)

    # --- Join and hash ---
    h_input = "||".join(parts).encode('utf-8')
    print("Hash input", h_input)
    return hashlib.sha256(h_input).hexdigest()

def str_to_hex(s: str) -> str:
    return s.encode('utf-8').hex()

def hex_to_str(hx: str) -> str:
    return bytes.fromhex(hx).decode('utf-8')
def h(*args):
    """
    The post-quantum secure hash function h(·).
    We use SHA-256 as specified.
    Returns a hex-encoded string.
    """
    h_input = "||".join(map(str, args)).encode('utf-8')
    return hashlib.sha256(h_input).hexdigest()

def xor_data(hex_str1, hex_str2):
    """
    Performs a stable XOR operation on two hex-encoded strings.
    This is the implementation of the ⊕ operator in the paper.
    """
    b1 = bytes.fromhex(hex_str1)
    b2 = bytes.fromhex(hex_str2)
    
    # Pad the shorter byte string to match lengths
    if len(b1) > len(b2):
        b2 = b2.ljust(len(b1), b'\0')
    elif len(b2) > len(b1):
        b1 = b1.ljust(len(b2), b'\0')
        
    xored_bytes = bytes([_a ^ _b for _a, _b in zip(b1, b2)])
    return xored_bytes.hex()

def get_timestamp():
    """Returns a fresh timestamp TS."""
    return datetime.now(timezone.utc).isoformat()

def check_timestamp(ts, delta_t=10):
    """
    Verifies freshness: |TS* - TS| < Delta_T.
    Returns True if fresh, False if stale (replayed).
    """
    try:
        ts_dt = datetime.fromisoformat(ts)
        now = datetime.now(timezone.utc)
        time_diff = abs((now - ts_dt).total_seconds())
        return time_diff < delta_t
    except (ValueError, TypeError):
        return False

# ==============================================================================
# REAL RLWE IMPLEMENTATION (Section III & IV-E)
# This section replaces the dummy string functions.
# ==============================================================================

# System Parameters
N = 1024
Q = 1073479681 # Paper's q = 1073479681
# Define the polynomial ring Rq = Zq[x] / (x^n + 1)
POLY_MOD = [1] + [0] * (N - 1) + [1] 
PARAMS_FILE = "rlwe_params.npz"

def make_random_A_poly(seed=None):
    # deterministic RNG if seed provided
    rng = np.random.default_rng(seed)
    A = (rng.integers(0, Q, size=(N))).astype(np.int64)
    # reduce modulo polynomial (x^N + 1)
    A = p.polydiv(A, POLY_MOD)[1].astype(int) % Q
    if len(A) < N:
        A = np.pad(A, (0, N - len(A)), 'constant').astype(int)
    return A


# Public parameter 'alpha' (called 'A' in the GitHub code)
# This is generated once and published by the MS
# We'll generate it here for simplicity.
A_poly = (np.random.random(size=(N)) * Q) % Q
A_poly = p.polydiv(A_poly, POLY_MOD)[1].astype(int)
if len(A_poly) < N:
    A_poly = np.pad(A_poly, (0, N - len(A_poly)), 'constant').astype(int)

def gen_poly():
    """
    Generates a secret/error polynomial from a discrete Gaussian
    distribution (χδ).
    """
    poly = np.floor(np.random.normal(0, 2, size=(N))).astype(int)
    remainder = p.polydiv(poly, POLY_MOD)[1] % Q
    if len(remainder) < N:
        remainder = np.pad(remainder, (0, N - len(remainder)), 'constant')
    return remainder.astype(int)

def rlwe_generate_keypair():
    """
    Generates a private key (s, e) and a public key (b).
    s = f (from paper)
    e = e (from paper)
    b = a_i or b_j (from paper)
    Logic: b = A*s + e (paper uses 2*e, this is a common variant)
    """
    s = gen_poly()
    e = gen_poly()
    
    b = p.polymul(A_poly, s)
    b = p.polyadd(b, e) % Q
    b = p.polydiv(b, POLY_MOD)[1] % Q
    
    if len(b) < N:
        b = np.pad(b, (0, N - len(b)), 'constant')
        
    return (s, e), b.astype(int)

def rlwe_compute_shared_secret(private_key_s, public_key_b):
    """
    Computes the shared secret.
    Logic: c = s * b
    """
    c = p.polymul(private_key_s, public_key_b) % Q
    c = p.polydiv(c, POLY_MOD)[1] % Q
    if len(c) < N:
        c = np.pad(c, (0, N - len(c)), 'constant')
    return c.astype(int)

# def Cha(poly):
#     """
#     This is the Characteristic function (dj = Cha(cj)).
#     It creates the "signal" bits for reconciliation.
#     """
#     signal_bits = np.zeros(N, dtype=int)
#     for i in range(N):
#         val = poly[i]
#         if 0 <= val < Q / 4:
#             signal_bits[i] = 0
#         elif Q / 4 <= val < Q / 2:
#             signal_bits[i] = 1
#         elif Q / 2 <= val < 3 * Q / 4:
#             signal_bits[i] = 0
#         elif 3 * Q / 4 <= val <= Q:
#             signal_bits[i] = 1
#     return signal_bits

# NEW FAST VERSION (Vectorized)
def Cha(poly):
    """
    This is the Characteristic function (dj = Cha(cj)).
    It runs in optimized C via Numpy, not a slow Python loop.
    """
    Q_4 = Q / 4
    Q_2 = Q / 2
    Q_3_4 = 3 * Q / 4

    # Create boolean masks
    cond1 = (poly >= Q_4) & (poly < Q_2)
    cond2 = (poly >= Q_3_4) & (poly <= Q)

    # Apply masks
    signal_bits = np.zeros(N, dtype=int)
    signal_bits[cond1] = 1
    signal_bits[cond2] = 1

    return signal_bits

# def Mod2(shared_poly, signal_bits):
#     """
#     This is the Modular function (wj = Mod2(cj, dj)).
#     It uses the signal to extract the shared key bits.
#     """
#     key_bits = np.zeros(N, dtype=int)
#     for i in range(N):
#         val = shared_poly[i]
#         if signal_bits[i] == 0: # Region 0 (0-Q/4) and (Q/2-3Q/4)
#             if (Q * 0.125 <= val < Q * 0.625): key_bits[i] = 1
#             else: key_bits[i] = 0
#         else: # Region 1 (Q/4-Q/2) and (3Q/4-Q)
#             if (Q * 0.375 <= val < Q * 0.875): key_bits[i] = 1
#             else: key_bits[i] = 0
    
#     # Convert the array of 0s/1s into a single hash
#     return h(np.array_str(key_bits))

# NEW FAST VERSION (Vectorized)
def Mod2(shared_poly, signal_bits):
    """
    This is the Modular function (wj = Mod2(cj, dj)).
    It runs in optimized C via Numpy, not a slow Python loop.
    """
    key_bits = np.zeros(N, dtype=int)

    # Define regions
    Q_0_125 = Q * 0.125
    Q_0_375 = Q * 0.375
    Q_0_625 = Q * 0.625
    Q_0_875 = Q * 0.875

    # Region 0 (where signal_bits == 0)
    mask_region0 = (signal_bits == 0)
    mask_key_is_1_region0 = (shared_poly >= Q_0_125) & (shared_poly < Q_0_625)
    key_bits[mask_region0 & mask_key_is_1_region0] = 1

    # Region 1 (where signal_bits == 1)
    mask_region1 = (signal_bits == 1)
    mask_key_is_1_region1 = (shared_poly >= Q_0_375) & (shared_poly < Q_0_875)
    key_bits[mask_region1 & mask_key_is_1_region1] = 1

    return h(np.array_str(key_bits))

# ==============================================================================
# NETWORK HELPER FUNCTIONS
# ==============================================================================

def send_message(sock, message):
    """Serializes and sends a JSON message."""
    # Convert numpy arrays to lists for JSON serialization
    for key, value in message.items():
        if isinstance(value, np.ndarray):
            message[key] = value.tolist()
    
    sock.sendall(json.dumps(message).encode('utf-8'))

def recv_message(sock):
    """Receives and deserializes a JSON message."""
    data = sock.recv(32768) # Increased buffer for large polynomials
    if not data:
        return None
    return json.loads(data.decode('utf-8'))

if os.path.exists(PARAMS_FILE):
    data = np.load(PARAMS_FILE)
    A_poly = data['A_poly'].astype(int)
else:
    # Option A: use a fixed seed for deterministic generation across test machines
    # seed = 123456789
    # Option B (safer): generate once and write file, so server and client share it
    A_poly = make_random_A_poly(seed=None)
    np.savez(PARAMS_FILE, A_poly=A_poly)
    
    
# ... (keep all your existing functions: h, xor_data, get_timestamp, etc.)
# ... (keep all the REAL RLWE IMPLEMENTATION functions)
# ... (keep the NETWORK HELPER FUNCTIONS)

# ==============================================================================
# NEW: AES-256-CBC ENCRYPTION (Section IV-F)
# ==============================================================================
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

def flip_one_bit(hex_string):
    """Flips the first bit of the first byte of a hex string."""
    try:
        b = bytearray.fromhex(hex_string)
        if not b: # Handle empty string
            return ""
        b[0] = b[0] ^ 1 # Flip the first bit
        return b.hex()
    except (ValueError, IndexError):
        # Fallback if hex is invalid or empty
        return h("flipped_fallback")

def encrypt_data(hex_key, hex_iv, plaintext_str):
    """
    Encrypts plaintext using AES-256-CBC with the session key.
    
    """
    backend = default_backend()
    key = bytes.fromhex(hex_key)
    iv = bytes.fromhex(hex_iv)
    
    # Pad the plaintext to be a multiple of the block size
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(plaintext_str.encode('utf-8')) + padder.finalize()
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=backend)
    encryptor = cipher.encryptor()
    ct = encryptor.update(padded_data) + encryptor.finalize()
    return ct.hex() # Return hex string for easy JSON transport

def decrypt_data(hex_key, hex_iv, ciphertext_hex):
    """
    Decrypts ciphertext using AES-256-CBC with the session key.
    [cite: 274]
    """
    backend = default_backend()
    key = bytes.fromhex(hex_key)
    iv = bytes.fromhex(hex_iv)
    ct = bytes.fromhex(ciphertext_hex)
    
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=backend)
    decryptor = cipher.decryptor()
    padded_plaintext = decryptor.update(ct) + decryptor.finalize()
    
    # Unpad the data
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
    return plaintext.decode('utf-8')