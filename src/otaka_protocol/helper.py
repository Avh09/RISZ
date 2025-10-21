import hashlib
import os
import json
from datetime import datetime, timezone

# ==============================================================================
# CRYPTOGRAPHIC PRIMITIVES (Section VI)
# ==============================================================================

def h(*args):
    """
    The post-quantum secure hash function h(·).
    [cite_start]We use SHA-256 as specified in the paper[cite: 210, 371].
    """
    h_input = "||".join(map(str, args)).encode('utf-8')
    return hashlib.sha256(h_input).hexdigest()

def get_timestamp():
    """Returns a fresh timestamp TS."""
    return datetime.now(timezone.utc).isoformat()

def check_timestamp(ts, delta_t=10):
    """
    Verifies freshness: |TS* - TS| [cite_start]< Delta_T[cite: 239].
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
# DUMMY RLWE STUB FUNCTIONS (Section III & IV-E)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !! WARNING: These are NOT cryptographically secure.                 !!
# !! You MUST replace these with a real RLWE library for your project.!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ==============================================================================

# [cite_start]Public parameter alpha, as string for this dummy example [cite: 209]
ALPHA = "dummy_alpha_public_parameter"

def rlwe_sample_from_chi_delta():
    """
    [cite_start]Represents sampling f or e from the discrete Gaussian distribution χδ[cite: 235].
    Returns a dummy random string.
    """
    return os.urandom(16).hex()

def rlwe_generate_public_key(f, e):
    """
    [cite_start]Represents ai = α*f1 + 2*e1 or bj = α*f2 + 2*e2[cite: 235, 242].
    This dummy function just concatenates them.
    """
    return f"{ALPHA}*{f}+2*{e}"

def rlwe_compute_shared_values(private_f, public_key):
    """
    [cite_start]Represents cj = ai*f2 or c'j = bj*f1[cite: 242, 244].
    This is the core of the key exchange.
    """
    return f"shared_secret_base_from_{private_f}_and_{public_key}"

def Mod2(c, d):
    """
    [cite_start]Represents wj = Mod2(cj, dj) or w'j = Mod2(c'j, dj)[cite: 242, 244].
    This is a placeholder for the 1-bit signal extraction.
    """
    return f"mod2_signal_from_{c}_and_{d}"

def Cha(c):
    """
    [cite_start]Represents dj = Cha(cj)[cite: 242].
    This is a placeholder for the characteristic function.
    """
    return f"cha_signal_from_{c}"

# ==============================================================================
# NETWORK HELPER FUNCTIONS
# ==============================================================================

def send_message(sock, message):
    """Serializes and sends a JSON message."""
    sock.sendall(json.dumps(message).encode('utf-8'))

def recv_message(sock):
    """Receives and deserializes a JSON message."""
    # [cite_start]Use a large buffer, as keys/polynomials are 4096 bits [cite: 507]
    data = sock.recv(8192) 
    if not data:
        return None
    return json.loads(data.decode('utf-8'))