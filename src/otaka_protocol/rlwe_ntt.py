# rlwe_ntt.py
import numpy as np
from numba import njit

MODULUS = 1073479681
N = 1024
ROOT = 3

# ---------- new helper ----------
def mod_pow(base, exp, mod):
    """Efficient modular exponentiation (Numba-safe)"""
    result = 1
    b = base % mod
    e = exp
    while e > 0:
        if e & 1:
            result = (result * b) % mod
        b = (b * b) % mod
        e >>= 1
    return result
# ---------------------------------

@njit
def bit_reverse_copy_numba(a):
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            tmp = a[i]
            a[i] = a[j]
            a[j] = tmp
    return a

@njit
def ntt_numba(a, n, q, wtable):
    """Forward NTT with precomputed twiddles"""
    res = a.copy()
    res = bit_reverse_copy_numba(res)
    step = 1
    while step < n:
        length = step * 2
        for i in range(0, n, length):
            w = 1
            for j in range(i, i + step):
                u = res[j]
                v = (res[j + step] * wtable[step] ) % q
                res[j] = (u + v) % q
                res[j + step] = (u - v) % q
                w = (w * wtable[1]) % q
        step = length
    return res

# Simpler, clearer negacyclic wrapper without pow() inside njit
def poly_mul_ntt(a, b):
    a = np.array(a, dtype=np.int64) % MODULUS
    b = np.array(b, dtype=np.int64) % MODULUS

    psi = mod_pow(ROOT, (MODULUS - 1) // (2 * N), MODULUS)
    psi_powers = np.array([mod_pow(psi, i, MODULUS) for i in range(N)], dtype=np.int64)
    inv_psi_powers = np.array([mod_pow(psi, -i % (2*N), MODULUS) for i in range(N)], dtype=np.int64)

    a_tilde = (a * psi_powers) % MODULUS
    b_tilde = (b * psi_powers) % MODULUS

    # Precompute wlen tables outside JIT (no pow inside)
    wtable = np.zeros(N + 1, dtype=np.int64)
    for step in range(1, N + 1):
        wtable[step] = mod_pow(ROOT, (MODULUS - 1) // (2 * step), MODULUS)

    A = ntt_numba(a_tilde, N, MODULUS, wtable)
    B = ntt_numba(b_tilde, N, MODULUS, wtable)
    C = (A * B) % MODULUS

    # Inverse NTT (simple direct version)
    c = C.copy()
    for i in range(N):
        c[i] = (c[i] * pow(N, MODULUS - 2, MODULUS)) % MODULUS
    return (c * inv_psi_powers) % MODULUS
