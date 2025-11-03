# client_benchmark.py
import os
from client import run_client

if __name__ == "__main__":
    mode = os.getenv("RLWE_MODE", "NTT")
    print(f"[Client] Running in {mode} mode")
    run_client()
