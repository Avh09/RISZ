# server_benchmark.py
import os
import subprocess
import matplotlib.pyplot as plt
import json
import time

def run_once(mode):
    print(f"\n=== Running SERVER in {mode} mode ===")
    env = os.environ.copy()
    env["RLWE_MODE"] = mode
    # Run the server and client simultaneously
    server_proc = subprocess.Popen(["python3", "-m", "src.otaka_protocol.server"], env=env)
    time.sleep(2)
    subprocess.run(["python3", "-m", "src.otaka_protocol.client"], env=env, check=True)
    server_proc.terminate()
    server_proc.wait()

def parse_logs():
    logs = []
    with open("rlwe_benchmark_log.json") as f:
        for line in f:
            logs.extend(json.loads(line))
    return logs

def visualize(logs):
    ntt_times = [x["duration"] for x in logs if "NTT" in x["event"]]
    naive_times = [x["duration"] for x in logs if "NAIVE" in x["event"]]
    plt.figure(figsize=(7,4))
    plt.boxplot([naive_times, ntt_times], labels=["Naive", "NTT"], showmeans=True)
    plt.ylabel("Time (seconds)")
    plt.title("RLWE Polynomial Multiplication Performance")
    plt.grid(True)
    plt.savefig("rlwe_comparison.png", dpi=150)
    plt.show()
    print(f"\nSaved plot to rlwe_comparison.png")

if __name__ == "__main__":
    # Clean up old logs
    if os.path.exists("rlwe_benchmark_log.json"):
        os.remove("rlwe_benchmark_log.json")
    run_once("NAIVE")
    time.sleep(3)
    run_once("NTT")
    logs = parse_logs()
    visualize(logs)
    print("Benchmark complete.")
