# simulate_hillclimb_whitebox.py
"""
White-box simulation of adaptive hill-climb style search for CA acceptance.
- This is a local simulation that uses the dataset to *model* an attacker's
  adaptive search ability under idealized feedback modes.
- It does NOT interact with any live authentication server and must only be
  used on synthetic/consenting datasets.

Outputs:
 - success rates and mean queries-to-success for each strategy
 - best-distance distributions
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import json
import math
import argparse
from tqdm import trange

# ----- CONFIG -----
DATASET_PATH = "features_extracted.csv"
FEATURE_COLUMNS = [
    'strokeDuration', 'startX', 'startY', 'stopX', 'stopY',
    'directEndToEndDistance', 'meanResultantLength', 'upDownLeftRightFlag',
    'directionOfEndToEndLine', 'largestDeviationFromEndToEndLine',
    'averageDirection', 'lengthOfTrajectory', 'averageVelocity',
    'midStrokePressure', 'midStrokeArea'
]
USER_ID_COL = "user_id"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Simulation hyperparams (tune these)
NUM_TRIALS = 200               # number of independent simulation trials per attacker/target
QUERY_BUDGET = 2000            # max allowed candidate queries per trial
STEP_STD_RATIO = 0.05          # gaussian step std as fraction of feature std
BINARY_THRESHOLD_PERCENTILE = 95  # threshold set at this percentile of genuine intra-distances
INITIALIZATION = "random_sample"  # "random_sample" or "target_template" or "perturbed_template"
# ------------------

def load_dataset(path):
    df = pd.read_csv(path)
    # factorize categorical if applicable
    if 'upDownLeftRightFlag' in df.columns and df['upDownLeftRightFlag'].dtype == object:
        df['upDownLeftRightFlag'], _ = pd.factorize(df['upDownLeftRightFlag'])
    # ensure columns
    for f in FEATURE_COLUMNS:
        if f not in df.columns:
            raise ValueError(f"Feature '{f}' not found in dataset.")
    return df

def get_template_for_user(df, user_id, agg="mean"):
    arr = df[df[USER_ID_COL] == user_id][FEATURE_COLUMNS].values
    if agg == "mean":
        return arr.mean(axis=0)
    elif agg == "median":
        return np.median(arr, axis=0)
    else:
        raise ValueError("agg must be 'mean' or 'median'")

def euclidean(a, b):
    return np.linalg.norm(a - b)

def random_sample_from_marginals(df):
    # sample each feature independently from its empirical marginals
    feat_vals = [df[f].values for f in FEATURE_COLUMNS]
    return np.array([np.random.choice(col) for col in feat_vals], dtype=float)

def noisy_candidate(base_vec, global_std, step_std_ratio):
    # gaussian perturbation applied independently per-dimension, clipped to +/- 3 sigma
    stds = global_std * step_std_ratio
    noise = np.random.normal(loc=0.0, scale=stds)
    cand = base_vec + noise
    return cand

def prepare_globals(df):
    # compute global per-feature std to scale step sizes
    data = df[FEATURE_COLUMNS].values.astype(float)
    feature_std = np.std(data, axis=0, ddof=1)
    return feature_std

# ----------------- Simulation strategies (white-box) -----------------

def random_sampling_strategy(df, target_vec, threshold, budget):
    """Random sampling from marginals until success or budget exhausted."""
    for q in range(1, budget+1):
        cand = random_sample_from_marginals(df)
        d = euclidean(cand, target_vec)
        if d <= threshold:
            return True, q, d
    return False, budget, None

def score_hillclimb_strategy(df, target_vec, threshold, budget, feature_std,
                             init_mode="random_sample", step_ratio=STEP_STD_RATIO):
    """
    Greedy hill-climb with *score* feedback (attacker sees real-valued distance).
    - candidate generation: gaussian perturb around current best
    - accept if candidate distance < current_best_distance
    """
    # initialize
    if init_mode == "random_sample":
        current = random_sample_from_marginals(df)
    elif init_mode == "target_template":
        current = target_vec.copy()
    elif init_mode == "perturbed_template":
        current = target_vec + np.random.normal(0, feature_std * 0.5)
    else:
        current = random_sample_from_marginals(df)
    cur_d = euclidean(current, target_vec)
    if cur_d <= threshold:
        return True, 0, cur_d
    queries = 0
    best_d = cur_d
    for q in range(1, budget+1):
        cand = noisy_candidate(current, feature_std, step_ratio)
        d = euclidean(cand, target_vec)
        queries += 1
        # greedy accept if improvement
        if d < best_d:
            current = cand
            best_d = d
            if best_d <= threshold:
                return True, queries, best_d
    return False, queries, best_d

def binary_hillclimb_strategy(df, target_vec, threshold, budget, feature_std,
                              init_mode="random_sample", step_ratio=STEP_STD_RATIO,
                              stagnation_patience=500):
    """
    Hill-climb with only binary accept/reject feedback:
    - attacker proposes candidate and learns only if it is accepted (d <= threshold) or not.
    - Without score, we use a strategy: propose local perturbations around a base
      and if accepted (rare), success; otherwise, if no acceptance after many attempts,
      occasionally reinitialize.
    This models a *limited* binary oracle attacker (conservative).
    """
    # This is intentionally conservative: without score info, we do local probing only.
    if init_mode == "random_sample":
        base = random_sample_from_marginals(df)
    elif init_mode == "target_template":
        base = target_vec.copy()
    elif init_mode == "perturbed_template":
        base = target_vec + np.random.normal(0, feature_std * 0.5)
    else:
        base = random_sample_from_marginals(df)
    queries = 0
    best_found_d = euclidean(base, target_vec)
    no_improve = 0
    for q in range(1, budget+1):
        # propose local perturbation around base
        cand = noisy_candidate(base, feature_std, step_ratio)
        d = euclidean(cand, target_vec)
        queries += 1
        if d <= threshold:
            return True, queries, d
        # If candidate is closer than base, move base occasionally (we simulate a weak heuristic)
        if d < best_found_d:
            best_found_d = d
            base = cand  # adopt the slightly better point
            no_improve = 0
        else:
            no_improve += 1
        # occasionally reinitialize base to escape local basins
        if no_improve >= stagnation_patience:
            base = random_sample_from_marginals(df)
            best_found_d = euclidean(base, target_vec)
            no_improve = 0
    return False, queries, best_found_d

# ----------------- Runner -----------------

def run_simulation(df, target_user, num_trials=NUM_TRIALS, budget=QUERY_BUDGET):
    # prepare
    feature_std = prepare_globals(df)
    # compute template and threshold (use 95th percentile of genuine intra distances)
    user_rows = df[df[USER_ID_COL] == target_user]
    if len(user_rows) < 4:
        raise ValueError("Not enough samples for user.")
    template = get_template_for_user(df, target_user, agg="mean")
    # compute intra distances
    uvecs = user_rows[FEATURE_COLUMNS].values.astype(float)
    intra_dists = np.linalg.norm(uvecs - template, axis=1)
    threshold = np.percentile(intra_dists, BINARY_THRESHOLD_PERCENTILE)

    print(f"Target user: {target_user}. Template from {len(uvecs)} samples. Threshold (p{BINARY_THRESHOLD_PERCENTILE}) = {threshold:.4f}")
    results = {
        "random": {"succ":0,"queries":[], "best_d":[]},
        "score_hc": {"succ":0,"queries":[], "best_d":[]},
        "binary_hc": {"succ":0,"queries":[], "best_d":[]}
    }

    # run trials
    for t in trange(num_trials, desc="Trials"):
        # Random sampling baseline
        ok, q, d = random_sampling_strategy(df, template, threshold, budget)
        if ok:
            results["random"]["succ"] += 1
            results["random"]["queries"].append(q)
        else:
            # record best as NaN or budget
            results["random"]["queries"].append(q)
        results["random"]["best_d"].append(d if d is not None else np.nan)

        # Score hill-climb
        ok, q, d = score_hillclimb_strategy(df, template, threshold, budget, feature_std, init_mode=INITIALIZATION)
        if ok:
            results["score_hc"]["succ"] += 1
            results["score_hc"]["queries"].append(q)
        else:
            results["score_hc"]["queries"].append(q)
        results["score_hc"]["best_d"].append(d if d is not None else np.nan)

        # Binary hill-climb (conservative)
        ok, q, d = binary_hillclimb_strategy(df, template, threshold, budget, feature_std, init_mode=INITIALIZATION)
        if ok:
            results["binary_hc"]["succ"] += 1
            results["binary_hc"]["queries"].append(q)
        else:
            results["binary_hc"]["queries"].append(q)
        results["binary_hc"]["best_d"].append(d if d is not None else np.nan)

    # summarize
    summary = {}
    for k in results:
        succ = results[k]["succ"]
        succ_rate = succ / num_trials
        q_list = np.array(results[k]["queries"], dtype=float)
        # only consider successful queries for mean queries-to-success
        succ_qs = [q for i,q in enumerate(q_list) if i < len(q_list) and not math.isnan(results[k]["best_d"][i]) and results[k]["best_d"][i] <= threshold]
        mean_q = np.mean(succ_qs) if len(succ_qs)>0 else None
        median_q = np.median(succ_qs) if len(succ_qs)>0 else None
        bestd = np.array([d for d in results[k]["best_d"] if not (d is None or (isinstance(d,float) and np.isnan(d)))])
        summary[k] = {
            "success_rate": succ_rate,
            "mean_queries_on_success": mean_q,
            "median_queries_on_success": median_q,
            "best_distance_stats": {
                "min": float(np.nanmin(bestd)) if bestd.size>0 else None,
                "median": float(np.nanmedian(bestd)) if bestd.size>0 else None,
                "mean": float(np.nanmean(bestd)) if bestd.size>0 else None
            }
        }
    return summary, results, threshold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--user", default=None, help="target user id (if not set, pick largest user)")
    parser.add_argument("--trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--budget", type=int, default=QUERY_BUDGET)
    args = parser.parse_args()

    df = load_dataset(args.dataset)
    print("Dataset rows:", len(df), "unique users:", df[USER_ID_COL].nunique())
    # pick target user
    if args.user is None:
        counts = df[USER_ID_COL].value_counts()
        target_user = counts.index[0]
    else:
        target_user = args.user
    summary, raw, threshold = run_simulation(df, target_user, num_trials=args.trials, budget=args.budget)


if __name__ == "__main__":
    main()
