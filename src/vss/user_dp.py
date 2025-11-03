# vss_dp_tradeoff_with_leakage.py
"""
DP tradeoff + privacy-leakage simulation (re-identification attack).

Requirements:
  - numpy, pandas, matplotlib, scikit-learn
  - data at '../../data/features_extracted.csv' (adjust DATA_PATH below)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split

# -----------------------------
# --- CONFIG
# -----------------------------
FEATURE_COLUMNS = [
    'strokeDuration', 'startX', 'startY', 'stopX', 'stopY',
    'directEndToEndDistance', 'meanResultantLength', 'upDownLeftRightFlag',
    'directionOfEndToEndLine', 'largestDeviationFromEndToEndLine',
    'averageDirection', 'lengthOfTrajectory', 'averageVelocity',
    'midStrokePressure', 'midStrokeArea'
]

# Privacy budgets to sweep; np.inf -> no DP
EPS_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0, np.inf]
SENSITIVITY = 1.0
RNG = np.random.default_rng(12345)

# Path to dataset (update if needed)
DATA_PATH = '../../data/features_extracted.csv'

# Attack config
ATTACK_TEST_SIZE = 0.3   # fraction of registration templates held-out for attack evaluation
ATTACK_TOP_K = 3

# -----------------------------
# --- DP NOISE FUNCTIONS
# -----------------------------
def add_dp_noise(v, epsilon, sensitivity=SENSITIVITY, mechanism='laplace'):
    if epsilon == np.inf:
        return v.copy()
    if mechanism == 'laplace':
        scale = sensitivity / float(epsilon)
        return v + RNG.laplace(0.0, scale, size=v.shape)
    else:
        sigma = sensitivity / float(epsilon)
        return v + RNG.normal(0.0, sigma, size=v.shape)

# -----------------------------
# --- Build DB (registration templates) and live data
# -----------------------------
def build_db_and_live(df, epsilon):
    """
    For each user: shuffle, split in half -> registration templates & live (test) samples
    Registration templates get DP noise based on epsilon (if epsilon != inf)
    """
    reg_vecs, reg_labels = [], []
    live = {}
    for uid in df['user_id'].unique():
        rows = df[df['user_id'] == uid][FEATURE_COLUMNS].values
        if len(rows) < 2:
            continue
        # shuffle deterministically
        RNG.shuffle(rows)
        split = len(rows) // 2
        reg = rows[:split].copy()
        live_samples = rows[split:].copy()
        # Apply DP to registration templates only (template protection)
        reg_noisy = np.array([add_dp_noise(v, epsilon) for v in reg])
        reg_vecs.extend(reg_noisy)
        reg_labels.extend([uid] * len(reg_noisy))
        live[uid] = live_samples
    return np.array(reg_vecs), np.array(reg_labels), live

# -----------------------------
# --- Robust threshold calibration (same as earlier)
# -----------------------------
def mad_normalized(dlist):
    if len(dlist) == 0:
        return 0.0
    med = np.median(dlist)
    mad = np.median(np.abs(dlist - med))
    return float(mad) * 1.4826 if mad > 0 else float(np.std(dlist) if np.std(dlist) > 1e-6 else 1e-6)

def robust_threshold_from_list(dlist, lambda_factor=0.7):
    if len(dlist) == 0:
        return np.inf
    med = np.median(dlist)
    mad = mad_normalized(dlist)
    sigma = mad if mad > 0 else (np.std(dlist) or 1e-6)
    return float(med + lambda_factor * sigma)

def calibrate_thresholds(db_scaled, db_labels, model, neighbors_for_calib=3, lambda_factor=0.7):
    thresholds = {}
    for u in np.unique(db_labels):
        idx = np.where(db_labels == u)[0]
        vecs = db_scaled[idx]
        dlist = []
        for v in vecs:
            n = min(neighbors_for_calib, len(db_scaled))
            d, _ = model.kneighbors(v.reshape(1, -1), n_neighbors=n)
            if d.shape[1] > 1:
                dlist.extend(list(d[0][1:]))
        thresholds[u] = robust_threshold_from_list(dlist, lambda_factor=lambda_factor) if len(dlist) > 0 else np.inf
    return thresholds

# -----------------------------
# --- VSS evaluation (FAR/FRR/ACC)
# -----------------------------
def query_vss_database(model, scaler, labels, qvec):
    q_scaled = scaler.transform(np.array(qvec).reshape(1, -1))
    dist, idx = model.kneighbors(q_scaled)
    return labels[idx[0][0]], float(dist[0][0])

def evaluate_vss(model, scaler, db_labels, thresholds, live):
    uids = list(live.keys())
    if len(uids) < 2:
        raise ValueError("Need ≥2 users for evaluation")
    target, imposter = uids[0], uids[1]  # keep deterministic
    # genuine
    succ, rej, fail = 0, 0, 0
    for v in live[target]:
        r, d = query_vss_database(model, scaler, db_labels, v)
        thr = thresholds.get(target, np.inf)
        if (r == target) and (d <= thr):
            succ += 1
        elif r != target:
            fail += 1
        else:
            rej += 1
    FRR = (rej + fail) / max(1, len(live[target])) * 100
    # imposter
    fooled = 0
    for v in live[imposter]:
        r, d = query_vss_database(model, scaler, db_labels, v)
        thr = thresholds.get(target, np.inf)
        if (r == target) and (d <= thr):
            fooled += 1
    FAR = fooled / max(1, len(live[imposter])) * 100
    ACC = (1 - (FAR + FRR) / 200) * 100
    return FAR, FRR, ACC

# -----------------------------
# --- Privacy Metrics: perturbation & pairwise distortion
# -----------------------------
def compute_privacy_metrics(original, noisy):
    # original, noisy: (N, D)
    perturb = np.linalg.norm(original - noisy, axis=1)
    mean_perturb = np.mean(perturb)
    # pairwise means
    def mean_pairwise(A):
        n = len(A)
        if n < 2:
            return 0.0
        s = 0.0
        cnt = 0
        for i in range(n):
            for j in range(i+1, n):
                s += np.linalg.norm(A[i] - A[j])
                cnt += 1
        return s / cnt if cnt > 0 else 0.0
    pair_orig = mean_pairwise(original)
    pair_noisy = mean_pairwise(noisy)
    distortion = 0.0 if pair_orig == 0 else abs(pair_noisy - pair_orig) / pair_orig
    return mean_perturb, distortion

# -----------------------------
# --- Re-identification attack (attacker trains on registration templates)
# -----------------------------
def run_reid_attack(reg_vecs, reg_labels):
    """
    Split registration templates into train/test for attack.
    Train a multiclass logistic regression to predict user_id from template vector.
    Return attack accuracy (top-1) and top-k accuracy.
    """
    # encode labels to ints
    le = LabelEncoder()
    y = le.fit_transform(reg_labels)
    X = reg_vecs.copy()
    # standardize features for classifier
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=ATTACK_TEST_SIZE, random_state=42, stratify=y)
    # attacker model (simple)
    clf = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='lbfgs', class_weight='balanced')
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred) * 100
    # top-k (works only if number classes > k)
    k = min(ATTACK_TOP_K, len(le.classes_))
    if k >= 2:
        topk = top_k_accuracy_score(yte, clf.predict_proba(Xte), k=k) * 100
    else:
        topk = acc
    return acc, topk, len(le.classes_)

# -----------------------------
# --- MAIN: sweep epsilons, evaluate metrics + attack
# -----------------------------
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    # factorize categorical flag
    df['upDownLeftRightFlag'], _ = pd.factorize(df['upDownLeftRightFlag'])

    # select a single user sample for privacy metric computation (take up to 50 samples)
    representative_uid = df['user_id'].unique()[0]
    rep_original = df[df['user_id'] == representative_uid][FEATURE_COLUMNS].values[:50]

    results = []
    reid_results = []

    for eps in EPS_VALUES:
        print(f"\n=== Running epsilon = {eps} ===")
        reg_vecs, reg_labels, live = build_db_and_live(df, eps)

        # If registration database too small, skip
        if len(reg_vecs) < 10 or len(np.unique(reg_labels)) < 2:
            print("Not enough registration data or users; skipping this ε.")
            continue

        # scale + nearest neighbors
        scaler = StandardScaler().fit(reg_vecs)
        db_scaled = scaler.transform(reg_vecs)
        model = NearestNeighbors(n_neighbors=3).fit(db_scaled)
        thresholds = calibrate_thresholds(db_scaled, reg_labels, model)

        # evaluate VSS performance
        FAR, FRR, ACC = evaluate_vss(model, scaler, reg_labels, thresholds, live)
        print(f"Authentication: FAR={FAR:.2f}%, FRR={FRR:.2f}%, ACC={ACC:.2f}%")

        # privacy metrics (on representative user)
        rep_noisy = np.array([add_dp_noise(v, eps) for v in rep_original])
        mean_perturb, distortion = compute_privacy_metrics(rep_original, rep_noisy)
        print(f"Privacy: mean_perturb={mean_perturb:.4f}, pairwise_distortion={distortion:.4f}")

        # re-identification attack on registration templates
        acc_reid, topk_reid, n_classes = run_reid_attack(reg_vecs, reg_labels)
        print(f"Re-ID attack: top1_acc={acc_reid:.2f}%, top{ATTACK_TOP_K}_acc={topk_reid:.2f}%, n_users={n_classes}")

        results.append({
            'epsilon': eps,
            'FAR': FAR, 'FRR': FRR, 'ACC': ACC,
            'mean_perturb': mean_perturb, 'distortion': distortion
        })
        reid_results.append({
            'epsilon': eps,
            'reid_top1': acc_reid, 'reid_topk': topk_reid, 'n_users': n_classes
        })

    # Convert to DataFrame
    res_df = pd.DataFrame(results)
    reid_df = pd.DataFrame(reid_results)
    print("\nSummary (auth metrics):\n", res_df)
    print("\nSummary (re-id attack):\n", reid_df)

    # === Plot: Accuracy vs Epsilon ===
    plt.figure(figsize=(8, 5))
    eps_plot = res_df['epsilon'].replace(np.inf, np.max([e for e in EPS_VALUES if e != np.inf]) * 1.2)
    plt.plot(eps_plot, res_df['ACC'], marker='o', label='Authentication Accuracy (ACC)')
    plt.xlabel('Privacy budget ε (note: ∞ plotted beyond last finite ε)')
    plt.ylabel('Accuracy (%)')
    plt.title('Privacy–Utility: Authentication Accuracy vs ε')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig('dp_tradeoff_accuracy.png', dpi=150)
    plt.show()

    # === Plot: Re-ID leakage vs epsilon ===
    plt.figure(figsize=(8, 5))
    plt.plot(eps_plot, reid_df['reid_top1'], marker='o', label='Re-ID top1 accuracy (attacker)')
    plt.plot(eps_plot, reid_df['reid_topk'], marker='s', label=f'Re-ID top{ATTACK_TOP_K} acc')
    plt.xlabel('Privacy budget ε')
    plt.ylabel('Attacker accuracy (%)')
    plt.title('Privacy Leakage: Re-identification Accuracy vs ε')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig('dp_reid_leakage.png', dpi=150)
    plt.show()

    # === Plot: Privacy metrics vs epsilon (mean perturbation & distortion) ===
    plt.figure(figsize=(8, 5))
    plt.plot(eps_plot, res_df['mean_perturb'], marker='o', label='Mean perturbation (L2 norm)')
    plt.plot(eps_plot, res_df['distortion'], marker='s', label='Pairwise distortion (relative)')
    plt.xlabel('Privacy budget ε')
    plt.title('Privacy preservation metrics vs ε')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('dp_privacy_metrics.png', dpi=150)
    plt.show()

    # === PCA visual examples for a subset of epsilons ===
    pca = PCA(n_components=2)
    rep_scaled = StandardScaler().fit_transform(rep_original)
    rep_2d = pca.fit_transform(rep_scaled)
    for e in [EPS_VALUES[0], EPS_VALUES[len(EPS_VALUES)//2], EPS_VALUES[-1]]:
        noisy = np.array([add_dp_noise(v, e) for v in rep_original])
        noisy_2d = pca.transform(StandardScaler().fit_transform(noisy))
        plt.figure(figsize=(6,5))
        plt.scatter(rep_2d[:,0], rep_2d[:,1], label='Original', s=50)
        plt.scatter(noisy_2d[:,0], noisy_2d[:,1], label=f'DP ε={e}', s=50, marker='x')
        plt.title(f'PCA: Original vs DP (ε={e})')
        plt.legend(); plt.grid(alpha=0.3)
        plt.savefig(f'dp_pca_eps_{e}.png', dpi=150)
        plt.show()

    print("\n✅ Finished DP tradeoff + leakage simulation. Check generated PNGs for visualizations.")
