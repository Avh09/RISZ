# updated_vss_dp_adaptive.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


FEATURE_COLUMNS = [
    'strokeDuration', 'startX', 'startY', 'stopX', 'stopY',
    'directEndToEndDistance', 'meanResultantLength', 'upDownLeftRightFlag',
    'directionOfEndToEndLine', 'largestDeviationFromEndToEndLine',
    'averageDirection', 'lengthOfTrajectory', 'averageVelocity',
    'midStrokePressure', 'midStrokeArea'
]

EPSILON = 5.0            
SENSITIVITY = 1.0         
LAMBDA_FACTOR = 0.7      
ADAPTIVE_WINDOW = 20
ADAPTIVE_SMOOTH_ALPHA = 0.3   
RNG = np.random.default_rng(42)  
DP_MECHANISM = 'laplace'

# -----------------------------
# --- Differential-privacy noise
# -----------------------------
def add_dp_noise(vector, epsilon=EPSILON, sensitivity=SENSITIVITY, mechanism='laplace'):
    """
    Add DP noise to a 1-D numpy vector.
    Laplace: scale = sensitivity / epsilon (L1)
    Gaussian (approx DP): sigma = sensitivity / epsilon (not exact Gaussian-DP formula but kept for experiments)
    """
    if mechanism == 'laplace':
        scale = float(sensitivity) / float(epsilon)
        noise = RNG.laplace(0.0, scale, size=vector.shape)
    elif mechanism == 'gaussian':
        sigma = float(sensitivity) / float(epsilon)
        noise = RNG.normal(0.0, sigma, size=vector.shape)
    else:
        raise ValueError("mechanism must be 'laplace' or 'gaussian'")
    return vector + noise

# -----------------------------
# --- Build registration database
# -----------------------------
def build_database(df, use_dp=False, dp_epsilon=EPSILON, dp_sensitivity=SENSITIVITY, dp_mechanism=DP_MECHANISM):
    """
    Build registration database. If use_dp is True, add noise to registration vectors only.
    (Live data is kept clean to reflect a real auth scenario.)
    """
    db_vectors, db_labels, live_sim_data = [], [], {}
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id][FEATURE_COLUMNS].values
        if len(user_data) < 2:
            continue
        RNG.shuffle(user_data)
        split = len(user_data) // 2
        reg = user_data[:split].copy()
        live = user_data[split:].copy()
        if use_dp:
            # Apply DP to registration templates only (typical for template protection)
            reg = np.array([add_dp_noise(v, epsilon=dp_epsilon, sensitivity=dp_sensitivity, mechanism=dp_mechanism)
                            for v in reg])
        db_vectors.extend(reg)
        db_labels.extend([user_id] * len(reg))
        live_sim_data[user_id] = live
    return np.array(db_vectors), np.array(db_labels), live_sim_data


def mad_normalized(dlist):
    if len(dlist) == 0:
        return 0.0
    med = np.median(dlist)
    mad = np.median(np.abs(dlist - med))
    return float(mad) * 1.4826 if mad > 0 else 0.0

def robust_threshold_from_list(dlist, lambda_factor=LAMBDA_FACTOR):
    if len(dlist) == 0:
        return np.inf
    med = np.median(dlist)
    mad = mad_normalized(dlist)
    if mad == 0:
        sigma = np.std(dlist) if np.std(dlist) > 1e-6 else 1e-6
    else:
        sigma = mad
    return float(med + lambda_factor * sigma)

def calibrate_thresholds(db_scaled, db_labels, model, lambda_factor=LAMBDA_FACTOR, neighbors_for_calib=3):

    thresholds = {}
    labels_unique = np.unique(db_labels)

    max_neighbors = min(neighbors_for_calib, len(db_scaled))
    for u in labels_unique:
        idx = np.where(db_labels == u)[0]
        vecs = db_scaled[idx]
        dlist = []
       
        for v in vecs:
            n = min(max_neighbors, len(db_scaled))
            d, _ = model.kneighbors(v.reshape(1, -1), n_neighbors=n)
            if d.shape[1] > 1:
                # take distances excluding the nearest (self)
                dlist.extend(list(d[0][1:]))
        if len(dlist) == 0:
            thresholds[u] = np.inf
        else:
            # robust threshold
            thresholds[u] = robust_threshold_from_list(dlist, lambda_factor=lambda_factor)
    return thresholds

def query_vss_database(model, scaler, labels, qvec, n_neighbors=None):
    q_scaled = scaler.transform(np.array(qvec).reshape(1, -1))
    if n_neighbors is None:
        dist, idx = model.kneighbors(q_scaled)
    else:
        dist, idx = model.kneighbors(q_scaled, n_neighbors=n_neighbors)
    return labels[idx[0][0]], float(dist[0][0])

def query_vss_with_rejection(model, scaler, labels, qvec, thresholds,
                             adaptive=False, user_hist=None, target=None, ground_truth=None,
                             override_threshold=None, update_on_genuine_only=True):
  
    retrieved, dist = query_vss_database(model, scaler, labels, qvec)
    threshold_user = target if target is not None else retrieved
    thr = thresholds.get(threshold_user, np.inf)

    if override_threshold is not None:
        thr = override_threshold

    accepted = (dist <= thr) and (retrieved == threshold_user)
    if user_hist is not None and target is not None and ground_truth == target:
        hist = user_hist.setdefault(target, [])
        if not update_on_genuine_only or (update_on_genuine_only and retrieved == target):
            hist.append(float(dist))
            if len(hist) > ADAPTIVE_WINDOW:
                hist.pop(0)

    if not accepted:
        return None, dist, thr
    return retrieved, dist, thr

def compute_adaptive_threshold(static_threshold, history_list, lambda_factor=LAMBDA_FACTOR, 
                               alpha=ADAPTIVE_SMOOTH_ALPHA, allow_tightening=True, allow_relaxation=True):

    if history_list is None or len(history_list) < 3:
        return static_threshold
    
    robust_thr = robust_threshold_from_list(history_list, lambda_factor=lambda_factor)
    
    # Calculate if we're tightening or relaxing
    is_tightening = robust_thr < static_threshold
    
    # Asymmetric smoothing: tighten faster, relax slower
    if is_tightening and allow_tightening:
        effective_alpha = alpha * 0.5  # Faster tightening
    elif not is_tightening and allow_relaxation:
        effective_alpha = alpha * 1.5  # Slower relaxation
        effective_alpha = min(effective_alpha, 0.9)  # Cap it
    else:
        return static_threshold
    
    new_thr = effective_alpha * static_threshold + (1.0 - effective_alpha) * robust_thr
    
    # Safety bounds: don't go too tight or too loose
    min_threshold = static_threshold * 0.5  
    max_threshold = static_threshold * 2.0  
    
    return float(np.clip(new_thr, min_threshold, max_threshold))

def evaluate_with_tracking_prebuilt(model, scaler, db_labels, thresholds_ref, live, adaptive=False,
                                    lambda_factor=LAMBDA_FACTOR, alpha_smooth=ADAPTIVE_SMOOTH_ALPHA,
                                    update_hist_on_accepted_only=False):

    labels = np.array(db_labels)
    thresholds = thresholds_ref.copy()

    uids = list(live.keys())
    if len(uids) < 2:
        raise ValueError("Need ≥2 users")
    target, imposter = uids[0], uids[1]
    hist = {} if adaptive else None

    genuine_outcomes, imposter_outcomes = [], []
    g_d, imp_d = [], []
    succ, rej, fail = 0, 0, 0

    current_thr = thresholds.get(target, np.inf)

    # Genuine attempts
    for i, v in enumerate(live[target]):
        if adaptive and hist and target in hist and len(hist[target]) >= 2:
            current_thr = compute_adaptive_threshold(thresholds.get(target, np.inf),
                                                     hist[target],
                                                     lambda_factor=lambda_factor,
                                                     alpha=alpha_smooth)
        else:
            current_thr = thresholds.get(target, np.inf)

        r, d, thr_used = query_vss_with_rejection(
            model, scaler, labels, v, thresholds,
            adaptive=adaptive, user_hist=hist, target=target,
            ground_truth=target, override_threshold=current_thr,
            update_on_genuine_only=update_hist_on_accepted_only
        )
        g_d.append(d)
        accepted = (r == target)
        genuine_outcomes.append((i+1, d, accepted, current_thr))

        if r == target:
            succ += 1
        elif r is None:
            rej += 1
        else:
            fail += 1

    FRR = (rej + fail) / max(1, len(live[target])) * 100

    # Imposter attempts
    fooled, detected = 0, 0
    for i, v in enumerate(live[imposter]):
        if adaptive and hist and target in hist and len(hist[target]) >= 2:
            imp_thr = compute_adaptive_threshold(thresholds.get(target, np.inf),
                                                 hist[target],
                                                 lambda_factor=lambda_factor,
                                                 alpha=alpha_smooth)
        else:
            imp_thr = thresholds.get(target, np.inf)

        r, d, thr_used = query_vss_with_rejection(
            model, scaler, labels, v, thresholds,
            adaptive=adaptive, user_hist=hist, target=target,
            ground_truth=imposter, override_threshold=imp_thr,
            update_on_genuine_only=update_hist_on_accepted_only
        )
        imp_d.append(d)
        accepted = (r == target)
        imposter_outcomes.append((i+1, d, accepted))

        if r == target:
            fooled += 1
        else:
            detected += 1

    FAR = fooled / max(1, len(live[imposter])) * 100
    ACC = (1 - (FAR + FRR) / 200) * 100

    return {
        "FAR": FAR, "FRR": FRR, "ACC": ACC,
        "genuine_outcomes": genuine_outcomes,
        "imposter_outcomes": imposter_outcomes,
        "genuine_distances": g_d,
        "imposter_distances": imp_d,
        "target_threshold": thresholds.get(target, np.nan),
        "final_adaptive_threshold": current_thr
    }

def evaluate_with_tracking(df, thresholds_ref, live, use_dp=False, adaptive=False,
                           dp_epsilon=EPSILON, dp_sensitivity=SENSITIVITY, dp_mechanism=DP_MECHANISM):
    db_vecs, db_labels, _ = build_database(df, use_dp=use_dp, dp_epsilon=dp_epsilon,
                                          dp_sensitivity=dp_sensitivity, dp_mechanism=dp_mechanism)
    scaler = StandardScaler().fit(db_vecs)
    db_scaled = scaler.transform(db_vecs)
    model = NearestNeighbors(n_neighbors=1).fit(db_scaled)
    db_labels = np.array(db_labels)
    thresholds = thresholds_ref.copy()

    uids = list(live.keys())
    if len(uids) < 2:
        raise ValueError("Need ≥2 users")
    target, imposter = uids[0], uids[1]
    hist = {} if adaptive else None

    genuine_outcomes, imposter_outcomes = [], []
    g_d, imp_d = [], []
    succ, rej, fail = 0, 0, 0

    for i, v in enumerate(live[target]):
        current_thr = thresholds.get(target, np.inf)
        if adaptive and hist and target in hist and len(hist[target]) >= 2:
            current_thr = compute_adaptive_threshold(thresholds.get(target, np.inf), hist[target])
        r, d, thr_used = query_vss_with_rejection(model, scaler, db_labels, v, thresholds,
                                        adaptive=adaptive, user_hist=hist, target=target,
                                        ground_truth=target, override_threshold=current_thr)
        g_d.append(d)
        accepted = (r == target)
        genuine_outcomes.append((i+1, d, accepted, current_thr))

        if r == target:
            succ += 1
        elif r is None:
            rej += 1
        else:
            fail += 1

    FRR = (rej + fail) / max(1, len(live[target])) * 100

    fooled, detected = 0, 0
    for i, v in enumerate(live[imposter]):
        r, d, thr_used = query_vss_with_rejection(model, scaler, db_labels, v, thresholds,
                                        adaptive=adaptive, user_hist=hist, target=target,
                                        ground_truth=imposter, override_threshold=thresholds.get(target, np.inf))
        imp_d.append(d)
        accepted = (r == target)
        imposter_outcomes.append((i+1, d, accepted))

        if r == target:
            fooled += 1
        else:
            detected += 1

    FAR = fooled / max(1, len(live[imposter])) * 100
    ACC = (1 - (FAR + FRR) / 200) * 100

    return {
        "FAR": FAR, "FRR": FRR, "ACC": ACC,
        "genuine_outcomes": genuine_outcomes,
        "imposter_outcomes": imposter_outcomes,
        "genuine_distances": g_d,
        "imposter_distances": imp_d,
        "target_threshold": thresholds.get(target, np.nan)
    }


def visualize_privacy_protection(df, use_dp_mechanism=DP_MECHANISM):
    """Show how DP protects individual records while maintaining utility"""
    user_id = df['user_id'].unique()[0]
    user_data = df[df['user_id'] == user_id][FEATURE_COLUMNS].values[:10]
    noisy_data = np.array([add_dp_noise(v, epsilon=EPSILON, sensitivity=SENSITIVITY, mechanism=use_dp_mechanism)
                           for v in user_data])

    # Calculate pairwise distances
    clean_dists = []
    noisy_dists = []
    for i in range(len(user_data)):
        for j in range(i+1, len(user_data)):
            clean_dists.append(np.linalg.norm(user_data[i] - user_data[j]))
            noisy_dists.append(np.linalg.norm(noisy_data[i] - noisy_data[j]))

    # Visualize in 2D using PCA
    scaler = StandardScaler()
    user_scaled = scaler.fit_transform(user_data)
    noisy_scaled = scaler.transform(noisy_data)

    pca = PCA(n_components=2)
    user_2d = pca.fit_transform(user_scaled)
    noisy_2d = pca.transform(noisy_scaled)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: 2D projection showing noise
    axes[0].scatter(user_2d[:, 0], user_2d[:, 1], s=100, alpha=0.7, label='Original', marker='o')
    axes[0].scatter(noisy_2d[:, 0], noisy_2d[:, 1], s=100, alpha=0.7, label=f'DP (ε={EPSILON})', marker='x')
    for i in range(len(user_2d)):
        axes[0].plot([user_2d[i, 0], noisy_2d[i, 0]],
                     [user_2d[i, 1], noisy_2d[i, 1]],
                     'k--', alpha=0.3, linewidth=0.5)
    axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
    axes[0].set_title('Privacy: Original vs DP-Protected Records')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Plot 2: Distance distribution comparison
    axes[1].hist(clean_dists, bins=15, alpha=0.6, label='Original')
    axes[1].hist(noisy_dists, bins=15, alpha=0.6, label=f'DP (ε={EPSILON})')
    axes[1].set_xlabel('Pairwise Distance'); axes[1].set_ylabel('Frequency')
    axes[1].set_title('Privacy: Distance Distributions')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Plot 3: Individual record perturbation
    perturbations = [np.linalg.norm(user_data[i] - noisy_data[i]) for i in range(len(user_data))]
    axes[2].bar(range(1, len(perturbations)+1), perturbations, alpha=0.7)
    axes[2].axhline(np.mean(perturbations), color='red', linestyle='--', label=f'Mean: {np.mean(perturbations):.3f}')
    axes[2].set_xlabel('Record Index'); axes[2].set_ylabel('L2 Norm of Noise')
    axes[2].set_title(f'Privacy: Per-Record Perturbation (ε={EPSILON})')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('privacy_protection_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n📊 Privacy Protection Metrics:")
    print(f"   Mean perturbation magnitude: {np.mean(perturbations):.4f}")
    print(f"   Original distance mean: {np.mean(clean_dists):.4f}")
    print(f"   DP distance mean: {np.mean(noisy_dists):.4f}")
    if np.mean(clean_dists) > 0:
        print(f"   Distance preservation: {(1 - abs(np.mean(noisy_dists) - np.mean(clean_dists))/np.mean(clean_dists)) * 100:.2f}%")

def visualize_adaptive_over_time(genuine_static, genuine_adaptive, 
                                 imposter_static, imposter_adaptive):
    """Show how adaptive thresholding evolves over authentication attempts"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data
    gen_static_attempts = [x[0] for x in genuine_static]
    gen_static_dists = [x[1] for x in genuine_static]
    gen_static_accepted = [x[2] for x in genuine_static]
    
    gen_adapt_attempts = [x[0] for x in genuine_adaptive]
    gen_adapt_dists = [x[1] for x in genuine_adaptive]
    gen_adapt_accepted = [x[2] for x in genuine_adaptive]
    gen_adapt_thresholds = [x[3] for x in genuine_adaptive]
    
    # --- Compute nice y-limits (zoom in for visibility) ---
    all_dists = np.array(gen_adapt_dists + gen_static_dists)
    all_thrs = np.array(gen_adapt_thresholds + [genuine_static[0][3]])
    y_min = min(all_dists.min(), all_thrs.min()) * 0.9
    y_max = max(all_dists.max(), all_thrs.max()) * 1.1
    if y_max - y_min < 0.1:  # prevent zero-range plot
        y_min -= 0.05
        y_max += 0.05
    
    # Plot 1: Static genuine distances
    colors_static = ['green' if x else 'red' for x in gen_static_accepted]
    axes[0, 0].scatter(gen_static_attempts, gen_static_dists, c=colors_static, 
                       alpha=0.6, s=50, label='Auth attempts')
    axes[0, 0].axhline(genuine_static[0][3], color='blue', linestyle='--', 
                       linewidth=2, label='Static threshold')
    axes[0, 0].set_xlabel('Authentication Attempt')
    axes[0, 0].set_ylabel('Distance to Template')
    axes[0, 0].set_ylim(y_min, y_max)
    axes[0, 0].set_title('Static Thresholding: Genuine User')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Adaptive genuine distances
    colors_adapt = ['green' if x else 'red' for x in gen_adapt_accepted]
    axes[0, 1].scatter(gen_adapt_attempts, gen_adapt_dists, c=colors_adapt,
                       alpha=0.6, s=50, label='Auth attempts')
    axes[0, 1].plot(gen_adapt_attempts, gen_adapt_thresholds, 'b-', 
                    linewidth=2, label='Adaptive threshold')
    axes[0, 1].set_xlabel('Authentication Attempt')
    axes[0, 1].set_ylabel('Distance to Template')
    axes[0, 1].set_ylim(y_min, y_max)
    axes[0, 1].set_title('Adaptive Thresholding: Genuine User')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cumulative acceptance rate
    static_cumulative = np.cumsum(gen_static_accepted) / np.arange(1, len(gen_static_accepted)+1) * 100
    adapt_cumulative = np.cumsum(gen_adapt_accepted) / np.arange(1, len(gen_adapt_accepted)+1) * 100
    
    axes[1, 0].plot(gen_static_attempts, static_cumulative, 'b-', 
                    linewidth=2, label='Static', marker='o', markersize=4)
    axes[1, 0].plot(gen_adapt_attempts, adapt_cumulative, 'r-',
                    linewidth=2, label='Adaptive', marker='s', markersize=4)
    axes[1, 0].set_xlabel('Authentication Attempt')
    axes[1, 0].set_ylabel('Cumulative Acceptance Rate (%)')
    axes[1, 0].set_title('Genuine User: Acceptance Rate Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 105])
    
    # Plot 4: Threshold evolution
    axes[1, 1].fill_between(gen_adapt_attempts, 0, gen_adapt_thresholds,
                            alpha=0.3, color='blue', label='Adaptive threshold band')
    axes[1, 1].plot(gen_adapt_attempts, gen_adapt_dists, 'ro-', 
                    alpha=0.6, markersize=4, label='Distances')
    axes[1, 1].axhline(genuine_static[0][3], color='green', linestyle='--',
                       linewidth=2, label='Initial static threshold')
    axes[1, 1].set_xlabel('Authentication Attempt')
    axes[1, 1].set_ylabel('Distance')
    axes[1, 1].set_ylim(y_min, y_max)
    axes[1, 1].set_title('Adaptive Threshold Evolution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adaptive_threshold_evolution_zoomed.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print stats
    print(f"\nAdaptive Thresholding Performance:")
    print(f"   Static acceptance rate: {static_cumulative[-1]:.2f}%")
    print(f"   Adaptive acceptance rate: {adapt_cumulative[-1]:.2f}%")
    print(f"   Improvement: {adapt_cumulative[-1] - static_cumulative[-1]:.2f} percentage points")
    print(f"   Initial threshold: {genuine_static[0][3]:.4f}")
    print(f"   Final adaptive threshold: {gen_adapt_thresholds[-1]:.4f}")
    print(f"   Threshold change: {((gen_adapt_thresholds[-1] - genuine_static[0][3]) / genuine_static[0][3] * 100):.2f}%")


if __name__ == "__main__":
    df = pd.read_csv('../../data/features_extracted.csv')
    df['upDownLeftRightFlag'], _ = pd.factorize(df['upDownLeftRightFlag'])

    clean_vecs, clean_labels, live_data_for_testing = build_database(df, use_dp=False)
    scaler_ref = StandardScaler().fit(clean_vecs)
    clean_scaled = scaler_ref.transform(clean_vecs)
    model_ref = NearestNeighbors(n_neighbors=3).fit(clean_scaled)   # use 3 neighbors for robust distances
    thresholds_ref = calibrate_thresholds(clean_scaled, clean_labels, model_ref, lambda_factor=LAMBDA_FACTOR)

    # Run experiments
    print("\n" + "="*60)
    print("VISUALIZATION 1: DIFFERENTIAL PRIVACY PROTECTION")
    print("="*60)
    visualize_privacy_protection(df)

    print("\n" + "="*60)
    print("RUNNING AUTHENTICATION EXPERIMENTS")
    print("="*60)

    results = {}
    experiments = {
        "Static (no DP)": (False, False),
        "Adaptive (no DP)": (False, True),
        f"Static (DP ε={EPSILON})": (True, False),
        f"Adaptive (DP ε={EPSILON})": (True, True),
    }

    for name, (dp, adapt) in experiments.items():
        print(f"\n=== {name} ===")

        db_vecs_exp, db_labels_exp, live_exp = build_database(df, use_dp=dp,
                                                             dp_epsilon=EPSILON, dp_sensitivity=SENSITIVITY,
                                                             dp_mechanism=DP_MECHANISM)
        # Fit scaler+model for this experiment and calibrate thresholds here
        scaler_exp = StandardScaler().fit(db_vecs_exp)
        db_scaled_exp = scaler_exp.transform(db_vecs_exp)
        model_exp = NearestNeighbors(n_neighbors=3).fit(db_scaled_exp)
        thresholds_exp = calibrate_thresholds(db_scaled_exp, db_labels_exp, model_exp, lambda_factor=LAMBDA_FACTOR,neighbors_for_calib=2)

        # Now evaluate using the prebuilt model/scaler/thresholds and the SAME live dataset
        results[name] = evaluate_with_tracking_prebuilt(model_exp, scaler_exp, db_labels_exp,
                                                        thresholds_exp, live_exp, adaptive=adapt,
                                                        lambda_factor=LAMBDA_FACTOR, alpha_smooth=ADAPTIVE_SMOOTH_ALPHA,
                                                        update_hist_on_accepted_only=False)
        print(f"FAR={results[name]['FAR']:.2f}%, FRR={results[name]['FRR']:.2f}%, "
              f"ACC={results[name]['ACC']:.2f}%")

    # Visualization 2: Adaptive over time
    print("\n" + "="*60)
    print("VISUALIZATION 2: ADAPTIVE THRESHOLDING EVOLUTION")
    print("="*60)
    visualize_adaptive_over_time(
        results["Static (no DP)"]["genuine_outcomes"],
        results["Adaptive (no DP)"]["genuine_outcomes"],
        results["Static (no DP)"]["imposter_outcomes"],
        results["Adaptive (no DP)"]["imposter_outcomes"]
    )

    # Summary comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON: ACCURACY vs PRIVACY")
    print("="*60)

    modes = list(results.keys())
    FARs = [results[m]["FAR"] for m in modes]
    FRRs = [results[m]["FRR"] for m in modes]
    ACCs = [results[m]["ACC"] for m in modes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Bar chart
    x = np.arange(len(modes))
    w = 0.25
    axes[0].bar(x - w, FARs, w, label='FAR')
    axes[0].bar(x, FRRs, w, label='FRR')
    axes[0].bar(x + w, ACCs, w, label='Accuracy')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(modes, rotation=15, ha='right')
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title(f"Authentication Performance Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Right: Scatter plot showing accuracy vs privacy tradeoff
    privacy_levels = [0, 0, EPSILON, EPSILON]  # 0 = no DP
    axes[1].scatter(privacy_levels, ACCs, s=200, alpha=0.7)
    for i, mode in enumerate(modes):
        axes[1].annotate(mode.split(' (')[0],
                        (privacy_levels[i], ACCs[i]),
                        xytext=(10, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
    axes[1].set_xlabel(f'Privacy Level (ε, lower = more private)')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Privacy-Utility Tradeoff')
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_xaxis()  # Lower epsilon = more privacy on right

    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n All visualizations complete!")
