# vae_similarity_evaluation_FINAL.py
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import entropy  # For entropy calculation
import argparse
import os
import random

# ----------------------- Config -----------------------
# (Using user_id 22.0 as in the original script)
DATASET_PATH = 'features_extracted.csv'        # full dataset (used for registration data)
ATTACKER_DATA_PATH = 'attacker_stolen_data.csv'  # stolen 20% set (used to train VAE)
FEATURE_COLUMNS = [
    'strokeDuration', 'startX', 'startY', 'stopX', 'stopY',
    'directEndToEndDistance', 'meanResultantLength', 'upDownLeftRightFlag',
    'directionOfEndToEndLine', 'largestDeviationFromEndToEndLine',
    'averageDirection', 'lengthOfTrajectory', 'averageVelocity',
    'midStrokePressure', 'midStrokeArea'
]
FEATURE_DIM = len(FEATURE_COLUMNS)
LATENT_DIM = 5
VAE_EPOCHS = 500
BATCH_SIZE = 16
NUM_ATTACKS = 200  # how many synthetic vectors to generate
RANDOM_SEED = 42

# ----------------------- VAE -----------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ----------------------- Utility / Training / Eval -----------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def prepare_data_and_scaler(dataset_path, attacker_path, feature_cols):
    """
    Loads data and prepares scaler.
    1. Loads full dataset to get 70% registration data for user 22.
    2. Fits 'server_scaler' on this 70% registration data.
    3. Loads the attacker's separate 20% stolen data.
    
    Returns:
        server_scaler: Scaler fitted on registration data.
        registration_vectors: (N_reg, D) numpy array of *unscaled* registration vectors.
        stolen_vectors: (N_stolen, D) numpy array of *unscaled* stolen vectors.
    """
    # 1) Load full dataset
    df_full = pd.read_csv(dataset_path)
    if 'user_id' not in df_full.columns:
        raise ValueError("'user_id' column required in dataset to simulate registration split.")
    
    # ensure categorical flag is numeric
    if 'upDownLeftRightFlag' in df_full.columns:
        df_full['upDownLeftRightFlag'], _ = pd.factorize(df_full['upDownLeftRightFlag'])

    # Filter to a single user (using 22.0 as in original script)
    user_df_full = df_full[df_full['user_id'] == 22.0]
    if user_df_full.empty:
        raise ValueError("No rows with user_id == 22.0 found in the full dataset.")

    # 2) split 70/30 to simulate registration (reg_df) and hold-out
    reg_df, _ = train_test_split(user_df_full, test_size=0.20, shuffle=False)
    
    if reg_df.empty:
        raise ValueError("Registration DF is empty after split.")
        
    registration_vectors = reg_df[feature_cols].values
    
    server_scaler = StandardScaler()
    server_scaler.fit(registration_vectors)
    print(f"[info] Fitted server scaler on registration set (n={len(registration_vectors)})")

    # 3) Load the attacker's stolen data (20% file)
    stolen_df = pd.read_csv(attacker_path)
    if 'upDownLeftRightFlag' in stolen_df.columns:
        stolen_df['upDownLeftRightFlag'], _ = pd.factorize(stolen_df['upDownLeftRightFlag'])
    
    stolen_vectors = stolen_df[feature_cols].values
    print(f"[info] Loaded attacker's stolen data (n={len(stolen_vectors)})")

    return server_scaler, registration_vectors, stolen_vectors

def train_vae_on_stolen(stolen_vectors, server_scaler, epochs=VAE_EPOCHS, batch_size=BATCH_SIZE):
    """
    Train VAE on the stolen vectors after scaling using server_scaler.
    """
    # *** THIS IS THE BUG FIX ***
    # Scale stolen vectors using server scaler (this simulates attacker knowing server scaler)
    # Must use .transform(), NOT .fit_transform()
    try:
        stolen_scaled = server_scaler.transform(stolen_vectors)
    except Exception as e:
        print(f"Error during scaling: {e}")
        print("This can happen if the stolen data has a different number of features than the registration data.")
        raise
        
    stolen_tensor = torch.tensor(stolen_scaled, dtype=torch.float32)

    dataset = TensorDataset(stolen_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    vae = VAE(FEATURE_DIM, LATENT_DIM)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    vae.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, mu, logvar = vae(batch)
            loss = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"[train] Epoch {epoch+1}/{epochs}  avg_loss={epoch_loss / max(len(loader),1):.4f}")
    
    vae.eval()
    # We return the unscaled stolen vectors for plotting
    return vae, stolen_vectors 

def evaluate_synthetic_similarity(vae_model, server_scaler, target_vectors, num_synth=NUM_ATTACKS):
    """
    Generates synthetic vectors and compares them against the *target_vectors*
    (the original registration data).
    
    target_vectors: The *unscaled* registration vectors. (N_target, D)
    """
    
    # --- Helper function for entropy ---
    def calculate_entropy(data, n_bins=50):
        """Calculates the avg entropy per feature by binning the data."""
        n_samples, n_features = data.shape
        total_entropy = 0.0
        
        for i in range(n_features):
            # 1. Create a histogram (probability distribution) for the feature
            # We use the min/max of the *entire* feature to set bin edges
            # to ensure a fair comparison between original and fake.
            feat_min = min(target_vectors[:, i].min(), data[:, i].min())
            feat_max = max(target_vectors[:, i].max(), data[:, i].max())
            
            # Create bins
            bins = np.linspace(feat_min, feat_max, n_bins + 1)
            hist, _ = np.histogram(data[:, i], bins=bins)
            
            # 2. Convert counts to probabilities
            pk = hist / float(n_samples)
            
            # 3. Calculate entropy for this feature
            # 'pk' is the probability distribution
            total_entropy += entropy(pk, base=2)
            
        # Return the average entropy across all features
        return total_entropy / n_features
    # --- End of helper function ---

    
    vae_model.eval()
    with torch.no_grad():
        z = torch.randn(num_synth, LATENT_DIM, dtype=torch.float32)
        synthetic_scaled = vae_model.decoder(z).cpu().numpy()

    # inverse transform to original space
    synthetic_unscaled = server_scaler.inverse_transform(synthetic_scaled)

    # *** THIS IS THE CONCEPTUAL FIX ***
    # We measure distance from synthetic vectors to the TARGET vectors
    d_synth_target = pairwise_distances(synthetic_unscaled, target_vectors, metric='euclidean') # (num_synth, n_target)
    min_dists_synth_to_target = d_synth_target.min(axis=1)

    # As a baseline, compute the nearest-neighbour distances *within the target set*
    d_target_target = pairwise_distances(target_vectors, target_vectors, metric='euclidean')
    np.fill_diagonal(d_target_target, np.inf)
    nn_dists_target = d_target_target.min(axis=1)

    # Stats
    def stats(arr):
        return {
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            '10%': float(np.quantile(arr, 0.1)),
            '25%': float(np.quantile(arr, 0.25)),
            '75%': float(np.quantile(arr, 0.75)),
            '90%': float(np.quantile(arr, 0.9))
        }

    synth_stats = stats(min_dists_synth_to_target)
    target_stats = stats(nn_dists_target)

    # Proportions
    p_leq_median = float((min_dists_synth_to_target <= target_stats['median']).mean())
    p_leq_mean = float((min_dists_synth_to_target <= target_stats['mean']).mean())
    p_leq_90 = float((min_dists_synth_to_target <= target_stats['90%']).mean())
    
    # *** NEW: Calculate Entropies ***
    entropy_original = calculate_entropy(target_vectors)
    entropy_fake = calculate_entropy(synthetic_unscaled)


    results = {
        'num_synth': num_synth,
        'n_target': target_vectors.shape[0],
        'synth_min_dists': min_dists_synth_to_target,
        'target_nn_dists': nn_dists_target,
        'synth_stats': synth_stats,
        'target_nn_stats': target_stats,
        'proportions': {
            'synth_leq_target_median': p_leq_median,
            'synth_leq_target_mean': p_leq_mean,
            'synth_leq_target_90pct': p_leq_90
        },
        'synthetic_vectors_unscaled': synthetic_unscaled,
        # Add new results
        'entropy_stats': {
            'original': entropy_original,
            'fake': entropy_fake
        }
    }
    return results

def generate_plots(results, target_vectors, stolen_vectors):
    """
    Generates and saves histograms and PCA plots.
    """
    print("\n[plot] Generating plots...")
    
    synth_dists = results['synth_min_dists']
    target_dists = results['target_nn_dists']
    synthetic_vectors = results['synthetic_vectors_unscaled']

    # 1. Histogram of Nearest-Neighbour Distances
    plt.figure(figsize=(10, 6))
    plt.hist(target_dists, bins=50, alpha=0.7, label='Target NN Dists (Baseline)', density=True, color='blue')
    plt.hist(synth_dists, bins=50, alpha=0.7, label='Synthetic-to-Target Dists (Attack)', density=True, color='red')
    plt.axvline(results['target_nn_stats']['median'], color='blue', linestyle='--', label=f"Target Median ({results['target_nn_stats']['median']:.2f})")
    plt.axvline(results['synth_stats']['median'], color='red', linestyle='--', label=f"Synth Median ({results['synth_stats']['median']:.2f})")
    plt.title('Nearest Neighbour Distance Comparison')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('distance_histogram.png')
    print("[plot] Saved distance_histogram.png")

    # 2. PCA Plot (2D)
    pca = PCA(n_components=2)
    
    # Combine all data for consistent PCA fitting
    all_data = np.vstack([target_vectors, stolen_vectors, synthetic_vectors])
    pca.fit(all_data)
    
    # Transform each set
    target_2d = pca.transform(target_vectors)
    stolen_2d = pca.transform(stolen_vectors)
    synth_2d = pca.transform(synthetic_vectors)
    
    plt.figure(figsize=(12, 8))
    # Plot target data first (as the "ground truth" cloud)
    plt.scatter(target_2d[:, 0], target_2d[:, 1], alpha=0.3, label='Target (Registration)', s=20, color='blue')
    # Plot stolen data
    plt.scatter(stolen_2d[:, 0], stolen_2d[:, 1], alpha=0.5, label='Stolen (VAE Training)', s=20, color='orange')
    # Plot synthetic data
    plt.scatter(synth_2d[:, 0], synth_2d[:, 1], alpha=0.7, label='Synthetic (Attack)', s=10, color='red', marker='x')
    
    plt.title('PCA 2D Visualization of Vector Sets')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.savefig('pca_2d_plot.png')
    print("[plot] Saved pca_2d_plot.png")
    plt.close('all')


# ----------------------- Main -----------------------
def main(args):
    set_seed(RANDOM_SEED)

    # Sanity checks for files
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"{DATASET_PATH} not found. Provide the full dataset to simulate server scaler.")
    if not os.path.exists(ATTACKER_DATA_PATH):
        raise FileNotFoundError(f"{ATTACKER_DATA_PATH} not found. Provide the attacker's stolen data (20%).")

    # 1. Prepare data
    # registration_vectors = The 70% set to mimic (The Target)
    # stolen_unscaled_vectors = The 20% set to train VAE on
    server_scaler, registration_vectors, stolen_unscaled_vectors = prepare_data_and_scaler(
        DATASET_PATH, ATTACKER_DATA_PATH, FEATURE_COLUMNS
    )

    # 2. Train VAE
    vae_model, stolen_vectors_for_plot = train_vae_on_stolen(
        stolen_unscaled_vectors, server_scaler, epochs=args.epochs, batch_size=args.batch_size
    )

    # 3. Evaluate
    results = evaluate_synthetic_similarity(
        vae_model, server_scaler, registration_vectors, num_synth=args.num_synth
    )

    # Print a readable summary
    print("\n=== MIMICRY ATTACK EVALUATION SUMMARY ===")
    print(f"Target set size (registration): {results['n_target']}")
    print(f"Training set size (stolen): {len(stolen_unscaled_vectors)}")
    print(f"Num synthetic samples: {results['num_synth']}\n")

    print("Target nearest-neighbour stats (distance to nearest *other* target point):")
    for k, v in results['target_nn_stats'].items():
        print(f"  {k}: {v:.6f}")
    print("\nSynthetic-to-Target nearest distances stats (each synth -> nearest target):")
    for k, v in results['synth_stats'].items():
        print(f"  {k}: {v:.6f}")
    print("\nProportions of synthetic vectors with distance <= target statistics:")
    for k, v in results['proportions'].items():
        print(f"  {k}: {v * 100:.2f}%")

    # Print Entropy Stats
    print("\nAverage Feature Entropy (bits):")
    print(f"  Original (Target): {results['entropy_stats']['original']:.6f}")
    print(f"  Synthetic (Fake):  {results['entropy_stats']['fake']:.6f}")


    # 4. Generate Plots
    generate_plots(results, registration_vectors, stolen_vectors_for_plot)
    
    # Optionally save arrays to disk for inspection
    if args.save:
        np.save('synth_min_dists_to_target.npy', results['synth_min_dists'])
        np.save('target_nn_dists.npy', results['target_nn_dists'])
        print("\nSaved synth_min_dists_to_target.npy and target_nn_dists.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE mimicry similarity evaluation.")
    parser.add_argument('--epochs', type=int, default=VAE_EPOCHS, help='VAE training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Training batch size')
    parser.add_argument('--num_synth', type=int, default=NUM_ATTACKS, help='Number of synthetic samples to generate')
    parser.add_argument('--save', action='store_true', help='Save distance arrays to disk')
    args = parser.parse_args()
    
    # Set global based on args (or pass args to main)
    NUM_ATTACKS = args.num_synth
    main(args)