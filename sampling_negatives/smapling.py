# Cell 1: Imports and setup
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import esm
from descriptastorus.descriptors import rdNormalizedDescriptors
import warnings
warnings.filterwarnings('ignore')

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable pandas progress bar
tqdm.pandas()

# Parameters
NUM_SAMPLES = None  # Will be set to match positive pairs
RANDOM_STATE = 42
PCA_COMPONENTS = 50  # Number of PCA components for DR
PLOT_COMPONENTS = [0, 1]  # Which PCs to use for 2D visualization
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

print("Setup complete!")

# Cell 2: Load datasets and get target sample size
print("Loading datasets...")
negatives = pd.read_csv('../datasets/cysdb_negatives.csv')
positives = pd.read_csv('../datasets/cysdb_positives.csv')

# Set target sample size to match positives
NUM_SAMPLES = len(positives)
print(f"Target sample size: {NUM_SAMPLES:,}")
print(f"Available negatives: {len(negatives):,}")
print(f"Sampling ratio: {NUM_SAMPLES / len(negatives):.3f}")

# Cell 3: ESM-2 protein embeddings (GPU accelerated)
print("Setting up ESM-2 model...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.eval().to(device)
batch_converter = alphabet.get_batch_converter()

# Get unique proteins
unique_proteins = negatives[['Entry', 'Sequence']].drop_duplicates()
print(f"Unique proteins to embed: {len(unique_proteins):,}")

# Embedding parameters
MAX_LEN = 512
BATCH_SIZE = 4  # Reduced batch size for larger model

# Storage for embeddings
protein_embeddings = {}
protein_sequences = {}

# Process proteins in batches
protein_list = unique_proteins.values.tolist()
print("Generating protein embeddings...")

for i in tqdm(range(0, len(protein_list), BATCH_SIZE), desc="ESM-2 batches"):
    batch = protein_list[i:i + BATCH_SIZE]
    
    # Prepare batch
    batch_data = []
    batch_entries = []
    
    for entry, seq in batch:
        seq_trunc = seq if len(seq) <= MAX_LEN else seq[:MAX_LEN]
        batch_data.append((entry, seq_trunc))
        batch_entries.append(entry)
    
    # Convert batch
    batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
    batch_tokens = batch_tokens.to(device)
    
    # Generate embeddings
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        representations = results['representations'][33]
    
    # Store embeddings (mean pooling)
    for j, entry in enumerate(batch_entries):
        seq_len = len(batch_strs[j])
        rep = representations[j, 1:seq_len+1]  # Remove CLS token
        protein_embeddings[entry] = rep.mean(0).cpu().numpy().astype(np.float32)
        protein_sequences[entry] = batch_strs[j]

print(f"Generated embeddings for {len(protein_embeddings)} proteins")

# Cell 4: RDKit molecular descriptors
print("Computing molecular descriptors...")
gen = rdNormalizedDescriptors.RDKit2DNormalized()

# Get unique molecules
unique_molecules = negatives['SMILES'].unique()
print(f"Unique molecules: {len(unique_molecules):,}")

# Storage for molecular features
molecule_features = {}

# Process molecules
for smiles in tqdm(unique_molecules, desc="RDKit descriptors"):
    try:
        desc = gen.process(smiles=smiles)
        if desc is not None and len(desc) > 1:
            molecule_features[smiles] = np.array(desc[1:], dtype=np.float32)
    except Exception as e:
        print(f"Error processing {smiles}: {e}")

print(f"Generated features for {len(molecule_features)} molecules")

# Cell 5: Create combined embeddings for all pairs
print("Creating combined embeddings...")

# Filter negatives to only include pairs with valid embeddings
valid_negatives = negatives[
    (negatives['Entry'].isin(protein_embeddings.keys())) & 
    (negatives['SMILES'].isin(molecule_features.keys()))
].copy()

print(f"Valid negative pairs: {len(valid_negatives):,}")

# Create combined feature vectors
combined_features = []
valid_indices = []

for idx, row in tqdm(valid_negatives.iterrows(), 
                     total=len(valid_negatives), 
                     desc="Combining features"):
    try:
        prot_feat = protein_embeddings[row['Entry']]
        mol_feat = molecule_features[row['SMILES']]
        
        # Concatenate protein and molecule features
        combined = np.concatenate([prot_feat, mol_feat])
        combined_features.append(combined)
        valid_indices.append(idx)
    except Exception as e:
        print(f"Error combining features for index {idx}: {e}")

# Convert to numpy array
combined_features = np.array(combined_features)
print(f"Combined feature shape: {combined_features.shape}")

# Update valid_negatives to only include successfully processed pairs
valid_negatives = valid_negatives.loc[valid_indices].reset_index(drop=True)

# Cell 6: Standardize features and perform dimensionality reduction
print("Standardizing features...")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(combined_features)

print(f"Performing PCA dimensionality reduction to {PCA_COMPONENTS} components...")
pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
pca_features = pca.fit_transform(scaled_features)

print(f"PCA embedding shape: {pca_features.shape}")
print(f"Explained variance ratio (first 10 components): {pca.explained_variance_ratio_[:10]}")
print(f"Cumulative explained variance (first 10): {pca.explained_variance_ratio_[:10].cumsum()}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# Extract 2D coordinates for visualization
pca_2d = pca_features[:, PLOT_COMPONENTS]
print(f"Using PC{PLOT_COMPONENTS[0]+1} and PC{PLOT_COMPONENTS[1]+1} for visualization")

# Cell 7: BIRCH clustering
print("Performing BIRCH clustering on PCA features...")

# Determine number of clusters based on sample size
n_clusters = min(max(NUM_SAMPLES // 10, 10), 100)  # Between 10-100 clusters
print(f"Using {n_clusters} clusters")

# Use full PCA features for clustering (more informative than just 2D)
birch = Birch(n_clusters=n_clusters, random_state=RANDOM_STATE)
cluster_labels = birch.fit_predict(pca_features)

print(f"Cluster distribution:")
unique_labels, counts = np.unique(cluster_labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"  Cluster {label}: {count} samples")

# Cell 8: Stratified sampling from clusters
print("Performing stratified sampling...")

# Calculate samples per cluster (proportional to cluster size)
cluster_sizes = np.bincount(cluster_labels)
samples_per_cluster = np.round(
    (cluster_sizes / len(cluster_labels)) * NUM_SAMPLES
).astype(int)

# Ensure we get exactly NUM_SAMPLES
diff = NUM_SAMPLES - samples_per_cluster.sum()
if diff > 0:
    # Add extra samples to largest clusters
    largest_clusters = np.argsort(cluster_sizes)[-diff:]
    samples_per_cluster[largest_clusters] += 1
elif diff < 0:
    # Remove samples from largest clusters
    largest_clusters = np.argsort(cluster_sizes)[diff:]
    samples_per_cluster[largest_clusters] -= 1

# Sample from each cluster
sampled_indices = []
for cluster_id in tqdm(range(n_clusters), desc="Sampling clusters"):
    cluster_mask = cluster_labels == cluster_id
    cluster_indices = np.where(cluster_mask)[0]
    
    n_samples = min(samples_per_cluster[cluster_id], len(cluster_indices))
    if n_samples > 0:
        selected = np.random.choice(cluster_indices, n_samples, replace=False)
        sampled_indices.extend(selected)

sampled_indices = np.array(sampled_indices)
print(f"Sampled {len(sampled_indices)} pairs")

# Create sampled dataset
sampled_negatives = valid_negatives.iloc[sampled_indices].copy()

# Cell 9: Calculate coverage score
print("Calculating coverage score...")

# Use 2D PCA coordinates for coverage calculation
all_points = pca_2d
sampled_points = pca_2d[sampled_indices]

# Calculate coverage using multiple metrics
def calculate_coverage_score(all_points, sampled_points):
    """Calculate coverage score using multiple metrics"""
    
    # 1. Minimum distance coverage
    distances = pairwise_distances(all_points, sampled_points)
    min_distances = distances.min(axis=1)
    mean_min_distance = min_distances.mean()
    
    # 2. Grid-based coverage
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    
    # Create grid
    grid_size = 50
    x_bins = np.linspace(x_min, x_max, grid_size)
    y_bins = np.linspace(y_min, y_max, grid_size)
    
    # Count occupied cells
    all_hist, _, _ = np.histogram2d(all_points[:, 0], all_points[:, 1], 
                                   bins=[x_bins, y_bins])
    sampled_hist, _, _ = np.histogram2d(sampled_points[:, 0], sampled_points[:, 1], 
                                       bins=[x_bins, y_bins])
    
    # Coverage = fraction of non-empty cells that contain at least one sample
    non_empty_cells = all_hist > 0
    covered_cells = (sampled_hist > 0) & non_empty_cells
    grid_coverage = covered_cells.sum() / non_empty_cells.sum()
    
    # 3. Density-based coverage
    density_coverage = len(sampled_points) / len(all_points)
    
    # Combined score
    coverage_score = (grid_coverage + (1 - mean_min_distance / distances.max())) / 2
    
    return {
        'grid_coverage': grid_coverage,
        'mean_min_distance': mean_min_distance,
        'density_coverage': density_coverage,
        'combined_score': coverage_score
    }

coverage_metrics = calculate_coverage_score(all_points, sampled_points)

print("Coverage Metrics:")
print(f"  Grid coverage: {coverage_metrics['grid_coverage']:.3f}")
print(f"  Mean min distance: {coverage_metrics['mean_min_distance']:.3f}")
print(f"  Density coverage: {coverage_metrics['density_coverage']:.3f}")
print(f"  Combined score: {coverage_metrics['combined_score']:.3f}")

# Cell 10: Create visualization
print("Creating visualization...")

plt.figure(figsize=(20, 6))

# Plot 1: All points colored by cluster
plt.subplot(1, 3, 1)
scatter = plt.scatter(pca_2d[:, 0], pca_2d[:, 1], 
                     c=cluster_labels, alpha=0.6, s=1, cmap='tab20')
plt.title(f'All Negative Pairs by Cluster (n={len(pca_2d):,})')
plt.xlabel(f'PC{PLOT_COMPONENTS[0]+1}')
plt.ylabel(f'PC{PLOT_COMPONENTS[1]+1}')
plt.colorbar(scatter, label='Cluster')

# Plot 2: All points
plt.subplot(1, 3, 2)
plt.scatter(pca_2d[:, 0], pca_2d[:, 1], 
           c='lightblue', alpha=0.5, s=1, label='All negatives')
plt.title(f'All Negative Pairs (n={len(pca_2d):,})')
plt.xlabel(f'PC{PLOT_COMPONENTS[0]+1}')
plt.ylabel(f'PC{PLOT_COMPONENTS[1]+1}')
plt.legend()

# Plot 3: Sampled points over all points
plt.subplot(1, 3, 3)
plt.scatter(pca_2d[:, 0], pca_2d[:, 1], 
           c='lightgray', alpha=0.3, s=1, label='All negatives')
plt.scatter(sampled_points[:, 0], sampled_points[:, 1], 
           c='red', alpha=0.7, s=10, label='Sampled negatives')
plt.title(f'Sampled Coverage (n={len(sampled_points):,})')
plt.xlabel(f'PC{PLOT_COMPONENTS[0]+1}')
plt.ylabel(f'PC{PLOT_COMPONENTS[1]+1}')
plt.legend()

plt.tight_layout()
plt.savefig('negative_sampling_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional PCA analysis plot
plt.figure(figsize=(12, 4))

# Plot explained variance
plt.subplot(1, 2, 1)
plt.plot(range(1, min(21, len(pca.explained_variance_ratio_)+1)), 
         pca.explained_variance_ratio_[:20], 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.grid(True)

# Plot cumulative explained variance
plt.subplot(1, 2, 2)
plt.plot(range(1, min(21, len(pca.explained_variance_ratio_)+1)), 
         pca.explained_variance_ratio_[:20].cumsum(), 'ro-')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Cumulative Explained Variance')
plt.grid(True)

plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Cell 11: Save results
print("Saving results...")

# Save sampled negatives
sampled_negatives.to_csv('../datasets/sampled_negatives.csv', index=False)

# Save sampling info
sampling_info = {
    'total_negatives': len(negatives),
    'valid_negatives': len(valid_negatives),
    'sampled_negatives': len(sampled_negatives),
    'n_clusters': n_clusters,
    'coverage_metrics': coverage_metrics,
    'pca_components': PCA_COMPONENTS,
    'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
    'cumulative_explained_variance': pca.explained_variance_ratio_.cumsum().tolist(),
    'protein_embedding_dim': len(list(protein_embeddings.values())[0]),
    'molecule_feature_dim': len(list(molecule_features.values())[0]),
    'combined_feature_dim': combined_features.shape[1]
}

import json
with open('../datasets/sampling_info.json', 'w') as f:
    json.dump(sampling_info, f, indent=2)

print("\nSummary:")
print(f"Original negatives: {len(negatives):,}")
print(f"Valid negatives: {len(valid_negatives):,}")
print(f"Sampled negatives: {len(sampled_negatives):,}")
print(f"Target (positives): {NUM_SAMPLES:,}")
print(f"Coverage score: {coverage_metrics['combined_score']:.3f}")

print("\nFiles created:")
print("- sampled_negatives.csv")
print("- sampling_info.json")
print("- negative_sampling_visualization.png")
print("- pca_analysis.png")

print("\nPipeline complete!")
