# AffinityDataProcessor Usage Guide

## Overview

The `AffinityDataProcessor` class provides a comprehensive solution for processing chemical-protein affinity data. It handles protein sequence embeddings, SMILES to graph conversion, and dataset preparation with efficient caching and stratified splitting.

## Key Features

- **Protein Sequence Embedding**: Uses ESM models with intelligent caching
- **SMILES to Graph Conversion**: Converts molecular SMILES to DGL graphs with caching
- **Data Cleaning**: Automatic validation and cleaning of chemical data
- **Stratified Splitting**: Supports stratification for balanced train/val/test splits
- **Multiple Export Formats**: Saves data as CSV, NumPy arrays, and pickle files
- **Backward Compatibility**: Works with existing code without breaking changes

## Basic Usage

### 1. Initialize the Processor

```python
from data_processing_utils.prep_data import AffinityDataProcessor

# Basic initialization
processor = AffinityDataProcessor()

# Advanced initialization with caching
processor = AffinityDataProcessor(
    input_col='SMILES',           # Column name for SMILES
    label_col='pCHEMBL',          # Column name for labels
    seq_col='Sequence',           # Column name for protein sequences
    device='cuda',                # Device for ESM model
    embedding_cache_path="datasets/protein_embeddings.pkl"  # Cache file
)
```

### 2. Process Your Dataset

```python
# Process dataset with all features
train_df, valid_df, test_df = processor.prepare_dataset(
    csv_path="datasets/your_data.csv",
    test_size=0.2,                # 20% for testing
    valid_size=0.1,               # 10% of training for validation
    random_state=42,              # For reproducibility
    stratify_col='cluster',       # Column for stratification (optional)
    output_dir="processed_data"   # Save processed files here
)
```

### 3. Create DataLoaders

```python
# Create DataLoaders for training
train_loader, valid_loader, test_loader = processor.create_dataloaders(
    train_df, valid_df, test_df,
    batch_size=32,
    include_protein_emb=True      # Include protein embeddings
)

# Use in training loop
for batch in train_loader:
    if len(batch) == 3:  # graph, protein_emb, labels
        batched_graph, protein_embs, labels = batch
        # Your model forward pass here
    else:  # graph, labels (no protein embeddings)
        batched_graph, labels = batch
        # Your model forward pass here
```

## Advanced Usage

### Custom Protein Embedding

```python
# Manual protein embedding
sequence = "MVLSPADKTNVKAAW"
embedding = processor.embed_sequence(sequence)
print(f"Embedding shape: {embedding.shape}")  # Should be (1152,) for ESM
```

### SMILES Processing with Caching

```python
# Convert SMILES to graphs (with automatic caching)
smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
for smiles in smiles_list:
    graph = processor.smiles_to_graph(smiles)
    if graph:
        print(f"{smiles}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

# Check cache status
print(f"Cached graphs: {len(processor.smiles_graph_cache)}")
print(f"Cached embeddings: {len(processor.protein_embeddings_cache)}")
```

### Batch Processing with Mapping

```python
# Efficient processing of dataframes
df_with_graphs = processor.create_smiles_graph_mapping(your_df)
df_with_embeddings = processor.extract_sequence_embeddings(your_df)
```

## File Outputs

When you run `prepare_dataset()` with an `output_dir`, the following files are created:

```
processed_data/
├── train_processed.csv              # Training data (no graphs)
├── train_protein_embeddings.npy     # Protein embeddings array
├── train_graphs.pkl                 # Molecular graphs (pickle)
├── train_labels.npy                 # Training labels
├── valid_processed.csv              # Validation data
├── valid_protein_embeddings.npy     # Validation embeddings
├── valid_graphs.pkl                 # Validation graphs
├── valid_labels.npy                 # Validation labels
├── test_processed.csv               # Test data
├── test_protein_embeddings.npy      # Test embeddings
├── test_graphs.pkl                  # Test graphs
└── test_labels.npy                  # Test labels
```

## Backward Compatibility

The class maintains compatibility with existing code:

```python
# Old way (still works)
from data_processing_utils.prep_data import prepare_dataset, collate
train_df, valid_df, test_df = prepare_dataset("data.csv")

# New way (recommended)
processor = AffinityDataProcessor()
train_df, valid_df, test_df = processor.prepare_dataset("data.csv")
```

## Performance Tips

1. **Use Caching**: Always provide `embedding_cache_path` to avoid recomputing embeddings
2. **Batch Processing**: Process large datasets in chunks if memory is limited
3. **GPU Usage**: Set `device='cuda'` if you have a GPU for faster embedding computation
4. **Unique SMILES**: The processor automatically handles duplicate SMILES efficiently

## Error Handling

The processor includes robust error handling:

- Invalid SMILES are automatically filtered out
- Failed graph conversions are logged and skipped
- Protein embedding errors fall back to zero vectors
- Missing columns trigger informative error messages

## Example: Complete Workflow

```python
# Complete example
from data_processing_utils.prep_data import AffinityDataProcessor
import torch

# 1. Initialize processor
processor = AffinityDataProcessor(
    embedding_cache_path="datasets/protein_embeddings.pkl"
)

# 2. Prepare dataset
train_df, valid_df, test_df = processor.prepare_dataset(
    csv_path="datasets/chembl_data.csv",
    stratify_col='cluster',
    output_dir="processed_chembl"
)

# 3. Create data loaders
train_loader, valid_loader, test_loader = processor.create_dataloaders(
    train_df, valid_df, test_df, batch_size=64
)

# 4. Training loop example
for epoch in range(10):
    for batch in train_loader:
        batched_graph, protein_embs, labels = batch
        
        # Your model forward pass
        # predictions = model(batched_graph, protein_embs)
        # loss = criterion(predictions, labels)
        # ... training code ...
        
        break  # Just show structure
    break
```

## Troubleshooting

### Common Issues

1. **ESM Model Not Found**: Ensure ESM is properly installed and GPU is available
2. **Memory Issues**: Reduce batch size or process data in smaller chunks
3. **Invalid SMILES**: Check your SMILES column for malformed entries
4. **Cache File Permissions**: Ensure write permissions for cache file location

### Debug Mode

```python
# Enable verbose output
processor = AffinityDataProcessor(embedding_cache_path="debug_cache.pkl")
# The processor will print progress updates during processing
```

This class provides a robust, efficient, and user-friendly way to prepare your chemical-protein affinity data while maintaining compatibility with existing code.
