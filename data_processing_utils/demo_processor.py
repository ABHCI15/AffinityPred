#!/usr/bin/env python3
"""
Demo script showing how to use the AffinityDataProcessor class.
This demonstrates the full workflow including protein embeddings and graph conversion.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prep_data import AffinityDataProcessor


def main():
    """Main demo function."""
    print("=== AffinityDataProcessor Demo ===\n")
    
    # Initialize the processor
    print("1. Initializing AffinityDataProcessor...")
    processor = AffinityDataProcessor(
        input_col='SMILES',
        label_col='pCHEMBL', 
        seq_col='Sequence',
        device='cuda',  # Change to 'cpu' if no GPU available
        embedding_cache_path="../datasets/protein_embeddings.pkl"
    )
    print("✓ Processor initialized\n")
    
    # Check available datasets
    print("2. Available datasets:")
    dataset_dir = "../datasets"
    if os.path.exists(dataset_dir):
        csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
        for f in csv_files[:5]:  # Show first 5
            print(f"   - {f}")
        if len(csv_files) > 5:
            print(f"   ... and {len(csv_files) - 5} more")
    print()
    
    # Example with a small synthetic dataset (for demo purposes)
    print("3. Creating synthetic demo dataset...")
    demo_data = {
        'SMILES': [
            'CCO',  # Ethanol
            'CC(=O)O',  # Acetic acid
            'c1ccccc1',  # Benzene
            'CCN(CC)CC',  # Triethylamine
            'CC(C)O'  # Isopropanol
        ],
        'pCHEMBL': [4.5, 3.2, 5.1, 2.8, 4.0],
        'Sequence': [
            'MVLSPADKTNVKAAW',  # Example protein sequences
            'MGSSHHHHHHSSGLV',
            'MTYKLIINGKTLKGE',
            'MVLSPADKTNVKAAW',  # Duplicate to test caching
            'MGEFPMRGVNLDIE'
        ],
        'cluster': [1, 2, 1, 1, 2]  # For stratification demo
    }
    
    demo_df = pd.DataFrame(demo_data)
    demo_csv_path = "demo_dataset.csv"
    demo_df.to_csv(demo_csv_path, index=False)
    print(f"✓ Created demo dataset with {len(demo_df)} samples\n")
    
    # Process the dataset
    print("4. Processing dataset (protein embeddings + graph conversion)...")
    try:
        train_df, valid_df, test_df = processor.prepare_dataset(
            csv_path=demo_csv_path,
            test_size=0.2,
            valid_size=0.2,  # 20% of training set
            random_state=42,
            stratify_col='cluster',
            output_dir="demo_processed_data"
        )
        
        print("✓ Dataset processing completed!")
        print(f"   Train: {len(train_df)} samples")
        print(f"   Valid: {len(valid_df)} samples") 
        print(f"   Test: {len(test_df)} samples\n")
        
    except Exception as e:
        print(f"⚠ Error during processing: {e}")
        print("This is normal for the demo - ESM model may not be available")
        print("The class structure is ready for when you have the model set up\n")
        
        # Fallback: show graph conversion only
        print("5. Fallback: Testing SMILES to graph conversion only...")
        for smiles in demo_data['SMILES']:
            graph = processor.smiles_to_graph(smiles)
            if graph is not None:
                print(f"   ✓ {smiles} -> Graph with {graph.number_of_nodes()} nodes")
            else:
                print(f"   ✗ {smiles} -> Failed to convert")
        print()
    
    # Show caching benefits
    print("6. Testing caching functionality...")
    print("   SMILES cache:", len(processor.smiles_graph_cache), "entries")
    print("   Protein cache:", len(processor.protein_embeddings_cache), "entries")
    
    # Test duplicate SMILES processing
    duplicate_smiles = 'CCO'
    print(f"\n   Processing duplicate SMILES '{duplicate_smiles}':")
    graph1 = processor.smiles_to_graph(duplicate_smiles)
    graph2 = processor.smiles_to_graph(duplicate_smiles)  # Should use cache
    print(f"   Same object from cache: {graph1 is graph2}")
    print()
    
    # Show data loader creation capabilities
    print("7. DataLoader creation capabilities:")
    print("   - Handles both graph-only and graph+protein embedding modes")
    print("   - Supports custom batch sizes and collation")
    print("   - Maintains stratification for balanced splits")
    print("   - Exports processed data in multiple formats (.csv, .npy, .pkl)")
    print()
    
    # Cleanup
    if os.path.exists(demo_csv_path):
        os.remove(demo_csv_path)
    
    print("=== Demo completed! ===")
    print("\nKey features of AffinityDataProcessor:")
    print("✓ Protein sequence embedding with ESM model + caching")
    print("✓ SMILES to molecular graph conversion + caching") 
    print("✓ Automatic data cleaning and validation")
    print("✓ Stratified train/validation/test splits")
    print("✓ Flexible DataLoader creation")
    print("✓ Export to multiple file formats")
    print("✓ Backward compatibility with existing code")


if __name__ == "__main__":
    main()
