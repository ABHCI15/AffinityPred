"""Generate Boltz YAML files for top 10 compound-protein pairs from predictions."""
import pandas as pd
from pathlib import Path


def create_boltz_yaml(sequence: str, smiles: str, rank: int, dataset: str) -> str:
    """
    Create Boltz YAML content for a compound-protein pair.
    
    Args:
        sequence: Protein sequence
        smiles: Compound SMILES string
        rank: Rank of the prediction (1-10)
        dataset: Dataset name (cysdb or mol_glue)
    
    Returns:
        YAML content as string
    """
    yaml_content = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {sequence}
  - ligand:
      id: B
      smiles: {smiles}
properties:
    - affinity:
        binder: B
"""
    return yaml_content


def generate_yaml_files():
    """Generate YAML files for top 10 predictions from each dataset."""
    # Load datasets
    df_cysdb = pd.read_csv('cysdb_predictions.csv')
    df_mol_glue = pd.read_csv('mol_glue_predictions.csv')
    
    # Sort by predicted pChEMBL (descending)
    df_cysdb = df_cysdb.sort_values(by='Predicted pChEMBL', ascending=False)
    df_mol_glue = df_mol_glue.sort_values(by='Predicted pChEMBL', ascending=False)
    
    # Get top 10 predictions for each protein
    top10_cysdb = df_cysdb.groupby('Sequence').head(10).copy()
    top10_mol_glue = df_mol_glue.groupby('Sequence').head(10).copy()
    
    # Add rank column
    top10_cysdb['Rank'] = top10_cysdb.groupby('Sequence')['Predicted pChEMBL'].rank(
        ascending=False, method='dense'
    ).astype(int)
    top10_mol_glue['Rank'] = top10_mol_glue.groupby('Sequence')['Predicted pChEMBL'].rank(
        ascending=False, method='dense'
    ).astype(int)
    
    # Generate YAML files for CysDB
    for _, row in top10_cysdb.iterrows():
        sequence = row['Sequence']
        smiles = row['SMILES']
        rank = row['Rank']
        
        # Create filename: rank_first5aa_dataset.yaml
        first_5aa = sequence[:5]
        filename = f"rank{rank}_{first_5aa}_cysdb.yaml"
        
        yaml_content = create_boltz_yaml(sequence, smiles, rank, 'cysdb')
        
        with open(filename, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created: {filename}")
    
    # Generate YAML files for mol_glue
    for _, row in top10_mol_glue.iterrows():
        sequence = row['Sequence']
        smiles = row['SMILES']
        rank = row['Rank']
        
        # Create filename: rank_first5aa_dataset.yaml
        first_5aa = sequence[:5]
        filename = f"rank{rank}_{first_5aa}_mol_glue.yaml"
        
        yaml_content = create_boltz_yaml(sequence, smiles, rank, 'mol_glue')
        
        with open(filename, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created: {filename}")
    
    print(f"\nGenerated {len(top10_cysdb)} YAML files for CysDB dataset")
    print(f"Generated {len(top10_mol_glue)} YAML files for mol_glue dataset")


if __name__ == "__main__":
    generate_yaml_files()
    
