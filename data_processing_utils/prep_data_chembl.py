import dgl
import dgl.function as fn
import matplotlib.pyplot as plt

# from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# from dgl.nn import GraphConv, GATConv, SAGEConv  
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

import pickle
import os
import esm
from esm.models.esmc import ESMC
from esm.sdk.api import (
    ESMProtein,
    LogitsConfig,
    LogitsOutput,
    ProteinType,
)

input_col = 'Smiles'
label_col = 'pCHEMBL'
seq_col = 'Sequence'


class AffinityDataProcessor:
    """
    A comprehensive data processor for chemical-protein affinity prediction.
    Handles protein sequence embeddings, SMILES to graph conversion, and dataset preparation.
    """
    
    def __init__(self, input_col='Smiles', label_col='pCHEMBL', seq_col='Sequence', 
                 device='cuda', embedding_cache_path='datasets/protein_chembl_embeddings.pkl', graph_cache_path=None):
        """
        Initialize the data processor.
        
        Args:
            input_col (str): Column name for SMILES data
            label_col (str): Column name for labels
            seq_col (str): Column name for protein sequences
            device (str): Device for ESM model ('cuda' or 'cpu')
            embedding_cache_path (str): Path to pickle file with precomputed embeddings
        """
        self.input_col = input_col
        self.label_col = label_col
        self.seq_col = seq_col
        self.device = device
        self.embedding_cache_path = embedding_cache_path
        self.graph_cache_path = graph_cache_path
        # Cache for embeddings and graphs
        self.protein_embeddings_cache = {}
        self.smiles_graph_cache = {}
        
        # ESM model (initialized lazily)
        self.esm_model = None
        
        # Load precomputed embeddings if available
        self._load_embedding_cache()
        
    def _load_embedding_cache(self):
        """Load precomputed protein embeddings from pickle file."""
        if self.embedding_cache_path and os.path.exists(self.embedding_cache_path):
            try:
                with open(self.embedding_cache_path, 'rb') as f:
                    self.protein_embeddings_cache = pickle.load(f)
                print(f"Loaded {len(self.protein_embeddings_cache)} precomputed embeddings from {self.embedding_cache_path}")
            except Exception as e:
                print(f"Warning: Could not load embedding cache from {self.embedding_cache_path}: {e}")
                self.protein_embeddings_cache = {}
        else:
            self.protein_embeddings_cache = {}
    
    def _save_embedding_cache(self):
        """Save protein embeddings cache to pickle file."""
        if self.embedding_cache_path:
            try:
                with open(self.embedding_cache_path, 'wb') as f:
                    pickle.dump(self.protein_embeddings_cache, f)
                print(f"Saved {len(self.protein_embeddings_cache)} embeddings to {self.embedding_cache_path}")
            except Exception as e:
                print(f"Warning: Could not save embedding cache to {self.embedding_cache_path}: {e}")
    
    def _init_esm_model(self):
        """Initialize ESM model lazily."""
        if self.esm_model is None:
            try:
                self.esm_model = ESMC.from_pretrained("esmc_600m").to(self.device)
                print("ESM model initialized successfully")
            except Exception as e:
                print(f"Error initializing ESM model: {e}")
                raise
    
    def embed_sequence(self, sequence: str) -> np.ndarray:
        """
        Embed a protein sequence using ESM model with caching.
        
        Args:
            sequence (str): Protein sequence
            
        Returns:
            np.ndarray: Mean embedding vector
        """
        # Check cache first
        if sequence in self.protein_embeddings_cache:
            return self.protein_embeddings_cache[sequence]
        
        # Initialize model if needed
        self._init_esm_model()
        
        try:
            protein = ESMProtein(sequence=sequence)
            
            # Use the model to get embeddings
            # with torch.no_grad():
            #     # Convert sequence to tokens
            #     tokens = self.esm_model.tokenize([sequence])
            #     if torch.cuda.is_available() and self.device == 'cuda':
            #         tokens = tokens.to('cuda')
                
            #     # Get embeddings
            #     output = self.esm_model(tokens)
                
            #     # Extract the embedding (usually last_hidden_state)
            #     if hasattr(output, 'last_hidden_state'):
            #         embeddings = output.last_hidden_state
            #     elif hasattr(output, 'embeddings'):
            #         embeddings = output.embeddings
            #     else:
            #         # Fallback: try to get the output tensor directly
            #         embeddings = output[0] if isinstance(output, tuple) else output
                
            #     # Average over sequence length (excluding special tokens)
            #     # Typically: [CLS] seq [SEP] -> take positions 1:-1
            #     if embeddings.dim() == 3:  # (batch, seq_len, hidden_dim)
            #         seq_embeddings = embeddings[0, 1:-1, :]  # Remove CLS and SEP tokens
            #         mean_embedding = seq_embeddings.mean(dim=0).cpu().numpy()
            #     else:
            #         mean_embedding = embeddings.mean(dim=0).cpu().numpy()

            protein_tensor = self.esm_model.encode(protein) # type: ignore
            EMBEDDING_CONFIG = LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=False)
            output = self.esm_model.logits(protein_tensor, EMBEDDING_CONFIG) # type: ignore
            # Cache the result
            mean_embedding = output.embeddings.mean(dim=1).squeeze(0).detach().cpu().numpy()  #type: ignore shape [1152]
            self.protein_embeddings_cache[sequence] = mean_embedding
            return mean_embedding
            
        except Exception as e:
            print(f"Error embedding sequence: {e}")
            # Return zero vector as fallback (adjust size based on your ESM model)
            fallback_size = 1152  # ESM2 typical size, adjust as needed
            zero_embedding = np.zeros(fallback_size)
            self.protein_embeddings_cache[sequence] = zero_embedding
            return zero_embedding
    
    def extract_sequence_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract embeddings for all sequences in dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with sequence column
            
        Returns:
            pd.DataFrame: DataFrame with added embedding column
        """
        if self.seq_col not in df.columns:
            raise ValueError(f"Sequence column '{self.seq_col}' not found in dataframe")
        
        print("Extracting protein sequence embeddings...")
        embeddings = []
        
        for idx, sequence in enumerate(df[self.seq_col]):
            if pd.isna(sequence):
                embeddings.append(np.zeros(1152))  # Default embedding size
            else:
                embeddings.append(self.embed_sequence(str(sequence)))
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)} sequences")
        
        df = df.copy()
        df['protein_embedding'] = embeddings
        
        # Save cache periodically
        self._save_embedding_cache()
        
        return df
    
    def get_atom_features(self, atom):
        """Extract atom features for graph construction."""
        atom_type = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
        atom_type_enc = [0.0] * len(atom_type)
        try:
            atom_type_enc[atom_type.index(atom.GetSymbol()) if atom.GetSymbol() in atom_type else -1] = 1.0
        except ValueError:
            atom_type_enc[-1] = 1.0

        hybridization_types = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
            Chem.rdchem.HybridizationType.S,
        ]
        hybridization = [0.0] * len(hybridization_types)
        try:
            hybridization_idx = hybridization_types.index(atom.GetHybridization())
            hybridization[hybridization_idx] = 1.0
        except ValueError:
            hybridization[-1] = 1.0

        atomic_num = [float(atom.GetAtomicNum())]
        degree = [float(atom.GetDegree())]
        total_num_hs = [float(atom.GetTotalNumHs())]
        formal_charge = [float(atom.GetFormalCharge())]
        is_aromatic = [float(atom.GetIsAromatic())]
        num_radical_electrons = [float(atom.GetNumRadicalElectrons())]
        in_ring = [float(atom.IsInRing())]

        features = atomic_num + degree + total_num_hs + is_aromatic + formal_charge + atom_type_enc + hybridization + num_radical_electrons + in_ring
        return torch.tensor(features, dtype=torch.float)

    def get_bond_features(self, bond):
        """Extract bond features for graph construction."""
        bond_type = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
        bond_feats = [int(bond.GetBondType() == bt) for bt in bond_type]
        bond_feats.extend([
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            int(bond.GetStereo() > 0),
        ])

        stereo_types = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS
        ]
        stereo_feats = [int(bond.GetStereo() == st) for st in stereo_types]
        bond_feats.extend(stereo_feats)
        return torch.tensor(bond_feats, dtype=torch.float)

    def smiles_to_graph(self, smiles):
        """
        Convert SMILES string to DGL graph with caching.
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            dgl.DGLGraph: Molecular graph
        """
        # Check cache first
        if smiles in self.smiles_graph_cache:
            return self.smiles_graph_cache[smiles]
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Get largest fragment
            fragments = Chem.GetMolFrags(mol, asMols=True)
            if len(fragments) > 1:
                mol = max(fragments, key=lambda m: m.GetNumAtoms())

            g = dgl.DGLGraph()
            g.add_nodes(mol.GetNumAtoms())

            node_features = [self.get_atom_features(atom) for atom in mol.GetAtoms()]
            g.ndata['feat'] = torch.stack(node_features)

            src_list = []
            dst_list = []
            edge_features = []
            for bond in mol.GetBonds():
                u = bond.GetBeginAtomIdx()
                v = bond.GetEndAtomIdx()
                src_list.extend([u, v])
                dst_list.extend([v, u])
                bond_feat = self.get_bond_features(bond).tolist()
                edge_features.extend([bond_feat, bond_feat])

            g.add_edges(src_list, dst_list)

            if edge_features:
                g.edata['feat'] = torch.tensor(edge_features, dtype=torch.float)
            else:
                num_bond_features = 13
                g.edata['feat'] = torch.zeros((0, num_bond_features), dtype=torch.float)
            
            # Cache the result
            self.smiles_graph_cache[smiles] = g
            return g

        except Exception as e:
            print(f"Error converting SMILES {smiles}: {str(e)}")
            return None
    
    def create_smiles_graph_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create graph representations for all unique SMILES in dataframe.
        
        Args:
            df (pd.DataFrame): DataFrame with SMILES column
            
        Returns:
            pd.DataFrame: DataFrame with added graph column
        """
        if self.input_col not in df.columns:
            raise ValueError(f"SMILES column '{self.input_col}' not found in dataframe")
        
        print("Converting SMILES to graphs...")
        
        # Get unique SMILES to avoid redundant computation
        unique_smiles = df[self.input_col].unique()
        print(f"Processing {len(unique_smiles)} unique SMILES...")
        
        # Convert unique SMILES to graphs
        for idx, smiles in enumerate(unique_smiles):
            if pd.notna(smiles):
                self.smiles_to_graph(smiles)
                
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(unique_smiles)} unique SMILES")
        
        # Map graphs to dataframe
        df = df.copy()
        df['graph'] = df[self.input_col].apply(lambda x: self.smiles_graph_cache.get(x, None) if pd.notna(x) else None)
        
        return df
    
    def prepare_dataset(self, csv_path, test_size=0.2, valid_size=0.1, random_state=5, 
                       stratify_col=None, output_dir="processed_data"):
        """
        Comprehensive dataset preparation including protein embeddings and graph conversion.
        
        Args:
            csv_path (str): Path to input CSV file
            test_size (float): Test set proportion
            valid_size (float): Validation set proportion (from training set)
            random_state (int): Random seed
            stratify_col (str): Column for stratification (optional)
            output_dir (str): Directory to save processed files
            
        Returns:
            tuple: (train_df, valid_df, test_df)
        """
        print(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path, sep='\t')
        print(f"Initial dataset size: {len(df)}")
        
        # Clean and validate data
        df = self._clean_and_validate_data(df)
        print(f"Dataset size after cleaning: {len(df)}")
        
        # Extract protein embeddings if sequence column exists
        if self.seq_col in df.columns:
            df = self.extract_sequence_embeddings(df)
        else:
            print(f"Warning: Sequence column '{self.seq_col}' not found. Skipping protein embeddings.")
        
        # Convert SMILES to graphs
        df = self.create_smiles_graph_mapping(df)
        
        # Remove rows with failed graph conversion
        df.dropna(subset=['graph'], inplace=True)
        print(f"Final dataset size: {len(df)}")
        
        # Prepare stratification
        stratify_data = None
        if stratify_col and stratify_col in df.columns:
            stratify_data = df[stratify_col]
            print(f"Using stratification based on column: {stratify_col}")
        
        # Split data
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, 
            stratify=stratify_data
        )
        
        # Further split training for validation
        stratify_train = train_df[stratify_col] if stratify_col and stratify_col in train_df.columns else None
        train_df, valid_df = train_test_split(
            train_df, test_size=valid_size / (1-test_size), 
            random_state=random_state, stratify=stratify_train
        )
        
        print(f"Train set size: {len(train_df)}")
        print(f"Validation set size: {len(valid_df)}")
        print(f"Test set size: {len(test_df)}")
        
        # Save processed datasets
        if output_dir:
            self._save_processed_datasets(train_df, valid_df, test_df, output_dir)
        
        return train_df, valid_df, test_df
    
    def _clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the input dataframe."""
        # Remove rows with missing essential data
        essential_cols = [self.input_col, self.label_col]
        df.dropna(subset=essential_cols, inplace=True)
        
        # Convert labels to numeric
        df[self.label_col] = pd.to_numeric(df[self.label_col], errors='coerce')
        df.dropna(subset=[self.label_col], inplace=True)
        
        # Validate SMILES
        valid_smiles = []
        for smiles in df[self.input_col]:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(smiles)
        
        invalid_count = len(df) - len(valid_smiles)
        print(f"Removed {invalid_count} invalid SMILES strings.")
        df = df[df[self.input_col].isin(valid_smiles)]
        
        # Clean SMILES (keep largest fragment)
        cleaned_smiles = []
        for smiles in df[self.input_col]:
            mol = Chem.MolFromSmiles(smiles) 
            if mol:
                fragments = Chem.GetMolFrags(mol, asMols=True)
                if len(fragments) > 1:
                    largest_fragment = max(fragments, key=lambda m: m.GetNumAtoms())
                    cleaned_smiles.append(Chem.MolToSmiles(largest_fragment))
                else:
                    cleaned_smiles.append(smiles)
            else:
                cleaned_smiles.append(None)
        
        df[self.input_col] = cleaned_smiles
        df.dropna(subset=[self.input_col], inplace=True)
        
        # Sort by label and clip outliers
        df.sort_values(self.label_col, ascending=False, inplace=True)
        lower_bound = df[self.label_col].quantile(0.01)
        upper_bound = df[self.label_col].quantile(0.99)
        df[self.label_col] = np.clip(df[self.label_col], lower_bound, upper_bound)
        
        return df
    
    def _save_processed_datasets(self, train_df, valid_df, test_df, output_dir):
        """Save processed datasets to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save dataframes
        for name, df in [('train', train_df), ('valid', valid_df), ('test', test_df)]:
            # Save CSV without graph column (not serializable)
            df.to_pickle(os.path.join(output_dir, f"{name}_processed.pkl"))
            df_save = df.drop(columns=['graph'], errors='ignore')
            df_save.to_csv(os.path.join(output_dir, f"{name}_processed.csv"), index=False)
            
            # Save protein embeddings if available
            if 'protein_embedding' in df.columns:
                embeddings = np.stack(df['protein_embedding'].values)
                np.save(os.path.join(output_dir, f"{name}_protein_embeddings.npy"), embeddings)
            
            # Save graphs as a dict
            df_unique = df.drop_duplicates(subset=[self.input_col])
            graphs = {df_unique.iloc[i][self.input_col]: df_unique.iloc[i]['graph'] for i in range(len(df_unique))}

            with open(os.path.join(output_dir, f"{name}_graphs.pkl"), 'wb') as f:
                pickle.dump(graphs, f)
            
            # Save labels
            labels = df[self.label_col].values
            np.save(os.path.join(output_dir, f"{name}_labels.npy"), labels)
        
        print(f"Saved processed datasets to {output_dir}")
    
    @staticmethod
    def collate(samples):
        """Collate function for DataLoader."""
        if len(samples[0]) == 2:  # (graph, label)
            graphs, labels = map(list, zip(*samples))
            batched_graph = dgl.batch(graphs)
            return batched_graph, torch.tensor(labels, dtype=torch.float32)
        elif len(samples[0]) == 3:  # (graph, protein_emb, label)
            graphs, protein_embs, labels = map(list, zip(*samples))
            batched_graph = dgl.batch(graphs)
            protein_embs = torch.stack([torch.tensor(emb) for emb in protein_embs])
            return batched_graph, protein_embs, torch.tensor(labels, dtype=torch.float32)
        else:
            raise ValueError("Unexpected sample format")
    
    def create_dataloaders(self, train_df, valid_df, test_df, batch_size=32, include_protein_emb=True):
        """
        Create DataLoaders for training, validation, and test sets.
        
        Args:
            train_df, valid_df, test_df: DataFrames with processed data
            batch_size (int): Batch size for DataLoaders
            include_protein_emb (bool): Whether to include protein embeddings
            
        Returns:
            tuple: (train_loader, valid_loader, test_loader)
        """
        def create_dataset(df):
            if include_protein_emb and 'protein_embedding' in df.columns:
                return list(zip(df['graph'], df['protein_embedding'], df[self.label_col]))
            else:
                return list(zip(df['graph'], df[self.label_col]))
        
        train_loader = DataLoader(
            create_dataset(train_df), 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=self.collate
        )
        
        valid_loader = DataLoader(
            create_dataset(valid_df), 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=self.collate
        )
        
        test_loader = DataLoader(
            create_dataset(test_df), 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=self.collate
        )
        
        return train_loader, valid_loader, test_loader


# Maintain backward compatibility with existing functions
def get_atom_features(atom):
    """Backward compatibility wrapper."""
    processor = AffinityDataProcessor()
    return processor.get_atom_features(atom)


def get_bond_features(bond):
    """Backward compatibility wrapper."""
    processor = AffinityDataProcessor()
    return processor.get_bond_features(bond)


def smiles_to_graph(smiles):
    """Backward compatibility wrapper."""
    processor = AffinityDataProcessor()
    return processor.smiles_to_graph(smiles)



# Example usage and backward compatibility
def prepare_dataset(csv_path, test_size=0.2, valid_size=0.1, random_state=5, stratification=False):
    """Backward compatibility wrapper for the old prepare_dataset function."""
    processor = AffinityDataProcessor()
    stratify_col = None if not stratification else 'cluster'  
    return processor.prepare_dataset(csv_path, test_size, valid_size, random_state, stratify_col)


def collate(samples):
    """Backward compatibility wrapper for collate function."""
    return AffinityDataProcessor.collate(samples)


# Main execution (for backward compatibility)
if __name__ == "__main__":
    # Example usage with new class
    processor = AffinityDataProcessor(
        embedding_cache_path="datasets/protein_chembl_embeddings.pkl", 
        label_col='pChEMBL Value'
    )
    
    csv_path = "datasets/chembl_uniprot_joined.tsv"
    train_df, valid_df, test_df = processor.prepare_dataset(
        csv_path, 
        output_dir="processed_data"
    )

    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(valid_df)}")
    print(f"Test set size: {len(test_df)}")

    # Create dataloaders
    train_loader, valid_loader, test_loader = processor.create_dataloaders(
        train_df, valid_df, test_df, batch_size=128
    )

    # Example usage
    for batch in train_loader:
        if len(batch) == 3:  # graph, protein_emb, labels
            batched_graph, protein_embs, labels = batch
            print("Batched graph:", batched_graph)
            print("Protein embeddings shape:", protein_embs.shape)
            print("Labels:", labels)
        else:  # graph, labels
            batched_graph, labels = batch
            print("Batched graph:", batched_graph)
            print("Labels:", labels)
        break
