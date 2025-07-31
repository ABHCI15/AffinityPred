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

def get_atom_features(atom):
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


def get_bond_features(bond):
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


def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles) # type: ignore
        if mol is None:
            return None

        # Get largest fragment
        fragments = Chem.GetMolFrags(mol, asMols=True) # type: ignore
        if len(fragments) > 1:
            mol = max(fragments, key=lambda m: m.GetNumAtoms())

        g = dgl.DGLGraph()
        g.add_nodes(mol.GetNumAtoms())

        node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        g.ndata['feat'] = torch.stack(node_features)

        src_list = []
        dst_list = []
        edge_features = []
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            src_list.extend([u, v])
            dst_list.extend([v, u])
            bond_feat = get_bond_features(bond).tolist()
            edge_features.extend([bond_feat, bond_feat])

        g.add_edges(src_list, dst_list)

        if edge_features:
            g.edata['feat'] = torch.tensor(edge_features, dtype=torch.float)
        else:
            num_bond_features = 10
            g.edata['feat'] = torch.zeros((0, num_bond_features), dtype=torch.float)
        return g

    except Exception as e:
        print(f"Error converting SMILES {smiles}: {str(e)}")
        return None


def prepare_dataset(csv_path, test_size=0.2, valid_size=0.1, random_state=42):
    df = pd.read_csv(csv_path)

    df.dropna(subset=['SMILES', 'Ratio'], inplace=True)
    df['Ratio'] = pd.to_numeric(df['Ratio'], errors='coerce')
    df.dropna(subset=['Ratio'], inplace=True)  
    valid_smiles = []
    for smiles in df['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
    invalid_smiles_count = len(df) - len(valid_smiles)
    print(f"Removed {invalid_smiles_count} invalid SMILES strings.")
    df = df[df['SMILES'].isin(valid_smiles)]
    cleaned_smiles = []
    for smiles in df['SMILES']:
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
    df['SMILES'] = cleaned_smiles
    df.dropna(subset=['SMILES'], inplace=True)

    df.sort_values('Ratio', ascending=False, inplace=True)
    # df.drop_duplicates(subset=['SMILES'], keep='first', inplace=True)

    lower_bound = df['Ratio'].quantile(0.01)
    upper_bound = df['Ratio'].quantile(0.99)
    df['Ratio'] = np.clip(df['Ratio'], lower_bound, upper_bound)



    df['graph'] = df['SMILES'].apply(smiles_to_graph)
    df.dropna(subset=['graph'], inplace=True)


    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, valid_df = train_test_split(train_df, test_size=valid_size / (1-test_size), random_state=random_state)

    return train_df, valid_df, test_df



def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels, dtype=torch.float32)



csv_path = "datasets/train.csv"
train_df, valid_df, test_df = prepare_dataset(csv_path)

print(f"Train set size: {len(train_df)}")
print(f"Validation set size: {len(valid_df)}")
print(f"Test set size: {len(test_df)}")



if not train_df.empty:
    print("Example Graph Features:")
    print("Node features:", train_df.iloc[0]['graph'].ndata['feat'])
    print("Edge features:", train_df.iloc[0]['graph'].edata['feat'])




train_loader = DataLoader(list(zip(train_df['graph'], train_df['Ratio'])), batch_size=32, shuffle=True, collate_fn=collate)
valid_loader = DataLoader(list(zip(valid_df['graph'], valid_df['Ratio'])), batch_size=32, shuffle=False, collate_fn=collate)
test_loader = DataLoader(list(zip(test_df['graph'], test_df['Ratio'])), batch_size=32, shuffle=False, collate_fn=collate)




for batched_graph, labels in train_loader:
   print("Batched graph:", batched_graph)
   print("Labels:", labels)
   break