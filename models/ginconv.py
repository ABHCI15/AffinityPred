
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw  
import dgl
import dgl.function as fn
from dgl.nn import GraphConv, GATConv, SAGEConv  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


class GINWithCrossAttention(nn.Module):
    def __init__(self, in_feats, h_feats, prot_dim, edge_feat_dim, emb_dim=512, num_classes=1, dropout_rate=0.10):
        super(GINWithCrossAttention, self).__init__()
        self.edge_proj1 = nn.Linear(edge_feat_dim, in_feats)
        self.edge_proj2 = nn.Linear(edge_feat_dim, h_feats)
        self.edge_proj3 = nn.Linear(edge_feat_dim, h_feats)
        self.edge_proj4 = nn.Linear(edge_feat_dim, h_feats)
        # GIN layers 
        self.mlp1 = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.BatchNorm1d(h_feats),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_feats, h_feats),
            nn.BatchNorm1d(h_feats),
        )
        self.conv1 = GINEConv(self.mlp1, learn_eps=True)
        self.bn1 = nn.BatchNorm1d(h_feats)

        self.mlp2 = nn.Sequential(
            nn.Linear(h_feats, h_feats),
            nn.BatchNorm1d(h_feats),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_feats, h_feats),
            nn.BatchNorm1d(h_feats),
        )
        self.conv2 = GINEConv(self.mlp2, learn_eps=True)
        self.bn2 = nn.BatchNorm1d(h_feats)

        self.mlp3 = nn.Sequential(
            nn.Linear(h_feats, h_feats),
            nn.BatchNorm1d(h_feats),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_feats, h_feats),
            nn.BatchNorm1d(h_feats),
        )
        self.conv3 = GINEConv(self.mlp3, learn_eps=True)
        self.bn3 = nn.BatchNorm1d(h_feats)

        self.mlp4 = nn.Sequential(
            nn.Linear(h_feats, h_feats),
            nn.BatchNorm1d(h_feats),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_feats, h_feats),
            nn.BatchNorm1d(h_feats),
        )
        self.conv4 = GINEConv(self.mlp4,learn_eps=True)
        self.bn4 = nn.BatchNorm1d(h_feats)

        # Graph pooling to get molecule embedding
        self.mol_proj = nn.Linear(h_feats * 3, emb_dim)
        
        # Protein projection
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_dim, emb_dim),
            nn.GELU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(emb_dim, emb_dim)
        )
        
        # Cross-attention mechanism
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=16,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, g, in_feat, edge_feat, x_prot):
        # GIN layers for molecule processing
        edge_feat1 = self.edge_proj1(edge_feat)
        

        h = self.conv1(g, in_feat, edge_feat1)
        h = self.bn1(h)
        h = F.gelu(h)
        h_res1 = h
        h = self.dropout(h)

        edge_feat2 = self.edge_proj2(edge_feat)
        h = self.conv2(g, h, edge_feat2)
        h = self.bn2(h)
        h = F.gelu(h)
        h = h + h_res1
        h_res2 = h
        h = self.dropout(h)
        
        edge_feat3 = self.edge_proj3(edge_feat)
        h = self.conv3(g, h, edge_feat3)
        h = self.bn3(h)
        h = F.gelu(h)
        h = h + h_res2
        h_res3 = h
        h = self.dropout(h)
        
        edge_feat4 = self.edge_proj4(edge_feat)
        h = self.conv4(g, h, edge_feat4)
        h = self.bn4(h)
        h = F.gelu(h)
        h = h + h_res3
        h = self.dropout(h)

        # Global pooling to get molecule representation
        g.ndata['h'] = h
        hg_mean = dgl.mean_nodes(g, 'h')
        hg_max = dgl.max_nodes(g, 'h')
        hg_sum = dgl.sum_nodes(g, 'h')
        mol_graph_emb = torch.cat([hg_mean, hg_max, hg_sum], dim=1)
        
        # Project molecule embedding to common space
        mol_emb = self.mol_proj(mol_graph_emb)
        
        # Project protein to common space
        prot_emb = self.prot_proj(x_prot)
        
        # Add sequence dimension for attention
        prot_seq = prot_emb.unsqueeze(1)
        mol_seq = mol_emb.unsqueeze(1)
        
        # Cross-attention: protein queries molecule
        prot_attended, _ = self.cross_attn(prot_seq, mol_seq, mol_seq)
        prot_attended = prot_attended.squeeze(1)
        
        # Combine attended protein with molecule embedding
        combined = torch.cat([prot_attended, mol_emb], dim=1)
        
        # Final prediction
        output = self.predictor(combined)
        return output.squeeze()