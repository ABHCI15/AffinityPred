
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw  
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import GINEConv
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for enhanced attention mechanisms."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class BiDirectionalCrossAttention(nn.Module):
    """Bidirectional cross-attention mechanism for protein-molecule interaction."""
    def __init__(self, embed_dim, num_heads=16, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Protein to Molecule attention
        self.prot_to_mol_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Molecule to Protein attention
        self.mol_to_prot_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Self-attention for protein and molecule
        self.prot_self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.mol_self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.prot_norm1 = nn.LayerNorm(embed_dim)
        self.prot_norm2 = nn.LayerNorm(embed_dim)
        self.mol_norm1 = nn.LayerNorm(embed_dim)
        self.mol_norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward networks
        self.prot_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.mol_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, prot_emb, mol_emb):
        """
        Args:
            prot_emb: [batch_size, seq_len, embed_dim] or [batch_size, embed_dim]
            mol_emb: [batch_size, seq_len, embed_dim] or [batch_size, embed_dim]
        """
        # Ensure inputs have sequence dimension
        if prot_emb.dim() == 2:
            prot_emb = prot_emb.unsqueeze(1)
        if mol_emb.dim() == 2:
            mol_emb = mol_emb.unsqueeze(1)
        
        # Self-attention for both modalities
        prot_self, _ = self.prot_self_attn(prot_emb, prot_emb, prot_emb)
        prot_emb = self.prot_norm1(prot_emb + self.dropout(prot_self))
        
        mol_self, _ = self.mol_self_attn(mol_emb, mol_emb, mol_emb)
        mol_emb = self.mol_norm1(mol_emb + self.dropout(mol_self))
        
        # Cross-attention: protein queries molecule
        prot_attended, prot_attn_weights = self.prot_to_mol_attn(
            prot_emb, mol_emb, mol_emb
        )
        prot_emb = self.prot_norm2(prot_emb + self.dropout(prot_attended))
        
        # Cross-attention: molecule queries protein
        mol_attended, mol_attn_weights = self.mol_to_prot_attn(
            mol_emb, prot_emb, prot_emb
        )
        mol_emb = self.mol_norm2(mol_emb + self.dropout(mol_attended))
        
        # Feed-forward networks
        prot_emb = prot_emb + self.prot_ffn(prot_emb)
        mol_emb = mol_emb + self.mol_ffn(mol_emb)
        
        return prot_emb.squeeze(1), mol_emb.squeeze(1), prot_attn_weights, mol_attn_weights


class AdaptivePooling(nn.Module):
    """Adaptive pooling mechanism for graph representations."""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.attention_pool = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=0)
        )
        
    def forward(self, g, node_feat):
        # Attention-based pooling
        g.ndata['h'] = node_feat
        attention_weights = self.attention_pool(node_feat)
        g.ndata['a'] = attention_weights
        
        # Weighted sum pooling
        weighted_sum = dgl.sum_nodes(g, 'h', weight='a')
        
        # Traditional pooling methods
        mean_pool = dgl.mean_nodes(g, 'h')
        max_pool = dgl.max_nodes(g, 'h')
        sum_pool = dgl.sum_nodes(g, 'h')
        
        # Combine all pooling methods
        return torch.cat([weighted_sum, mean_pool, max_pool, sum_pool], dim=1)


class GINWithBidirectionalAttention(nn.Module):
    """
    Enhanced GIN model with bidirectional cross-attention and SOTA techniques.
    Designed for ESM-C protein embeddings and DGL molecular graphs.
    
    Features:
    - Residual connections in GIN layers
    - Bidirectional cross-attention between protein and molecule
    - Adaptive pooling for graph representations
    - Multi-scale feature fusion
    - Advanced regularization techniques
    """
    def __init__(self, node_feat_dim, edge_feat_dim, h_feats=256, prot_dim=1152, 
                 emb_dim=256, num_classes=1, dropout_rate=0.15, num_attention_layers=2):
        super(GINWithBidirectionalAttention, self).__init__()
        
        self.num_attention_layers = num_attention_layers
        self.h_feats = h_feats
        
        # Input projections for node and edge features
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, h_feats),
            nn.BatchNorm1d(h_feats),
            nn.GELU()
        )
        
        # Edge feature projections for each GIN layer
        self.edge_proj1 = nn.Linear(edge_feat_dim, h_feats)
        self.edge_proj2 = nn.Linear(edge_feat_dim, h_feats)
        self.edge_proj3 = nn.Linear(edge_feat_dim, h_feats)
        self.edge_proj4 = nn.Linear(edge_feat_dim, h_feats)
        self.edge_proj5 = nn.Linear(edge_feat_dim, h_feats)
        
        # Enhanced GIN layers with deeper architecture
        self.mlp1 = nn.Sequential(
            nn.Linear(h_feats, h_feats),
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
        self.conv4 = GINEConv(self.mlp4, learn_eps=True)
        self.bn4 = nn.BatchNorm1d(h_feats)
        
        # Additional GIN layer for deeper representation
        self.mlp5 = nn.Sequential(
            nn.Linear(h_feats, h_feats),
            nn.BatchNorm1d(h_feats),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_feats, h_feats),
            nn.BatchNorm1d(h_feats),
        )
        self.conv5 = GINEConv(self.mlp5, learn_eps=True)
        self.bn5 = nn.BatchNorm1d(h_feats)

        # Adaptive pooling mechanism
        self.adaptive_pool = AdaptivePooling(h_feats, hidden_dim=h_feats//2)
        
        # Molecule projection with skip connections
        self.mol_proj = nn.Sequential(
            nn.Linear(h_feats * 4, emb_dim * 2),  # 4 pooling methods
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        
        # Enhanced protein projection
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim)
        )
        
        # Bidirectional cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            BiDirectionalCrossAttention(emb_dim, num_heads=16, dropout=dropout_rate)
            for _ in range(num_attention_layers)
        ])
        
        # Multi-scale fusion
        self.fusion_gate = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Sigmoid()
        )
        
        # Enhanced prediction layers with residual connections
        self.predictor = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim // 2, emb_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(emb_dim // 4, num_classes)
        )
        
        # Additional components for regularization
        self.dropout = nn.Dropout(dropout_rate)
        self.feature_dropout = nn.Dropout(dropout_rate * 0.5)
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, g, x_prot):
        """
        Forward pass with enhanced architecture.
        
        Args:
            g: DGL graph with node features g.ndata['feat'] and edge features g.edata['feat']
            x_prot: Protein ESM-C embeddings [batch_size, 1152]
        """
        # Extract node and edge features from graph
        node_feat = g.ndata['feat']  # [num_nodes, node_feat_dim]
        edge_feat = g.edata['feat']  # [num_edges, edge_feat_dim]
        
        # Project node features to hidden dimension
        h = self.node_proj(node_feat)  # [num_nodes, h_feats]
        
        # Enhanced GIN layers with residual connections
        edge_feat1 = self.edge_proj1(edge_feat)
        h = self.conv1(g, h, edge_feat1)
        h = self.bn1(h)
        h = F.gelu(h)
        h_res1 = h
        h = self.dropout(h)

        edge_feat2 = self.edge_proj2(edge_feat)
        h = self.conv2(g, h, edge_feat2)
        h = self.bn2(h)
        h = F.gelu(h)
        h = h + h_res1  # Residual connection
        h_res2 = h
        h = self.dropout(h)
        
        edge_feat3 = self.edge_proj3(edge_feat)
        h = self.conv3(g, h, edge_feat3)
        h = self.bn3(h)
        h = F.gelu(h)
        h = h + h_res2  # Residual connection
        h_res3 = h
        h = self.dropout(h)
        
        edge_feat4 = self.edge_proj4(edge_feat)
        h = self.conv4(g, h, edge_feat4)
        h = self.bn4(h)
        h = F.gelu(h)
        h = h + h_res3  # Residual connection
        h_res4 = h
        h = self.dropout(h)
        
        # Additional layer for deeper representation
        edge_feat5 = self.edge_proj5(edge_feat)
        h = self.conv5(g, h, edge_feat5)
        h = self.bn5(h)
        h = F.gelu(h)
        h = h + h_res4  # Residual connection
        h = self.dropout(h)

        # Adaptive pooling for molecule representation
        mol_graph_emb = self.adaptive_pool(g, h)
        
        # Project to common embedding space
        mol_emb = self.mol_proj(mol_graph_emb)
        prot_emb = self.prot_proj(x_prot)
        
        # Apply feature dropout for regularization
        mol_emb = self.feature_dropout(mol_emb)
        prot_emb = self.feature_dropout(prot_emb)
        
        # Store original embeddings for fusion
        mol_emb_orig = mol_emb
        prot_emb_orig = prot_emb
        
        # Apply multiple bidirectional cross-attention layers
        attention_weights_list = []
        for cross_attn_layer in self.cross_attention_layers:
            prot_emb, mol_emb, prot_attn_weights, mol_attn_weights = cross_attn_layer(
                prot_emb, mol_emb
            )
            attention_weights_list.append((prot_attn_weights, mol_attn_weights))
        
        # Gated fusion of original and attended embeddings
        gate = self.fusion_gate(torch.cat([prot_emb_orig, mol_emb_orig], dim=1))
        prot_fused = gate * prot_emb + (1 - gate) * prot_emb_orig
        mol_fused = gate * mol_emb + (1 - gate) * mol_emb_orig
        
        # Combine for final prediction
        combined = torch.cat([prot_fused, mol_fused], dim=1)
        
        # Final prediction
        output = self.predictor(combined)
        
        return output.squeeze()

    def get_attention_weights(self, g, x_prot):
        """
        Get attention weights for interpretability.
        
        Args:
            g: DGL graph with node and edge features
            x_prot: Protein ESM-C embeddings
        """
        with torch.no_grad():
            # Extract features
            node_feat = g.ndata['feat']
            edge_feat = g.edata['feat']
            
            # Forward pass up to attention
            h = self.node_proj(node_feat)
            
            edge_feat1 = self.edge_proj1(edge_feat)
            h = self.conv1(g, h, edge_feat1)
            h = self.bn1(h)
            h = F.gelu(h)
            h_res1 = h

            edge_feat2 = self.edge_proj2(edge_feat)
            h = self.conv2(g, h, edge_feat2)
            h = self.bn2(h)
            h = F.gelu(h)
            h = h + h_res1
            h_res2 = h
            
            edge_feat3 = self.edge_proj3(edge_feat)
            h = self.conv3(g, h, edge_feat3)
            h = self.bn3(h)
            h = F.gelu(h)
            h = h + h_res2
            h_res3 = h
            
            edge_feat4 = self.edge_proj4(edge_feat)
            h = self.conv4(g, h, edge_feat4)
            h = self.bn4(h)
            h = F.gelu(h)
            h = h + h_res3
            h_res4 = h
            
            edge_feat5 = self.edge_proj5(edge_feat)
            h = self.conv5(g, h, edge_feat5)
            h = self.bn5(h)
            h = F.gelu(h)
            h = h + h_res4

            mol_graph_emb = self.adaptive_pool(g, h)
            mol_emb = self.mol_proj(mol_graph_emb)
            prot_emb = self.prot_proj(x_prot)
            
            # Get attention weights from all layers
            all_attention_weights = []
            for cross_attn_layer in self.cross_attention_layers:
                prot_emb, mol_emb, prot_attn_weights, mol_attn_weights = cross_attn_layer(
                    prot_emb, mol_emb
                )
                all_attention_weights.append({
                    'protein_to_molecule': prot_attn_weights,
                    'molecule_to_protein': mol_attn_weights
                })
            
            return all_attention_weights

