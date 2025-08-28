import pickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
import torch.optim as optim
import dgl
from dgl.nn.pytorch.conv import GINEConv
from clearml import Task
from models.ginconvBiDirectional import GINWithBidirectionalAttention
import pandas as pd
import os
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import torch

os.environ["DGLBACKEND"] = "pytorch"

p_embed = 'protein_embedding'
graph_col = 'graph'
y_col = 'pChEMBL Value'

batch_size = 1024

class AffinityDataset(DGLDataset):
    def __init__(self, df):
        super().__init__(name='affinity_dataset')
        self.df = df
        self.graphs = self.df[graph_col]
        self.prot = self.df[p_embed]
        self.y = self.df[y_col]

    # def process(self):
    #     self.graphs = self.df[graph_col]
    #     self.prot = self.df[p_embed]
    #     self.y = self.df[y_col]
        
    def __getitem__(self, idx):
        return self.graphs[idx], self.prot[idx], self.y[idx]
    def __len__(self):
        return len(self.df)
        

df_train = AffinityDataset(pd.read_pickle("processed_data/train_processed.pkl"))
train_loader = GraphDataLoader(df_train, batch_size=batch_size, shuffle=True, num_workers=5)
# del df_train
df_valid = AffinityDataset(pd.read_pickle("processed_data/valid_processed.pkl"))
valid_loader = GraphDataLoader(df_valid, batch_size=batch_size, shuffle=False, num_workers=5)
# del df_valid
df_test = AffinityDataset(pd.read_pickle("processed_data/test_processed.pkl"))
test_loader = GraphDataLoader(df_test, batch_size=batch_size, shuffle=False, num_workers=5)
# del df_test





task = Task.init(project_name="AffinityPrediction", task_name="GNN Training", task_type=Task.TaskTypes.optimizer)


# train step 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
test_df = pd.read_pickle("processed_data/test_processed.pkl")
in_feat = test_df[graph_col].iloc[0].ndata['feat'].shape[1]
prot_dim = test_df[p_embed].iloc[0].shape[0]
num_classes = 1
dropout = 0.2
lr = 1e-3
weight_decay = 1e-5
num_epochs = 750
edge_feat_dim = test_df[graph_col].iloc[0].edata['feat'].shape[1]
h_features = 256
model = GINWithBidirectionalAttention()