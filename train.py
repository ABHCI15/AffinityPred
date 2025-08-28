import pickle
import time
from matplotlib.pyplot import step
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import AdamW
import dgl
from dgl.nn.pytorch.conv import GINEConv
from clearml import Logger, Task
from models.ginconvBiDirectional import GINWithBidirectionalAttention
import pandas as pd
import os
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import torch
import torch.nn as nn
from datetime import datetime
import numpy as np

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
model = GINWithBidirectionalAttention(in_feat, edge_feat_dim, h_features, prot_dim, emb_dim=256, num_classes=1, dropout_rate=dropout, num_attention_layers=2)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20, verbose=True, min_lr=1e-6, cooldown=10)
loss_fn = nn.MSELoss()

# Store metrics
train_losses = []
valid_mses = []
valid_rmses = []
valid_r2s = []
best_val_loss = float('inf')
best_model_state = None
patience = 500
early_stopping_counter = 0

for epoch in range(num_epochs):
    start_time = time.time()
    print(datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"))
    model.train()
    total_loss = 0 
    for batch_idx, (batched_graph, protein_embs, labels) in enumerate(train_loader):
        batched_graph = batched_graph.to(device)
        protein_embs = protein_embs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['feat'], protein_embs)
        loss = loss_fn(predictions, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    train_losses.append(avg_loss)

    model.eval()
    with torch.no_grad():
        val_predictions = []
        val_labels = []
        for batched_graph, protein_embs, labels in valid_loader:
            batched_graph = batched_graph.to(device)
            protein_embs = protein_embs.to(device)
            labels = labels.to(device)
            predictions = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['feat'], protein_embs)
            val_predictions.extend(predictions.cpu().tolist())
            val_labels.extend(labels.cpu().tolist())

        val_mse = mean_squared_error(val_labels, val_predictions)
        val_rmse = np.sqrt(val_mse)
        val_r2 = r2_score(val_labels, val_predictions)
        valid_mses.append(val_mse)
        valid_rmses.append(val_rmse)
        valid_r2s.append(val_r2)
        Logger.current_logger().report_scalar("MSE Loss", "Validation", iteration=epoch, value=val_mse)
        Logger.current_logger().report_scalar("RMSE", "Validation", iteration=epoch, value=val_rmse)
        Logger.current_logger().report_scalar("R-squared", "Validation", iteration=epoch, value=val_r2)
        Logger.current_logger().report_scalar("MSE Loss", "Train", iteration=epoch, value=avg_loss)
        Logger.current_logger().report_scalar("RMSE", "Train", iteration=epoch, value=np.sqrt(avg_loss))
        Logger.current_logger().report_scalar("Learning Rate", "LR", iteration=epoch, value=scheduler.get_last_lr()[0])

    scheduler.step(val_rmse)
    
    end_time = time.time()
    epoch_duration = end_time - start_time
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val MSE: {val_mse:.4f}, Val RMSE: {val_rmse:.4f}, Val R-squared: {val_r2:.4f}, Time: {epoch_duration:.2f}s, lr: {scheduler.get_last_lr()[0]:.4f}")

    if val_rmse < best_val_loss:
        best_val_loss = val_rmse
        best_model_state = model.state_dict()
        best_epoch = epoch
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

if best_model_state:
    model.load_state_dict(best_model_state)

# Evaluation on Test Set
model.eval()
test_predictions = []
test_labels = []
with torch.no_grad():
    for batched_graph, protein_embs, labels in test_loader:
        batched_graph = batched_graph.to(device)
        protein_embs = protein_embs.to(device)
        labels = labels.to(device)
        predictions = model(batched_graph, batched_graph.ndata['feat'], batched_graph.edata['feat'], protein_embs)
        test_predictions.extend(predictions.cpu().tolist())
        test_labels.extend(labels.cpu().tolist())

test_mse = mean_squared_error(test_labels, test_predictions)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(test_labels, test_predictions)
print(f"Test MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R-squared: {test_r2:.4f}, lr: {scheduler.get_last_lr()[0]:.4f}")
Logger.current_logger().report_scalar("Test MSE Loss", "Test", iteration=0, value=test_mse)
Logger.current_logger().report_scalar("Test RMSE", "Test", iteration=0, value=test_rmse)
Logger.current_logger().report_scalar("Test R-squared", "Test", iteration=0, value=test_r2)

date = datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(model.state_dict(), f"weights/last_model_{date}.pth")
torch.save(best_model_state, f"weights/best_model_{date}.pth")