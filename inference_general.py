import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import dgl
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from matplotlib import pyplot as plt
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from models.ginconvBiDirectionalNoAttPool import GINWithBidirectionalAttention


os.environ["DGLBACKEND"] = "pytorch"


# Column names as used in training
GRAPH_COL = "graph"
PROT_EMB_COL = "protein_embedding"
Y_COL = "pChEMBL Value"


class AffinityDataset(DGLDataset):
    """Dataset wrapper mirroring train_caAtt.py behavior."""

    def __init__(self, df: pd.DataFrame):
        super().__init__(name="affinity_dataset")
        self.df = df.reset_index(drop=True)
        self.graphs = self.df[GRAPH_COL]
        self.prot = self.df[PROT_EMB_COL]
        self.y = self.df[Y_COL]

    def __getitem__(self, idx):
        return self.graphs[idx], self.prot[idx], self.y[idx]

    def __len__(self) -> int:
        return len(self.df)


def _latest_best_weights(weights_dir: str = "weights") -> str:
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")
    candidates = [
        os.path.join(weights_dir, f)
        for f in os.listdir(weights_dir)
        if f.startswith("best_model_") and f.endswith(".pth")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No best_model_*.pth files found under {weights_dir}. Provide --weights."
        )
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


@torch.no_grad()
def _predict(
    model: nn.Module, loader: GraphDataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: List[float] = []
    trues: List[float] = []
    for batched_graph, protein_embs, labels in loader:
        batched_graph = batched_graph.to(device)
        protein_embs = protein_embs.float().to(device)
        labels = labels.float().to(device)
        outputs = model(batched_graph, protein_embs)
        preds.extend(outputs.detach().cpu().view(-1).numpy().tolist())
        trues.extend(labels.detach().cpu().view(-1).numpy().tolist())
    return np.array(trues, dtype=np.float32), np.array(preds, dtype=np.float32)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    # Pearson correlation via numpy (avoid SciPy dependency)
    if y_true.std() > 0 and y_pred.std() > 0:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = float("nan")
    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mae),
        "median_ae": float(medae),
        "r2": float(r2),
        "explained_variance": float(evs),
        "pearson_r": pearson_r,
    }


def _plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: str,
    metrics: Dict[str, float],
) -> None:
    plt.figure(figsize=(6, 6), dpi=150)
    sns.set_style("whitegrid")
    ax = sns.scatterplot(x=y_true, y=y_pred, s=14, alpha=0.6, edgecolor=None)
    min_v = float(np.nanmin([y_true.min(), y_pred.min()]))
    max_v = float(np.nanmax([y_true.max(), y_pred.max()]))
    ax.plot([min_v, max_v], [min_v, max_v], ls="--", c="red", lw=1)
    ax.set_xlabel("Actual pChEMBL")
    ax.set_ylabel("Predicted pChEMBL")
    ax.set_title(title)
    # Metrics box
    txt = (
        f"RMSE: {metrics['rmse']:.3f}\n"
        f"MAE: {metrics['mae']:.3f}\n"
        f"R2: {metrics['r2']:.3f}\n"
        f"Pearson r: {metrics['pearson_r']:.3f}"
    )
    ax.text(
        0.05,
        0.95,
        txt,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title_prefix: str,
    out_dir: str,
) -> None:
    residuals = y_true - y_pred

    # Residuals vs predicted
    plt.figure(figsize=(6, 4), dpi=150)
    sns.set_style("whitegrid")
    ax = sns.scatterplot(x=y_pred, y=residuals, s=12, alpha=0.6, edgecolor=None)
    ax.axhline(0.0, ls="--", c="red", lw=1)
    ax.set_xlabel("Predicted pChEMBL")
    ax.set_ylabel("Residual (Actual - Pred)")
    ax.set_title(f"{title_prefix}: Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{title_prefix.lower()}_residuals_vs_pred.png"))
    plt.close()

    # Residuals distribution
    plt.figure(figsize=(6, 4), dpi=150)
    sns.histplot(residuals, kde=True, bins=40, color="steelblue")
    plt.xlabel("Residual (Actual - Pred)")
    plt.title(f"{title_prefix}: Residuals Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{title_prefix.lower()}_residuals_hist.png"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference for train_caAtt model")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to best_model_*.pth. Defaults to latest in weights/",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for evaluation (default: 1024)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save reports/plots (default: reports/caAtt_inference_<date>)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets similar to train_caAtt.py
    valid_df = pd.read_pickle("processed_data/valid_processed.pkl")
    test_df = pd.read_pickle("processed_data/test_processed.pkl")

    valid_ds = AffinityDataset(valid_df)
    test_ds = AffinityDataset(test_df)
    valid_loader = GraphDataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Derive model dims from a sample graph and protein embedding
    sample = test_df.iloc[0]
    in_feat = sample[GRAPH_COL].ndata["feat"].shape[1]
    edge_feat_dim = sample[GRAPH_COL].edata["feat"].shape[1]
    prot_dim = int(sample[PROT_EMB_COL].shape[0])

    # Mirror train_caAtt.py hyperparameters
    h_features = 256
    emb_dim = 256
    dropout = 0.2
    num_classes = 1
    num_attention_layers = 2

    model = GINWithBidirectionalAttention(
        in_feat,
        edge_feat_dim,
        h_features,
        prot_dim,
        emb_dim=emb_dim,
        num_classes=num_classes,
        dropout_rate=dropout,
        num_attention_layers=num_attention_layers,
    ).to(device)

    weights_path = args.weights or _latest_best_weights("weights")
    print(f"Loading weights: {weights_path}")
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)

    # Evaluate
    y_valid, yhat_valid = _predict(model, valid_loader, device)
    y_test, yhat_test = _predict(model, test_loader, device)

    metrics_valid = _metrics(y_valid, yhat_valid)
    metrics_test = _metrics(y_test, yhat_test)

    # Output directory
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join("reports", f"caAtt_inference_{date}")
    os.makedirs(out_dir, exist_ok=True)

    # Save metrics JSON
    metrics = {"validation": metrics_valid, "test": metrics_test}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions CSV
    pd.DataFrame(
        {
            "y_true": y_valid,
            "y_pred": yhat_valid,
            "residual": y_valid - yhat_valid,
        }
    ).to_csv(os.path.join(out_dir, "valid_predictions.csv"), index=False)

    pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": yhat_test,
            "residual": y_test - yhat_test,
        }
    ).to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    # Plots: Pred vs Actual
    _plot_pred_vs_actual(
        y_valid,
        yhat_valid,
        title="Validation: Predicted vs Actual",
        out_path=os.path.join(out_dir, "valid_pred_vs_actual.png"),
        metrics=metrics_valid,
    )
    _plot_pred_vs_actual(
        y_test,
        yhat_test,
        title="Test: Predicted vs Actual",
        out_path=os.path.join(out_dir, "test_pred_vs_actual.png"),
        metrics=metrics_test,
    )

    # Plots: Residuals
    _plot_residuals(y_valid, yhat_valid, title_prefix="Validation", out_dir=out_dir)
    _plot_residuals(y_test, yhat_test, title_prefix="Test", out_dir=out_dir)

    # Console summary
    print("Validation metrics:", json.dumps(metrics_valid, indent=2))
    print("Test metrics:", json.dumps(metrics_test, indent=2))
    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
