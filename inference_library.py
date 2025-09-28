from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from IPython import embed
import dgl
import numpy as np
from sympy import Li
import torch
import pandas as pd
from rdkit import Chem

from data_processing_utils.prep_data_cysdb import AffinityDataProcessor
from models.ginconvBiDirectionalNoAttPool import GINWithBidirectionalAttention

os.environ["DGLBACKEND"] = "pytorch"
@dataclass
class InferenceConfig:
    """Configuration bundle for single-pair inference."""

    weights_path: Optional[Path] = None
    device: Optional[str] = 'cuda'
    hidden_features: int = 256
    protein_emb_dim: int = 256
    dropout_rate: float = 0.2
    attention_layers: int = 2
    embedding_cache_path: Optional[Path] = Path("datasets/protein_chembl_embeddings.pkl")

class LibraryInferencer:
    """Score protein–compound pairs with a trained caAtt model."""

    def __init__(self, config: Optional[InferenceConfig] = None) -> None:
        self.config = config or InferenceConfig()
        device_str = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = torch.device(device_str)

        processor_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.processor = AffinityDataProcessor(
            input_col="SMILES",
            label_col="Activity",
            seq_col="Sequence",
            device=processor_device,
        )

        self.model: Optional[GINWithBidirectionalAttention] = None
        self.model_ready = False
        self._weights_path: Optional[Path] = self.config.weights_path

    def _latest_best_weights(self) -> Path:
        weights_dir = Path("weights")
        if not weights_dir.is_dir():
            raise FileNotFoundError(
                "Weights directory not found. Provide weights_path in InferenceConfig."
            )
        candidates = sorted(
            weights_dir.glob("best_model_*.pth"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                "No best_model_*.pth files located. Please train the model or "
                "specify a weights_path."
            )
        return candidates[0]

    def _ensure_model(
        self,
        graph: dgl.DGLGraph,
        protein_embedding: torch.Tensor,
    ) -> None:
        """Instantiate and load the model if needed."""
        if self.model_ready:
            return

        node_feat_dim = graph.ndata["feat"].shape[1]
        edge_feat = graph.edata.get("feat")
        edge_feat_dim = edge_feat.shape[1] if edge_feat is not None else 13
        prot_dim = protein_embedding.shape[0]

        weights_path = self._weights_path or self._latest_best_weights()
        state_dict = torch.load(weights_path, map_location=self.device)

        model = GINWithBidirectionalAttention(
            node_feat_dim,
            edge_feat_dim,
            h_feats=self.config.hidden_features,
            prot_dim=prot_dim,
            emb_dim=self.config.protein_emb_dim,
            num_classes=1,
            dropout_rate=self.config.dropout_rate,
            num_attention_layers=self.config.attention_layers,
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.model = model
        self.model_ready = True
        self._weights_path = weights_path

    def predict(
        self,
        protein_sequence: str,
        smiles: str,
        protein_embedding: Optional[np.ndarray] = None,
    ) -> float:
        """Return predicted pChEMBL for a protein sequence and SMILES string."""
        if not protein_sequence:
            raise ValueError("protein_sequence must be a non-empty string")
        if not smiles:
            raise ValueError("smiles must be a non-empty string")

        if protein_embedding is None:
            embedding_np = self.processor.embed_sequence(protein_sequence)
        else:
            embedding_np = protein_embedding
        protein_tensor = torch.from_numpy(np.asarray(embedding_np)).float()

        graph = self.processor.smiles_to_graph(smiles)
        if graph is None:
            raise ValueError(f"Unable to convert SMILES to graph: {smiles}")

        # Clone to avoid mutating cached graphs when moving to device
        graph = graph.clone()
        self._ensure_model(graph, protein_tensor)
        assert self.model is not None

        batched_graph = dgl.batch([graph]).to(self.device)
        protein_tensor = protein_tensor.to(self.device).unsqueeze(0)

        with torch.inference_mode():
            prediction = self.model(batched_graph, protein_tensor)
        return float(prediction.squeeze().detach().cpu().item())
    
    def predict_lib(self, smiles, proteins):
        chem_list = []
        prot_list = []
        pred_list = []
        for i in proteins:
            protein_sequence = i
            for j in smiles:
                pred = self.predict(protein_sequence, j)
                chem_list.append(j)
                prot_list.append(i)
                pred_list.append(pred)
        out_df = pd.DataFrame({'SMILES':chem_list, 'Sequence':prot_list, 'Predicted pChEMBL':pred_list})
        return out_df

if __name__ == "__main__":
    df_cys = pd.read_csv("/home/nroethler/Code/abhiram/chemmap/AffinityPred/datasets/cysdb_complete_with_sequences.csv")
    unique_SMILES = df_cys["SMILES"].unique()
    
    # Filter out invalid SMILES
    from rdkit import Chem
    valid_SMILES = [s for s in unique_SMILES if Chem.MolFromSmiles(str(s)) is not None]
    print(f"Filtered to {len(valid_SMILES)} valid SMILES out of {len(unique_SMILES)}")
    
    prot_list = ["MAVRELPGAWNFRDVADTATALRPGRLFRSSELSRLDDAGRATLRRLGITDVADLRSSREVARRGPGRVPDGIDVHLLPFPDLADDDADDSAPHETAFKRLLTNDGSNGESGESSQSINDAATRYMTDEYRQFPTRNGAQRALHRVVTLLAAGRPVLTHCFAGKDRTGFVVALVLEAVGLDRDVIVADYLRSNDSVPQLRARISEMIQQRFDTELAPEVVTFTKARLSDGVLGVRAEYLAAARQTIDETYGSLGGYLRDAGISQATVNRMRGVLLG","MTPSQWLDIAVLAVAFIAAISGWRAGALGSMLSFGGVLLGATAGVLLAPHIVSQISAPRAKLFAALFLILALVVVGEVAGVVLGRAVRGAIRNRPIRLIDSVIGVGVQLVVVLTAAWLLAMPLTQSKEQPELAAAVKGSRVLARVNEAAPTWLKTVPKRLSALLNTSGLPAVLEPFSRTPVIPVASPDPALVNNPVVAATEPSVVKIRSLAPRCQKVLEGTGFVISPDRVMTNAHVVAGSNNVTVYAGDKPFEATVVSYDPSVDVAILAVPHLPPPPLVFAAEPAKTGADVVVLGYPGGGNFTATPARIREAIRLSGPDIYGDPEPVTRDVYTIRADVEQGDSGGPLIDLNGQVLGVVFGAAIDDAETGFVLTAGEVAGQLAKIGATQPVGTGACVS", "MLRGIQALSRPLTRVYRALAVIGVLAASLLASWVGAVPQVGLAASALPTFAHVVIVVEENRSQAAIIGNKSAPFINSLAANGAMMAQAFAETHPSEPNYLALFAGNTFGLTKNTCPVNGGALPNLGSELLSAGYTFMGFAEDLPAVGSTVCSAGKYARKHVPWVNFSNVPTTLSVPFSAFPKPQNYPGLPTVSFVIPNADNDMHDGSIAQGDAWLNRHLSAYANWAKTNNSLLVVTWDEDDGSSRNQIPTVFYGAHVRPGTYNETISHYNVLSTLEQIYGLPKTGYATNAPPITDIWGD"]
    inference_engine = LibraryInferencer(InferenceConfig(weights_path=Path("/home/nroethler/Code/abhiram/chemmap/AffinityPred/weights/best_model_20250909_113311.pth")))
    inference_engine.predict_lib(valid_SMILES, prot_list).to_csv("cysdb_predictions.csv", index=False)
    mol_glues = pd.read_csv("/home/nroethler/Code/abhiram/chemmap/AffinityPred/datasets/MolGlueDB_full.csv")['SMILES'].unique()
    inference_engine.predict_lib(mol_glues, prot_list).to_csv("mol_glue_predictions.csv", index=False)
