# TubercuProbe: Cross-Attention Graph–Sequence Affinity Prediction

TubercuProbe pairs a ligand graph encoder (edge-aware GINE with residual connections) and frozen
ESM-C protein embeddings fused by bi-directional cross-attention to predict compound–protein
affinity (pChEMBL). The repository bundles data preparation utilities, training scripts, baseline
comparisons, Boltz structural follow-up, and analysis artifacts for the TubercuProbe paper.

## Project Highlights
- Reproducible DGL/PyTorch implementation of the cross-attention affinity model in `train_caAtt.py`.
- Automated data pipeline for ChEMBL and downstream probe libraries via
  `scripts/prepare_affinity_data.py`.
- Baseline GraphDTA reproductions (`baseline/GraphDTA`) and Boltz/AlphaFold follow-up analyses
  (`Boltz-pred/`, `AF3_*`).
- ClearML logging hooks for experiment tracking plus ready-to-run SLURM submission (`train.sh`).
- Library-level inference workflows for CysDB and MolGlueDB screens (`inference_library.py`).

## Repository Layout
- `data_processing_utils/`: AffinityDataProcessor implementations for ChEMBL and CysDB, plus demos.
- `scripts/prepare_affinity_data.py`: CLI wrapper for generating processed train/valid/test splits.
- `models/`: Model definitions (TubercuProbe encoder, baselines, prototypes).
- `train_caAtt.py` / `train.py`: Training entry points (the `_caAtt` variant reflects paper results).
- `train.sh`: SLURM job file targeting the `gen_ca` conda environment on RTX A6000 nodes.
- `inference_general.py`: Batch evaluation on processed splits with diagnostics and plots.
- `inference_library.py`: Convenience class for scoring protein–compound libraries.
- `baseline/GraphDTA/`: Upstream baseline models, logs, and notebooks.
- `Boltz-pred/`: YAML runners, logging, and analysis of Boltz structural affinity comparisons.
- `reports/`: Generated plots (loss curves, residuals, correlation plots) saved by inference scripts.
- `datasets/`: Raw/tabulated sources (ChEMBL exports, CysDB, MolGlueDB, BindingDB snapshots).
- `processed_data*/`: Pickled splits used for training and evaluation (ignored from version control).
- `weights/`: Serialized checkpoints saved by `train_caAtt.py`.
- `notebooks (*.ipynb)`: EDA, sampling studies, and result visualisation for the paper figures.

## Environment Setup
1. Install Miniconda (or micromamba) and create the project environment:
   ```bash
   conda create -n gen_ca python=3.10
   conda activate gen_ca
   pip install -r requirements.txt
   ```
   `req.txt` freezes the full production stack (CUDA 12.6, PyTorch 2.7, RDKit 2025.x) and can be
   used with `pip install -r req.txt` when replicating the exact HPC environment.
2. Install DGL built for your CUDA toolchain (example for CUDA 12):
   ```bash
   pip install dgl-cu121
   ```
3. (Optional) Initialise ClearML if you want remote experiment tracking:
   ```bash
   clearml-init
   ```
   Without credentials, set `CLEARML_NO_DEFAULT_SERVER=1` to silence remote logging attempts.
4. Ensure RDKit, FAISS, and ESM assets are accessible; Hugging Face ESM-C weights are fetched via
   the `esm` Python package and require internet access on first run.

## Data Preparation
Raw datasets are staged under `datasets/` (the directory is `.gitignore`d to prevent committing
artifacts). The paper relies on the following sources:

| Dataset | Location | Notes |
| --- | --- | --- |
| ChEMBL affinity table | `datasets/chembl_combined.tsv` | Tab-separated export with `Smiles`, `pCHEMBL`, `Sequence`, optional cluster IDs. |
| CysDB electrophiles | `datasets/cysdb_complete_with_sequences.csv` | Contains curated cysteine-reactive probes and sequence mappings. |
| MolGlueDB | `datasets/MolGlueDB_full.csv` | Used for molecular glue screening. |
| BindingDB snapshots | `datasets/BindingDB_*` | Auxiliary benchmarking; not required for the main model. |

Use the CLI wrapper to build processed splits with cached graphs and embeddings:

```bash
# ChEMBL (paper training set)
python scripts/prepare_affinity_data.py \
  --dataset chembl \
  --input datasets/chembl_combined.tsv \
  --output-dir processed_data \
  --stratify-col cluster

# CysDB probe library (for downstream screening pipelines)
python scripts/prepare_affinity_data.py \
  --dataset cysdb \
  --input datasets/cysdb_complete_with_sequences.csv \
  --output-dir processed_data_cysdb \
  --stratify-col cluster
```

Key outputs (`*_processed.pkl`, `*_graphs.npy`, `*_labels.npy`) are written to the chosen output
folder and consumed directly by the training and inference scripts. The processor caches ESM-C
embeddings under `datasets/protein_*.pkl`; delete these files to recompute from scratch.

## Training
- **Local training**: ensure `processed_data/` exists, then run
  ```bash
  python train_caAtt.py
  ```
  Default hyperparameters: hidden size 512, dropout 0.23, batch size 512, AdamW (`lr=1e-3`,
  `weight_decay=1e-5`), ReduceLROnPlateau patience 20, early stopping patience 150. Checkpoints are
  saved to `weights/` (both `best_*.pth` and `last_*.pth`).
- **SLURM cluster**: edit `train.sh` if your module system differs, then submit with
  ```bash
  sbatch train.sh
  ```
  (Current configuration: 1×RTX A6000 GPU, 15 CPU cores, 120 GB RAM, env `gen_ca`).
- **Alternative script**: `train.py` mirrors the above with slightly different hyperparameters and
  is retained for comparison runs logged earlier in the project.

### Baseline Models
Baselines were reproduced with the upstream GraphDTA code under `baseline/GraphDTA/`. Refer to its
`README.md` for dataset formatting and usage (`python training.py`). Result CSVs and plots are kept
alongside the code for transparency.

## Evaluation and Reporting
- **Hold-out metrics**: run `inference_general.py` to compute regression metrics and generate plots.
  ```bash
  python inference_general.py --weights weights/best_model_<timestamp>.pth
  ```
  By default, outputs are saved under `reports/caAtt_inference_<date>/` (scatter plots, residual
  histograms, JSON metrics).
- **Library screening**: `inference_library.py` exposes the `LibraryInferencer` for scoring custom
  SMILES/protein lists. Example usage:
  ```python
  from pathlib import Path
  from inference_library import InferenceConfig
  from inference_library import LibraryInferencer

  engine = LibraryInferencer(InferenceConfig(weights_path=Path("weights/best_model_20250909_113311.pth")))
  preds = engine.predict_lib(smiles_list, protein_seq_list)
  preds.to_csv("cysdb_predictions.csv", index=False)
  ```
  Update the hard-coded protein list or refactor into your workflow as needed.
- **Boltz comparison**: `Boltz-pred/` contains YAML files and shell wrappers used to submit Boltz
  affinity evaluations for top-k TubercuProbe predictions. Generated scatter/correlation plots
  (`boltz_affinity_*.png`) underpin the manuscript’s comparison figures.
- **AlphaFold/AF3 follow-up**: pre-computed structure ranking CSVs and plots are stored at the root
  (`AF3_*`). See the relevant notebooks for reproduction.

## Notebooks and Analysis
The root-level notebooks capture exploratory and paper figure workflows:
- `chembl_eda.ipynb`, `eda.ipynb`: dataset exploration and cleaning decisions.
- `sampling_chembl*.ipynb`: negative sampling experiments.
- `baseline_results_summary.ipynb`: aggregates GraphDTA metrics.
- `boltz-results-analysis.ipynb`: fuses Boltz outputs with TubercuProbe scores.

Notebooks expect the processed data and metrics generated by the scripts above. Re-run them in the
`gen_ca` environment to regenerate plot assets under `reports/` or the notebook directories.

## ClearML and Logging
`train_caAtt.py` initialises a ClearML task (`AffinityPrediction/GNN Training`). Set the following
environment variable if you want to disable remote logging during offline runs:
```bash
export CLEARML_NO_DEFAULT_SERVER=1
```
Training and inference logs are additionally mirrored to stdout and saved in `train.log` /
`reports/` directories for audit trails.

## Testing and Quality Checks
Formal pytest suites are not yet included. Recommended next steps:
- Add smoke tests that construct a tiny DGL graph and exercise `GINWithBidirectionalAttention`.
- Validate `AffinityDataProcessor` output columns via lightweight fixtures.
- Automate linting/formatting (black, isort, ruff) within the `gen_ca` env.

## Reproducibility Checklist (Quick Reference)
- Raw data locations and preprocessing commands documented above.
- Exact environment specification captured in `req.txt` (pip freeze) and `train.sh` (SLURM setup).
- Model checkpoints stored under `weights/` with timestamps.
- Metrics and plots archived under `reports/` and `Boltz-pred/` for manuscript figures.
- Downstream screening outputs (`cysdb_predictions.csv`, `mol_glue_predictions.csv`) are reproducible
  via `inference_library.py` using the released weights.

## License and Data Usage
Ensure compliance with the respective dataset licenses (ChEMBL, CysDB, MolGlueDB, BindingDB).
Generated model weights and processed artifacts should remain private unless the underlying data
permits redistribution.

## Questions
For clarifications or contributions, open an issue or reach out to the Maintainers listed in the
manuscript author list.

## Current Organization is a Work in progress