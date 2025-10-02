# Boltz Affinity Prediction Analysis Summary

## Overview
This analysis matches the top 10 protein-compound pairs from two datasets (CysDB and mol_glue) with their experimentally-derived Boltz affinity predictions.

## Data Summary

### Datasets Analyzed
- **CysDB**: 30 protein-compound pairs (10 per protein)
- **mol_glue**: 30 protein-compound pairs (10 per protein)
- **Total pairs**: 60

### Proteins Analyzed
Three proteins were evaluated, identified by their first 5 amino acids:
1. **MAVRE** - Full sequence starting with MAVRELPGAWNFRDVADTATALRPGRLF...
2. **MTPSQ** - Full sequence starting with MTPSQWLDIAVLAVAFIAAISGWRAGAL...
3. **MLRGI** - Full sequence starting with MLRGIVITSTFGLALLSFGASVALEGAQ...

## File Structure

### Input Files
- `cysdb_predictions.csv` - Predicted pChEMBL values for CysDB compounds
- `mol_glue_predictions.csv` - Predicted pChEMBL values for mol_glue compounds
- `boltz_results_rank{N}_{PROTEIN}_{DATASET}/` - Boltz prediction folders (60 total)

### Output Files
- `boltz_complete_results.csv` - Complete merged dataset with all predictions
- `boltz_top_candidates.csv` - Top candidates based on combined scoring
- `boltz_affinity_correlation.png` - Scatter plots showing correlations
- `boltz_affinity_heatmaps.png` - Heatmaps of affinity values by protein and rank

### Boltz Results Structure
Each Boltz result folder contains:
```
boltz_results_rank{N}_{PROTEIN_ID}_{DATASET}/
├── predictions/
│   └── rank{N}_{PROTEIN_ID}_{DATASET}/
│       ├── affinity_rank{N}_{PROTEIN_ID}_{DATASET}.json  ← Affinity predictions
│       ├── confidence_rank{N}_{PROTEIN_ID}_{DATASET}_model_0.json
│       ├── pae_rank{N}_{PROTEIN_ID}_{DATASET}_model_0.npz
│       ├── pde_rank{N}_{PROTEIN_ID}_{DATASET}_model_0.npz
│       ├── plddt_rank{N}_{PROTEIN_ID}_{DATASET}_model_0.npz
│       ├── pre_affinity_rank{N}_{PROTEIN_ID}_{DATASET}.npz
│       └── rank{N}_{PROTEIN_ID}_{DATASET}_model_0.cif
```

## Key Findings

### Statistical Summary

#### Predicted pChEMBL Values
- **CysDB**: Mean = 6.33 ± 0.20 (range: 6.04 - 6.85)
- **mol_glue**: Mean = 7.67 ± 0.22 (range: 7.25 - 8.26)
- mol_glue compounds show higher predicted binding affinity

#### Boltz Affinity Values
- **CysDB**: Mean = 1.47 ± 0.75 (range: -0.05 - 3.76)
- **mol_glue**: Mean = 0.27 ± 0.49 (range: -0.48 - 1.68)
- Lower Boltz affinity values indicate stronger predicted binding
- mol_glue compounds show lower (better) Boltz affinity values

#### Boltz Probability Binary
- **CysDB**: Mean = 0.23 ± 0.16 (range: 0.05 - 0.76)
- **mol_glue**: Mean = 0.19 ± 0.11 (range: 0.07 - 0.62)
- Lower probability values suggest higher confidence in binding

### Correlation Analysis

#### Within-Dataset Correlations
- **CysDB**: 
  - Pearson correlation: -0.063
  - Spearman correlation: -0.016
  - Very weak negative correlation

- **mol_glue**: 
  - Pearson correlation: -0.274
  - Spearman correlation: -0.273
  - Weak negative correlation

#### Overall Correlation
- **Pearson correlation: -0.692**
- **Spearman correlation: -0.679**
- Strong negative correlation across all data

**Interpretation**: The negative correlation indicates that higher predicted pChEMBL values (better predicted binding) tend to correspond with lower Boltz affinity values (which also indicate better binding). This is expected as Boltz affinity measures binding strength inversely.

## Boltz Affinity JSON Structure

Each affinity JSON file contains multiple prediction values:
```json
{
    "affinity_pred_value": <primary prediction>,
    "affinity_probability_binary": <confidence>,
    "affinity_pred_value1": <alternative prediction 1>,
    "affinity_probability_binary1": <confidence 1>,
    "affinity_pred_value2": <alternative prediction 2>,
    "affinity_probability_binary2": <confidence 2>
}
```

## Analysis Workflow

1. **Data Loading**: Load predicted pChEMBL values from CSV files
2. **Ranking**: Rank compounds within each protein based on predicted affinity
3. **Boltz Extraction**: Parse Boltz affinity JSON files from result folders
4. **Merging**: Match predictions with Boltz results by rank, protein, and dataset
5. **Statistical Analysis**: Calculate correlations and summary statistics
6. **Visualization**: Generate scatter plots and heatmaps
7. **Top Candidates**: Identify best compounds using combined scoring

## Naming Convention

### YAML Files
Format: `rank{N}_{FIRST_5_AA}_{DATASET}.yaml`
- Example: `rank1_MAVRE_cysdb.yaml`
- N: Rank from 1-10
- FIRST_5_AA: First 5 amino acids of protein sequence
- DATASET: Either `cysdb` or `mol_glue`

### Result Folders
Format: `boltz_results_rank{N}_{FIRST_5_AA}_{DATASET}/`
- Matches YAML file naming for easy correlation

## Usage

### Run Complete Analysis
```bash
cd /home/nroethler/Code/abhiram/chemmap/AffinityPred/Boltz-pred
jupyter notebook boltz-results-analysis.ipynb
```

### Execute All Cells
The notebook will:
1. Load and prepare data
2. Extract Boltz affinities from all result folders
3. Merge and analyze correlations
4. Generate visualizations
5. Export results to CSV

### Key Outputs
- Complete dataset with matched predictions
- Top candidate compounds
- Correlation plots
- Affinity heatmaps

## Interpretation Guide

### Predicted pChEMBL
- Higher values = stronger predicted binding
- Range typically 4-10
- Values > 6 indicate good binding affinity

### Boltz Affinity
- **Lower values = stronger binding**
- Negative values indicate very strong binding
- Values > 2 indicate weaker binding

### Combined Scoring
The analysis includes a normalized combined score:
- Normalizes both metrics to 0-1 scale
- Inverts Boltz affinity (lower is better → higher score)
- Averages normalized scores
- Higher combined score = better overall candidate

## Next Steps

1. **Structural Analysis**: Examine 3D structures from .cif files
2. **Confidence Assessment**: Review pLDDT and PAE scores
3. **Experimental Validation**: Test top candidates experimentally
4. **Further Filtering**: Apply drug-likeness filters (Lipinski's rule, etc.)

## Files Generated

All results are saved in the Boltz-pred directory:
- `boltz_complete_results.csv`
- `boltz_top_candidates.csv`
- `boltz_affinity_correlation.png`
- `boltz_affinity_heatmaps.png`
