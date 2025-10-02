#!/bin/bash
#SBATCH --job-name=alphafold3               # Job name
#SBATCH --ntasks=1
#SBATCH --output=alphafold3_mol_glue.log       # Output file
#SBATCH --gpus=1                     # Request 1 GPU
#SBATCH --cpus-per-task=10               # Request 20 CPUs
#SBATCH --mem=50G

module load alphafold/3.0.1

alphafold \
    --input_dir=/home/nroethler/Code/abhiram/chemmap/AffinityPred/AF3-T10/mol_glue \
    --run_inference=True \
    --save_embeddings=True \
    --output_dir=/home/nroethler/Code/abhiram/chemmap/AffinityPred/AF3-T10/mol_glue_output \