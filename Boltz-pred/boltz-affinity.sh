#!/bin/bash
#SBATCH --job-name=boltz_affinity            # Job name
#SBATCH --ntasks=1
#SBATCH --output=boltz-affinity.log          # Output file (all stdout/stderr goes here)
#SBATCH --gpus=1                             # Request 1 GPU
#SBATCH --cpus-per-task=10                   # Number of CPUs
#SBATCH --mem=50G                            # Memory


# Loop through all yaml files and run boltz predict
for filename in *.yaml; do
    echo "Processing $filename..."
    boltz predict "$filename" \
        --use_potentials \
        --use_msa_server \
        --recycling_steps 20 \
        --sampling_steps_affinity 1000
done
