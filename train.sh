#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 3
#SBATCH --gres=gpu:1
#SBATCH --mem 22gb
#SBATCH --output=trainca_%A.log

conda activate gen_ca
python /home/nroethler/Code/abhiram/chemmap/AffinityPred/train.py
