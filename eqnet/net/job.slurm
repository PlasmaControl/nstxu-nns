#!/bin/bash

#SBATCH -n 1
#SBATCH -p general
#SBATCH -t 30:00:00
#SBATCH -J eqnet
#SBATCH --mem=4000
#SBATCH -o a.out
#SBATCH -e a.err
#SBATCH --export=ALL

source /etc/profile.d/modules.sh
module purge
module load anaconda3/2020.02
conda init tcsh
source /usr/pppl/anaconda3/2020.02/etc/profile.d/conda.sh
conda deactivate
conda activate torch-env
module load mdsplus

python -u eqnet_batch.py > a.out
