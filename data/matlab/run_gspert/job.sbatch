#!/bin/bash

#SBATCH -n 1
#SBATCH -p general
#SBATCH -t 08:00:00
#SBATCH -J sparc_sim
#SBATCH --mem=4000
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --export=ALL

echo '<<>> BEGINNING <<>>'

matlab -nodisplay < $1

echo -e '\n\n<<>> END <<>>'
