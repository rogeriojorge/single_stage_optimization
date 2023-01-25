#!/bin/bash

#SBATCH -N 4
#SBATCH -n 25
#SBATCH -A FUA36_OHARS
#SBATCH -p skl_fua_prod
#SBATCH --mem=160000 
#SBATCH --time 24:00:00
#SBATCH --job-name=CNTcirc
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rogerio.jorge@tecnico.ulisboa.pt
cd /marconi_scratch/userexternal/rjorge00/single_stage_optimization

mkdir -p outputs

for len in {3.4,3.6,3.8,4.0}; do \
        mpirun -np 25 ./main.py CNT --MAXITER_stage_2 500 --MAXITER_single_stage 500 --max_modes 2 3 4 5 --lengthbound $len > outputs/CNT_length${len}_circular.txt &
done
wait

