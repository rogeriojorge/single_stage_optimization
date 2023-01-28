#!/bin/bash

#SBATCH -N 4
#SBATCH -n 25
#SBATCH -A FUA36_OHARS
#SBATCH -p skl_fua_prod
#SBATCH --mem=160000 
#SBATCH --time 24:00:00
#SBATCH --job-name=CNTnoncirc
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rogerio.jorge@tecnico.ulisboa.pt
cd /marconi_scratch/userexternal/rjorge00/single_stage_optimization

mkdir -p outputs

for len in {3.4,3.6,3.8,4.0}; do \
        mpirun -np 25 ./main.py CNT --stage1 --stage2 --single_stage --MAXITER_stage_1 50 --MAXITER_stage_2 1500 --MAXITER_single_stage 500 --max_modes 3 --lengthbound $len --FREE_TOP_BOTTOM_CNT > outputs/CNT_length${len}_noncircular.txt &
done
wait

