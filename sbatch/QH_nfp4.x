#!/bin/bash

#SBATCH -N 4
#SBATCH -n 25
#SBATCH -A FUA36_OHARS
#SBATCH -p skl_fua_prod
#SBATCH --mem=160000 
#SBATCH --time 24:00:00
#SBATCH --job-name=QHnfp4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rogerio.jorge@tecnico.ulisboa.pt
cd /marconi_scratch/userexternal/rjorge00/single_stage_optimization

mkdir -p outputs

nfp=4
for ncoils in {3,4,5}; do \
    for len in {3.5,4.0,4.5,5.0}; do \
        mpirun -np 25 ./main.py QH --stage1 --stage2 --single_stage --MAXITER_stage_1 50 --MAXITER_stage_2 1500 --MAXITER_single_stage 500 --max_modes 3 --ncoils $ncoils --lengthbound $len --vmec_input_start input.nfp${nfp}_QH_optimized > outputs/QH_nfp${nfp}_length${len}_${ncoils}coils.txt &
    done
    wait
done
wait

