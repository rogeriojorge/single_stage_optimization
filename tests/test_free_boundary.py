#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.mhd import Vmec
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import curves_to_vtk, create_equally_spaced_curves
from simsopt.field import BiotSavart
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import CurveLength
from simsopt.mhd import VirtualCasing
from simsopt import load, save
from simsopt.util import MpiPartition
from simsopt.field.coil import coils_to_makegrid
from subprocess import run
mpi = MpiPartition()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=1)
args = parser.parse_args()
## SELECT CONFIGURATIONS TO LOAD
CONFIG = {
    1: {"dir_name": 'QA_Stage123_Lengthbound5.5_ncoils3_nfp2'},
    2: {"dir_name": 'Paper_CNT_Stage123_Lengthbound3.8_ncoils4'},
}
nphi = 32   # surface vmec phi resolution
ntheta = 32 # surface vmec theta resolution
## FOLDER STRUCTURE
results_folder = 'results'
coils_folder = 'coils'
wout_final_name = 'wout_final.nc'
vmec_input_final_name = 'input.final'
vmec_input_final_name_freeb = 'input.final_freeb'
mgrid_executable = '/Users/rogeriojorge/bin/xgrid'
## LOAD RESULTS
this_path = Path(__file__).parent.resolve()
dir_name = CONFIG[args.type]["dir_name"]
output_dir = os.path.join(this_path, '..', results_folder, dir_name)
vmec_file_input = os.path.join(output_dir, vmec_input_final_name)
bs = load(os.path.join(output_dir, coils_folder, "biot_savart_opt.json"))
vmec = Vmec(os.path.join(output_dir, wout_final_name), verbose=False, mpi=mpi)
s = vmec.boundary
opt_coil_location = os.path.join(output_dir, coils_folder, 'coils.opt_coils')
xgrid_output_location = os.path.join(output_dir, coils_folder, 'input_xgrid.dat')
## GENERATE MAKEGRID
r0 = np.sqrt(s.gamma()[:, :, 0] ** 2 + s.gamma()[:, :, 1] ** 2)
z0 = s.gamma()[:, :, 2]
nzeta = 32
nr = 64
nz = 64
rmin=0.9*np.min(r0)
rmax=1.1*np.max(r0)
zmin=1.1*np.min(z0)
zmax=1.1*np.max(z0)
def find_ncoils(string):
    parts = string.split("_")
    for part in parts:
        if part[:5] == "ncoils":
            return int(part[5:])
ncoils = find_ncoils(output_dir)
coils_to_makegrid(opt_coil_location, [c.curve for c in bs.coils[0:ncoils]], [c.current for c in bs.coils[0:ncoils]], nfp=s.nfp, stellsym=True)
with open(xgrid_output_location, 'w') as f:
    f.write('opt_coils\n')
    f.write('R\n') # R puts extcur to 1 (more convenient) instead of manually defining them when using S
    f.write('y\n')
    f.write(f'{rmin}\n')
    f.write(f'{rmax}\n')
    f.write(f'{zmin}\n')
    f.write(f'{zmax}\n')
    f.write(f'{nzeta}\n')
    f.write(f'{nr}\n')
    f.write(f'{nz}\n')
print("Running makegrid")
run_string = f"{mgrid_executable} < {xgrid_output_location} > {os.path.join('log_xgrid.opt_coils')}"
# run(run_string, shell=True, check=True)
print(" done")

print(vmec_file_input)
vmec_final = Vmec(vmec_file_input, verbose=True, nphi=nphi, ntheta=ntheta, mpi=mpi)
vmec_final.indata.lfreeb = True
vmec_final.indata.mgrid_file = 'mgrid_opt_coils.nc'
vmec_final.indata.extcur[0:2*s.nfp*ncoils] = [1]*2*s.nfp*ncoils
# vmec_final.indata.nvacskip = 6
vmec_final.indata.nzeta = nzeta
# vmec_final.indata.phiedge = vmec_final.indata.phiedge

vmec_final.indata.ns_array[:]    = [0]*len(vmec_final.indata.ns_array)
vmec_final.indata.niter_array[:] = [0]*len(vmec_final.indata.niter_array)
vmec_final.indata.ftol_array[:]  = [0]*len(vmec_final.indata.ftol_array)

vmec_final.indata.ns_array[:2]    = [   9,    29]
vmec_final.indata.niter_array[:2] = [1000,  1000]
vmec_final.indata.ftol_array[:2]  = [1e-9, 1e-11]
vmec_final.indata.mpol = 6
vmec_final.indata.ntor = 6
vmec_final.indata.phiedge = np.abs(vmec_final.indata.phiedge)

vmec_final.write_input(os.path.join(output_dir, vmec_input_final_name_freeb))

vmec_final.run()