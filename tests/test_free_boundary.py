#!/usr/bin/env python3
import os
import re
import sys
import glob
import shutil
import numpy as np
import booz_xform as bx
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import warnings
import matplotlib.cbook
matplotlib.use('Agg') 
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
from simsopt.mhd import Vmec, Boozer
from simsopt import load
from simsopt.util import MpiPartition
from simsopt.field.coil import coils_to_makegrid
mpi = MpiPartition()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=1)
args = parser.parse_args()
## SELECT CONFIGURATIONS TO LOAD
CONFIG = {
    1: {"dir_name": 'QA_Stage123_Lengthbound4.5_ncoils4_nfp2'},
    2: {"dir_name": 'Paper_CNT_Stage123_Lengthbound3.8_ncoils4'},
}
nphi = 32   # surface vmec phi resolution
ntheta = 32 # surface vmec theta resolution
boozxform_nsurfaces = 15
## FOLDER STRUCTURE
results_folder = 'results'
coils_folder = 'coils'
figures_folder = 'figures'
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
match = re.search(r'ncoils(\d+)', output_dir)
ncoils = int(match.group(1)) if match else None
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
vmec_final.indata.extcur[0:2*int(s.nfp)*int(ncoils)] = [1]*2*int(s.nfp)*int(ncoils)
# vmec_final.indata.nvacskip = 6
vmec_final.indata.nzeta = nzeta
# vmec_final.indata.phiedge = vmec_final.indata.phiedge

vmec_final.indata.ns_array[:]    = [0]*len(vmec_final.indata.ns_array)
vmec_final.indata.niter_array[:] = [0]*len(vmec_final.indata.niter_array)
vmec_final.indata.ftol_array[:]  = [0]*len(vmec_final.indata.ftol_array)

vmec_final.indata.ns_array[:1]    = [  151]
vmec_final.indata.niter_array[:1] = [10000]
vmec_final.indata.ftol_array[:1]  = [1e-14]
vmec_final.indata.mpol = 6
vmec_final.indata.ntor = 6
vmec_final.indata.phiedge = np.abs(vmec_final.indata.phiedge)

os.chdir(output_dir)
vmec_final.write_input(os.path.join(output_dir, vmec_input_final_name_freeb))
vmec_freeb = Vmec(os.path.join(output_dir, vmec_input_final_name_freeb), mpi=mpi, verbose=True)
vmec_freeb.run()
if mpi.proc0_world:
    shutil.move("wout_final_freeb_000_000000.nc", "wout_final_freeb.nc")
    # os.remove("wout_final_freeb_000_000000.nc")
    sys.path.insert(1, os.path.join(this_path, '../src'))
    from vmecPlot2 import main as vmecPlot2_main
    vmecPlot2_main(file="wout_final_freeb.nc", name="vmec_freeb", figures_folder=figures_folder, coils_curves=[c.curve for c in bs.coils[0:ncoils]])
    print('Creating Boozer class for vmec_final')
    b1 = Boozer(vmec_final, mpol=64, ntor=64)
    print('Defining surfaces where to compute Boozer coordinates')
    booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
    print(f' booz_surfaces={booz_surfaces}')
    b1.register(booz_surfaces)
    print('Running BOOZ_XFORM')
    b1.run()
    b1.bx.write_boozmn("boozmn_"+dir_name+"_freeb.nc")
    print("Plot BOOZ_XFORM")
    fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
    plt.savefig(os.path.join(figures_folder, "Boozxform_surfplot_1_"+dir_name+'_freeb.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
    plt.savefig(os.path.join(figures_folder, "Boozxform_surfplot_2_"+dir_name+'_freeb.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
    plt.savefig(os.path.join(figures_folder, "Boozxform_surfplot_3_"+dir_name+'_freeb.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
    if 'QH' in dir_name:
        helical_detail = True
    else:
        helical_detail = False
    fig = plt.figure(); bx.symplot(b1.bx, helical_detail = helical_detail, sqrts=True)
    plt.savefig(os.path.join(figures_folder, "Boozxform_symplot_"+dir_name+'_freeb.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
    fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
    plt.savefig(os.path.join(figures_folder, "Boozxform_modeplot_"+dir_name+'_freeb.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
if mpi.proc0_world:
    for objective_file in glob.glob(f"*000_*"): os.remove(objective_file)
    for objective_file in glob.glob(f"parvmec*"): os.remove(objective_file)
    for objective_file in glob.glob(f"threed*"): os.remove(objective_file)
os.chdir(this_path)