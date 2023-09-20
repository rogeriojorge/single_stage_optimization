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
mpi = MpiPartition()
this_path = Path(__file__).parent.resolve()
#############################################
config_name = 'nfp4_QH_finitebeta'
vmec_input_file = os.path.join(this_path, 'vmec_inputs', f'input.{config_name}')
ncoils = 5
R0 = 1.0
R1 = 0.15
order = 8
LENGTH_PENALTY = 1e-3
lengthbound = 3.5
MAXITER = 250
nphi = 32
ntheta = 32
vc_src_nphi = 80
out_dir = os.path.join(this_path, 'results', f'{config_name}_Lengthbound{lengthbound:.1f}_ncoils{int(ncoils)}')
out_dir.mkdir(parents=True, exist_ok=True)
os.chdir(out_dir)
#############################################
vmec = Vmec(vmec_input_file, mpi=mpi, verbose=False)
surf = vmec.surface
#############################################
head, tail = os.path.split(vmec_file)
vc_filename = os.path.join(head, tail.replace('wout', 'vcasing'))
print('virtual casing data file:', vc_filename)
if os.path.isfile(vc_filename):
    print('Loading saved virtual casing result')
    vc = VirtualCasing.load(vc_filename)
else:
    # Virtual casing must not have been run yet.
    print('Running the virtual casing calculation')
    vc = VirtualCasing.from_vmec(vmec_file, src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta)
s = SurfaceRZFourier.from_wout(vmec_file, range="half period", nphi=nphi, ntheta=ntheta)
total_current = Vmec(vmec_file).external_current() / (2 * s.nfp)
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
base_currents = [Current(total_current / ncoils * 1e-5) * 1e5 for _ in range(ncoils-1)]
total_current = Current(total_current)
total_current.fix_all()
base_currents += [total_current - sum(base_currents)]
coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))
curves = [c.curve for c in coils]
curves_to_vtk(curves, out_dir / "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(out_dir / "surf_init", extra_data=pointData)
#############################################
# Define the individual terms in the objective function
Jf = SquaredFlux(surf, bs, definition="local")
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, inputs.CC_THRESHOLD, num_basecurves=inputs.ncoils)
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jcs = [LpCurveCurvature(c, 2, inputs.CURVATURE_THRESHOLD) for c in base_curves]
J_MSC = inputs.MSC_WEIGHT * sum(QuadraticPenalty(J, inputs.MSC_THRESHOLD) for J in Jmscs)
J_LENGTH_PENALTY = inputs.LENGTH_CON_WEIGHT * sum(QuadraticPenalty(Jls[i], inputs.LENGTHBOUND) for i in range(len(base_curves)))
Jals = [ArclengthVariation(c) for c in base_curves]
J_LENGTH = inputs.LENGTH_WEIGHT * sum(Jls)
J_CC = inputs.CC_WEIGHT * Jccdist
J_CURVATURE = inputs.CURVATURE_WEIGHT * sum(Jcs)
J_ALS = inputs.ARCLENGTH_WEIGHT * sum(Jals)
JF = Jf + J_ALS + J_CC + J_CURVATURE + J_MSC + J_LENGTH_PENALTY# + J_LENGTH + J_CS



# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    Bbs = bs.B().reshape((nphi, ntheta, 3))
    BdotN = np.abs(np.sum(Bbs * s.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
    BdotN_mean = np.mean(BdotN)
    BdotN_max = np.max(BdotN)
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨|B·n|⟩={BdotN_mean:.1e}, max(|B·n|)={BdotN_max:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return 1e-4*J, 1e-4*grad


# print("""
# ################################################################################
# ### Perform a Taylor test ######################################################
# ################################################################################
# """)
f = fun
dofs = JF.x
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
# for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
#     J1, _ = f(dofs + eps*h)
#     J2, _ = f(dofs - eps*h)
#     print("err", (J1-J2)/(2*eps) - dJh)

print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300, 'ftol': 1e-20, 'gtol': 1e-20}, tol=1e-20)
dofs = res.x
curves_to_vtk(curves, out_dir / "curves_opt")
Bbs = bs.B().reshape((nphi, ntheta, 3))
BdotN = np.abs(np.sum(Bbs * s.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
pointData = {"B_N": BdotN[:, :, None]}
s.to_vtk(out_dir / "surf_opt", extra_data=pointData)

###############################

bs.save("biot_savart_opt.json")

# bs = load("biot_savart_opt.json")

from simsopt.field.coil import coils_to_makegrid
from subprocess import run

r0 = np.sqrt(s.gamma()[:, :, 0] ** 2 + s.gamma()[:, :, 1] ** 2)
z0 = s.gamma()[:, :, 2]
nzeta = 32
nr = 64
nz = 64

rmin=0.9*np.min(r0)
rmax=1.1*np.max(r0)
zmin=1.1*np.min(z0)
zmax=1.1*np.max(z0)

base_curves = [c.curve for c in bs.coils][:len(base_curves)]
base_currents = [c.current for c in bs.coils][:len(base_curves)]
coils_to_makegrid(os.path.join('coils.opt_coils'), base_curves, base_currents, nfp=s.nfp, stellsym=True)

with open(os.path.join('input_xgrid.dat'), 'w') as f:
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
mgrid_executable = '/Users/rogeriojorge/bin/xgrid'
run_string = f"{mgrid_executable} < {os.path.join('input_xgrid.dat')} > {os.path.join('log_xgrid.opt_coils')}"
# run(run_string, shell=True, check=True)
print(" done")

print(vmec_file_input)
vmec_final = Vmec(vmec_file_input, verbose=True, nphi=nphi, ntheta=ntheta, mpi=mpi)
vmec_final.indata.lfreeb = True
vmec_final.indata.mgrid_file = 'mgrid_opt_coils.nc'
vmec_final.indata.extcur[0:2*s.nfp*len(base_curves)] = [1]*2*s.nfp*len(base_curves)
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

vmec_final.write_input('input.final_freeb')

vmec_final.run()