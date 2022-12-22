#!/usr/bin/env python3
import os
import sys
import time
import glob
import shutil
import logging
import numpy as np
import pandas as pd
from mpi4py import MPI
from math import isnan
import booz_xform as bx
from pathlib import Path
from simsopt import load
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from simsopt.util import MpiPartition
from simsopt._core.derivative import Derivative
from simsopt.solve import least_squares_mpi_solve
from simsopt._core.optimizable import make_optimizable
from simsopt._core.finite_difference import MPIFiniteDifference
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.mhd import Vmec, Boozer, QuasisymmetryRatioResidual, VirtualCasing
from simsopt.objectives import SquaredFlux, QuadraticPenalty, LeastSquaresProblem
from simsopt.geo import (CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                        LpCurveCurvature, ArclengthVariation, curves_to_vtk, create_equally_spaced_curves)
logging.basicConfig()
logger = logging.getLogger('single_stage')
logger.setLevel(1)
comm = MPI.COMM_WORLD
def pprint(*args, **kwargs):
    if comm.rank == 0:
        print(*args, **kwargs)
mpi = MpiPartition()
parent_path = str(Path(__file__).parent.resolve())
os.chdir(parent_path)
start = time.time()
##########################################################################################
############## Input parameters
##########################################################################################
max_modes = [2]
stage_1=True
single_stage=True
MAXITER_stage_1 = 100
MAXITER_stage_2 = 1500
MAXITER_single_stage = 350
finite_beta=False
mercier_stability=False
ncoils = 4
# beta_target = 0.03
aspect_ratio_target = 8.0
CC_THRESHOLD = 0.10
LENGTH_THRESHOLD = 3.1
CURVATURE_THRESHOLD = 6
MSC_THRESHOLD = 12
nphi_VMEC=26
ntheta_VMEC=30
vc_src_nphi=ntheta_VMEC
nmodes_coils = 7
coils_objective_weight = 5e+3
use_previous_results_if_available = True
mercier_threshold=3e-5
mercier_weight=1e-4
quasisymmetry_helicity_m = 1
quasisymmetry_helicity_n = -1
aspect_ratio_weight = 1
diff_method="forward"
R0 = 1.0
R1 = 0.6
quasisymmetry_target_surfaces = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
debug_coils_outputtxt = True
debug_output_file = 'output.txt'
boozxform_nsurfaces = 10
helical_detail = True
finite_difference_abs_step = 1e-7
finite_difference_rel_step = 1e-5
JACOBIAN_THRESHOLD = 100
LENGTH_CON_WEIGHT = 0.1 # Weight on the quadratic penalty for the curve length
CC_WEIGHT = 1e+0 # Weight for the coil-to-coil distance penalty in the objective function
CURVATURE_WEIGHT = 1e-3 # Weight for the curvature penalty in the objective function
MSC_WEIGHT = 1e-3 # Weight for the mean squared curvature penalty in the objective function
ARCLENGTH_WEIGHT = 1e-7 # Weight for the arclength variation penalty in the objective function
plot_result = True
##########################################################################################
##########################################################################################
directory = f'optimization_QH'
if mercier_stability: directory +='_mercier'
if finite_beta: directory +='_finitebeta'
# if stage_1: directory +='_stage1'
vmec_verbose=False
vmec_input_filename=f'input.preciseQH'
if finite_beta: vmec_input_filename+='_finiteBeta'
# Create output directories
if not use_previous_results_if_available:
    if comm.rank == 0:
        if os.path.isdir(directory):
            shutil.copytree(directory, f"{directory}_backup", dirs_exist_ok=True)
            shutil.rmtree(directory)
this_path = os.path.join(parent_path, directory)
os.makedirs(this_path, exist_ok=True)
os.chdir(this_path)
OUT_DIR = os.path.join(this_path, "output")
vmec_results_path = os.path.join(this_path, "vmec")
coils_results_path = os.path.join(this_path, "coils")
if comm.rank == 0:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(vmec_results_path, exist_ok=True)
    os.makedirs(coils_results_path, exist_ok=True)
##########################################################################################
##########################################################################################
# Stage 1
if use_previous_results_if_available and os.path.isfile(os.path.join(this_path, "input.final")):
    vmec_input = os.path.join(this_path,"input.final")
else: vmec_input = os.path.join(parent_path,vmec_input_filename)
pprint(f' Using vmec input file {vmec_input}')
vmec = Vmec(vmec_input, mpi=mpi, verbose=vmec_verbose, nphi=nphi_VMEC, ntheta=ntheta_VMEC, range_surface='half period')
# if finite_beta:
#     try:
#         vmec.run()
#         pprint('betatotal before change =',vmec.wout.betatotal)
#         pprint('vmec.indata.am[0:2] before change =',vmec.indata.am[0:2])
#         pprint('aspect ratio =',vmec.aspect())
#         vmec.indata.am[0:2]=np.array([1,-1])*vmec.wout.am[0]*beta_target/vmec.wout.betatotal
#         vmec.need_to_run_code = True
#         vmec.run()
#         pprint('vmec.indata.am[0:2] after change =',vmec.indata.am[0:2])
#         pprint('betatotal after change =',vmec.wout.betatotal)
#         pprint('aspect ratio =',vmec.aspect())
#     except Exception as e:
#         pprint(e)
# exit()
surf = vmec.boundary
##########################################################################################
##########################################################################################
# Finite Beta Virtual Casing Principle
if finite_beta:
    vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC)
    total_current_vmec = vmec.external_current() / (2 * surf.nfp)
    pprint(f' Total current = {total_current_vmec}')
    pprint(f' max(B_external_normal) = {np.max(vc.B_external_normal)}')
    pprint(f' betatotal =',vmec.wout.betatotal)
    pprint(f' vmec.indata.am[0:2] =',vmec.indata.am[0:2])
    pprint(f' aspect ratio =',vmec.aspect())
##########################################################################################
##########################################################################################
#Stage 2
if use_previous_results_if_available and os.path.isfile(os.path.join(coils_results_path, "biot_savart_opt.json")):
    bs = load(os.path.join(coils_results_path, "biot_savart_opt.json"))
    curves = [coil._curve for coil in bs.coils]
    base_curves = curves[0:ncoils]
    if finite_beta:
        total_current = Current(total_current_vmec)
        total_current.fix_all()
        currents = [Current(coil._current.x[0])*1e5 for coil in bs.coils]
        base_currents = currents[0:ncoils-1]
        base_currents += [total_current - sum(base_currents)]
        # base_currents = currents[0:ncoils]
    else:
        currents = [Current(coil._current.x[0])*1e5 for coil in bs.coils]
        base_currents = currents[0:ncoils]
else:
    base_curves = create_equally_spaced_curves(ncoils, surf.nfp, stellsym=True, R0=R0, R1=R1, order=nmodes_coils, numquadpoints=128)
    if finite_beta:
        base_currents = [Current(total_current_vmec / ncoils * 1e-5) * 1e5 for _ in range(ncoils-1)]
        total_current = Current(total_current_vmec)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
    else:
        base_currents = [Current(1) * 1e5 for _ in range(ncoils)]
##########################################################################################
##########################################################################################
# Save initial surface and coil data
coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)
bs.set_points(surf.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
if finite_beta: BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal
else: BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
if comm.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_init"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_init"), extra_data=pointData)
##########################################################################################
##########################################################################################
if finite_beta: Jf = SquaredFlux(surf, bs, local=True, target=vc.B_external_normal)
else: Jf = SquaredFlux(surf, bs, local=True)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=len(curves))
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for i, c in enumerate(base_curves)]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jals = [ArclengthVariation(c) for c in base_curves]
J_CC = CC_WEIGHT * Jccdist
J_CURVATURE = CURVATURE_WEIGHT * sum(Jcs)
J_MSC = MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for i, J in enumerate(Jmscs))
J_ALS = ARCLENGTH_WEIGHT * sum(Jals)
J_LENGTH_PENALTY = LENGTH_CON_WEIGHT * sum([QuadraticPenalty(Jls[i], LENGTH_THRESHOLD) for i in range(len(base_curves))])
JF = Jf + J_CC + J_LENGTH_PENALTY + J_CURVATURE + J_MSC + J_ALS
##########################################################################################
##########################################################################################
# Initial stage 2 optimization
def fun_coils(dofss, info, oustr_dict=[]):
    info['Nfeval'] += 1
    if info['Nfeval'] == 2: pprint('Iteration #: ', end='', flush=True)
    pprint(info['Nfeval'], ' ', end='', flush=True)
    JF.x = dofss
    J = JF.J()
    grad = JF.dJ()
    if mpi.proc0_world:
        jf = Jf.J()
        Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
        if finite_beta:
            BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - Jf.target
        else:
            BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
        BdotN = np.mean(np.abs(BdotN_surf))
        BdotNmax = np.max(np.abs(BdotN_surf))
        outstr = f"\nfun_coils#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}, B·n max={BdotNmax:.1e}"
        dict1 = {}
        dict1.update({
            'Nfeval': info['Nfeval'], 'J':float(J), 'Jf': float(jf),
            'Lengths':float(sum(j.J() for j in Jls)), 'J_CC':float(J_CC.J()), 'J_LENGTH_PENALTY': float(J_LENGTH_PENALTY.J()),
            'J_CURVATURE':float(J_CURVATURE.J()), 'J_MSC':float(J_MSC.J()), 'J_ALS':float(J_ALS.J()),
            'curvatures':float(np.sum([np.max(c.kappa()) for c in base_curves])), 'msc':float(np.sum([j.J() for j in Jmscs])),
            'B.n':float(BdotN), 'gradJcoils':float(np.linalg.norm(JF.dJ())), 'C-C-Sep':float(Jccdist.shortest_distance()),
        })
        if debug_coils_outputtxt:
            outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}"
            outstr += f" J_CC={(J_CC.J()):.1e}, J_CURVATURE={J_CURVATURE.J():.1e}, J_MSC={J_MSC.J():.1e}, J_ALS={J_ALS.J():.1e}, J_LENGTH_PENALTY={J_LENGTH_PENALTY.J():.1e}"
            cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
            kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
            msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
            outstr += f" Coil lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curvature=[{kap_string}], mean squared curvature=[{msc_string}]"
            outstr += f", coils dofs="+", ".join([f"{pr}" for pr in dofss[0:6]])
        with open(debug_output_file, "a") as myfile:
            myfile.write(outstr)
        oustr_dict.append(dict1)
    return J, grad
def plot_df_stage2(df, max_mode):
    df = pd.DataFrame(oustr_dict)
    df.to_csv(f'output_stage2_max_mode_{max_mode}.csv', index_label='index')
    ax=df.plot(kind='line', logy=True, y=['J','Jf','J_CC','J_CURVATURE','J_MSC','J_ALS','J_LENGTH_PENALTY'], linewidth=0.8)
    ax.set_ylim(bottom=1e-9, top=None)
    ax.set_xlabel('Number of function evaluations')
    ax.set_ylabel('Objective function')
    # plt.axvline(x=info_coils['Nfeval'], linestyle='dashed', color='k', label='simple-loop', linewidth=0.8)
    plt.legend(loc=3, prop={'size': 6})
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'optimization_stage2_max_mode_{max_mode}.pdf'), bbox_inches = 'tight', pad_inches = 0)
    plt.close()
##########################################################################################
##########################################################################################
#############################################################
## Define main optimization function with gradients
#############################################################
##########################################################################################
##########################################################################################
pprint(f'  Starting optimization')
##########################################################################################
##########################################################################################
def Mercier_objective(v, mercier_smin=0.2):
    v.run()
    sDMerc = v.wout.DMerc * v.s_full_grid
    # Discard the inner part of the volume, since vmec's DGeod is inaccurate there.                            
    mask = np.logical_and(v.s_full_grid > mercier_smin, v.s_full_grid < 0.95)
    sDMerc = sDMerc[mask]
    x = np.maximum(mercier_threshold - sDMerc, 0)
    residuals = x / (np.sqrt(len(sDMerc)))# * mercier_threshold)
    return residuals
##########################################################################################
##########################################################################################
def fun_J(dofs_vmec, dofs_coils):
    run_vcasing = False

    if np.sum(prob.x!=dofs_vmec)>0:
        prob.x = dofs_vmec
        run_vcasing = True
    J_stage_1 = prob.objective()

    dofs_coils = np.ravel(dofs_coils)
    if np.sum(JF.x!=dofs_coils)>0:
        JF.x = dofs_coils

    if finite_beta and run_vcasing:
        try:
            logger.info('Running virtual casing')
            vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC)
            Jf = SquaredFlux(surf, bs, local=True, target=vc.B_external_normal)
            if np.sum(Jf.x!=dofs_coils)>0: Jf.x = dofs_coils
            JF.opts[0].opts[0].opts[0].opts[0].opts[0] = Jf
            if np.sum(JF.x!=dofs_coils)>0: JF.x = dofs_coils
        except Exception as e:
            print(e)
            J = JACOBIAN_THRESHOLD
            Jf = JF.opts[0].opts[0].opts[0].opts[0].opts[0]
    bs.set_points(surf.gamma().reshape((-1, 3)))

    J_stage_2 = coils_objective_weight * JF.J()
    J = J_stage_1 + J_stage_2

    return J
##########################################################################################
##########################################################################################
def fun(dofss, prob_jacobian=None, info={'Nfeval':0}, max_mode=1, oustr_dict=[]):
    logger.info('Entering fun')
    info['Nfeval'] += 1
    os.chdir(vmec_results_path)

    dofs_vmec = dofss[-number_vmec_dofs:]
    dofs_coils = dofss[:-number_vmec_dofs]

    J = fun_J(dofs_vmec,dofs_coils)

    if J > JACOBIAN_THRESHOLD or isnan(J):
        logger.info(f"Exception caught during function evaluation with J={J}. Returning J={JACOBIAN_THRESHOLD}")
        J = JACOBIAN_THRESHOLD
        
    logger.info('Writing result')
    jF = JF.J()
    jf = JF.opts[0].opts[0].opts[0].opts[0].opts[0].J()
    Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
    if finite_beta:
        BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - JF.opts[0].opts[0].opts[0].opts[0].opts[0].target
    else:
        BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
    BdotN = np.mean(np.abs(BdotN_surf))
    BdotNmax = np.max(np.abs(BdotN_surf))
    outstr = f"\n\nfun#{info['Nfeval']} on {time.strftime('%Y/%m/%d - %H:%M:%S')} - J={J:.3e}, JF={jF:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}, B·n max={BdotNmax:.1e}"
    dict1 = {}
    dict1.update({
        'Nfeval': info['Nfeval'], 'J':float(J), 'Jf': float(jf),
        'Lengths':float(sum(j.J() for j in Jls)), 'J_CC':float(J_CC.J()), 'J_LENGTH_PENALTY': float(J_LENGTH_PENALTY.J()),
        'J_CURVATURE':float(J_CURVATURE.J()), 'J_MSC':float(J_MSC.J()), 'J_ALS':float(J_ALS.J()),
        'curvatures':float(np.sum([np.max(c.kappa()) for c in base_curves])), 'msc':float(np.sum([j.J() for j in Jmscs])),
        'B.n':float(BdotN), 'gradJcoils':float(np.linalg.norm(JF.dJ())), 'C-C-Sep':float(Jccdist.shortest_distance()),
    })

    # Computing gradients
    coils_dJ = JF.dJ()
    grad_with_respect_to_coils = coils_objective_weight * coils_dJ
    if J<JACOBIAN_THRESHOLD:
        logger.info(f'Objective function {J} is smaller than the threshold {JACOBIAN_THRESHOLD}')
        logger.info(f'Now calculating the gradient')
        if finite_beta:
            grad_with_respect_to_surface = prob_jacobian.jac(dofs_vmec, dofs_coils)[0]
            fun_J(dofs_vmec,dofs_coils)
        else:
            prob_dJ = prob_jacobian.jac(dofs_vmec)
            surface = surf
            bs.set_points(surface.gamma().reshape((-1, 3)))
            ## Mixed term - derivative of squared flux with respect to the surface shape
            n = surface.normal()
            absn = np.linalg.norm(n, axis=2)
            B = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
            dB_by_dX = bs.dB_by_dX().reshape((nphi_VMEC, ntheta_VMEC, 3, 3))
            Bcoil = bs.B().reshape(n.shape)
            unitn = n * (1./absn)[:, :, None]
            Bcoil_n = np.sum(Bcoil*unitn, axis=2)
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            B_n = Bcoil_n
            B_diff = Bcoil
            B_N = np.sum(Bcoil * n, axis=2)
            assert Jf.local
            dJdx = (B_n/mod_Bcoil**2)[:, :, None] * (np.sum(dB_by_dX*(n-B*(B_N/mod_Bcoil**2)[:, :, None])[:, :, None, :], axis=3))
            dJdN = (B_n/mod_Bcoil**2)[:, :, None] * B_diff - 0.5 * (B_N**2/absn**3/mod_Bcoil**2)[:, :, None] * n
            deriv = surface.dnormal_by_dcoeff_vjp(dJdN/(nphi_VMEC*ntheta_VMEC)) + surface.dgamma_by_dcoeff_vjp(dJdx/(nphi_VMEC*ntheta_VMEC))
            mixed_dJ = Derivative({surface: deriv})(surface)

            ## Put both gradients together
            grad_with_respect_to_surface = np.ravel(prob_dJ) + coils_objective_weight * mixed_dJ
    else:
        logger.info(f'Objective function {J} is greater than or equal to the threshold {JACOBIAN_THRESHOLD}')
        grad_with_respect_to_surface = [0] * number_vmec_dofs
        grad_with_respect_to_coils = [0] * len(dofs_coils)
    grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))

    if mpi.proc0_world:
        if debug_coils_outputtxt:
            # if not finite_beta: outstr += f", ║∇J coils║={np.linalg.norm(grad_with_respect_to_coils/coils_objective_weight):.1e}"
            outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
            outstr += f"\n J_CC={(J_CC.J()):.1e}, J_LENGTH_PENALTY={J_LENGTH_PENALTY.J():.1e}"
            outstr += f", J_CURVATURE={J_CURVATURE.J():.1e}, J_MSC={J_MSC.J():.1e}, J_ALS={J_ALS.J():.1e}"
            cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
            kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
            msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
            outstr += f"\n Coil lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}"
            outstr += f", curvature=[{kap_string}], mean squared curvature=[{msc_string}]"
        try:
            outstr += f"\n surface dofs="+", ".join([f"{pr}" for pr in dofss[-number_vmec_dofs:]])
            coilsdofs = dofss[:-number_vmec_dofs]
            outstr += f"\n coils dofs="+", ".join([f"{pr}" for pr in coilsdofs[0:6]])
            if J<JACOBIAN_THRESHOLD:
                outstr += f"\n Quasisymmetry objective={qs.total()}"
                outstr += f"\n Aspect={vmec.aspect()}"
                outstr += f"\n Mean iota={vmec.mean_iota()}"
                outstr += f"\n Magnetic well={vmec.vacuum_well()}"
                outstr += f"\n Mercier objective={np.max(Mercier_objective(vmec))}"
                outstr += f"\n Total beta={vmec.wout.betatotal}"
                dict1.update({'Jquasisymmetry':float(qs.total()),
                              'Jmercier':float(np.max(Mercier_objective(vmec))),
                              'Jaspect':float((vmec.aspect()-aspect_ratio_target)**2)})
            else:
                dict1.update({'Jquasisymmetry':0, 'Jaspect':0, 'Jmercier': 0})
        except Exception as e:
            pprint(e)

        # Remove spurious files
        for vcasing_file in glob.glob("vcasing*"): os.remove(vcasing_file)
        for jac_file in glob.glob("jac_log_*"): os.remove(jac_file)
        os.chdir(parent_path)
        for vcasing_file in glob.glob("vcasing*"): os.remove(vcasing_file)
        for jac_file in glob.glob("jac_log_*"): os.remove(jac_file)
        os.chdir(this_path)
        for vcasing_file in glob.glob("vcasing*"): os.remove(vcasing_file)
        for jac_file in glob.glob("jac_log_*"): os.remove(jac_file)
        with open(debug_output_file, "a") as myfile:
            myfile.write(outstr)
        oustr_dict.append(dict1)
        if np.mod(info['Nfeval'],5)==0:
            if finite_beta: pointData = {"B_N": (np.sum(bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3)) * surf.unitnormal(), axis=2) - vc.B_external_normal)[:, :, None]}
            else: pointData = {"B_N":  np.sum(bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
            surf.to_vtk(os.path.join(coils_results_path,f"surf_intermediate_max_mode_{max_mode}_{info['Nfeval']}"), extra_data=pointData)
            curves_to_vtk(curves, os.path.join(coils_results_path,f"curves_intermediate_max_mode_{max_mode}_{info['Nfeval']}"))

    return J, grad
##########################################################################################
##########################################################################################
#############################################################
## Perform optimization
#############################################################
oustr_dict_outer=[]
ran_stage1 = False # Only running stage1 optimization once
for max_mode in max_modes:
    oustr_dict_inner=[]
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    number_vmec_dofs = int(len(surf.x))
    qs = QuasisymmetryRatioResidual(vmec, quasisymmetry_target_surfaces, helicity_m=quasisymmetry_helicity_m, helicity_n=quasisymmetry_helicity_n)
    objective_tuple = [(vmec.aspect, aspect_ratio_target, aspect_ratio_weight), (qs.residuals, 0, 1)]
    opt_Mercier = make_optimizable(Mercier_objective, vmec)
    if mercier_stability: objective_tuple.append((opt_Mercier.J, 0, mercier_weight))
    prob = LeastSquaresProblem.from_tuples(objective_tuple)

    dofs = np.concatenate((JF.x, vmec.x))
    bs.set_points(surf.gamma().reshape((-1, 3)))

    if stage_1 and not ran_stage1:
        # if finite_beta: vmec.indata.am[0:2]=np.array([1,-1])*vmec.wout.am[0]*beta_target/vmec.wout.betatotal;vmec.need_to_run_code=True
        pprint(f'  Performing stage 1 optimization with {MAXITER_stage_1} iterations')
        os.chdir(vmec_results_path)
        least_squares_mpi_solve(prob, mpi, grad=True, rel_step=finite_difference_rel_step, abs_step=finite_difference_abs_step, max_nfev=MAXITER_stage_1)
        dofs[-number_vmec_dofs:] = prob.x
        os.chdir(this_path)
        vmec.write_input(os.path.join(this_path, f'input.stage1'))
        pprint(f"Aspect ratio at max_mode {max_mode}: {vmec.aspect()}")
        pprint(f"Mean iota at {max_mode}: {vmec.mean_iota()}")
        pprint(f"Quasisymmetry objective at max_mode {max_mode}: {qs.total()}")
        pprint(f"Magnetic well at max_mode {max_mode}: {vmec.vacuum_well()}")
        pprint(f"Mercier objective at max_mode {max_mode}: {np.max(opt_Mercier.J())}")
        pprint(f"Total beta at max_mode {max_mode}: {vmec.wout.betatotal}")
        pprint(f"Squared flux at max_mode {max_mode}: {Jf.J()}")
        if mpi.proc0_world:
            with open(debug_output_file, "a") as myfile:
                try:
                    myfile.write(f"\nAspect ratio at max_mode {max_mode}: {vmec.aspect()}")
                    myfile.write(f"\nMean iota at {max_mode}: {vmec.mean_iota()}")
                    myfile.write(f"\nQuasisymmetry objective at max_mode {max_mode}: {qs.total()}")
                    myfile.write(f"\nMagnetic well at max_mode {max_mode}: {vmec.vacuum_well()}")
                    myfile.write(f"\nMercier objective at max_mode {max_mode}: {np.max(opt_Mercier.J())}")
                    myfile.write(f"\nTotal beta at at max_mode {max_mode}: {vmec.wout.betatotal}")
                    myfile.write(f"\nSquared flux at max_mode {max_mode}: {Jf.J()}")
                except Exception as e:
                    myfile.write(e)
        ran_stage1 = True

    if finite_beta:
        # vmec.indata.am[0:2]=np.array([1,-1])*vmec.wout.am[0]*beta_target/vmec.wout.betatotal;vmec.need_to_run_code=True
        vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC)
        Jf = SquaredFlux(surf, bs, local=True, target=vc.B_external_normal)
        JF.opts[0].opts[0].opts[0].opts[0].opts[0] = Jf
    if mpi.proc0_world:
        info_coils={'Nfeval':0}
        oustr_dict=[]
        pprint(f'  Performing stage 2 optimization with {MAXITER_stage_2} iterations')
        res = minimize(fun_coils, dofs[:-number_vmec_dofs], jac=True, args=(info_coils,oustr_dict), method='L-BFGS-B', options={'maxiter': MAXITER_stage_2, 'maxcor': 300}, tol=1e-12)
        dofs[:-number_vmec_dofs] = res.x
        JF.x = dofs[:-number_vmec_dofs]
        Jf = JF.opts[0].opts[0].opts[0].opts[0].opts[0]
        if finite_beta: pointData = {"B_N": (np.sum(bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3)) * surf.unitnormal(), axis=2) - Jf.target)[:, :, None]}
        else: pointData = {"B_N":  np.sum(bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
        surf.to_vtk(os.path.join(coils_results_path,f'surf_after_inner_loop_max_mode_{max_mode}'), extra_data=pointData)
        curves_to_vtk(curves, os.path.join(coils_results_path,f"curves_after_inner_loop_max_mode_{max_mode}"))
        bs.save(os.path.join(coils_results_path,f"biot_savart_inner_loop_max_mode_{max_mode}.json"))
        df = pd.DataFrame(oustr_dict)
        plot_df_stage2(df, max_mode)
        with open(debug_output_file, "a") as myfile:
            try:
                myfile.write(f"\nAspect ratio at max_mode {max_mode}: {vmec.aspect()}")
                myfile.write(f"\nMean iota at {max_mode}: {vmec.mean_iota()}")
                myfile.write(f"\nQuasisymmetry objective at max_mode {max_mode}: {qs.total()}")
                myfile.write(f"\nMagnetic well at max_mode {max_mode}: {vmec.vacuum_well()}")
                myfile.write(f"\nMercier objective at max_mode {max_mode}: {np.max(opt_Mercier.J())}")
                myfile.write(f"\nTotal beta at at max_mode {max_mode}: {vmec.wout.betatotal}")
                myfile.write(f"\nSquared flux at max_mode {max_mode}: {Jf.J()}")
            except Exception as e:
                myfile.write(e)
    mpi.comm_world.Bcast(dofs, root=0)

    if single_stage:
        pprint(f'  Performing single stage optimization with {MAXITER_single_stage} iterations')
        if finite_beta:
            # If in finite beta, MPI is used to compute the gradients of surface dofs of J=J_stage1+J_stage2
            opt = make_optimizable(fun_J, dofs[-number_vmec_dofs:], dofs[:-number_vmec_dofs], dof_indicators=["dof","non-dof"])
            # opt = make_optimizable(fun_J, dofs, dof_indicators=["dof"])
            with MPIFiniteDifference(opt.J, mpi, diff_method=diff_method, abs_step=finite_difference_abs_step, rel_step=finite_difference_rel_step) as prob_jacobian:
                if mpi.proc0_world:
                    res = minimize(fun, dofs, args=(prob_jacobian,{'Nfeval':0},max_mode,oustr_dict_inner), jac=True, method='BFGS', options={'maxiter': MAXITER_single_stage}, tol=1e-9)
                    dofs = res.x
        else:
            # If in vacuum, MPI is used to compute the gradients of J=J_stage1 only
            with MPIFiniteDifference(prob.objective, mpi, rel_step=finite_difference_rel_step, abs_step=finite_difference_abs_step, diff_method=diff_method) as prob_jacobian:
                if mpi.proc0_world:
                    res = minimize(fun, dofs, args=(prob_jacobian,{'Nfeval':0},max_mode,oustr_dict_inner), jac=True, method='BFGS', options={'maxiter': MAXITER_single_stage}, tol=1e-9)
    
    mpi.comm_world.Bcast(dofs, root=0)

    if mpi.proc0_world:
        if finite_beta: pointData = {"B_N": (np.sum(bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3)) * surf.unitnormal(), axis=2) - vc.B_external_normal)[:, :, None]}
        else: pointData = {"B_N":  np.sum(bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
        surf.to_vtk(os.path.join(coils_results_path,'surf_opt_max_mode_'+str(max_mode)), extra_data=pointData)
        curves_to_vtk(curves, os.path.join(coils_results_path,'curves_opt_max_mode_'+str(max_mode)))
        bs.save(os.path.join(coils_results_path,'biot_savart_opt_max_mode_'+str(max_mode)+'.json'))
        vmec.write_input(os.path.join(vmec_results_path, f'input.maxmode{max_mode}'))

    os.chdir(vmec_results_path)
    try:
        pprint(f"Aspect ratio at max_mode {max_mode}: {vmec.aspect()}")
        pprint(f"Mean iota at {max_mode}: {vmec.mean_iota()}")
        pprint(f"Quasisymmetry objective at max_mode {max_mode}: {qs.total()}")
        pprint(f"Magnetic well at max_mode {max_mode}: {vmec.vacuum_well()}")
        pprint(f"Mercier objective at max_mode {max_mode}: {np.max(opt_Mercier.J())}")
        pprint(f"Total beta at max_mode {max_mode}: {vmec.wout.betatotal}")
        pprint(f"Squared flux at max_mode {max_mode}: {Jf.J()}")
    except Exception as e:
        pprint(e)
    os.chdir(this_path)
    if mpi.proc0_world:
        try:
            df = pd.DataFrame(oustr_dict_inner)
            df.to_csv(os.path.join(this_path, f'output_max_mode_{max_mode}.csv'), index_label='index')
            if mercier_stability: ax=df.plot(kind='line', logy=True, y=['J','Jf','Jquasisymmetry', 'Jmercier','Jaspect','J_CC','J_LENGTH_PENALTY','J_CURVATURE'], linewidth=0.8)
            else: ax=df.plot(kind='line', logy=True, y=['J','Jf','Jquasisymmetry','Jaspect','J_CC','J_LENGTH_PENALTY','J_CURVATURE'], linewidth=0.8)
            ax.set_ylim(bottom=1e-9, top=None)
            ax.set_xlabel('Number of function evaluations')
            ax.set_ylabel('Objective function')
            plt.legend(loc=3, prop={'size': 6})
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f'optimization_stage3_max_mode_{max_mode}.pdf'), bbox_inches = 'tight', pad_inches = 0)
            plt.close()
        except Exception as e:
            pprint(e)

####################################################################################
# Extra stage 2 optimization at the end if single_stage = True
####################################################################################
if single_stage:
    if finite_beta:
        vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC, trgt_ntheta=ntheta_VMEC)
        Jf = SquaredFlux(surf, bs, local=True, target=vc.B_external_normal)
        JF.opts[0].opts[0].opts[0].opts[0].opts[0] = Jf
    if mpi.proc0_world:
        info_coils={'Nfeval':0}
        oustr_dict=[]
        pprint(f'  Performing stage 2 optimization with {MAXITER_stage_2} iterations')
        res = minimize(fun_coils, dofs[:-number_vmec_dofs], jac=True, args=(info_coils,oustr_dict), method='L-BFGS-B', options={'maxiter': MAXITER_stage_2, 'maxcor': 300}, tol=1e-12)
        dofs[:-number_vmec_dofs] = res.x
        JF.x = dofs[:-number_vmec_dofs]
        Jf = JF.opts[0].opts[0].opts[0].opts[0].opts[0]
        df = pd.DataFrame(oustr_dict)
        plot_df_stage2(df, 0)
##########################################################################################
##########################################################################################
#############################################################
## Save figures for coils, surfaces and objective over time
#############################################################
if mpi.proc0_world:
    try:
        if finite_beta: pointData = {"B_N": (np.sum(bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3)) * surf.unitnormal(), axis=2) - vc.B_external_normal)[:, :, None]}
        else: pointData = {"B_N":  np.sum(bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
        surf.to_vtk(os.path.join(coils_results_path,'surf_opt'), extra_data=pointData)
        curves_to_vtk(curves, os.path.join(coils_results_path,'curves_opt'))
        bs.save(os.path.join(coils_results_path,"biot_savart_opt.json"))
        vmec.write_input(os.path.join(this_path, f'input.final'))
        df = pd.DataFrame(oustr_dict_outer[0])
        df.to_csv(os.path.join(this_path, f'output_final.csv'), index_label='index')
        ax=df.plot(kind='line',
            logy=True,
            y=['J','Jf','B.n','Jquasisymmetry','Jmercier','Jaspect','J_CC','J_LENGTH_PENALTY','J_CURVATURE'],
            linewidth=0.8)
        ax.set_ylim(bottom=1e-9, top=None)
        plt.legend(loc=3, prop={'size': 6})
        ax.set_xlabel('Number of function evaluations')
        ax.set_ylabel('Objective function')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'optimization_stage3_final.pdf'), bbox_inches = 'tight', pad_inches = 0)
        plt.close()
    except Exception as e:
        pprint(e)
##########################################################################################
##########################################################################################
#############################################
## Print main results
#############################################
os.chdir(vmec_results_path)
try:
    pprint(f"Aspect ratio after optimization: {vmec.aspect()}")
    pprint(f"Mean iota after optimization: {vmec.mean_iota()}")
    pprint(f"Quasisymmetry objective after optimization: {qs.total()}")
    pprint(f"Magnetic well after optimization: {vmec.vacuum_well()}")
    pprint(f"Mercier objective: {np.max(opt_Mercier.J())}")
    pprint(f"Squared flux after optimization: {Jf.J()}")
except Exception as e: pprint(e)
##########################################################################################
##########################################################################################
#############################################
## Final VMEC equilibrium
#############################################
if not plot_result: exit()
os.chdir(this_path)
try:
    vmec_final = Vmec(os.path.join(this_path, f'input.final'))
    vmec_final.indata.ns_array[:3]    = [  16,     51,   101]#,   151,   201]
    vmec_final.indata.niter_array[:3] = [ 4000,  6000, 10000]#,  8000, 10000]
    vmec_final.indata.ftol_array[:3]  = [1e-12, 1e-13, 1e-16]#, 1e-15, 1e-15]
    vmec_final.run()
    if mpi.proc0_world:
        shutil.move(os.path.join(this_path, f"wout_final_000_000000.nc"), os.path.join(this_path, f"wout_final.nc"))
        os.remove(os.path.join(this_path, f'input.final_000_000000'))
except Exception as e:
    pprint('Exception when creating final vmec file:')
    pprint(e)
if stage_1:
    try:
        vmec_stage1 = Vmec(os.path.join(this_path, f'input.stage1'))
        vmec_stage1.indata.ns_array[:3]    = [  16,     51,   101]#,   151,   201]
        vmec_stage1.indata.niter_array[:3] = [ 4000,  6000, 10000]#,  8000, 10000]
        vmec_stage1.indata.ftol_array[:3]  = [1e-12, 1e-13, 1e-16]#, 1e-15, 1e-15]
        vmec_stage1.run()
        if mpi.proc0_world:
            shutil.move(os.path.join(this_path, f"wout_stage1_000_000000.nc"), os.path.join(this_path, f"wout_stage1.nc"))
            os.remove(os.path.join(this_path, f'input.stage1_000_000000'))
    except Exception as e:
        pprint('Exception when creating stage1 vmec file:')
        pprint(e)
##########################################################################################
##########################################################################################
#############################################
## Create results figures
#############################################
if os.path.isfile(os.path.join(this_path, f"wout_final.nc")):
    pprint('Found final vmec file')
    if mpi.proc0_world:
        pprint("Plot VMEC result")
        import vmecPlot2
        vmecPlot2.main(file=os.path.join(this_path, f"wout_final.nc"), name='single_stage', figures_folder=OUT_DIR, coils_curves=curves)
        pprint('Creating Boozer class for vmec_final')
        b1 = Boozer(vmec_final, mpol=64, ntor=64)
        pprint('Defining surfaces where to compute Boozer coordinates')
        booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
        pprint(f' booz_surfaces={booz_surfaces}')
        b1.register(booz_surfaces)
        pprint('Running BOOZ_XFORM')
        b1.run()
        if mpi.proc0_world:
            b1.bx.write_boozmn(os.path.join(vmec_results_path,"boozmn_single_stage.nc"))
            pprint("Plot BOOZ_XFORM")
            fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
            plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_1_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
            plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_2_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
            plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_3_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.symplot(b1.bx, helical_detail = helical_detail, sqrts=True)
            plt.savefig(os.path.join(OUT_DIR, "Boozxform_symplot_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
            plt.savefig(os.path.join(OUT_DIR, "Boozxform_modeplot_single_stage.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
if os.path.isfile(os.path.join(this_path, f"wout_stage1.nc")) and vmec_stage1:
    pprint('Found stage1 vmec file')
    if mpi.proc0_world:
        pprint("Plot VMEC result")
        import vmecPlot2
        vmecPlot2.main(file=os.path.join(this_path, f"wout_stage1.nc"), name='stage1', figures_folder=OUT_DIR)
        pprint('Creating Boozer class for vmec_stage1')
        b1 = Boozer(vmec_stage1, mpol=64, ntor=64)
        pprint('Defining surfaces where to compute Boozer coordinates')
        booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
        pprint(f' booz_surfaces={booz_surfaces}')
        b1.register(booz_surfaces)
        pprint('Running BOOZ_XFORM')
        b1.run()
        if mpi.proc0_world:
            b1.bx.write_boozmn(os.path.join(vmec_results_path,"boozmn_stage1.nc"))
            pprint("Plot BOOZ_XFORM")
            fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
            plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_1_stage1.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
            plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_2_stage1.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
            plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_3_stage1.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.symplot(b1.bx, helical_detail = helical_detail, sqrts=True)
            plt.savefig(os.path.join(OUT_DIR, "Boozxform_symplot_stage1.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
            plt.savefig(os.path.join(OUT_DIR, "Boozxform_modeplot_stage1.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
##########################################################################################
##########################################################################################
stop = time.time()
##########################################################################################
##########################################################################################
pprint("============================================")
pprint("Finished optimization")
pprint(f"Took {stop-start} seconds")
pprint("============================================")
