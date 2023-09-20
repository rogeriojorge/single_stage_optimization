#!/usr/bin/env python3
import os
import sys
import glob
import shutil
import numpy as np
from math import isnan
import booz_xform as bx
from pathlib import Path
from subprocess import run
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from simsopt import make_optimizable
from simsopt._core.util import ObjectiveFailure
from simsopt.field.coil import coils_to_makegrid
from src.vmecPlot2 import main as vmecPlot2_main
from simsopt._core.finite_difference import MPIFiniteDifference
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual, VirtualCasing, Boozer
from simsopt.geo import (CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, ArclengthVariation, curves_to_vtk, create_equally_spaced_curves)
from simsopt.objectives import SquaredFlux, QuadraticPenalty, LeastSquaresProblem
from simsopt.util import MpiPartition, proc0_print, comm_world
from simsopt import load, save
mpi = MpiPartition()
this_path = Path(__file__).parent.resolve()
#############################################
run_optimization = True
test_free_boundary = True
config_name = 'nfp4_QH_finitebeta'
aspect_ratio_target = 8
aspect_ratio_weight = 1e1
max_modes = [1,2]
coils_objective_weight = 1e+3
MAXITER_stage_2 = 120
MAXITER_single_stage = 10
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
boozxform_nsurfaces = 14
#############################################
vmec_input_file = os.path.join(this_path, 'vmec_inputs', f'input.{config_name}')
out_dir = os.path.join(this_path, 'results', f'{config_name}_Lengthbound{lengthbound:.1f}_ncoils{int(ncoils)}')
vmec_results_path = os.path.join(this_path, "vmec")
coils_results_path = os.path.join(this_path, "cois")
figures_results_path = os.path.join(this_path, "figures")
if comm_world.rank == 0:
    out_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(vmec_results_path, exist_ok=True)
    os.makedirs(coils_results_path, exist_ok=True)
    os.makedirs(figures_results_path, exist_ok=True)
os.chdir(out_dir)
#############################################
quasisymmetry_target_surfaces = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
finite_difference_abs_step = 1e-7
finite_difference_rel_step = 0
JACOBIAN_THRESHOLD = 100
CC_THRESHOLD = 0.08
LENGTH_THRESHOLD = 3.3
CURVATURE_THRESHOLD = 7
MSC_THRESHOLD = 10
LENGTH_CON_WEIGHT = 0.1  # Weight on the quadratic penalty for the curve length
LENGTH_WEIGHT = 1e-8  # Weight on the curve lengths in the objective function
CC_WEIGHT = 1e+0  # Weight for the coil-to-coil distance penalty in the objective function
CURVATURE_WEIGHT = 1e-3  # Weight for the curvature penalty in the objective function
MSC_WEIGHT = 1e-3  # Weight for the mean squared curvature penalty in the objective function
ARCLENGTH_WEIGHT = 1e-9  # Weight for the arclength variation penalty in the objective function
#############################################
vmec = Vmec(vmec_input_file, mpi=mpi, verbose=False, ntheta=ntheta, nphi=nphi)
surf = vmec.surface
vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta, filename=None)
total_current_vmec = vmec.external_current() / (2 * surf.nfp)
base_curves = create_equally_spaced_curves(ncoils, surf.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
base_currents = [Current(total_current_vmec / ncoils * 1e-5) * 1e5 for _ in range(ncoils-1)]
total_current = Current(total_current_vmec)
total_current.fix_all()
base_currents += [total_current - sum(base_currents)]
coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)
bs.set_points(surf.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi, ntheta, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal
if comm_world.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_init"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_init"), extra_data=pointData)
Jf = SquaredFlux(surf, bs, definition="local", target=vc.B_external_normal)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=len(curves))
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for i, c in enumerate(base_curves)]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jals = [ArclengthVariation(c) for c in base_curves]
J_LENGTH = LENGTH_WEIGHT * sum(Jls)
J_CC = CC_WEIGHT * Jccdist
J_CURVATURE = CURVATURE_WEIGHT * sum(Jcs)
J_MSC = MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for i, J in enumerate(Jmscs))
J_ALS = ARCLENGTH_WEIGHT * sum(Jals)
J_LENGTH_PENALTY = LENGTH_CON_WEIGHT * sum([QuadraticPenalty(Jls[i], LENGTH_THRESHOLD) for i in range(len(base_curves))])
JF = Jf + J_CC + J_LENGTH + J_LENGTH_PENALTY + J_CURVATURE + J_MSC
#############################################
def fun_coils(dofss, info):
    info['Nfeval'] += 1
    JF.x = dofss
    J = JF.J()
    grad = JF.dJ()
    if mpi.proc0_world:
        jf = Jf.J()
        Bbs = bs.B().reshape((nphi, ntheta, 3))
        BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - Jf.target
        BdotN = np.mean(np.abs(BdotN_surf))
        # BdotNmax = np.max(np.abs(BdotN_surf))
        outstr = f"fun_coils#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"  # , B·n max={BdotNmax:.1e}"
        outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}"
        cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
        outstr += f" lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curv=[{kap_string}],msc=[{msc_string}]"
        print(outstr)
    return J, grad
#############################################
def fun_J(prob, coils_prob):
    global previous_surf_dofs
    J_stage_1 = prob.objective()
    if np.any(previous_surf_dofs != prob.x):  # Only run virtual casing if surface dofs have changed
        previous_surf_dofs = prob.x
        try:
            vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta, filename=None)
            Jf.target = vc.B_external_normal
        except ObjectiveFailure as e:
            pass
    bs.set_points(surf.gamma().reshape((-1, 3)))
    J_stage_2 = coils_objective_weight * JF.J()
    J = J_stage_1 + J_stage_2
    return J
#############################################
def fun(dofss, prob_jacobian, info={'Nfeval': 0}):
    info['Nfeval'] += 1
    os.chdir(vmec_results_path)
    prob.x = dofss[-number_vmec_dofs:]
    coil_dofs = dofss[:-number_vmec_dofs]
    # Un-fix the desired coil dofs so they can be updated:
    JF.full_unfix(free_coil_dofs)
    JF.x = coil_dofs
    J = fun_J(prob, JF)
    if J > JACOBIAN_THRESHOLD or isnan(J):
        proc0_print(f"fun#{info['Nfeval']}: Exception caught during function evaluation with J={J}. Returning J={JACOBIAN_THRESHOLD}")
        J = JACOBIAN_THRESHOLD
        grad_with_respect_to_surface = [0] * number_vmec_dofs
        grad_with_respect_to_coils = [0] * len(coil_dofs)
    else:
        proc0_print(f"fun#{info['Nfeval']}: Objective function = {J:.4f}")
        coils_dJ = JF.dJ()
        grad_with_respect_to_coils = coils_objective_weight * coils_dJ
        JF.fix_all()  # Must re-fix the coil dofs before beginning the finite differencing.
        grad_with_respect_to_surface = prob_jacobian.jac(prob.x)[0]
    JF.fix_all()
    grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))
    return J, grad
##########################################################################################
qs = QuasisymmetryRatioResidual(vmec, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=-1)
if run_optimization:
    for max_mode in max_modes:
        print(f"Running optimization with max_mode={max_mode}")
        vmec.indata.mpol = max_mode+2
        vmec.indata.ntor = max_mode+2
        surf.fix_all()
        surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
        surf.fix("rc(0,0)")
        number_vmec_dofs = int(len(surf.x))
        objective_tuple = [(vmec.aspect, aspect_ratio_target, aspect_ratio_weight), (qs.residuals, 0, 1)]
        prob = LeastSquaresProblem.from_tuples(objective_tuple)
        previous_surf_dofs = prob.x
        dofs = np.concatenate((JF.x, vmec.x))
        bs.set_points(surf.gamma().reshape((-1, 3)))
        vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta, filename=None)
        Jf = SquaredFlux(surf, bs, definition="local", target=vc.B_external_normal)
        proc0_print(f"Aspect ratio before optimization: {vmec.aspect()}")
        proc0_print(f"Mean iota before optimization: {vmec.mean_iota()}")
        proc0_print(f"Quasisymmetry objective before optimization: {qs.total()}")
        proc0_print(f"Magnetic well before optimization: {vmec.vacuum_well()}")
        proc0_print(f"Squared flux before optimization: {Jf.J()}")
        proc0_print(f'  Performing stage 2 optimization with ~{MAXITER_stage_2} iterations')
        res = minimize(fun_coils, dofs[:-number_vmec_dofs], jac=True, args=({'Nfeval': 0}), method='L-BFGS-B', options={'maxiter': MAXITER_stage_2, 'maxcor': 300}, tol=1e-12)
        bs.set_points(surf.gamma().reshape((-1, 3)))
        Bbs = bs.B().reshape((nphi, ntheta, 3))
        BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal
        if comm_world.rank == 0:
            curves_to_vtk(curves, os.path.join(coils_results_path, "curves_after_stage2"))
            pointData = {"B_N": BdotN_surf[:, :, None]}
            surf.to_vtk(os.path.join(coils_results_path, "surf_after_stage2"), extra_data=pointData)
        proc0_print(f'  Performing single stage optimization with ~{MAXITER_single_stage} iterations')
        dofs[:-number_vmec_dofs] = res.x
        JF.x = dofs[:-number_vmec_dofs]
        mpi.comm_world.Bcast(dofs, root=0)
        opt = make_optimizable(fun_J, prob, JF)
        free_coil_dofs = JF.dofs_free_status
        JF.fix_all()
        with MPIFiniteDifference(opt.J, mpi, diff_method="forward", abs_step=finite_difference_abs_step, rel_step=finite_difference_rel_step) as prob_jacobian:
            if mpi.proc0_world:
                res = minimize(fun, dofs, args=(prob_jacobian, {'Nfeval': 0}), jac=True, method='BFGS', options={'maxiter': MAXITER_single_stage}, tol=1e-9)
        Bbs = bs.B().reshape((nphi, ntheta, 3))
        BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2) - vc.B_external_normal
        if comm_world.rank == 0:
            curves_to_vtk(curves, os.path.join(coils_results_path, "curves_opt"))
            pointData = {"B_N": BdotN_surf[:, :, None]}
            surf.to_vtk(os.path.join(coils_results_path, "surf_opt"), extra_data=pointData)
        bs.save(os.path.join(coils_results_path, "biot_savart_opt.json"))
        proc0_print(f"Aspect ratio after optimization: {vmec.aspect()}")
        proc0_print(f"Mean iota after optimization: {vmec.mean_iota()}")
        proc0_print(f"Quasisymmetry objective after optimization: {qs.total()}")
        proc0_print(f"Magnetic well after optimization: {vmec.vacuum_well()}")
        proc0_print(f"Squared flux after optimization: {Jf.J()}")
        JF.full_unfix(free_coil_dofs)  # Needed to evaluate JF.dJ
        BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
        BdotN = np.mean(np.abs(BdotN_surf))
        BdotNmax = np.max(np.abs(BdotN_surf))
        outstr = f"Coil parameters: ⟨B·n⟩={BdotN:.1e}, B·n max={BdotNmax:.1e}"
        outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}"
        cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
        outstr += f" lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curv=[{kap_string}], msc=[{msc_string}]"
        proc0_print(outstr)
        #### Create final VMEC files
        vmec.write_input(os.path.join(this_path, f'input.final'))
        vmec_final = os.path.join(this_path, f'input.final', verbose=True, ntheta=ntheta, nphi=nphi, mpi=mpi)
        vmec_final.indata.ns_array[:1]    = [  151]
        vmec_final.indata.niter_array[:1] = [10000]
        vmec_final.indata.ftol_array[:1]  = [1e-14]
        vmec_final.run()
### PLOT
    if mpi.proc0_world:
        vmecPlot2_main(file="wout_final.nc", name=config_name, figures_folder=figures_results_path, coils_curves=[c.curve for c in bs.coils[0:ncoils]])
        print('Creating Boozer class for vmec_final')
        b1 = Boozer(vmec, mpol=64, ntor=64)
        print('Defining surfaces where to compute Boozer coordinates')
        booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
        print(f' booz_surfaces={booz_surfaces}')
        b1.register(booz_surfaces)
        print('Running BOOZ_XFORM')
        b1.run()
        b1.bx.write_boozmn("boozmn_"+config_name+".nc")
        print("Plot BOOZ_XFORM")
        fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
        plt.savefig(os.path.join(figures_results_path, "Boozxform_surfplot_1_"+config_name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
        plt.savefig(os.path.join(figures_results_path, "Boozxform_surfplot_2_"+config_name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
        plt.savefig(os.path.join(figures_results_path, "Boozxform_surfplot_3_"+config_name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
        if 'QH' in config_name:
            helical_detail = True
        else:
            helical_detail = False
        fig = plt.figure(); bx.symplot(b1.bx, helical_detail = helical_detail, sqrts=True)
        plt.savefig(os.path.join(figures_results_path, "Boozxform_symplot_"+config_name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
        plt.savefig(os.path.join(figures_results_path, "Boozxform_modeplot_"+config_name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
    if mpi.proc0_world:
        for objective_file in glob.glob(f"*000_*"): os.remove(objective_file)
        for objective_file in glob.glob(f"parvmec*"): os.remove(objective_file)
        for objective_file in glob.glob(f"threed*"): os.remove(objective_file)
#############################################
if test_free_boundary:
    bs = load(os.path.join(coils_results_path, "biot_savart_opt.json"))
    # bs = load("biot_savart_opt.json")
    vmec_final = Vmec(os.path.join(this_path, f'wout_final.nc'), verbose=True, ntheta=ntheta, nphi=nphi, mpi=mpi)
    s = vmec_final.surface
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
    vmec_final = Vmec(os.path.join(this_path, f'input.final'), verbose=True, ntheta=ntheta, nphi=nphi, mpi=mpi)
    vmec_final.indata.lfreeb = True
    vmec_final.indata.mgrid_file = 'mgrid_opt_coils.nc'
    vmec_final.indata.extcur[0:2*s.nfp*len(base_curves)] = [1]*2*s.nfp*len(base_curves)
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
    vmec_final.write_input('input.final_freeb')
    vmec_final.write_input(os.path.join(out_dir, 'input.final_freeb'))
    vmec_freeb = Vmec(os.path.join(out_dir, 'input.final_freeb'), mpi=mpi, verbose=True)
    vmec_freeb.run()
    if mpi.proc0_world:
        shutil.move("wout_final_freeb_000_000000.nc", "wout_final_freeb.nc")
        # os.remove("wout_final_freeb_000_000000.nc")
        sys.path.insert(1, os.path.join(this_path, '../src'))
        vmecPlot2_main(file="wout_final_freeb.nc", name="vmec_freeb", figures_folder=figures_results_path, coils_curves=[c.curve for c in bs.coils[0:ncoils]])
        print('Creating Boozer class for vmec_freeb')
        b1 = Boozer(vmec_freeb, mpol=64, ntor=64)
        print('Defining surfaces where to compute Boozer coordinates')
        booz_surfaces = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
        print(f' booz_surfaces={booz_surfaces}')
        b1.register(booz_surfaces)
        print('Running BOOZ_XFORM')
        b1.run()
        b1.bx.write_boozmn("boozmn_"+config_name+"_freeb.nc")
        print("Plot BOOZ_XFORM")
        fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
        plt.savefig(os.path.join(figures_results_path, "Boozxform_surfplot_1_"+config_name+'_freeb.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.surfplot(b1.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
        plt.savefig(os.path.join(figures_results_path, "Boozxform_surfplot_2_"+config_name+'_freeb.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.surfplot(b1.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
        plt.savefig(os.path.join(figures_results_path, "Boozxform_surfplot_3_"+config_name+'_freeb.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
        if 'QH' in config_name:
            helical_detail = True
        else:
            helical_detail = False
        fig = plt.figure(); bx.symplot(b1.bx, helical_detail = helical_detail, sqrts=True)
        plt.savefig(os.path.join(figures_results_path, "Boozxform_symplot_"+config_name+'_freeb.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
        plt.savefig(os.path.join(figures_results_path, "Boozxform_modeplot_"+config_name+'_freeb.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
    if mpi.proc0_world:
        for objective_file in glob.glob(f"*000_*"): os.remove(objective_file)
        for objective_file in glob.glob(f"parvmec*"): os.remove(objective_file)
        for objective_file in glob.glob(f"threed*"): os.remove(objective_file)