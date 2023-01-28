#!/usr/bin/env python3
"""
Single stage stellarator optimization (3SO) code
Repository: https://github.com/rogeriojorge/single_stage_optimization
TEMPORARY NOTE: use single_stage branch from SIMSOPT while it is not merged to main branch
"""
import os
import sys
import glob
import json
import time
import shutil
import logging
import argparse
import numpy as np
import pandas as pd
import booz_xform as bx
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from simsopt.mhd import Vmec, Boozer
from simsopt.util import MpiPartition
from simsopt._core.derivative import Derivative
from simsopt.solve import least_squares_mpi_solve
from simsopt._core.finite_difference import MPIFiniteDifference
from simsopt.geo import curves_to_vtk
from src.vmecPlot2 import main as vmecPlot2_main
from src.field_from_coils import main as field_from_coils_main
from src.initialization_functions import (pprint, recalculate_inputs,
                                     create_results_folders, create_initial_coils)
from src.stage_1 import form_stage_1_objective_function
from src.stage_2 import form_stage_2_objective_function, inner_coil_loop
# Start the timer
start = time.time()
# Start mpi process
mpi = MpiPartition()
# Set the logger to print debugs
logger = logging.getLogger(__name__)
# from simsopt.util.mpi import log
# log(level=logging.DEBUG)
# log(level=logging.INFO)
# Check if user selected QA or QH when launching main.py
QAQHQICNTselected=False
if len(sys.argv) > 1:
    if sys.argv[1]=='QA' or sys.argv[1]=='QH' or sys.argv[1]=='QI' or sys.argv[1]=='CNT':
        QAorQHorQIorCNT = sys.argv[1]
        QAQHQICNTselected=True
if not QAQHQICNTselected:
    pprint('First line argument (QA, QH, QI or CNT) not selected. Defaulting to QA.')
    QAorQHorQIorCNT = 'QA'
###############
# Parse the command line arguments and overwrite inputs.py if needed
parser = argparse.ArgumentParser()
inputs = recalculate_inputs(parser, QAQHQICNTselected, QAorQHorQIorCNT, sys.argv)
# Create results folders
parent_path = str(Path(__file__).parent.resolve())
current_path = os.path.join(parent_path, 'results', f'{inputs.name}')
if mpi.proc0_world and inputs.remove_previous_results and os.path.isdir(current_path):
    shutil.copytree(current_path, current_path + '_backup', dirs_exist_ok=True)
    shutil.rmtree(current_path)
if mpi.proc0_world: Path(current_path).mkdir(parents=True, exist_ok=True)
current_path = str(Path(current_path).resolve())
if mpi.proc0_world: shutil.copy(os.path.join(parent_path,'src','inputs.py'), os.path.join(current_path,f'{inputs.name}.py'))
time.sleep(0.5)
try: os.chdir(current_path)
except:
    time.sleep(3)
    os.chdir(current_path)
coils_results_path, vmec_results_path, figures_results_path = create_results_folders(inputs)
# Write inputs to file f'inputs_{inputs.name}.json'
inputs_dict = dict([(att, getattr(inputs,att)) for att in dir(inputs) if '__' not in att])
with open(os.path.join(current_path, f'inputs_{inputs.name}.json'), 'w', encoding='utf-8') as f:
    json.dump(inputs_dict, f, ensure_ascii=False, indent=4)
# If the user did not specify the stage, do single stage by default
if (not inputs.stage_1) and (not inputs.stage_2) and (not inputs.single_stage):
    inputs.single_stage = True
######################################################
pprint("============================================")
pprint("Starting single stage optimization")
pprint("============================================")
######################################################
# Check if a previous optimization has already taken place and load it if exists
vmec_files_list = os.listdir(vmec_results_path)
if os.path.exists(os.path.join(current_path, f'input.{inputs.name}_final')):
    vmec_input_filename = os.path.join(current_path, f'input.{inputs.name}_final')
elif len(vmec_files_list)==0:
    vmec_input_filename = os.path.join(parent_path, 'vmec_inputs', inputs.vmec_input_start)
else:
    vmec_input_files = [file for file in vmec_files_list if 'input.' in file]
    vmec_input_files.sort(key=lambda item: (len(item), item), reverse=False)
    vmec_input_filename = vmec_input_files[-1]
pprint(f' Using vmec input file {os.path.join(vmec_results_path,vmec_input_filename)}')
if inputs.use_half_period:
    vmec = Vmec(os.path.join(vmec_results_path,vmec_input_filename), mpi=mpi, verbose=inputs.vmec_verbose, nphi=inputs.nphi, ntheta=inputs.ntheta, range_surface='half period')
    vmec_full_boundary = Vmec(os.path.join(vmec_results_path,vmec_input_filename), mpi=mpi, verbose=inputs.vmec_verbose, nphi=inputs.nphi, ntheta=inputs.ntheta)
else:
    vmec = Vmec(os.path.join(vmec_results_path,vmec_input_filename), mpi=mpi, verbose=inputs.vmec_verbose, nphi=inputs.nphi, ntheta=inputs.ntheta)
    vmec_full_boundary = vmec
surf = vmec.boundary
surf_full_boundary = vmec_full_boundary.boundary
bs, coils, curves, base_curves = create_initial_coils(vmec, parent_path, coils_results_path, inputs, surf_full_boundary, mpi)
################################################################
######### DEFINE OBJECTIVE FUNCTION AND ITS DERIVATIVES ########
################################################################
class single_stage_obj_and_der():
    def __init__(self) -> None:
        pass
    def fun(self, dofs, prob_jacobian=None, info={'Nfeval':0}, max_mode=1, oustr_dict=[]):
        logger.info('Entering fun')
        info['Nfeval'] += 1
        JF.x = dofs[:-number_vmec_dofs]
        prob.x = dofs[-number_vmec_dofs:]
        bs.set_points(surf.gamma().reshape((-1, 3)))
        os.chdir(vmec_results_path)
        J_stage_1 = prob.objective()
        J_stage_2 = inputs.coils_objective_weight * JF.J()
        J = J_stage_1 + J_stage_2
        if J > inputs.JACOBIAN_THRESHOLD:
            logger.info(f"Exception caught during function evaluation with J={J}. Returning J={inputs.JACOBIAN_THRESHOLD}")
            J = inputs.JACOBIAN_THRESHOLD
        jf = Jf.J()
        BdotN = np.mean(np.abs(np.sum(bs.B().reshape((inputs.nphi, inputs.ntheta, 3)) * surf.unitnormal(), axis=2)))
        outstr = f"\n\nfun#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
        dict1 = {}
        dict1.update({
            'Nfeval': info['Nfeval'], 'J':float(J), 'Jf': float(jf),'J_length':float(J_LENGTH.J()),
            'J_CC':float(J_CC.J()),'J_CURVATURE':float(J_CURVATURE.J()), 'J_LENGTH_PENALTY': float(J_LENGTH_PENALTY.J()),
            # ,'J_CS':float(J_CS.J())
            'J_MSC':float(J_MSC.J()), 'J_ALS':float(J_ALS.J()), 'Lengths':float(sum(j.J() for j in Jls)),
            'curvatures':float(np.sum([np.max(c.kappa()) for c in base_curves])),'msc':float(np.sum([j.J() for j in Jmscs])),
            'B.n':float(BdotN),
            # 'gradJcoils':float(np.linalg.norm(JF.dJ())),
            'C-C-Sep':float(Jccdist.shortest_distance())
            #, 'C-S-Sep':float(Jcsdist.shortest_distance())
        })
        if inputs.debug_coils_outputtxt:
            # outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}"
            outstr += f", , C-C-Sep={Jccdist.shortest_distance():.2f}"#, C-S-Sep={Jcsdist.shortest_distance():.2f}"
            outstr += f"\nJf={jf:.1e}, J_length={J_LENGTH.J():.1e}, J_CC={(J_CC.J()):.1e}, J_CURVATURE={J_CURVATURE.J():.1e}, J_MSC={J_MSC.J():.1e}, J_ALS={J_ALS.J():.1e}, J_LENGTH_PENALTY={J_LENGTH_PENALTY.J():.1e}"#, J_CS={J_CS.J():.1e}
            cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
            kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
            msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
            outstr += f"\n Coil lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curvature=[{kap_string}], mean squared curvature=[{msc_string}]"
        try:
            outstr += f"\n surface dofs="+", ".join([f"{pr}" for pr in dofs[-number_vmec_dofs:]])
            if J<inputs.JACOBIAN_THRESHOLD:
                dict1.update({'Jquasisymmetry':float(qs.total()), 'Jiota':float((vmec.mean_iota()-inputs.iota_target)**2), 'Jaspect':float((vmec.aspect()-inputs.aspect_ratio_target)**2)})
                if not QAorQHorQIorCNT=='QI': outstr += f"\n Quasisymmetry objective={qs.total()}"
                outstr += f"\n aspect={vmec.aspect()}"
                outstr += f"\n mean iota={vmec.mean_iota()}"
            else:
                dict1.update({'Jquasisymmetry':0, 'Jiota':0,'Jaspect':0})
        except Exception as e:
            pprint(e)
        if J<inputs.JACOBIAN_THRESHOLD:
            logger.info(f'Objective function {J} is smaller than the threshold {inputs.JACOBIAN_THRESHOLD}')
            ## Finite differences for the first-stage objective function
            prob_dJ = prob_jacobian.jac(prob.x)

            if not inputs.finite_beta:
                ## Finite differences for the second-stage objective function
                coils_dJ = JF.dJ()
                ## Mixed term - derivative of squared flux with respect to the surface shape
                n = surf.normal()
                absn = np.linalg.norm(n, axis=2)
                B = bs.B().reshape((inputs.nphi, inputs.ntheta, 3))
                dB_by_dX = bs.dB_by_dX().reshape((inputs.nphi, inputs.ntheta, 3, 3))
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
                deriv = surf.dnormal_by_dcoeff_vjp(dJdN/(inputs.nphi*inputs.ntheta)) + surf.dgamma_by_dcoeff_vjp(dJdx/(inputs.nphi*inputs.ntheta))
                mixed_dJ = Derivative({surf: deriv})(surf)
                ## Put both gradients together
                grad_with_respect_to_coils = inputs.coils_objective_weight * coils_dJ
                grad_with_respect_to_surface = np.ravel(prob_dJ) + inputs.coils_objective_weight * mixed_dJ
            else:
                pprint('Finite beta needs to be implemented')
                exit()
            grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))
        else:
            logger.info(f'Objective function {J} is greater than the threshold {inputs.JACOBIAN_THRESHOLD}')
            grad = [0] * len(dofs)
        os.chdir(current_path)
        with open(inputs.debug_output_file, "a") as myfile:
            myfile.write(outstr)
        oustr_dict.append(dict1)
        if np.mod(info['Nfeval'],inputs.output_interval)==0:
            pointData = {"B_N": np.sum(bs.B().reshape((inputs.nphi, inputs.ntheta, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
            surf_full_boundary.x = surf.x
            surf_full_boundary.to_vtk(os.path.join(coils_results_path,f"surf_intermediate_max_mode_{max_mode}_{info['Nfeval']}"), extra_data=pointData)
            curves_to_vtk(curves, os.path.join(coils_results_path,f"curves_intermediate_max_mode_{max_mode}_{info['Nfeval']}"))
        return J, grad
# Loop over the number of predefined maximum poloidal/toroidal modes
if inputs.stage_1 or inputs.stage_2 or inputs.single_stage:
    oustr_dict_outer=[]
    ran_stage1 = False # Only running stage1 optimization once
    previous_max_mode=0
    for max_mode in inputs.max_modes:
        if max_mode != previous_max_mode: oustr_dict_inner=[]
        pprint(f' Starting optimization with max_mode={max_mode}')
        pprint(f'  Forming stage 1 objective function')
        vmec, vmec_full_boundary, surf, surf_full_boundary, qs, qi, number_vmec_dofs, prob = form_stage_1_objective_function(vmec, vmec_full_boundary, surf, surf_full_boundary, max_mode, inputs)
        pprint(f'  Forming stage 2 objective function')
        JF_simple, JF, Jls, Jmscs, Jccdist, Jcsdist, Jf, \
            J_LENGTH, J_CC, J_CS, J_CURVATURE, J_MSC, J_ALS, J_LENGTH_PENALTY = form_stage_2_objective_function(surf, bs, base_curves, curves, inputs)
        dofs = np.concatenate((JF.x, vmec.x))
        # Stage 1 Optimization
        if inputs.stage_1:
            os.chdir(vmec_results_path)
            pprint(f'  Performing Stage 1 optimization with {inputs.MAXITER_stage_1} iterations')
            least_squares_mpi_solve(prob, mpi, grad=True, rel_step=inputs.finite_difference_rel_step, abs_step=inputs.finite_difference_abs_step, max_nfev=inputs.MAXITER_stage_1, ftol=inputs.ftol)
            os.chdir(current_path)
            vmec.write_input(os.path.join(current_path, f'input.stage1'))
            if mpi.proc0_world:
                with open(inputs.debug_output_file, "a") as myfile:
                    try:
                        myfile.write(f"\nAspect ratio at max_mode {max_mode}: {vmec.aspect()}")
                        myfile.write(f"\nMean iota at {max_mode}: {vmec.mean_iota()}")
                        if not QAorQHorQIorCNT=='QI': myfile.write(f"\nQuasisymmetry objective at max_mode {max_mode}: {qs.total()}")
                        myfile.write(f"\nSquared flux at max_mode {max_mode}: {Jf.J()}")
                    except Exception as e:
                        myfile.write(e)
            ran_stage1 = True
        # Stage 2 Optimization
        if (inputs.stage_2 and inputs.stage_1) or previous_max_mode==0:
            pprint(f'  Performing Stage 2 optimization with {inputs.MAXITER_stage_2+inputs.MAXITER_stage_2_simple} iterations')
            surf = vmec.boundary
            bs.set_points(surf.gamma().reshape((-1, 3)))
            if mpi.proc0_world:
                dofs, bs, JF = inner_coil_loop(mpi, JF_simple, JF, Jls, Jmscs, Jccdist, Jcsdist, Jf, J_LENGTH, J_CC, J_CS, J_CURVATURE, J_MSC, J_ALS, J_LENGTH_PENALTY, vmec, curves, base_curves, surf, coils_results_path, number_vmec_dofs, bs, max_mode, inputs, figures_results_path, surf_full_boundary)
                with open(inputs.debug_output_file, "a") as myfile:
                    try:
                        myfile.write(f"\nAspect ratio at max_mode {max_mode}: {vmec.aspect()}")
                        myfile.write(f"\nMean iota at {max_mode}: {vmec.mean_iota()}")
                        if not QAorQHorQIorCNT=='QI': myfile.write(f"\nQuasisymmetry objective at max_mode {max_mode}: {qs.total()}")
                        myfile.write(f"\nSquared flux at max_mode {max_mode}: {Jf.J()}")
                    except Exception as e:
                        myfile.write(e)
            mpi.comm_world.Bcast(dofs, root=0)
        # Single stage Optimization
        if inputs.single_stage:
            pprint(f'  Performing Single Stage optimization with {inputs.MAXITER_single_stage} iterations')
            x0 = np.copy(np.concatenate((JF.x, vmec.x)))
            dofs = np.concatenate((JF.x, vmec.x))
            obj_and_der = single_stage_obj_and_der()
            with MPIFiniteDifference(prob.objective, mpi, rel_step=inputs.finite_difference_rel_step, abs_step=inputs.finite_difference_abs_step, diff_method=inputs.diff_method) as prob_jacobian:
                if mpi.proc0_world:
                    res = minimize(obj_and_der.fun, dofs, args=(prob_jacobian,{'Nfeval':0},max_mode,oustr_dict_inner), jac=True, method='BFGS', options={'maxiter': inputs.MAXITER_single_stage}, tol=1e-15)
                    with open(inputs.debug_output_file, "a") as myfile:
                        try:
                            myfile.write(f"\nAspect ratio at max_mode {max_mode}: {vmec.aspect()}")
                            myfile.write(f"\nMean iota at {max_mode}: {vmec.mean_iota()}")
                            if not QAorQHorQIorCNT=='QI': myfile.write(f"\nQuasisymmetry objective at max_mode {max_mode}: {qs.total()}")
                            myfile.write(f"\nSquared flux at max_mode {max_mode}: {Jf.J()}")
                        except Exception as e:
                            myfile.write(e)
            mpi.comm_world.Bcast(dofs, root=0)
        # Stage 2 Optimization after single_stage
        if (inputs.stage_2 and inputs.single_stage) or (inputs.stage_2 and not inputs.single_stage and not inputs.stage_1 and not previous_max_mode==0):
            pprint(f'  Performing Stage 2 optimization with {inputs.MAXITER_stage_2+inputs.MAXITER_stage_2_simple} iterations')
            surf = vmec.boundary
            bs.set_points(surf.gamma().reshape((-1, 3)))
            if mpi.proc0_world:
                dofs, bs, JF = inner_coil_loop(mpi, JF_simple, JF, Jls, Jmscs, Jccdist, Jcsdist, Jf, J_LENGTH, J_CC, J_CS, J_CURVATURE, J_MSC, J_ALS, J_LENGTH_PENALTY, vmec, curves, base_curves, surf, coils_results_path, number_vmec_dofs, bs, max_mode, inputs, figures_results_path, surf_full_boundary)
                with open(inputs.debug_output_file, "a") as myfile:
                    try:
                        myfile.write(f"\nAspect ratio at max_mode {max_mode}: {vmec.aspect()}")
                        myfile.write(f"\nMean iota at {max_mode}: {vmec.mean_iota()}")
                        if not QAorQHorQIorCNT=='QI': myfile.write(f"\nQuasisymmetry objective at max_mode {max_mode}: {qs.total()}")
                        myfile.write(f"\nSquared flux at max_mode {max_mode}: {Jf.J()}")
                    except Exception as e:
                        myfile.write(e)
            mpi.comm_world.Bcast(dofs, root=0)
        if mpi.proc0_world:
            pointData = {"B_N": np.sum(bs.B().reshape((inputs.nphi, inputs.ntheta, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
            surf_full_boundary.x = surf.x
            surf_full_boundary.to_vtk(os.path.join(coils_results_path,inputs.resulting_surface+'max_mode_'+str(max_mode)), extra_data=pointData)
            curves_to_vtk(curves, os.path.join(coils_results_path,inputs.resulting_coils+'max_mode_'+str(max_mode)))
            bs.save(os.path.join(coils_results_path,inputs.resulting_field_json+'max_mode_'+str(max_mode)+'.json'))
            vmec.write_input(os.path.join(vmec_results_path, f'input.{inputs.name}_maxmode{max_mode}'))
        os.chdir(vmec_results_path)
        try:
            pprint(f"Aspect ratio at max_mode {max_mode}: {vmec.aspect()}")
            pprint(f"Mean iota at {max_mode}: {vmec.mean_iota()}")
            if not QAorQHorQIorCNT=='QI': pprint(f"Quasisymmetry objective at max_mode {max_mode}: {qs.total()}")
            pprint(f"Squared flux at max_mode {max_mode}: {Jf.J()}")
        except Exception as e:
            pprint(e)
        os.chdir(current_path)
        if inputs.single_stage and mpi.proc0_world:
            try:
                df = pd.DataFrame(oustr_dict_inner)
                df.to_csv(os.path.join(current_path, f'output_max_mode_{max_mode}.csv'), index_label='index')
                ax=df.plot(
                    kind='line',
                    logy=True,
                    y=['J','Jf','J_length','J_CC','J_CURVATURE','J_MSC','J_ALS','J_LENGTH_PENALTY','Jquasisymmetry','Jiota','Jaspect'],#,'C-C-Sep','C-S-Sep'],
                    linewidth=0.8)
                ax.set_ylim(bottom=1e-9, top=None)
                ax.set_xlabel('Number of function evaluations')
                ax.set_ylabel('Objective function')
                plt.legend(loc=3, prop={'size': 6})
                plt.tight_layout()
                plt.savefig(os.path.join(figures_results_path, f'optimization_stage3_max_mode_{max_mode}.pdf'), bbox_inches = 'tight', pad_inches = 0)
                plt.close()
            except Exception as e:
                pprint(e)
        previous_max_mode = max_mode
        oustr_dict_outer.append(oustr_dict_inner)
    if mpi.proc0_world:
        pointData = {"B_N": np.sum(bs.B().reshape((inputs.nphi, inputs.ntheta, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
        surf_full_boundary.x = surf.x
        surf_full_boundary.to_vtk(os.path.join(coils_results_path,inputs.resulting_surface), extra_data=pointData)
        curves_to_vtk(curves, os.path.join(coils_results_path,inputs.resulting_coils))
        bs.save(os.path.join(coils_results_path,inputs.resulting_field_json))
        vmec.write_input(os.path.join(current_path, f'input.{inputs.name}_final'))
        if inputs.single_stage:
            try:
                df = pd.DataFrame(oustr_dict_outer)
                df.to_csv(os.path.join(current_path, f'output_{inputs.name}_final.csv'), index_label='index')
                ax=df.plot(kind='line',
                    logy=True,
                    y=['J','Jf','J_length','J_CC','J_CURVATURE','J_MSC','J_ALS','J_LENGTH_PENALTY','Jquasisymmetry', 'Jiota','Jaspect'],#,'C-C-Sep','C-S-Sep'],
                    linewidth=0.8)
                ax.set_ylim(bottom=1e-9, top=None)
                plt.legend(loc=3, prop={'size': 6})
                ax.set_xlabel('Number of function evaluations')
                ax.set_ylabel('Objective function')
                plt.tight_layout()
                plt.savefig(os.path.join(figures_results_path, f'optimization_stage3_final.pdf'), bbox_inches = 'tight', pad_inches = 0)
                plt.close()
            except Exception as e:
                pprint(e)
    os.chdir(vmec_results_path)
    try:
        pprint(f"Aspect ratio after optimization: {vmec.aspect()}")
        pprint(f"Mean iota after optimization: {vmec.mean_iota()}")
        if not QAorQHorQIorCNT=='QI': pprint(f"Quasisymmetry objective after optimization: {qs.total()}")
        pprint(f"Squared flux after optimization: {Jf.J()}")
    except Exception as e:
        pprint(e)
os.chdir(current_path)
if inputs.create_wout_final:
    try:
        vmec_final = Vmec(os.path.join(current_path, f'input.{inputs.name}_final'))
        vmec_final.indata.ns_array[:3]    = [  16,    51,    101]
        vmec_final.indata.niter_array[:3] = [ 2000,  3000, 20000]
        vmec_final.indata.ftol_array[:3]  = [1e-14, 1e-14, 1e-14]
        vmec_final.run()
        if mpi.proc0_world:
            shutil.move(os.path.join(current_path, f"wout_{inputs.name}_final_000_000000.nc"), os.path.join(current_path, f"wout_{inputs.name}_final.nc"))
            os.remove(os.path.join(current_path, f'input.{inputs.name}_final_000_000000'))
    except Exception as e:
        pprint('Exception when creating final vmec file:')
        pprint(e)
## Create results figures
if os.path.isfile(os.path.join(current_path, f"wout_{inputs.name}_final.nc")):
    pprint('Found final vmec file')
    if mpi.proc0_world:
        if inputs.vmec_plot_result:
            pprint("Plot VMEC result")
            vmecPlot2_main(file=os.path.join(current_path, f"wout_{inputs.name}_final.nc"), name=inputs.name, figures_folder=figures_results_path, coils_curves=curves)
    if inputs.booz_xform_plot_result:
        pprint('Creating Boozer class for vmec_final')
        b1 = Boozer(vmec_final, mpol=64, ntor=64)
        pprint('Defining surfaces where to compute Boozer coordinates')
        booz_surfaces = np.linspace(0,1,inputs.boozxform_nsurfaces,endpoint=False)
        pprint(f' booz_surfaces={booz_surfaces}')
        b1.register(booz_surfaces)
        pprint('Running BOOZ_XFORM')
        b1.run()
        if mpi.proc0_world:
            b1.bx.write_boozmn(os.path.join(vmec_results_path,"boozmn_"+inputs.name+".nc"))
            pprint("Plot BOOZ_XFORM")
            fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
            plt.savefig(os.path.join(figures_results_path, "Boozxform_surfplot_1_"+inputs.name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.surfplot(b1.bx, js=int(inputs.boozxform_nsurfaces/2), fill=False, ncontours=35)
            plt.savefig(os.path.join(figures_results_path, "Boozxform_surfplot_2_"+inputs.name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.surfplot(b1.bx, js=inputs.boozxform_nsurfaces-1, fill=False, ncontours=35)
            plt.savefig(os.path.join(figures_results_path, "Boozxform_surfplot_3_"+inputs.name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
            if inputs.name[0:2] == 'QH':
                helical_detail = True
            else:
                helical_detail = False
            fig = plt.figure(); bx.symplot(b1.bx, helical_detail = helical_detail, sqrts=True)
            plt.savefig(os.path.join(figures_results_path, "Boozxform_symplot_"+inputs.name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
            plt.savefig(os.path.join(figures_results_path, "Boozxform_modeplot_"+inputs.name+'.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
os.chdir(parent_path)
# Stop the timer
stop = time.time()
# Remove spurious files
if mpi.proc0_world:
    for objective_file in glob.glob(os.path.join(current_path,f"jac_*")): os.remove(objective_file)
    for objective_file in glob.glob(os.path.join(vmec_results_path,f"jac_*")): os.remove(objective_file)
    for objective_file in glob.glob(os.path.join(vmec_results_path,f"objective_*")): os.remove(objective_file)
    for objective_file in glob.glob(os.path.join(vmec_results_path,f"residuals_*")): os.remove(objective_file)
    for objective_file in glob.glob(os.path.join(vmec_results_path,f"*000_*")): os.remove(objective_file)
    for objective_file in glob.glob(os.path.join(vmec_results_path,f"parvmec*")): os.remove(objective_file)
    for objective_file in glob.glob(os.path.join(vmec_results_path,f"threed*")): os.remove(objective_file)
# Finish
pprint("============================================")
pprint("End of single stage optimization")
pprint(f"Took {stop-start} seconds")
pprint("============================================")