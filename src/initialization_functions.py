import os
from . import inputs
import numpy as np
from pathlib import Path
from simsopt.geo import curves_to_vtk
from simsopt.field import BiotSavart
from simsopt.field import coils_via_symmetries

# Define a print function that only prints on one processor
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    def pprint(*args, **kwargs):
        if comm.rank == 0:
            print(*args, **kwargs)
except ImportError:
    comm = None
    pprint = print

# Recalculate input parameters from command line arguments
def recalculate_inputs(parser, QAQHQIselected, QAorQHorQI, sysargv):
    parser.add_argument("--use_half_period", dest="use_half_period", default=inputs.use_half_period, action="store_true")
    parser.add_argument("--finite_beta", dest="finite_beta", default=inputs.finite_beta, action="store_true")
    parser.add_argument("--vmec_input_start", default=inputs.vmec_input_start_QA if QAorQHorQI=='QA' else inputs.vmec_input_start_QH if QAorQHorQI=='QH' else inputs.vmec_input_start_QI)
    parser.add_argument("--lengthbound", type=float, default=inputs.LENGTHBOUND_QA if QAorQHorQI=='QA' else inputs.LENGTHBOUND_QH if QAorQHorQI=='QH' else inputs.LENGTHBOUND_QI)
    parser.add_argument("--cc_threshold", type=float, default=inputs.CC_THRESHOLD_QA if QAorQHorQI=='QA' else inputs.CC_THRESHOLD_QH if QAorQHorQI=='QH' else inputs.CC_THRESHOLD_QI)
    parser.add_argument("--msc_threshold", type=float, default=inputs.MSC_THRESHOLD_QA if QAorQHorQI=='QA' else inputs.MSC_THRESHOLD_QH if QAorQHorQI=='QH' else inputs.MSC_THRESHOLD_QI)
    parser.add_argument("--curvature_threshold", type=float, default=inputs.CURVATURE_THRESHOLD_QA if QAorQHorQI=='QA' else inputs.CURVATURE_THRESHOLD_QH if QAorQHorQI=='QH' else inputs.CURVATURE_THRESHOLD_QI)
    parser.add_argument("--ncoils", type=float, default=inputs.ncoils_QA if QAorQHorQI=='QA' else inputs.ncoils_QH if QAorQHorQI=='QH' else inputs.ncoils_QI)
    parser.add_argument("--order", type=float, default=inputs.order)
    parser.add_argument("--MAXITER_stage_1", type=float, default=inputs.MAXITER_stage_1)
    parser.add_argument("--MAXITER_stage_2_simple", type=float, default=inputs.MAXITER_stage_2_simple)
    parser.add_argument("--MAXITER_stage_2", type=float, default=inputs.MAXITER_stage_2)
    parser.add_argument("--MAXITER_single_stage", type=float, default=inputs.MAXITER_single_stage)
    parser.add_argument("--quasisymmetry_helicity_n", type=float, default=inputs.quasisymmetry_helicity_n_QA if QAorQHorQI=='QA' else inputs.quasisymmetry_helicity_n_QH)
    parser.add_argument("--aspect_ratio_target", type=float, default=inputs.aspect_ratio_target_QA if QAorQHorQI=='QA' else inputs.aspect_ratio_target_QH if QAorQHorQI=='QH' else inputs.aspect_ratio_target_QI)
    parser.add_argument("--stage1", dest="stage1", default=inputs.stage_1, action="store_true")
    parser.add_argument("--stage2", dest="stage2", default=inputs.stage_2, action="store_true")
    parser.add_argument("--single_stage", dest="single_stage", default=inputs.single_stage, action="store_true")
    parser.add_argument("--include_iota_target", dest="include_iota_target", default=inputs.include_iota_target_QA if QAorQHorQI=='QA' else inputs.include_iota_target_QH if QAorQHorQI=='QH' else inputs.include_iota_target_QI, action="store_true")
    parser.add_argument("--iota_target", type=float, default=inputs.iota_target_QA if QAorQHorQI=='QA' else inputs.iota_target_QI)
    parser.add_argument('--max_modes', nargs='+',dest="max_modes", default=inputs.max_modes, type=int)
    if QAQHQIselected: args = parser.parse_args(sysargv[2:])
    else: args = parser.parse_args()
    inputs.use_half_period = args.use_half_period
    inputs.max_modes = args.max_modes
    inputs.finite_beta = args.finite_beta
    inputs.order = args.order
    inputs.MAXITER_stage_1 = args.MAXITER_stage_1
    inputs.MAXITER_stage_2_simple = args.MAXITER_stage_2_simple
    inputs.MAXITER_stage_2 = args.MAXITER_stage_2
    inputs.MAXITER_single_stage = args.MAXITER_single_stage
    inputs.iota_target = args.iota_target
    inputs.ncoils = int(args.ncoils)
    inputs.LENGTHBOUND = args.lengthbound
    inputs.CC_THRESHOLD = args.cc_threshold
    inputs.MSC_THRESHOLD = args.msc_threshold
    inputs.vmec_input_start = args.vmec_input_start
    inputs.CURVATURE_THRESHOLD = args.curvature_threshold
    inputs.stage_1 = args.stage1
    inputs.stage_2 = args.stage2
    inputs.single_stage = args.single_stage
    inputs.quasisymmetry_helicity_n = args.quasisymmetry_helicity_n
    inputs.include_iota_target = args.include_iota_target
    inputs.aspect_ratio_target = args.aspect_ratio_target
    inputs.QAorQHorQI = QAorQHorQI
    stage_string = ''
    if args.stage1: stage_string+='1'
    if args.stage2: stage_string+='2'
    if args.single_stage: stage_string+='3'
    if stage_string == '': stage_string = '123'
    inputs.name = f'{QAorQHorQI}_Stage{stage_string}_Lengthbound{args.lengthbound:.1f}_ncoils{int(args.ncoils)}_nfp{args.vmec_input_start[9:10]}'
    return inputs

def create_results_folders(inputs):
    Path(inputs.coils_folder).mkdir(parents=True, exist_ok=True)
    coils_results_path = str(Path(inputs.coils_folder).resolve())
    Path(inputs.vmec_folder).mkdir(parents=True, exist_ok=True)
    vmec_results_path = str(Path(inputs.vmec_folder).resolve())
    Path(inputs.figures_folder).mkdir(parents=True, exist_ok=True)
    figures_results_path = str(Path(inputs.figures_folder).resolve())
    if inputs.remove_previous_debug_output and inputs.single_stage:
        try: os.remove(inputs.debug_output_file)
        except OSError: pass
    return coils_results_path, vmec_results_path, figures_results_path

def create_initial_coils(base_curves, base_currents, nfp, surf, coils_results_path, inputs, mpi):
    coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))
    curves = [c.curve for c in coils]
    if mpi.proc0_world:
        curves_to_vtk(curves, os.path.join(coils_results_path, inputs.initial_coils))
        pointData = {"B_N": np.sum(bs.B().reshape((inputs.nphi, inputs.ntheta, 3)) * surf.unitnormal(), axis=2)[:, :, None]}
        surf.to_vtk(os.path.join(coils_results_path, inputs.initial_surface), extra_data=pointData)
    return bs, coils, curves
