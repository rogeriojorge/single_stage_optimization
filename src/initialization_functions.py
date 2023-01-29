import os
from . import inputs
import numpy as np
from pathlib import Path
from simsopt import load
from simsopt.field import BiotSavart
from simsopt.geo import curves_to_vtk
from simsopt.geo.curve import RotatedCurve
from simsopt.field.coil import ScaledCurrent
from simsopt.field import coils_via_symmetries
from simsopt.field import Current, Coil, BiotSavart
from simsopt.geo import create_equally_spaced_curves
from simsopt.geo.curvexyzfourier import CurveXYZFourier
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
def recalculate_inputs(parser, QAQHQICNTselected, QAorQHorQIorCNT, sysargv):
    parser.add_argument("--use_half_period", dest="use_half_period", default=False if QAorQHorQIorCNT=='CNT' else inputs.use_half_period, action="store_true")
    parser.add_argument("--finite_beta", dest="finite_beta", default=inputs.finite_beta, action="store_true")
    parser.add_argument("--diff_method", default=inputs.diff_method)
    parser.add_argument("--ncoils", type=float,               default=inputs.ncoils_QA if QAorQHorQIorCNT=='QA' else inputs.ncoils_QH           if QAorQHorQIorCNT=='QH' else inputs.ncoils_QI           if QAorQHorQIorCNT=='QI' else 4)
    parser.add_argument("--vmec_input_start",       default=inputs.vmec_input_start_QA if QAorQHorQIorCNT=='QA' else inputs.vmec_input_start_QH if QAorQHorQIorCNT=='QH' else inputs.vmec_input_start_QI if QAorQHorQIorCNT=='QI' else inputs.vmec_input_start_CNT)
    parser.add_argument("--lengthbound",   type=float,  default=inputs.LENGTHBOUND_QA  if QAorQHorQIorCNT=='QA' else inputs.LENGTHBOUND_QH      if QAorQHorQIorCNT=='QH' else inputs.LENGTHBOUND_QI      if QAorQHorQIorCNT=='QI' else inputs.LENGTHBOUND_CNT     )
    parser.add_argument("--cc_threshold",   type=float, default=inputs.CC_THRESHOLD_QA if QAorQHorQIorCNT=='QA' else inputs.CC_THRESHOLD_QH     if QAorQHorQIorCNT=='QH' else inputs.CC_THRESHOLD_QI     if QAorQHorQIorCNT=='QI' else inputs.CC_THRESHOLD_CNT    )
    parser.add_argument("--msc_threshold", type=float, default=inputs.MSC_THRESHOLD_QA if QAorQHorQIorCNT=='QA' else inputs.MSC_THRESHOLD_QH    if QAorQHorQIorCNT=='QH' else inputs.MSC_THRESHOLD_QI    if QAorQHorQIorCNT=='QI' else inputs.MSC_THRESHOLD_CNT   )
    parser.add_argument("--iota_target", type=float,     default=inputs.iota_target_QA if QAorQHorQIorCNT=='QA' else inputs.iota_target_QH      if QAorQHorQIorCNT=='QH' else inputs.iota_target_QI      if QAorQHorQIorCNT=='QI' else inputs.iota_target_CNT     )
    parser.add_argument("--curvature_threshold", type=float, default=inputs.CURVATURE_THRESHOLD_QA                 if QAorQHorQIorCNT=='QA' else inputs.CURVATURE_THRESHOLD_QH      if QAorQHorQIorCNT=='QH' else inputs.CURVATURE_THRESHOLD_QI      if QAorQHorQIorCNT=='QI' else inputs.CURVATURE_THRESHOLD_CNT     )
    parser.add_argument("--quasisymmetry_helicity_n", type=float, default=inputs.quasisymmetry_helicity_n_QA       if QAorQHorQIorCNT=='QA' else inputs.quasisymmetry_helicity_n_QH if QAorQHorQIorCNT=='QH' else inputs.quasisymmetry_helicity_n_QI if QAorQHorQIorCNT=='QI' else inputs.quasisymmetry_helicity_n_CNT)
    parser.add_argument("--aspect_ratio_target", type=float, default=inputs.aspect_ratio_target_QA                 if QAorQHorQIorCNT=='QA' else inputs.aspect_ratio_target_QH      if QAorQHorQIorCNT=='QH' else inputs.aspect_ratio_target_QI      if QAorQHorQIorCNT=='QI' else inputs.aspect_ratio_target_CNT     )
    parser.add_argument("--include_iota_target", dest="include_iota_target", default=inputs.include_iota_target_QA if QAorQHorQIorCNT=='QA' else inputs.include_iota_target_QH      if QAorQHorQIorCNT=='QH' else inputs.include_iota_target_QI      if QAorQHorQIorCNT=='QI' else inputs.include_iota_target_CNT, action="store_true")
    parser.add_argument("--order", type=float, default=inputs.order_CNT if QAorQHorQIorCNT=='CNT' else inputs.order)
    parser.add_argument("--MAXITER_stage_1", type=float, default=inputs.MAXITER_stage_1)
    parser.add_argument("--MAXITER_stage_2_simple", type=float, default=inputs.MAXITER_stage_2_simple)
    parser.add_argument("--MAXITER_stage_2", type=float, default=inputs.MAXITER_stage_2)
    parser.add_argument("--MAXITER_single_stage", type=float, default=inputs.MAXITER_single_stage)
    parser.add_argument("--stage1", dest="stage1", default=inputs.stage_1, action="store_true")
    parser.add_argument("--stage2", dest="stage2", default=inputs.stage_2, action="store_true")
    parser.add_argument("--single_stage", dest="single_stage", default=inputs.single_stage, action="store_true")
    parser.add_argument('--max_modes', nargs='+',dest="max_modes", default=inputs.max_modes, type=int)
    parser.add_argument("--FREE_TOP_BOTTOM_CNT", dest="FREE_TOP_BOTTOM_CNT", default=inputs.FREE_TOP_BOTTOM_CNT, action="store_true")
    if QAQHQICNTselected: args = parser.parse_args(sysargv[2:])
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
    inputs.diff_method = args.diff_method
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
    inputs.QAorQHorQIorCNT = QAorQHorQIorCNT
    inputs.FREE_TOP_BOTTOM_CNT = args.FREE_TOP_BOTTOM_CNT
    if QAorQHorQIorCNT=='CNT': inputs.nphi = inputs.nphi_CNT
    stage_string = ''
    if args.stage1: stage_string+='1'
    if args.stage2: stage_string+='2'
    if args.single_stage: stage_string+='3'
    if stage_string == '': stage_string = '123'
    inputs.name = f'{QAorQHorQIorCNT}_Stage{stage_string}_Lengthbound{args.lengthbound:.1f}_ncoils{int(args.ncoils)}'
    if not QAorQHorQIorCNT=='CNT': inputs.name += f'_nfp{args.vmec_input_start[9:10]}'
    if QAorQHorQIorCNT=='CNT' and not inputs.FREE_TOP_BOTTOM_CNT: inputs.name += f'_circular'
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

def create_initial_modular_coils(base_curves, base_currents, nfp):
    coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
    bs = BiotSavart(coils)
    curves = [c.curve for c in coils]
    return bs, coils, curves

def create_initial_coils(vmec, parent_path, coils_results_path, inputs, surf_full_boundary, mpi):
    bs_json_files = [file for file in os.listdir(coils_results_path) if '.json' in file]
    if inputs.QAorQHorQIorCNT=='CNT':
        gamma=0.57555024
        alpha=0.23910724
        center1 = 0.405
        center2 = 0.315
        radius1 = 1.08
        radius2 = 0.405
        current2 = 1.7
        if len(bs_json_files)==0:
            # if inputs.finite_beta:
            #     base_currents = [Current(total_current_vmec/4*1e-5)*1e5, Current(total_current_vmec/4*1e-5)*1e5]
            # else:
            current1 = alpha*current2
            base_currents = [Current(current1)*1e5,Current(current2)*1e5]
            base_curves = [CurveXYZFourier(128, inputs.order) for i in range(2)]
            base_curves[0].set_dofs(np.concatenate(([       0, 0, radius1],np.zeros(2*(inputs.order-1)),[0,                radius1, 0],np.zeros(2*(inputs.order-1)),[-center1,                     0, 0],np.zeros(2*(inputs.order-1)))))
            base_curves[1].set_dofs(np.concatenate(([ center2, 0, radius2],np.zeros(2*(inputs.order-1)),[0, -radius2*np.sin(gamma), 0],np.zeros(2*(inputs.order-1)),[       0, radius2*np.cos(gamma), 0],np.zeros(2*(inputs.order-1)))))
        else:
            if os.path.exists(os.path.join(coils_results_path, inputs.resulting_field_json)):
                bs_temporary = load(os.path.join(coils_results_path, inputs.resulting_field_json))
            else:
                bs_temporary = load(os.path.join(coils_results_path, bs_json_files[-1]))
            curves = [coil._curve for coil in bs_temporary.coils]
            currents = [Current(coil._current.x[0])*1e5 for coil in bs_temporary.coils]
            base_curves = curves[0:2]
            base_currents = currents[0:2]
        if not inputs.FREE_TOP_BOTTOM_CNT:
            base_curves[0].fix_all()
        rotcurve1 = RotatedCurve(base_curves[0], phi=2*np.pi/2, flip=True)
        rotcurrent1 = ScaledCurrent(base_currents[0],-1.e-5)*1.e5
        rotcurve2 = RotatedCurve(base_curves[1], phi=2*np.pi/2, flip=False)
        rotcurrent2 = ScaledCurrent(base_currents[1],1.e-5)*1.e5
        curves = np.concatenate((base_curves,[rotcurve1,rotcurve2]))
        currents = np.concatenate((base_currents,[rotcurrent1,rotcurrent2]))
        coils = [Coil(curv, curr) for (curv, curr) in zip(curves, currents)]
        curves = [c.curve for c in coils]
        bs = BiotSavart(coils)
    else:
        bs_json_files = [file for file in os.listdir(coils_results_path) if '.json' in file]
        if len(bs_json_files)==0:
            bs_initial_file = os.path.join(parent_path, 'coil_inputs', f'biot_savart_nfp{vmec.indata.nfp}_{inputs.QAorQHorQIorCNT}_ncoils{inputs.ncoils}.json')
            if inputs.use_initial_coils_if_available and os.path.isfile(bs_initial_file):
                bs_temporary = load(bs_initial_file)
                base_curves = [bs_temporary.coils[i]._curve for i in range(inputs.ncoils)]
                base_currents = [bs_temporary.coils[i]._current for i in range(inputs.ncoils)]
            else:
                base_curves = create_equally_spaced_curves(inputs.ncoils, vmec.indata.nfp, stellsym=True, R0=inputs.R0, R1=inputs.R1, order=inputs.order)
                base_currents = [Current(inputs.initial_current*1e-5)*1e5 for i in range(inputs.ncoils)]
        else:
            if os.path.exists(os.path.join(coils_results_path, inputs.resulting_field_json)):
                bs_temporary = load(os.path.join(coils_results_path, inputs.resulting_field_json))
            else:
                bs_temporary = load(os.path.join(coils_results_path, bs_json_files[-1]))
            base_curves = [bs_temporary.coils[i]._curve for i in range(inputs.ncoils)]
            base_currents = [bs_temporary.coils[i]._current for i in range(inputs.ncoils)]
        # Create the initial coils
        base_currents[0].fix_all()
        bs, coils, curves = create_initial_modular_coils(base_curves, base_currents, vmec.indata.nfp)
    if mpi.proc0_world:
        bs.set_points(surf_full_boundary.gamma().reshape((-1, 3)))
        curves_to_vtk(curves, os.path.join(coils_results_path, inputs.initial_coils))
        pointData = {"B_N": np.sum(bs.B().reshape((inputs.nphi, inputs.ntheta, 3)) * surf_full_boundary.unitnormal(), axis=2)[:, :, None]}
        surf_full_boundary.to_vtk(os.path.join(coils_results_path, inputs.initial_surface), extra_data=pointData)
    return bs, coils, curves, base_curves