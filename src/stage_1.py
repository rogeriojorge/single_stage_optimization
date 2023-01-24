from functools import partial
from simsopt import make_optimizable
from simsopt.mhd import QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from .qi_functions import QuasiIsodynamicResidual, MirrorRatioPen, MaxElongationPen

def form_stage_1_objective_function(vmec, vmec_full_boundary, surf, surf_full_boundary, max_mode, inputs):
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    surf_full_boundary.fix_all()
    surf_full_boundary.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf_full_boundary.fix("rc(0,0)")
    number_vmec_dofs = int(len(surf.x))
    objective_tuple = [(vmec.aspect, inputs.aspect_ratio_target, inputs.aspect_ratio_weight)]
    qs = QuasisymmetryRatioResidual(vmec, inputs.quasisymmetry_target_surfaces, helicity_m=inputs.quasisymmetry_helicity_m, helicity_n=inputs.quasisymmetry_helicity_n)
    optQI = partial(QuasiIsodynamicResidual,snorms=inputs.snorms, nphi=inputs.nphi_QI, nalpha=inputs.nalpha_QI, nBj=inputs.nBj_QI, mpol=inputs.mpol_QI, ntor=inputs.ntor_QI, nphi_out=inputs.nphi_out_QI, arr_out=inputs.arr_out_QI)
    qi = make_optimizable(optQI, vmec)
    partial_MaxElongationPen = partial(MaxElongationPen,t=inputs.maximum_elongation)
    optElongation = make_optimizable(partial_MaxElongationPen, vmec)
    partial_MirrorRatioPen = partial(MirrorRatioPen,t=inputs.maximum_mirror)
    optMirror = make_optimizable(partial_MirrorRatioPen, vmec)
    if inputs.QAorQHorQIorCNT == 'QI':
        objective_tuple.append((qi.J, 0, inputs.qsqi_weight))
        objective_tuple.append((optElongation.J, 0, inputs.elongation_weight))
        objective_tuple.append((optMirror.J, 0, inputs.mirror_weight))
    else:
        objective_tuple.append((qs.residuals, 0, inputs.qsqi_weight))
    if inputs.include_iota_target:
        objective_tuple.append((vmec.mean_iota, inputs.iota_target, inputs.iota_weight))
    prob = LeastSquaresProblem.from_tuples(objective_tuple)
    return vmec, vmec_full_boundary, surf, surf_full_boundary, qs, qi, number_vmec_dofs, prob