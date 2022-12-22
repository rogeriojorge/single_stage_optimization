from simsopt.mhd import QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem

def form_stage_1_objective_function(vmec, surf, max_mode, inputs):
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    number_vmec_dofs = int(len(surf.x))
    qs = QuasisymmetryRatioResidual(vmec, inputs.quasisymmetry_target_surfaces, helicity_m=inputs.quasisymmetry_helicity_m, helicity_n=inputs.quasisymmetry_helicity_n)
    objective_tuple = [(vmec.aspect, inputs.aspect_ratio_target, inputs.aspect_ratio_weight), (qs.residuals, 0, inputs.quasisymmetry_weight)]
    if inputs.include_iota_target: objective_tuple.append((vmec.mean_iota, inputs.iota_target, inputs.iota_weight))
    prob = LeastSquaresProblem.from_tuples(objective_tuple)
    return surf, vmec, qs, number_vmec_dofs, prob