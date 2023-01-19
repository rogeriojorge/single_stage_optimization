#!/usr/bin/env python
import os
import time
import glob
import logging
import matplotlib
import warnings
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from simsopt.mhd import Vmec
from simsopt.util import MpiPartition
from simsopt.mhd import VirtualCasing
from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable, make_optimizable
from simsopt.geo import create_equally_spaced_curves
from simsopt.field import Current, coils_via_symmetries, BiotSavart
from simsopt.geo import (CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                        LpCurveCurvature, ArclengthVariation, CurveSurfaceDistance)
from simsopt.objectives import SquaredFlux, LeastSquaresProblem, QuadraticPenalty
from simsopt._core.finite_difference import finite_difference_steps, FiniteDifference, MPIFiniteDifference
matplotlib.use('Agg') 
warnings.filterwarnings("ignore",category=matplotlib.MatplotlibDeprecationWarning)
logger = logging.getLogger(__name__)
mpi = MpiPartition()


# Input parameters
QA_or_QHs = ['QH','QA']
derivative_algorithms = ["forward","centered"]
LENGTHBOUND=20
LENGTH_CON_WEIGHT=1e-2
JACOBIAN_THRESHOLD=50
CURVATURE_THRESHOLD=5
CURVATURE_WEIGHT=1e-6
MSC_THRESHOLD=5
MSC_WEIGHT=1e-6
CC_THRESHOLD=0.1
CC_WEIGHT = 1e-3
CS_THRESHOLD=0.05
CS_WEIGHT = 1e-3
ARCLENGTH_WEIGHT = 1e-7
max_mode=1
coils_objective_weight=1
ncoils=3
R0=1
R1=0.5
order=2
nphi=30
ntheta=30
finite_beta = False
vc_src_nphi = nphi
OUT_DIR = f"output"
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(OUT_DIR)

for QA_or_QH in QA_or_QHs:
    if finite_beta: vmec_file = f'../input.precise{QA_or_QH}_FiniteBeta'
    else: vmec_file = f'../input.precise{QA_or_QH}'
    vmec = Vmec(vmec_file, nphi=nphi, ntheta=ntheta, mpi=mpi, verbose=False, range_surface='half period')
    surf = vmec.boundary
    objective_tuple = [(vmec.aspect, 4, 1)]
    if QA_or_QH: objective_tuple.append((vmec.mean_iota, 0.4, 1))
    prob = LeastSquaresProblem.from_tuples(objective_tuple)
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    number_vmec_dofs = int(len(surf.x))

    # Finite Beta Virtual Casing Principle
    if finite_beta:
        if mpi.proc0_world: print('Running the virtual casing calculation')
        vc = VirtualCasing.from_vmec(vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta)
        total_current = vmec.external_current() / (2 * surf.nfp)
        initial_current = total_current / ncoils * 1e-5
    else:
        initial_current = 1

    # Stage 2
    base_curves = create_equally_spaced_curves(ncoils, vmec.indata.nfp, stellsym=True, R0=R0, R1=R1, order=order)
    if finite_beta:
        base_currents = [Current(initial_current) * 1e5 for i in range(ncoils-1)]
        total_current = Current(total_current)
        total_current.fix_all()
        base_currents += [total_current - sum(base_currents)]
    else:
        base_currents = [Current(initial_current) * 1e5 for i in range(ncoils)]
        base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, vmec.indata.nfp, True)
    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))
    curves = [c.curve for c in coils]
    if finite_beta: Jf = SquaredFlux(surf, bs, local=True, target=vc.B_external_normal)
    else: Jf = SquaredFlux(surf, bs, local=True)


    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    surf_full_boundary.fix_all()
    surf_full_boundary.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf_full_boundary.fix("rc(0,0)")
    number_vmec_dofs = int(len(surf.x))
    qs = QuasisymmetryRatioResidual(vmec, inputs.quasisymmetry_target_surfaces, helicity_m=inputs.quasisymmetry_helicity_m, helicity_n=inputs.quasisymmetry_helicity_n)
    objective_tuple = [(vmec.aspect, inputs.aspect_ratio_target, inputs.aspect_ratio_weight), (qs.residuals, 0, inputs.quasisymmetry_weight)]
    if inputs.include_iota_target: objective_tuple.append((vmec.mean_iota, inputs.iota_target, inputs.iota_weight))
    prob = LeastSquaresProblem.from_tuples(objective_tuple)



    # Define the individual terms in the objective function
    Jf = SquaredFlux(surf, bs)
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, inputs.CC_THRESHOLD, num_basecurves=inputs.ncoils)
    # Jcsdist = CurveSurfaceDistance(curves, surf, inputs.CS_THRESHOLD)
    class Jcsdist:
        def __init__(self) -> None: pass
        def shortest_distance(self): return 0
    Jcs = [LpCurveCurvature(c, 2, inputs.CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    Jals = [ArclengthVariation(c) for c in base_curves]

    # Form the total objective function. To do this, we can exploit the
    # fact that Optimizable objects with J() and dJ() functions can be
    # multiplied by scalars and added
    J_LENGTH = inputs.LENGTH_WEIGHT * sum(Jls)
    J_CC = inputs.CC_WEIGHT * Jccdist
    # J_CS = inputs.CS_WEIGHT * Jcsdist
    class J_CS:
        def __init__(self) -> None: pass
        def J(self): return 0
        def dJ(self): return 0
    J_CURVATURE = inputs.CURVATURE_WEIGHT * sum(Jcs)
    J_MSC = inputs.MSC_WEIGHT * sum(QuadraticPenalty(J, inputs.MSC_THRESHOLD) for J in Jmscs)
    J_ALS = inputs.ARCLENGTH_WEIGHT * sum(Jals)
    J_LENGTH_PENALTY = inputs.LENGTH_CON_WEIGHT * sum(QuadraticPenalty(Jls[i], inputs.LENGTHBOUND/len(base_curves)) for i in range(len(base_curves)))

    JF_simple = Jf + J_LENGTH_PENALTY + J_MSC + J_CC

    JF = Jf + J_ALS + J_CC + J_CURVATURE + J_MSC + J_LENGTH_PENALTY# + J_LENGTH + J_CS


    J_stage_1 = prob.objective()
    J_stage_2 = coils_objective_weight * JF.J()
    J = J_stage_1 + J_stage_2




    pprint("""
    ################################################################################
    ### Perform a Taylor test ######################################################
    ################################################################################
    """)
    f = fun
    dofs = JF.x
    np.random.seed(1)
    h = np.random.uniform(size=dofs.shape)
    J0, dJ0 = f(dofs)
    dJh = sum(dJ0 * h)
    for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        J1, _ = f(dofs + eps*h)
        J2, _ = f(dofs - eps*h)
        pprint("err", (J1-J2)/(2*eps) - dJh)