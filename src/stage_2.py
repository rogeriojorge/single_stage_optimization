import os
import numpy as np
import pandas as pd
from .initialization_functions import pprint
import matplotlib.pyplot as plt
from simsopt.geo import (CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                        LpCurveCurvature, ArclengthVariation)
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import curves_to_vtk
from scipy.optimize import minimize

def form_stage_2_objective_function(surf, bs, base_curves, curves, inputs):
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
    J_LENGTH_PENALTY = inputs.LENGTH_CON_WEIGHT * sum(QuadraticPenalty(Jls[i], inputs.LENGTHBOUND) for i in range(len(base_curves)))

    JF_simple = Jf + J_LENGTH_PENALTY + J_MSC + J_CC

    JF = Jf + J_ALS + J_CC + J_CURVATURE + J_MSC + J_LENGTH_PENALTY# + J_LENGTH + J_CS
    
    return JF_simple, JF, Jls, Jmscs, Jccdist, Jcsdist(), Jf, J_LENGTH, J_CC, J_CS(), J_CURVATURE, J_MSC, J_ALS, J_LENGTH_PENALTY

def inner_coil_loop(mpi, JF_simple, JF, Jls, Jmscs, Jccdist, Jcsdist, Jf, J_LENGTH, J_CC, J_CS, J_CURVATURE, J_MSC, J_ALS, J_LENGTH_PENALTY, vmec, curves, base_curves, surf, coils_results_path, number_vmec_dofs, bs, max_mode, inputs, figures_results_path):

    def fun_coils_simple(dofss, info, oustr_dict=[]):
        info['Nfeval'] += 1
        if info['Nfeval'] == 2: pprint('Iteration #: ', end='', flush=True)
        pprint(info['Nfeval'], ' ', end='', flush=True)
        JF.x = dofss
        J = JF_simple.J()
        grad = JF_simple.dJ()
        if mpi.proc0_world:
            jf = Jf.J()
            BdotN = np.mean(np.abs(np.sum(bs.B().reshape((inputs.nphi, inputs.ntheta, 3)) * surf.unitnormal(), axis=2)))
            outstr = f"\nfun_coils_simple#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
            dict1 = {}
            dict1.update({
                'Nfeval': info['Nfeval'], 'J':float(J), 'Jf': float(jf),'J_length':float(J_LENGTH.J()),
                'J_CC':float(J_CC.J()),'J_CURVATURE':float(J_CURVATURE.J()), 'J_LENGTH_PENALTY': float(J_LENGTH_PENALTY.J()),#,'J_CS':float(J_CS.J())
                'J_MSC':float(J_MSC.J()), 'J_ALS':float(J_ALS.J()), 'Lengths':float(sum(j.J() for j in Jls)),
                'curvatures':float(np.sum([np.max(c.kappa()) for c in base_curves])),'msc':float(np.sum([j.J() for j in Jmscs])),
                'B.n':float(BdotN),
                #'gradJcoils':float(np.linalg.norm(JF.dJ())),
                'C-C-Sep':float(Jccdist.shortest_distance())#, 'C-S-Sep':float(Jcsdist.shortest_distance())
            })
            if inputs.debug_coils_outputtxt:
                # outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}"
                outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"#, C-S-Sep={Jcsdist.shortest_distance():.2f}
                outstr += f" Jf={jf:.1e}, J_length={J_LENGTH.J():.1e}, J_CC={(J_CC.J()):.1e}, J_CURVATURE={J_CURVATURE.J():.1e}, J_MSC={J_MSC.J():.1e}, J_ALS={J_ALS.J():.1e}, J_LENGTH_PENALTY={J_LENGTH_PENALTY.J():.1e}"#, J_CS={J_CS.J():.1e}
                cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
                kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
                msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
                outstr += f" Coil lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curvature=[{kap_string}], mean squared curvature=[{msc_string}]"
            with open(inputs.debug_output_file, "a") as myfile:
                myfile.write(outstr)
            oustr_dict.append(dict1)
        return J, grad

    def fun_coils(dofss, info, oustr_dict=[]):
        info['Nfeval'] += 1
        if info['Nfeval'] == 2: pprint('Iteration #: ', end='', flush=True)
        pprint(info['Nfeval'], ' ', end='', flush=True)
        JF.x = dofss
        J = JF.J()
        grad = JF.dJ()
        if mpi.proc0_world:
            jf = Jf.J()
            BdotN = np.mean(np.abs(np.sum(bs.B().reshape((inputs.nphi, inputs.ntheta, 3)) * surf.unitnormal(), axis=2)))
            outstr = f"\nfun_coils#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
            dict1 = {}
            dict1.update({
                'Nfeval': info['Nfeval'], 'J':float(J), 'Jf': float(jf),'J_length':float(J_LENGTH.J()),
                'J_CC':float(J_CC.J()),'J_CURVATURE':float(J_CURVATURE.J()), 'J_LENGTH_PENALTY': float(J_LENGTH_PENALTY.J()),#,'J_CS':float(J_CS.J())
                'J_MSC':float(J_MSC.J()), 'J_ALS':float(J_ALS.J()), 'Lengths':float(sum(j.J() for j in Jls)),
                'curvatures':float(np.sum([np.max(c.kappa()) for c in base_curves])),'msc':float(np.sum([j.J() for j in Jmscs])),
                'B.n':float(BdotN),
                # 'gradJcoils':float(np.linalg.norm(JF.dJ())),
                # 'C-S-Sep':float(Jcsdist.shortest_distance())
                'C-C-Sep':float(Jccdist.shortest_distance())
            })
            if inputs.debug_coils_outputtxt:
                # outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}"
                outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"#, C-S-Sep={Jcsdist.shortest_distance():.2f}"
                outstr += f" Jf={jf:.1e}, J_length={J_LENGTH.J():.1e}, J_CC={(J_CC.J()):.1e}, J_CURVATURE={J_CURVATURE.J():.1e}, J_MSC={J_MSC.J():.1e}, J_ALS={J_ALS.J():.1e}, J_LENGTH_PENALTY={J_LENGTH_PENALTY.J():.1e}"#, J_CS={J_CS.J():.1e}
                cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
                kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
                msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
                outstr += f" Coil lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curvature=[{kap_string}], mean squared curvature=[{msc_string}]"
            with open(inputs.debug_output_file, "a") as myfile:
                myfile.write(outstr)
            oustr_dict.append(dict1)
        return J, grad

    dofs = np.concatenate((JF.x, vmec.x))
    oustr_dict=[]
    curves_to_vtk(curves, os.path.join(coils_results_path,f"curves_before_inner_loop_max_mode_{max_mode}"))
    pprint(f'\n  Running simple intermediate coil loop with {inputs.MAXITER_stage_2_simple} iterations:')
    info_simple={'Nfeval':0}
    res = minimize(fun_coils_simple, dofs[:-number_vmec_dofs], jac=True, args=(info_simple,oustr_dict), method='L-BFGS-B', options={'maxiter': inputs.MAXITER_stage_2_simple, 'maxcor': 300}, tol=1e-10)
    dofs[:-number_vmec_dofs] = res.x
    JF.x = dofs[:-number_vmec_dofs]
    pprint(f'\n  Running more complete intermediate coil loop with {inputs.MAXITER_stage_2} iterations:')
    info_not_simple={'Nfeval':0}
    res = minimize(fun_coils, dofs[:-number_vmec_dofs], jac=True, args=(info_not_simple,oustr_dict), method='L-BFGS-B', options={'maxiter': inputs.MAXITER_stage_2, 'maxcor': 300}, tol=1e-10)
    dofs[:-number_vmec_dofs] = res.x
    JF.x = dofs[:-number_vmec_dofs]
    curves_to_vtk(curves, os.path.join(coils_results_path,f"curves_after_inner_loop_max_mode_{max_mode}"))
    bs.save(os.path.join(coils_results_path,f"biot_savart_inner_loop_max_mode_{max_mode}.json"))
    df = pd.DataFrame(oustr_dict)
    df.to_csv(f'output_stage2_max_mode_{max_mode}.csv', index_label='index')
    ax=df.plot(kind='line', logy=True, y=['J','Jf','J_length','J_CC','J_CURVATURE','J_MSC','J_ALS','J_LENGTH_PENALTY','C-C-Sep'], linewidth=0.8)
    ax.set_ylim(bottom=1e-9, top=None)
    ax.set_xlabel('Number of function evaluations')
    ax.set_ylabel('Objective function')
    plt.axvline(x=info_simple['Nfeval'], linestyle='dashed', color='k', label='simple-loop', linewidth=0.8)
    plt.legend(loc=3, prop={'size': 6})
    plt.tight_layout()
    plt.savefig(os.path.join(figures_results_path, f'optimization_stage2_max_mode_{max_mode}.pdf'), bbox_inches = 'tight', pad_inches = 0)

    return dofs, bs, JF