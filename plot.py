#!/usr/bin/env python
import os
import sys
import time
import shutil
import argparse
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
import booz_xform as bx
from pathlib import Path
parent_path = str(Path(__file__).parent.resolve())
from math import ceil, sqrt
import matplotlib.pyplot as plt
from simsopt import load
from simsopt.mhd import Vmec, Boozer
from simsopt.geo import QfmSurface, SurfaceRZFourier
from simsopt.geo import QfmResidual, Volume, curves_to_vtk
from simsopt.field import particles_to_vtk, compute_fieldlines
import logging
logging.basicConfig()
logger = logging.getLogger('PlotSingleStage')
logger.setLevel(1)
def pprint(*args, **kwargs): print(*args, **kwargs) if comm.rank == 0 else 1

parser = argparse.ArgumentParser()
parser.add_argument("--configuration",default='QA_Stage123_Lengthbound4.0_ncoils4_nfp2')
parser.add_argument("--create_Poincare", dest="create_Poincare", default=False, action="store_true")
parser.add_argument("--create_QFM", dest="create_QFM", default=False, action="store_true")
parser.add_argument("--volume_scale", type=float, default=1.0)
parser.add_argument("--nfieldlines", type=int, default=8)
parser.add_argument("--tmax_fl", type=int, default=400)
parser.add_argument("--nphi_QFM", type=int, default=38)
parser.add_argument("--ntheta_QFM", type=int, default=38)
parser.add_argument("--mpol", type=int, default=18)
parser.add_argument("--ntor", type=int, default=18)
parser.add_argument("--tol_qfm", type=float, default=1e-14)
parser.add_argument("--tol_poincare", type=float, default=1e-14)
parser.add_argument("--maxiter_qfm", type=int, default=1000)
parser.add_argument("--constraint_weight", type=float, default=1e+0)
parser.add_argument("--ntheta_VMEC", type=int, default=300)
parser.add_argument("--boozxform_nsurfaces", type=int, default=10)
parser.add_argument("--bs_file", default='biot_savart_opt.json')
args = parser.parse_args()

helical_detail = False
this_path = os.path.join(parent_path, args.configuration)
OUT_DIR = os.path.join(this_path, "output")
os.chdir(this_path)
bs = load(os.path.join(this_path, f"coils/{args.bs_file}"))

vmec_ran_QFM = False
if args.create_QFM:
    vmec = Vmec(os.path.join(this_path,f'wout_final.nc'))
    s = SurfaceRZFourier.from_wout(os.path.join(this_path,f'wout_final.nc'), nphi=args.nphi_QFM, ntheta=args.ntheta_QFM, range="half period")
    s.change_resolution(args.mpol, args.ntor)
    s_original_VMEC = SurfaceRZFourier.from_wout(os.path.join(this_path,f'wout_final.nc'), nphi=args.nphi_QFM, ntheta=args.ntheta_QFM, range="half period")
    nfp = vmec.wout.nfp
    s.to_vtk(os.path.join(OUT_DIR, 'QFM_original_VMEC'))
    pprint('Obtaining QFM surface')
    bs.set_points(s.gamma().reshape((-1, 3)))
    curves = [coil.curve for coil in bs.coils]
    curves_to_vtk(curves, os.path.join(OUT_DIR, "curves_QFM_test"))
    pointData = {"B_N": np.sum(bs.B().reshape((args.nphi_QFM, args.ntheta_QFM, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    s.to_vtk(os.path.join(OUT_DIR, "surf_QFM_test"), extra_data=pointData)
    # Optimize at fixed volume
    qfm = QfmResidual(s, bs)
    pprint(f"Initial qfm.J()={qfm.J()}")
    vol = Volume(s)
    vol_target = Volume(s).J()*args.volume_scale
    qfm_surface = QfmSurface(bs, s, vol, vol_target)
    t1=time.time()
    pprint(f"Initial ||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
    res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=args.tol_qfm, maxiter=args.maxiter_qfm, constraint_weight=args.constraint_weight)
    pprint(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
    res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=args.tol_qfm, maxiter=args.maxiter_qfm/10)
    pprint(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
    pprint(f"Found QFM surface in {time.time()-t1}s.")
    s.to_vtk(os.path.join(OUT_DIR, 'QFM_found'))
    s_gamma = s.gamma()
    s_R = np.sqrt(s_gamma[:, :, 0]**2 + s_gamma[:, :, 1]**2)
    s_Z = s_gamma[:, :, 2]
    s_gamma_original = s_original_VMEC.gamma()
    s_R_original = np.sqrt(s_gamma_original[:, :, 0]**2 + s_gamma_original[:, :, 1]**2)
    s_Z_original = s_gamma_original[:, :, 2]

    # Plot QFM surface
    fig = plt.figure()
    ax = fig.add_subplot(111,aspect='equal')
    plt.plot(s_R[0,:],s_Z[0,:], label = 'QFM')
    plt.plot(s_R_original[0,:],s_Z_original[0,:], label = 'VMEC')
    plt.xlabel('R')
    plt.ylabel('Z')
    ax.axis('equal')
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, 'QFM_surface.pdf'), bbox_inches = 'tight', pad_inches = 0)

    # Create QFM VMEC equilibrium
    os.chdir(OUT_DIR)
    vmec_QFM = Vmec(os.path.join(this_path,f'input.final'))
    vmec_QFM.indata.mpol = args.mpol
    vmec_QFM.indata.ntor = args.ntor
    vmec_QFM.boundary = s
    vmec_QFM.indata.ns_array[:2]    = [   16,     51]
    vmec_QFM.indata.niter_array[:2] = [ 5000,  10000]
    vmec_QFM.indata.ftol_array[:2]  = [1e-14,  1e-14]
    vmec_QFM.indata.am[0:10] = [0]*10
    vmec_QFM.write_input(os.path.join(this_path,f'input.qfm'))
    vmec_QFM = Vmec(os.path.join(this_path,f'input.qfm'))
    try:
        vmec_QFM.run()
        vmec_ran_QFM = True
    except Exception as e:
        pprint('VMEC QFM did not converge')
        pprint(e)
    try:
        shutil.move(os.path.join(OUT_DIR, f"wout_qfm_000_000000.nc"), os.path.join(this_path, f"wout_qfm.nc"))
        os.remove(os.path.join(OUT_DIR, f'input.qfm_000_000000'))
    except Exception as e:
        print(e)

if vmec_ran_QFM or os.path.isfile(os.path.join(this_path, f"wout_QFM.nc")):
    vmec_QFM = Vmec(os.path.join(this_path,f'wout_QFM.nc'))
    nfp = vmec_QFM.wout.nfp
    sys.path.insert(1, os.path.join(parent_path, '../single_stage/plotting'))
    if vmec_ran_QFM or not os.path.isfile(os.path.join(OUT_DIR, "QFM_VMECparams.pdf")):
        import vmecPlot2
        vmecPlot2.main(file=os.path.join(this_path, f"wout_QFM.nc"), name='QFM', figures_folder=OUT_DIR)
    nzeta=4
    zeta = np.linspace(0,2*np.pi/nfp,num=nzeta,endpoint=False)
    theta = np.linspace(0,2*np.pi,num=args.ntheta_VMEC)
    iradii = np.linspace(0,vmec_QFM.wout.ns-1,num=args.nfieldlines).round()
    iradii = [int(i) for i in iradii]
    R = np.zeros((nzeta,args.nfieldlines,args.ntheta_VMEC))
    Z = np.zeros((nzeta,args.nfieldlines,args.ntheta_VMEC))
    Raxis = np.zeros(nzeta)
    Zaxis = np.zeros(nzeta)
    phis = zeta

    pprint("Obtain VMEC QFM surfaces")
    for itheta in range(args.ntheta_VMEC):
        for izeta in range(nzeta):
            for iradius in range(args.nfieldlines):
                for imode, xnn in enumerate(vmec_QFM.wout.xn):
                    angle = vmec_QFM.wout.xm[imode]*theta[itheta] - xnn*zeta[izeta]
                    R[izeta,iradius,itheta] += vmec_QFM.wout.rmnc[imode, iradii[iradius]]*np.cos(angle)
                    Z[izeta,iradius,itheta] += vmec_QFM.wout.zmns[imode, iradii[iradius]]*np.sin(angle)
    for izeta in range(nzeta):
        for n in range(vmec_QFM.wout.ntor+1):
            angle = -n*nfp*zeta[izeta]
            Raxis[izeta] += vmec_QFM.wout.raxis_cc[n]*np.cos(angle)
            Zaxis[izeta] += vmec_QFM.wout.zaxis_cs[n]*np.sin(angle)

    if vmec_ran_QFM or not os.path.isfile(os.path.join(OUT_DIR,"boozmn_QFM.nc")):
        pprint('Creating Boozer class for vmec_final')
        b1 = Boozer(vmec_QFM, mpol=64, ntor=64)
        pprint('Defining surfaces where to compute Boozer coordinates')
        booz_surfaces = np.linspace(0,1,args.boozxform_nsurfaces,endpoint=False)
        pprint(f' booz_surfaces={booz_surfaces}')
        b1.register(booz_surfaces)
        pprint('Running BOOZ_XFORM')
        b1.run()
        b1.bx.write_boozmn(os.path.join(OUT_DIR,"boozmn_QFM.nc"))
        pprint("Plot BOOZ_XFORM")
        fig = plt.figure(); bx.surfplot(b1.bx, js=1,  fill=False, ncontours=35)
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_1_QFM.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.surfplot(b1.bx, js=int(args.boozxform_nsurfaces/2), fill=False, ncontours=35)
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_2_QFM.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.surfplot(b1.bx, js=args.boozxform_nsurfaces-1, fill=False, ncontours=35)
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_surfplot_3_QFM.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.symplot(b1.bx, helical_detail = helical_detail, sqrts=True)
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_symplot_QFM.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()
        fig = plt.figure(); bx.modeplot(b1.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
        plt.savefig(os.path.join(OUT_DIR, "Boozxform_modeplot_QFM.pdf"), bbox_inches = 'tight', pad_inches = 0); plt.close()

if args.create_Poincare:
    def trace_fieldlines(bfield, R0, Z0):
        t1 = time.time()
        phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
        fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
            bfield, R0, Z0, tmax=args.tmax_fl, tol=args.tol_poincare, comm=comm,
            phis=phis, stopping_criteria=[])
        t2 = time.time()
        pprint(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//args.nfieldlines}", flush=True)
        # if comm is None or comm.rank == 0:
        #     particles_to_vtk(fieldlines_tys, os.path.join(OUT_DIR,f'fieldlines_optimized_coils'))
        return fieldlines_tys, fieldlines_phi_hits, phis

    if vmec_ran_QFM or os.path.isfile(os.path.join(this_path, f"wout_QFM.nc")):
        R0 = R[0,:,0]
        Z0 = Z[0,:,0]
    else:
        pprint('R0 and Z0 not found.')
        exit()
    pprint('Beginning field line tracing')
    fieldlines_tys, fieldlines_phi_hits, phis = trace_fieldlines(bs, R0, Z0)
    pprint('Creating Poincare plot R, Z')
    r = []
    z = []
    for izeta in range(len(phis)):
        r_2D = []
        z_2D = []
        for iradius in range(len(fieldlines_phi_hits)):
            lost = fieldlines_phi_hits[iradius][-1, 1] < 0
            data_this_phi = fieldlines_phi_hits[iradius][np.where(fieldlines_phi_hits[iradius][:, 1] == izeta)[0], :]
            if data_this_phi.size == 0:
                pprint(f'No Poincare data for iradius={iradius} and izeta={izeta}')
                continue
            r_2D.append(np.sqrt(data_this_phi[:, 2]**2+data_this_phi[:, 3]**2))
            z_2D.append(data_this_phi[:, 4])
        r.append(r_2D)
        z.append(z_2D)
    r = np.array(r, dtype=object)
    z = np.array(z, dtype=object)
    pprint('Plotting Poincare plot')
    nrowcol = ceil(sqrt(len(phis)))
    fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(12, 8))
    for i in range(len(phis)):
        row = i//nrowcol
        col = i % nrowcol
        axs[row, col].set_title(f"$\\phi = {phis[i]/np.pi:.3f}\\pi$ ", loc='right', y=0.0)
        axs[row, col].set_xlabel("$R$")
        axs[row, col].set_ylabel("$Z$")
        axs[row, col].set_aspect('equal')
        axs[row, col].tick_params(direction="in")
        for j in range(args.nfieldlines):
            if j== 0 and i == 0:
                legend1 = 'Poincare plot'
                legend2 = 'VMEC QFM'
            else:
                legend1 = legend2 = '_nolegend_'
            try: axs[row, col].scatter(r[i][j], z[i][j], marker='o', s=0.7, linewidths=0, c='b', label = legend1)
            except Exception as e: pprint(e, i, j)
            if vmec_ran_QFM or os.path.isfile(os.path.join(this_path, f"wout_QFM.nc")):
                axs[row, col].scatter(R[i,j], Z[i,j], marker='o', s=0.7, linewidths=0, c='r', label = legend2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'poincare_QFM_fieldline_all.pdf'), bbox_inches = 'tight', pad_inches = 0)



#!/usr/bin/env python3
import os
import numpy as np
from coilpy import Coil
from simsopt import load
from simsopt.mhd import Vmec, VirtualCasing
from simsopt.util import MpiPartition
mpi = MpiPartition()

# dir = 'optimization_CNT'
dir = 'optimization_CNT_circular'
whole_torus = True
stage1 = False

# dir = 'optimization_QH'
# whole_torus = False
# stage1 = True

coils_stage1 = "biot_savart_inner_loop_max_mode_3.json"

nphi = 256
ntheta = 128
ncoils = 4

finite_beta = True

if finite_beta: dir += '_finitebeta'
filename_final = dir+'/input.final'
filename_stage1 = dir+'/input.stage1'
outdir = dir+'/coils/'

def coilpy_plot(curves, filename, height=0.1, width=0.1):
    def wrap(data):
        return np.concatenate([data, [data[0]]])
    xx = [wrap(c.gamma()[:, 0]) for c in curves]
    yy = [wrap(c.gamma()[:, 1]) for c in curves]
    zz = [wrap(c.gamma()[:, 2]) for c in curves]
    II = [1. for _ in curves]
    names = [i for i in range(len(curves))]
    coils = Coil(xx, yy, zz, II, names, names)
    coils.toVTK(filename, line=False, height=height, width=width)

if whole_torus: vmec_final = Vmec(filename_final, mpi=mpi, verbose=True, nphi=nphi, ntheta=ntheta)
else: vmec_final = Vmec(filename_final, mpi=mpi, verbose=True, nphi=nphi, ntheta=ntheta, range_surface='half period')
vmec_final.indata.ns_array[:3]    = [  16,     51,   101]
vmec_final.indata.niter_array[:3] = [ 4000,  6000, 10000]
vmec_final.indata.ftol_array[:3]  = [1e-12, 1e-13, 1e-16]
s_final = vmec_final.boundary
vc_src_nphi = int(nphi/2/vmec_final.indata.nfp) if whole_torus else nphi
if finite_beta: vc_final = VirtualCasing.from_vmec(vmec_final, src_nphi=vc_src_nphi, src_ntheta=ntheta)

bs_final = load(outdir + "biot_savart_opt.json")
B_on_surface_final = bs_final.set_points(s_final.gamma().reshape((-1, 3))).AbsB()
norm_final = np.linalg.norm(s_final.normal().reshape((-1, 3)), axis=1)
meanb_final = np.mean(B_on_surface_final * norm_final)/np.mean(norm_final)
absb_final = bs_final.AbsB().reshape(s_final.gamma().shape[:2] + (1,))
Bbs = bs_final.B().reshape((nphi, ntheta, 3))
if finite_beta:
    if whole_torus: BdotN_surf = np.sum(Bbs * s_final.unitnormal(), axis=2) - vc_final.B_external_normal_extended
    else: BdotN_surf = np.sum(Bbs * s_final.unitnormal(), axis=2) - vc_final.B_external_normal
else:
    BdotN_surf = np.sum(Bbs * s_final.unitnormal(), axis=2)
pointData_final = {"B·n/|B|": BdotN_surf[:, :, None]/absb_final,
             "|B|": bs_final.AbsB().reshape(s_final.gamma().shape[:2] + (1,))/meanb_final}
if whole_torus: coilpy_plot([c.curve for c in bs_final.coils], outdir + "coils_optPlot.vtu", height=0.05, width=0.05)
else: coilpy_plot([c.curve for c in bs_final.coils[0:ncoils]], outdir + "coils_optPlot.vtu", height=0.05, width=0.05)
s_final.to_vtk(outdir + "surf_optPlot", extra_data=pointData_final)

if stage1:
    if whole_torus: vmec_stage1 = Vmec(filename_stage1, mpi=mpi, verbose=True, nphi=nphi, ntheta=ntheta)
    else: vmec_stage1 = Vmec(filename_stage1, mpi=mpi, verbose=True, nphi=nphi, ntheta=ntheta, range_surface='half period')
    vmec_stage1.indata.ns_array[:3]    = [  16,     51,   101]
    vmec_stage1.indata.niter_array[:3] = [ 4000,  6000, 10000]
    vmec_stage1.indata.ftol_array[:3]  = [1e-12, 1e-13, 1e-16]
    s_stage1 = vmec_stage1.boundary
    if finite_beta: vc_stage1 = VirtualCasing.from_vmec(vmec_stage1, src_nphi=vc_src_nphi, src_ntheta=ntheta)

    bs_stage1 = load(outdir + coils_stage1)
    B_on_surface_stage1 = bs_stage1.set_points(s_stage1.gamma().reshape((-1, 3))).AbsB()
    norm_stage1 = np.linalg.norm(s_stage1.normal().reshape((-1, 3)), axis=1)
    meanb_stage1 = np.mean(B_on_surface_stage1 * norm_stage1)/np.mean(norm_stage1)
    absb_stage1 = bs_stage1.AbsB().reshape(s_stage1.gamma().shape[:2] + (1,))
    Bbs = bs_stage1.B().reshape((nphi, ntheta, 3))
    if finite_beta:
        if whole_torus: BdotN_surf = np.sum(Bbs * s_stage1.unitnormal(), axis=2) - vc_stage1.B_external_normal_extended
        else: BdotN_surf = np.sum(Bbs * s_stage1.unitnormal(), axis=2) - vc_stage1.B_external_normal
    else:
        BdotN_surf = np.sum(Bbs * s_stage1.unitnormal(), axis=2)
    pointData_stage1 = {"B·n/|B|": BdotN_surf[:, :, None]/absb_stage1,
                "|B|": bs_stage1.AbsB().reshape(s_stage1.gamma().shape[:2] + (1,))/meanb_stage1}
    if whole_torus: coilpy_plot([c.curve for c in bs_stage1.coils], outdir + "coils_stage1Plot.vtu", height=0.05, width=0.05)
    else: coilpy_plot([c.curve for c in bs_stage1.coils[0:ncoils]], outdir + "coils_stage1Plot.vtu", height=0.05, width=0.05)
    s_stage1.to_vtk(outdir + "surf_stage1Plot", extra_data=pointData_stage1)

files_to_remove = ['input.final_000_000000','input.stage1_000_000000','parvmecinfo.txt','threed1.final','threed1.stage1',
                   'vcasing_final_000_000000.nc','vcasing_stage1_000_000000.nc','wout_final_000_000000.nc','wout_stage1_000_000000.nc']
for file in files_to_remove:
    try: os.remove(file)
    except Exception as e: print(e)

print(f"Created coils_optPlot.vtu, surf_optPlot.vts, coils_stage1Plot.vtu and surf_stage1Plot.vts in directory {outdir}")

