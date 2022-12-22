#!/usr/bin/env python

import os
import sys
import time
import shutil
import numpy as np
import booz_xform as bx
from math import ceil, sqrt
import matplotlib.pyplot as plt
from opt_funcs import plot_qfm_poincare
from simsopt import load
from simsopt.mhd import Vmec, Boozer
from simsopt.geo import SurfaceRZFourier
from simsopt.field import InterpolatedField
from simsopt.geo import QfmSurface, QfmResidual, Volume
from simsopt.field import SurfaceClassifier, \
    particles_to_vtk, compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

def main(folder, OUT_DIR='figures', coils_folder='coils', vmec_folder='vmec', mpi=None,
    fieldlines_phi_hits=None, qfm_poincare_plot=False, nzeta = 8, nradius = 3,
    tol_qfm = 1e-14, maxiter_qfm = 1000, vmec_input_start = '', name_manual='',
    constraint_weight = 1e-0, ntheta = 300, mpol_qfm = 16, ntor_qfm = 16, create_QFM=True,
    nphi_qfm = 25, ntheta_qfm = 40, tmax_fl = 1000, degree = 4, tol_tracing = 1e-12, diagnose_QFM=True, savefig=True):
    if mpi==None:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        except ImportError:
            comm = None
    else:
       comm = mpi.comm_world

    def pprint(*args, **kwargs):
        if comm == None or comm.rank == 0:
            print(*args, **kwargs)

    pprint("Start field_from_coils.py")
    pprint("=======================")

    if name_manual=='':
        name=os.path.basename(os.path.normpath(folder))
    else:
        name=name_manual

    ## Check paths and files

    pprint('Reading Biot-Savart file:')
    pprint(os.path.join(folder, coils_folder, 'biot_savart_opt.json'))
    biot_savart_file = os.path.join(folder, coils_folder, 'biot_savart_opt.json')
    bs = load(biot_savart_file)

    os.chdir(vmec_folder)
    if os.path.isfile(os.path.join(folder,f'wout_{name}_qfm.nc')) and not create_QFM:
        # Load vmec output with the qfm surface
        pprint(f'Reading QFM surfaces from vmec output file wout_{name}_qfm.nc')
        vmec = Vmec(os.path.join(folder,f'wout_{name}_qfm.nc'), mpi=mpi)
        s = SurfaceRZFourier.from_wout(os.path.join(folder,f'wout_{name}_qfm.nc'), nphi=nphi_qfm, ntheta=ntheta_qfm, range="half period")
        nfp = vmec.wout.nfp
        os.chdir(folder)
    else:
        # Find qfm surface closest to vmec input
        pprint('Reading from vmec inputfile:')
        if vmec_input_start == '':
            pprint(os.path.join(folder, f'input.{name}_final'))
            original_vmec_input = os.path.join(folder, f'input.{name}_final')
        else:
            pprint(vmec_input_start)
            original_vmec_input = vmec_input_start
        vmec = Vmec(original_vmec_input, mpi=mpi)
        nfp = vmec.indata.nfp
        s = SurfaceRZFourier.from_vmec_input(original_vmec_input, nphi=nphi_qfm, ntheta=ntheta_qfm, range="half period")
        s_original = SurfaceRZFourier.from_vmec_input(original_vmec_input, nphi=nphi_qfm, ntheta=ntheta_qfm, range="half period")
        s.change_resolution(mpol_qfm, ntor_qfm)
        os.chdir(folder)

        if comm is None or comm.rank == 0:
            # Optimize at fixed volume
            qfm = QfmResidual(s, bs)
            pprint(f"Initial qfm.J()={qfm.J()}")
            vol = Volume(s)
            vol_target = vol.J()
            qfm_surface = QfmSurface(bs, s, vol, vol_target)
            t1 = time.time()
            pprint(f"Initial ||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
            res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=tol_qfm, maxiter=maxiter_qfm, constraint_weight=constraint_weight)
            pprint(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
            res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=tol_qfm, maxiter=maxiter_qfm)
            pprint(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")
            t2 = time.time()
            pprint(f"Time for finding QFM={t2-t1:.3f}s.")

            s_gamma = s.gamma()
            s_R = np.sqrt(s_gamma[:, :, 0]**2 + s_gamma[:, :, 1]**2)
            s_Z = s_gamma[:, :, 2]

            s_gamma_original = s_original.gamma()
            s_R_original = np.sqrt(s_gamma_original[:, :, 0]**2 + s_gamma_original[:, :, 1]**2)
            s_Z_original = s_gamma_original[:, :, 2]

            # Plot difference between original VMEC boundary and QFM surface
            fig = plt.figure()
            ax = fig.add_subplot(111,aspect='equal')
            plt.plot(s_R[0,:],s_Z[0,:], label = 'New QFM')
            plt.plot(s_R_original[0,:],s_Z_original[0,:], label = 'Original VMEC')
            plt.xlabel('R')
            plt.ylabel('Z')
            ax.axis('equal')
            plt.legend()
            if savefig: plt.savefig(os.path.join(OUT_DIR, name+'_VMECboundary_vs_QFM.pdf'), bbox_inches = 'tight', pad_inches = 0)

        os.chdir(vmec_folder)
        vmec.boundary = s
        vmec.indata.ns_array[:3]    = [  16,    51,    101]
        vmec.indata.niter_array[:3] = [ 2000,  3000,  8000]
        vmec.indata.ftol_array[:3]  = [1e-14, 1e-14, 1e-14]
        os.chdir(folder)
        vmec.write_input(f'input.{name}_qfm')
        vmec = Vmec(f'input.{name}_qfm', mpi=mpi)
        os.chdir(vmec_folder)
        os.chdir(folder)
        try:
            vmec.run()
        except Exception as e:
            pprint('VMEC QFM did not converge')
            return 0, 0, 0, 0, vmec
        if comm is None or comm.rank == 0:
            try:
                shutil.move(f"wout_{name}_qfm_000_000000.nc", f"wout_{name}_qfm.nc")
                os.remove(f'input.{name}_qfm_000_000000')
            except Exception as e:
                print(e)
        os.chdir(folder)
    
    if diagnose_QFM:
        boozxform_nsurfaces = 14
        sys.path.insert(1, os.path.join(folder, '../plotting'))
        import vmecPlot2
        figures_results_path = OUT_DIR
        vmecPlot2.main(file=f"wout_{name}_qfm.nc", name=name+'_qfm', figures_folder=figures_results_path)

        pprint('Creating Boozer class for vmec_qfm')
        b1_qfm = Boozer(vmec, mpol=64, ntor=64)
        pprint('Defining surfaces where to compute Boozer coordinates')
        booz_surfaces_qfm = np.linspace(0,1,boozxform_nsurfaces,endpoint=False)
        pprint(f' booz_surfaces={booz_surfaces_qfm}')
        b1_qfm.register(booz_surfaces_qfm)
        pprint('Running BOOZ_XFORM')
        b1_qfm.run()
        if comm is None or comm.rank == 0:
            b1_qfm.bx.write_boozmn(os.path.join('vmec',"boozmn_"+name+"_qfm.nc"))
            pprint("Plot BOOZ_XFORM")
            fig = plt.figure(); bx.surfplot(b1_qfm.bx, js=1,  fill=False, ncontours=35)
            plt.savefig(os.path.join(figures_results_path, "Boozxform_surfplot_1_"+name+'_qfm.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.surfplot(b1_qfm.bx, js=int(boozxform_nsurfaces/2), fill=False, ncontours=35)
            plt.savefig(os.path.join(figures_results_path, "Boozxform_surfplot_2_"+name+'_qfm.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.surfplot(b1_qfm.bx, js=boozxform_nsurfaces-1, fill=False, ncontours=35)
            plt.savefig(os.path.join(figures_results_path, "Boozxform_surfplot_3_"+name+'_qfm.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
            if name[0:2] == 'QH':
                helical_detail = True
            else:
                helical_detail = False
            fig = plt.figure(); bx.symplot(b1_qfm.bx, helical_detail = helical_detail, sqrts=True)
            plt.savefig(os.path.join(figures_results_path, "Boozxform_symplot_"+name+'_qfm.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()
            fig = plt.figure(); bx.modeplot(b1_qfm.bx, sqrts=True); plt.xlabel(r'$s=\psi/\psi_b$')
            plt.savefig(os.path.join(figures_results_path, "Boozxform_modeplot_"+name+'_qfm.pdf'), bbox_inches = 'tight', pad_inches = 0); plt.close()

    ## Tracing
    s.to_vtk(os.path.join(OUT_DIR, f'{name}_surface'))
    # sc_fieldline = SurfaceClassifier(s, h=0.03, p=2)
    # sc_fieldline.to_vtk(os.path.join(OUT_DIR, f'{name}_levelset'), h=0.03)

    # Initialize arrays
    zeta = np.linspace(0,2*np.pi/nfp,num=nzeta,endpoint=False)
    nfieldlines = nradius
    theta = np.linspace(0,2*np.pi,num=ntheta)
    iradii = np.linspace(0,vmec.wout.ns-1,num=nradius).round()
    iradii = [int(i) for i in iradii]
    R = np.zeros((nzeta,nradius,ntheta))
    Z = np.zeros((nzeta,nradius,ntheta))
    Raxis = np.zeros(nzeta)
    Zaxis = np.zeros(nzeta)
    phis = zeta

    ## Obtain VMEC QFM surfaces
    for itheta in range(ntheta):
        for izeta in range(nzeta):
            for iradius in range(nradius):
                for imode, xnn in enumerate(vmec.wout.xn):
                    angle = vmec.wout.xm[imode]*theta[itheta] - xnn*zeta[izeta]
                    R[izeta,iradius,itheta] += vmec.wout.rmnc[imode, iradii[iradius]]*np.cos(angle)
                    Z[izeta,iradius,itheta] += vmec.wout.zmns[imode, iradii[iradius]]*np.sin(angle)
    for izeta in range(nzeta):
        for n in range(vmec.wout.ntor+1):
            angle = -n*nfp*zeta[izeta]
            Raxis[izeta] += vmec.wout.raxis_cc[n]*np.cos(angle)
            Zaxis[izeta] += vmec.wout.zaxis_cs[n]*np.sin(angle)

    if qfm_poincare_plot:
        ## Field line tracing function
        def trace_fieldlines(bfield, label):
            t1 = time.time()
            R0 = R[0,:,0]
            Z0 = Z[0,:,0]
            pprint(f'Initial radii for field line tracer: {R0}')
            pprint(f'Starting particle tracer')
            phis = zeta
            fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
                bfield, R0, Z0, tmax=tmax_fl, tol=tol_tracing, comm=comm,
                phis=phis, stopping_criteria=[
                    # LevelsetStoppingCriterion(sc_fieldline.dist)
                ])
            t2 = time.time()
            pprint(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
            if comm is None or comm.rank == 0:
                # particles_to_vtk(fieldlines_tys, os.path.join(OUT_DIR, f'{name}_fieldlines_{label}'))
                plot_poincare_data(fieldlines_phi_hits, phis, os.path.join(OUT_DIR, f'{name}_poincare_fieldline_{label}.png'), dpi=150)
            return fieldlines_tys, fieldlines_phi_hits, phis
        # n = 20
        # rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
        # zs = s.gamma()[:, :, 2]
        # rrange = (0.9*np.min(rs), 1.1*np.max(rs), n)
        # phirange = (0, 2*np.pi/nfp, n*2)
        # zrange = (0, 1.1*np.max(zs), n//2)
        # def skip(rs, phis, zs):
        #     rphiz = np.asarray([rs, phis, zs]).T.copy()
        #     dists = sc_fieldline.evaluate_rphiz(rphiz)
        #     skip = list((dists < -0.05).flatten())
        #     pprint("Skip", sum(skip), "cells out of", len(skip), flush=True)
        #     return skip
        # pprint('Initializing InterpolatedField')
        # bsh = InterpolatedField(
        #     bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True, skip=skip
        # )
        # pprint('Done initializing InterpolatedField.')
        # bsh.set_points(s.gamma().reshape((-1, 3)))
        # bs.set_points(s.gamma().reshape((-1, 3)))
        # Bh = bsh.B()
        # B = bs.B()
        # pprint("Mean(|B|) on plasma surface =", np.mean(bs.AbsB()))
        # pprint("|B-Bh| on surface:", np.sort(np.abs(B-Bh).flatten()))
        pprint('Beginning field line tracing')
        # fieldlines_tys, fieldlines_phi_hits, phis = trace_fieldlines(bsh, 'bsh')
        ## Not using an interpolated field takes much longer
        fieldlines_tys, fieldlines_phi_hits, phis = trace_fieldlines(bs, 'bs')
        if fieldlines_phi_hits is not None:
            if comm is None or comm.rank == 0:
                pprint('Plotting magnetic surfaces and Poincare plots')
                plot_qfm_poincare(phis=phis, fieldlines_phi_hits=fieldlines_phi_hits, R=R, Z=Z, OUT_DIR=OUT_DIR, name=name)

    pprint("End field_from_coils.py")
    pprint("=======================")

    return R, Z, Raxis, Zaxis, vmec

if __name__ == "__main__":
    # Create results folders if not present
    try:
        Path(sys.argv[2]).mkdir(parents=True, exist_ok=True)
        figures_results_path = str(Path(sys.argv[2]).resolve())
        main(folder = sys.argv[1], OUT_DIR = sys.argv[2], qfm_poincare_plot=True)
    except:
        try:
            main(folder = sys.argv[1], qfm_poincare_plot=True)
        except:
            main(folder = os.getcwd(), qfm_poincare_plot=True)