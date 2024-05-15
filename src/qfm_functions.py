import os
import numpy as np
from math import ceil, sqrt
import matplotlib.pyplot as plt

# Define a print function that only prints on one processor
from simsopt.util import proc0_print as pprint
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

def plot_qfm_poincare(phis, fieldlines_phi_hits, R, Z, OUT_DIR, name):
    if comm is None or comm.rank == 0:
        nradius = len(fieldlines_phi_hits)
        r = []
        z = []
        # Obtain Poincare plot
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

        # Plot figure
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
            for j in range(nradius):
                if j== 0 and i == 0:
                    legend1 = 'Poincare plot'
                    legend2 = 'VMEC QFM'
                else:
                    legend1 = legend2 = '_nolegend_'
                try: axs[row, col].scatter(r[i][j], z[i][j], marker='o', s=0.7, linewidths=0, c='b', label = legend1)
                except Exception as e: pprint(e, i, j)
                axs[row, col].scatter(R[i,j], Z[i,j], marker='o', s=0.7, linewidths=0, c='r', label = legend2)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'{name}_poincare_VMEC_fieldline_all.pdf'), bbox_inches = 'tight', pad_inches = 0)

        fig, axs = plt.subplots(1, 1, figsize=(12, 8))
        axs.set_title(f"$\\phi = {phis[0]/np.pi:.3f}\\pi$ ", loc='right', y=0.0)
        axs.set_xlabel("$Z$")
        axs.set_ylabel("$R$")
        axs.set_aspect('equal')
        axs.tick_params(direction="in")
        for j in range(nradius):
            if j== 0:
                legend1 = 'Poincare plot'
                legend2 = 'VMEC QFM'
            else:
                legend1 = legend2 = '_nolegend_'
            try: axs.scatter(r[0][j], z[0][j], marker='o', s=1.5, linewidths=0.5, c='b', label = legend1)
            except Exception as e: pprint(e, 0, j)
            axs.scatter(R[0,j], Z[0,j], marker='o', s=1.5, linewidths=0.5, c='r', label = legend2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'{name}_poincare_VMEC_fieldline_0.pdf'), bbox_inches = 'tight', pad_inches = 0)
