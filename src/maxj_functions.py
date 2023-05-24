import numpy as np
import time
import math
from simsopt.mhd import Boozer
#import booz_xform as bx
from .maxj_src import find_turning_points,refine_turning_points
from .maxj_src import int_phi_Vpar,Btot_alpha_phi
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

# Returns the variation of J|| with the distance to the axis
def maxJ_Residual(vmec, snorms, nlambda = 8,
                nphi=64, ntheta=256,
                mpol=64, try_alpha=np.pi/2,
                ntor=48, phi_max=10*np.pi,
                pppi=64, eps=10 ** -4, flatten=False):
    """
    Calculate the variation of J|| with the distance to the magnetic axis
    vmec        -   VMEC object
    snorms      -   Flux surfaces at which the derivative will be calculated
    nlambda     -   Number of different values of lambda to test
    nphi        -   Resolution in the phi angle to build the 2D grid in theta and phi angle
    ntheta      -   Resolution in the theta angle to build the 2D grid in theta and phi angle              
    mpol        -   Poloidal modes in Boozer transformation
                    Should be ~20 for low VMEC Fourier modes, ~40 for more VMEC Fourier modes
    ntor        -   Toroidal modes in Boozer transformation
                    Should be ~20 for low VMEC Fourier modes, ~40 for more VMEC Fourier modes
    try_alpha   -   Value of the boozer mixed angle alpha for the calculation of J||
    phi_max     -   Maximum value of phi angle for obtaining turning points (v|| = 0)
    pppi        -   Points per pi rad 
    eps         -   Difference between points for the calculation of the derivative
    """
    ################################################################
    ############################ SETUP #############################
    ################################################################
    print("Entrei na função.")
    t1 = time.time()
    vmec.run()
 
    ns = vmec.wout.ns - 1

    surfaces = []
    for i in range(ns):
        surfaces.append((1/(2*ns)) + (i/ns)) #Ver se aqui é mesmo 0.005 ou se secalhar é melhor 1/(2ns).

    surfaces_np = np.array(surfaces)

    b1 = Boozer(vmec,mpol,ntor,verbose=False)
    b1.register(surfaces)
    b1.run()

    interpolators = [UnivariateSpline(surfaces_np,b1.bx.bmnc_b[i,:]) for i in range(len(b1.bx.bmnc_b))]
    interpolator_iota = UnivariateSpline(surfaces_np,b1.bx.iota)

    J_psi_surfaces = [] #hosts the results of the minimum of the derivative for each surface and corresponding lambda and surface
 
    #Calculation of J|| for the different flux surfaces
    for js0 in snorms:
        #print("Normalised toroidal flux s="+str("{:.3f}".format(b1.s_b[js0])))
        #print("Iota for flux surface iota="+str("{:.3f}".format(b1.iota[js0])))

        iota=interpolator_iota(js0)
        iota_dif=interpolator_iota(js0 + eps)

        #Obtaining the maximum and minimum values of |B| to set the limits for lambda
        modB = Btot_alpha_phi(b1.bx,ntheta,nphi,js0,phi_max,interpolators,interpolator_iota)[2]
        Bmax = np.max(modB)
        Bmin = np.min(modB)

        lam_interval = ((1/Bmin) - (1/Bmax))/(nlambda + 1)
        lamb = []

        for i in range(nlambda):
            lamb.append((1/Bmax) + (i+1)*lam_interval)

        J_psi = [] #hosts the results of the derivative for different lambdas
        J_psi_lam = [] #hosts the results of the derivative and the corresponding lambda

        #Calculations of J|| for the different lambdas
        for lam in lamb:

            #Coarse calculation of the turning points (careful if they are very close to each other...)
            #print("Calculating the turning points for lam="+str(lam))
            turning_points_set=find_turning_points(b1.bx,lam,js0,iota,try_alpha,phi_max,pppi,interpolators)
            turning_points_set_dif=find_turning_points(b1.bx,lam,(js0 + eps),iota_dif,try_alpha,phi_max,pppi,interpolators)

            
            #Now refine the calculation.... 
            #print("Refining the turning points for lam="+str(lam))
            turning_points_set_refine=refine_turning_points(b1.bx,turning_points_set,lam,js0,iota,try_alpha,interpolators)
            turning_points_set_refine_dif=refine_turning_points(b1.bx,turning_points_set_dif,lam,(js0 + eps),iota_dif,try_alpha,interpolators)

            #Now let's fill in the list of J// calculations for each "well"
            Jpar = [] #hosts value from the integration
            Jpar_dif = [] #hosts value from the integration of the very close next surface
            for idx in range(len(turning_points_set_refine)):
                intermediate = int_phi_Vpar(b1.bx,turning_points_set_refine[idx][0], 
                                            turning_points_set_refine[idx][1],
                                            lam,js0,iota,try_alpha,interpolators)[0]
                if not np.isnan(intermediate):
                    Jpar.append(intermediate)


            for idx in range(len(turning_points_set_refine_dif)):
                intermediate = int_phi_Vpar(b1.bx,turning_points_set_refine_dif[idx][0], 
                                            turning_points_set_refine_dif[idx][1],
                                            lam,(js0 + eps),iota_dif,try_alpha,interpolators)[0]
                if not np.isnan(intermediate):
                    Jpar_dif.append(intermediate)

            if len(Jpar)!=0:
                av_Jpar = sum(Jpar)/len(Jpar)
            else:
                av_Jpar = 0

            if len(Jpar_dif)!=0 and av_Jpar!=0:
                av_Jpar_dif = sum(Jpar_dif)/len(Jpar_dif)
            else:
                av_Jpar_dif = 0

            if av_Jpar!=0 and av_Jpar_dif!=0:
                dj_dpsi = (av_Jpar_dif - av_Jpar)/eps
                J_psi.append(dj_dpsi)
                J_psi_lam.append([dj_dpsi, lam, js0])

        #min_derivative = min(J_psi)
        #index = J_psi.index(min_derivative)
        #J_psi_surfaces.append([min_derivative, J_psi_lam[index][1], J_psi_lam[index][2]]) 
        J_psi_surfaces.append(J_psi)

    if len(J_psi_surfaces) == 0:
        J_psi_surfaces.append(100)

    if flatten:
        J_psi_surfaces = np.array(J_psi_surfaces).flatten()

    t2 = time.time()
    print("Saí da função")
    print("J_psi_surfaces =", J_psi_surfaces)
    print("Tempo:", t2-t1, "segundos.")

    return J_psi_surfaces
        

            
    
    