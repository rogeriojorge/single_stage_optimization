#! /usr/bin/env python3

#Set of routines primarily aimed at calculating the
#second abiabatic moment J// from a (phi,alpha) map of
#the |B| in any stellarator equilibrium.
#The J// will depend on the values of lambda=Eperp/Etotal,
#the iota and the |B| at the flux surface where we want
#to perform the calculation.
#
#Author: R. Coelho
#License: TBD
#Release Date: 14/04/2023

import numpy as np
import math
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.optimize import fsolve
import booz_xform as bx

def plot_Btot_surf_xform(b1,ntheta,nphi,js0):
  #Plotar
  bx.surfplot(b1, ntheta=ntheta,nphi=nphi,js=js0,ncontours=30)
  plt.savefig("ModB_surfplot_routine_js="+str(js0)+".eps")
  plt.show()
  bx.symplot(b1, log=False, sqrts=True)
  plt.show()

  #Ler ficheiro de output
  #b2 = bx.Booz_xform()
  #b2.read_boozmn("boozmn.nc")
  return
###################################################
###################################################
def Btot_theta_phi(b1,ntheta,nphi,js0):
  #Como calcular o módulo de B em js=js0
  #Para um conjunto de thetas e phis, podemos encontrar o array 2D em meshgrid de theta e phi
  theta1d = np.linspace(0, 2 * np.pi, ntheta)
  phi1d = np.linspace(0, 2 * np.pi / b1.nfp, nphi)
  phi, theta = np.meshgrid(phi1d, theta1d)
  modB = np.zeros((ntheta,nphi))
  #Para cada theta e phi, escolhendo o índice em psi como js
  for jmn in range(len(b1.xm_b)):
    m = b1.xm_b[jmn]
    n = b1.xn_b[jmn]
    angle = m * theta - n * phi
    modB += b1.bmnc_b[jmn, js0] * np.cos(angle)
    #(Caso seja stellarator asymmetric, raramente)
    if b1.asym:
      modB += b1.bmns_b[jmn, js0] * np.sin(angle)
  return [theta,phi,modB]

def plot_Btot_theta_phi(b1,modB,theta,phi,js0):
  #Para plotar
  plt.contourf(phi, theta, modB,levels=30)
  plt.title("|B| [T] on surface s="+str("{:.3f}".format(b1.s_b[js0])))
  plt.xlabel(r"Boozer toroidal angle $\phi$ [rad]")
  plt.ylabel(r"Boozer poloidal angle $\theta$  [rad]")
  plt.colorbar()
  plt.savefig("ModB_surfplot_mine_js="+str(js0)+".eps")
  plt.show()
  return
###################################################
###################################################
def Btot_alpha_phi(b1,ntheta,nphi,js0,phi_max,interpolators,interpolator_iota): #js0 is now the value of s (surface)
  #Now let's make the |B|(s,alpha,phiB) equivalent......
  iota=interpolator_iota(js0)
  alpha=np.linspace(0, 2 * np.pi, ntheta)
  phi1d_scaled = np.linspace(0, phi_max, nphi)
  modB_alpha_phi = np.zeros((ntheta,nphi))
  #Para cada theta e phi, escolhendo o índice em psi como js
  for ida in range(len(alpha)):
    modB_phi=np.zeros(nphi)
    for jmn in range(len(b1.xm_b)):
      m = b1.xm_b[jmn]
      n = b1.xn_b[jmn]
      angle = m * alpha[ida] + (m*iota - n) * phi1d_scaled
      #modB_phi += b1.bmnc_b[jmn, js0] * np.cos(angle)
      modB_phi += interpolators[jmn](js0) * np.cos(angle)
      #(Caso seja stellarator asymmetric, raramente)
      if b1.asym:
        modB_phi += b1.bmns_b[jmn, js0] * np.sin(angle)
    modB_alpha_phi[ida,:]=modB_phi
  return [alpha,phi1d_scaled,modB_alpha_phi]

def plot_Btot_alpha_phi(b1,modB_alpha_phi,alpha,phi1d_scaled,js0):
  #Para plotar
  plt.contourf(phi1d_scaled, alpha, modB_alpha_phi, levels=30)
  plt.title("|B| in [T] on surface s="+str("{:.3f}".format(b1.s_b[js0])))
  plt.xlabel(r"Boozer toroidal angle $\phi$ [rad]")
  plt.ylabel(r"Boozer poloidal angle $\alpha$  [rad]")
  plt.colorbar()
  plt.savefig("ModB_surfplot_alpha_phi_js="+str(js0)+".eps")
  plt.show()
  return
###################################################
###################################################
def Btot_js0_alpha(b1,phi,js0,iota,alpha,interpolators):
  modB_phi=0
  for jmn in range(len(b1.xm_b)):
    m = b1.xm_b[jmn]
    n = b1.xn_b[jmn]
    angle = m * alpha + (m*iota - n) * phi
    #modB_phi += b1.bmnc_b[jmn, js0] * np.cos(angle)
    modB_phi += interpolators[jmn](js0) * np.cos(angle)
    #(Caso seja stellarator asymmetric, raramente)
    if b1.asym:
      modB_phi += b1.bmns_b[jmn, js0] * np.sin(angle)
  return modB_phi

def find_turning_points(b1,lam,js0,iota,alpha,phi_max,pppi,interpolators):
  phi_list_of_tuples=[]
  #Use large enough range in phi to find rough solutions
  npoints=int(pppi/np.pi*phi_max) #pppi: points per pi rad
  phi1d = np.linspace(0, phi_max, npoints)
  modB=np.array([Btot_js0_alpha(b1,phi,js0,iota,alpha,interpolators) for phi in phi1d])
  #Now define the argument of the square root for Vpar
  tmp=1-lam*modB
  #Find the indices with sign crossings in tmp array. The zero lies in between idx and idx+1
  zero_crossings = np.where(np.diff(np.sign(tmp)))[0]
  for idx,value in enumerate(zero_crossings):
    try:
      if ( (tmp[zero_crossings[idx]] < 0) and (tmp[zero_crossings[idx+1]] > 0) ):
        phi_list_of_tuples.append((phi1d[zero_crossings[idx]],phi1d[zero_crossings[idx+1]]))
    except:
      pass
      #print("no more matching turning point pairs")
  return phi_list_of_tuples

def refine_turning_points(b1,turning_points_input,lam,js0,iota,alpha,interpolators):
  phi_list_of_tuples=[]
  def f(phi,b1,lam,js0,iota,alpha,interpolators):
    modB=Btot_js0_alpha(b1,phi,js0,iota,alpha,interpolators)
    tmp=1-lam*modB
    return tmp
  for idx in range(len(turning_points_input)):
    phi_list_of_tuples.append( fsolve(f, 
                    [turning_points_input[idx][0],turning_points_input[idx][1]],
                    args=(b1,lam,js0,iota,alpha,interpolators,)) )
  return phi_list_of_tuples

def Vpar_div_B(phi,b1,lam,js0,iota,alpha,interpolators):
  modB=Btot_js0_alpha(b1,phi,js0,iota,alpha,interpolators)
  vpa=np.sqrt(1-lam*modB)/modB
  return vpa

def int_phi_Vpar(b1,phi_min, phi_max,lam,js0,iota,alpha,interpolators):
  tmp=quad(Vpar_div_B, phi_min, phi_max, args=(b1,lam,js0,iota,alpha,interpolators,))
  return tmp
