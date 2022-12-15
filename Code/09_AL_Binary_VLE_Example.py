# -*- coding: utf-8 -*-
"""
Python script to test AL on binary activity coefficient data by computing
VLE phase diagrams.

Sections:
    . Imports
    . Configuration
    . Ground Truth VLE
    . Active Learning VLE
    . Publication Plots

Last edit: 2022-10-27
Author: Dinis Abranches
"""

# =============================================================================
# Imports
# =============================================================================

# Specific
import numpy
from matplotlib import pyplot as plt

# Local
from lib import thermoAux as thermo
from lib import mlAux as ml

# =============================================================================
# Configuration
# =============================================================================

# System definition
components=['Acetone','Cyclohexane']

# GP Configuration
gpConfig={'kernel':'RQ',
          'useWhiteKernel':False,
          'doLogY':True,
          'indepDim':False,
          'trainLikelihood':False
          }

# VLE Pressure
P=101325

# =============================================================================
# Ground Truth VLE
# =============================================================================

# Get NRTL parameters
parameters=thermo.NRTL_Parameters(components[0],components[1])
# Define gamma functions
def F_Gamma_1_NRTL(x1,T):
    gammas=thermo.NRTL_Binary_getGamma(parameters,x1,T)
    return gammas[0]
def F_Gamma_2_NRTL(x1,T):
    gammas=thermo.NRTL_Binary_getGamma(parameters,x1,T)
    return gammas[1]
Fs_gamma_NRTL=[F_Gamma_1_NRTL,F_Gamma_2_NRTL]
# Get Antoine parameters
antoine1=thermo.antoineParameters(components[0])
antoine2=thermo.antoineParameters(components[1])
# Define Vapor Pressure Functions
def F_VP_1(T):
    P=thermo.antoineEquation(antoine1,T,getVar='P')
    return P
def F_VP_2(T):
    P=thermo.antoineEquation(antoine2,T,getVar='P')
    return P  
def F_Inverse_VP_1(P):
    T=thermo.antoineEquation(antoine1,P,getVar='T')
    return T
def F_Inverse_VP_2(P):
    T=thermo.antoineEquation(antoine2,P,getVar='T')
    return T
Fs_VP=[F_VP_1,F_VP_2]
Fs_Inverse_VP=[F_Inverse_VP_1,F_Inverse_VP_2]
# Compute VLE
bubble_NRTL,dew_NRTL,gammas_NRTL=thermo.compute_Tx_VLE_Binary(Fs_gamma_NRTL,
                                              Fs_VP,
                                              Fs_Inverse_VP,
                                              P,
                                              z1_range=numpy.linspace(0,1,101),
                                              do_Bubble_Only=True)

# =============================================================================
# Active Learning VLE
# =============================================================================

# Testing Set Grid
x1_range=numpy.linspace(0,1,101)
# Perform AL
bubble_gp,dew_gp,gamma_gp,MAF_Vector,X_AL \
    =ml.AL_VLE_Binary_Type1(F_Gamma_1_NRTL,F_Gamma_2_NRTL,gpConfig,P,Fs_VP,
                            Fs_Inverse_VP,x1_range,maxIter=100,min_AF=0.01,
                            bubbleTruth=bubble_NRTL,dewTruth=dew_NRTL,
                            plot_VLE_GIF=r'C:\Users\dinis\Desktop\testes\vle.gif',
                            title=None)

# =============================================================================
# Pulbication Plots
# =============================================================================

# Plot Configuration
plt.rcParams['figure.dpi'] = 600
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 8
plt.rcParams['figure.titlesize'] = 8
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['ytick.labelsize'] = 7
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.it'] = 'serif:italic'
plt.rcParams['mathtext.bf'] = 'serif:bold'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams["savefig.pad_inches"] = 0.02
plt.rcParams['savefig.dpi'] = 600
# Create figure
plt.figure(figsize=(3,1.7))
# Plots
plt.plot(bubble_NRTL[:,0],bubble_NRTL[:,1],'-k',linewidth=1,
         label='Ground Truth')
plt.plot(dew_NRTL[:,0],dew_NRTL[:,1],'-k',linewidth=1)
plt.plot(bubble_gp[:,0],bubble_gp[:,1],'--r',linewidth=1,
         label='GP-Predicted')
plt.plot(dew_gp[:,0],dew_gp[:,1],'--r',linewidth=1)
plt.plot(X_AL[:,0],X_AL[:,1],'ob',markersize=2,label='AL Training Data')
plt.xlabel('$\mathregular{z_{'+components[0]+'}}$')
plt.ylabel('T /K',fontsize = 7)
# plt.text(0.18,0.05,'VLE (P = '+str(int(P/101325))+' atm)',
#          color='black',
#          horizontalalignment='center',
#          verticalalignment='center',
#          transform=plt.gca().transAxes)
plt.text(0.02,0.05,components[0]+'/'+components[1],
         color='black',
         horizontalalignment='left',
         verticalalignment='center',
         transform=plt.gca().transAxes)
plt.legend(prop={'size': 6})
