# -*- coding: utf-8 -*-
"""
Python script to test AL on binary activity coefficient data by computing
SLE phase diagrams.

Sections:
    . Imports
    . Configuration
    . Ground Truth SLE
    . Active Learning SLE
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
components=['Acetone','Water']

# GP Configuration
gpConfig={'kernel':'RQ',
          'useWhiteKernel':False,
          'doLogY':True,
          'indepDim':False,
          'trainLikelihood':False
          }

# =============================================================================
# Ground Truth SLE
# =============================================================================

# Get NRTL parameters
parameters=thermo.NRTL_Parameters(components[0],components[1])
# Define gamma functions
def F_Truth_1(x1,T):
    gammas=thermo.NRTL_Binary_getGamma(parameters,x1,T)
    return gammas[0]
def F_Truth_2(x1,T):
    gammas=thermo.NRTL_Binary_getGamma(parameters,x1,T)
    return gammas[1]
Fs_gamma_NRTL=[F_Truth_1,F_Truth_2]
# Get melting properties
properties_1=thermo.meltingProperties(components[0])
properties_2=thermo.meltingProperties(components[1])
# Compute SLE
SLE_NRTL,gammas_NRTL=thermo.compute_Tx_SLE_Binary(Fs_gamma_NRTL,
                                                  properties_1,properties_2,
                                            x1_range=numpy.linspace(0,1,101))

# =============================================================================
# Active Learning SLE
# =============================================================================

# Testing Set Grid
x1_range=numpy.linspace(0,1,101)
# Perform AL
SLE_gp,gamma_gp,MAF_Vector,X_AL=ml.AL_SLE_Binary_Type1(
    F_Truth_1,
    F_Truth_2,
    gpConfig,
    properties_1,
    properties_2,
    x1_range,
    maxIter=100,min_AF=0.5,
    plot_SLE_GIF=r'C:\Users\dinis\Desktop\testes\SLE.gif',
    title=None,SLE_Truth=SLE_NRTL)

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
plt.plot(SLE_NRTL[:,0],SLE_NRTL[:,1],'-k',linewidth=1,
         label='Ground Truth')
plt.plot(SLE_gp[:,0],SLE_gp[:,1],'--r',linewidth=1,
         label='GP-Predicted')
plt.plot(X_AL[:,0],X_AL[:,1],'ob',markersize=2,label='AL Training Data')
plt.xlabel('$\mathregular{x_{'+components[0]+'}}$')
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

print(len(X_AL))