# -*- coding: utf-8 -*-
"""
Python script to test the performance of GPs on binary activity coefficient
data by computing LLE phase diagrams.

Sections:
    . Imports
    . Configuration
    . Ground Truth LLE
    . GP-Predicted LLE
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

# Title prefix for plots
titlePrefix=components[0]+'/'+components[1]

# Temperature range
Tmin=250
Tmax=550

# Training Grid type
trainGridType=2 # 1 for 5x5, 2 for 4x5+100

# GP Configuration
gpConfig={'kernel':'RBF',
          'useWhiteKernel':False,
          'doLogY':True,
          'indepDim':False,
          'trainLikelihood':False
          }

# Temperature range for LLE
LLE_range=numpy.linspace(150,250,20)

# =============================================================================
# Ground Truth LLE
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
# Compute F_Gamma_1_NRTL
LLE_NRTL,__=thermo.compute_Tx_LLE_Binary(Fs_gamma_NRTL,LLE_range)

# =============================================================================
# GP-Predicted LLE
# =============================================================================

# Testing Set Grid
x1_range=numpy.linspace(0,1,101)
T_range=numpy.linspace(Tmin,Tmax,101)
# Build Training Dataset (Comp. 1)
X_Train=ml.build_X_Train_Binary(trainGridType,1,Tmin,Tmax)
__,Y_Train=ml.buildDataset_Binary(F_Gamma_1_NRTL,X=X_Train)
# Build Testing Dataset
X_Test,Y_Test=ml.buildDataset_Binary(F_Gamma_1_NRTL,x1_range=x1_range,
                                     T_range=T_range)
# Define feature normalization
__,X_Scaler=ml.normalize(X_Test,method='MinMax')
# Build GP
model_1=ml.buildGP(X_Train,Y_Train,X_Scaler=X_Scaler,gpConfig=gpConfig)
# Build Training Dataset (Comp. 2)
X_Train=ml.build_X_Train_Binary(trainGridType,2,Tmin,Tmax)
__,Y_Train=ml.buildDataset_Binary(F_Gamma_2_NRTL,X=X_Train)
# Build Testing Dataset
X_Test,Y_Test=ml.buildDataset_Binary(F_Gamma_2_NRTL,x1_range=x1_range,
                                     T_range=T_range)
# Define feature normalization
__,X_Scaler=ml.normalize(X_Test,method='MinMax')
# Build GP
model_2=ml.buildGP(X_Train,Y_Train,X_Scaler=X_Scaler,gpConfig=gpConfig)
# Build posteriors to decrease computational cost
model_1=model_1.posterior()
model_2=model_2.posterior()
# Define GP-based gamma functions
def F_Gamma_1_GP(x1,T):
    X=numpy.array([x1,T]).reshape(1,2)
    pred,__=ml.gpPredict(model_1,X,X_Scaler=X_Scaler,gpConfig=gpConfig)
    return pred[0,0]
def F_Gamma_2_GP(x1,T):
    X=numpy.array([x1,T]).reshape(1,2)
    pred,__=ml.gpPredict(model_2,X,X_Scaler=X_Scaler,gpConfig=gpConfig)
    return pred[0,0]
Fs_gamma_GP=[F_Gamma_1_GP,F_Gamma_2_GP]
# Compute LLE
LLE_GP,__=thermo.compute_Tx_LLE_Binary(Fs_gamma_GP,LLE_range)

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
plt.plot(LLE_NRTL[:,1],LLE_NRTL[:,0],'-k',linewidth=1,label='Ground Truth')
plt.plot(numpy.flip(LLE_NRTL[:,2]),numpy.flip(LLE_NRTL[:,0]),'-k',linewidth=1)
plt.plot(LLE_GP[:,1],LLE_GP[:,0],'--r',linewidth=1,label='GP-Predicted')
plt.plot(numpy.flip(LLE_GP[:,2]),numpy.flip(LLE_GP[:,0]),'--r',linewidth=1)
plt.xlabel('$\mathregular{x_{'+components[0]+'}}$')
plt.ylabel('T /K',fontsize = 7)
# plt.text(0.18,0.05,'VLE (P = '+str(int(P/101325))+' atm)',
#          color='black',
#          horizontalalignment='center',
#          verticalalignment='center',
#          transform=plt.gca().transAxes)
# plt.text(0.02,0.05,components[0]+'/'+components[1],
#          color='black',
#          horizontalalignment='left',
#          verticalalignment='center',
#          transform=plt.gca().transAxes)
plt.legend(prop={'size': 6})

