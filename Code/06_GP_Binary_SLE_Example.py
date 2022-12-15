# -*- coding: utf-8 -*-
"""
Python script to test the performance of GPs on binary activity coefficient
data by computing SLE phase diagrams.

Sections:
    . Imports
    . Configuration
    . Ground Truth SLE
    . GP-Predicted SLE
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
components=['Acetone','Hexane']

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
# Get melting properties
properties_1=thermo.meltingProperties(components[0])
properties_2=thermo.meltingProperties(components[1])
# Compute SLE
SLE_NRTL,gammas_NRTL=thermo.compute_Tx_SLE_Binary(Fs_gamma_NRTL,
                                                  properties_1,properties_2,
                                            x1_range=numpy.linspace(0,1,101))

# =============================================================================
# GP-Predicted SLE
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
# Compute SLE
SLE_GP,gammas_GP=thermo.compute_Tx_SLE_Binary(Fs_gamma_GP,
                                              properties_1,properties_2,
                                              x1_range=x1_range)

plt.plot(gammas_NRTL[:,0],gammas_NRTL[:,1],'-k',linewidth=1,label='Ground Truth')
plt.plot(gammas_GP[:,0],gammas_GP[:,1],'--r',linewidth=1,label='GP-Predicted')



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
plt.plot(SLE_NRTL[:,0],SLE_NRTL[:,1],'-k',linewidth=1,label='Ground Truth')
plt.plot(SLE_GP[:,0],SLE_GP[:,1],'--r',linewidth=1,label='GP-Predicted')
plt.xlabel('$\mathregular{x_{'+components[0]+'}}$')
plt.ylabel('T /K',fontsize = 7)
#plt.ylim(bottom=100)
# plt.text(0.18,0.05,'VLE (P = '+str(int(P/101325))+' atm)',
#          color='black',
#          horizontalalignment='center',
#          verticalalignment='center',
#          transform=plt.gca().transAxes)
plt.text(0.5,0.93,components[0]+'/'+components[1],
         color='black',
         horizontalalignment='center',
         verticalalignment='center',
         transform=plt.gca().transAxes)
plt.legend(prop={'size': 6},
           loc='lower left')
