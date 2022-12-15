# -*- coding: utf-8 -*-
"""
Python script to test active learning on activity coefficient data.

Sections:
    . Imports
    . Configuration
    . Datasets Generation
    . Active Learning
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
components=['Acetone','Methanol']

# Component of interest for gamma calculation
targetComp=1 # 1 or 2

# Temperature range
Tmin=250
Tmax=550

# GP Configuration
gpConfig={'kernel':'RQ',
          'useWhiteKernel':False,
          'doLogY':True,
          'indepDim':False,
          'trainLikelihood':False
          }

# Title prefix for plots
titlePrefix=components[0]+'/'+components[1]+', Gamma_'+str(targetComp)

# =============================================================================
# Datasets Generation
# =============================================================================

# Get NRTL parameters
parameters=thermo.NRTL_Parameters(components[0],components[1])
# Define truth function
def F_Truth(x1,T):
    gammas=thermo.NRTL_Binary_getGamma(parameters,x1,T)
    return gammas[targetComp-1]
# Build Training Dataset
X_Init=ml.build_X_Train_Binary(3,targetComp,Tmin,Tmax)
__,Y_Init=ml.buildDataset_Binary(F_Truth,X=X_Init)
# Build Testing Dataset
X_Test,Y_Test=ml.buildDataset_Binary(F_Truth,x1_range=numpy.linspace(0,1,101),
                                     T_range=numpy.linspace(Tmin,Tmax,101))
# T Slices
T_Slices=numpy.array([265,340,430,535])

# =============================================================================
# Active Learning
# =============================================================================

# Define feature normalization
__,X_Scaler=ml.normalize(X_Test,method='MinMax')
# Perform Active Learning
model,X_Train,scores=ml.AL_Independent_Binary(targetComp,X_Init,Y_Init,
                                              X_Test,Y_Test,gpConfig,
                                              X_Scaler=X_Scaler,
                                              maxIter=100,min_MRE=0.5)
# Perform predictions
Y_Pred,Y_STD=ml.gpPredict(model,X_Test,X_Scaler=X_Scaler,gpConfig=gpConfig)

# =============================================================================
# Publication Plots
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
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.it'] = 'serif:italic'
plt.rcParams['mathtext.bf'] = 'serif:bold'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams["savefig.pad_inches"] = 0.02
plt.rcParams['savefig.dpi'] = 600
# Get axis ranges from X
x1_range=numpy.unique(X_Test[:,0])
T_range=X_Test[:len(x1_range),1]
# Reshape Y
ZMatrix=numpy.log(Y_Pred.reshape(len(T_range),len(x1_range),order='F'))
# Create figure
plt.figure(figsize=(3,1.7))
# Heatmap
hm=plt.pcolormesh(x1_range,T_range,ZMatrix)
cb=plt.colorbar(hm)
cb.set_label('$\mathregular{log(\gamma)_{'+components[targetComp-1]+'}}$')
# Virtual Points
plt.plot(X_Train[:100,0],X_Train[:100,1],'or',markersize=2)
# AL points
for n in range(len(X_Train[100:,0])):
    if X_Train[100+n,1]>500:
        plt.plot(X_Train[100+n,0],X_Train[100+n,1],'or',markersize=2)
        plt.text(X_Train[100+n,0]+0.01,X_Train[100+n,1]-20,str(n),color="r",
                 fontsize=8)
    else:
        plt.plot(X_Train[100+n,0],X_Train[100+n,1],'or',markersize=2)
        plt.text(X_Train[100+n,0]+0.01,X_Train[100+n,1],str(n),color="r",
                 fontsize=8)
# Labels
plt.xlabel('$\mathregular{x_{'+components[0]+'}}$')
plt.ylabel('T /K')
plt.show()

# Create figure
plt.figure(figsize=(3,1.7))
# Plots
N=len(scores['MPE'])
plt.semilogy(scores['MPE'],'--k',linewidth=1,label='True MRE')
plt.semilogy(scores['GP_MPE'],'--r',linewidth=1,label='GP-Predicted MRE')
plt.semilogy([0,N-1],[0.5,0.5],'-k',linewidth=1,label='Target (0.5%)')
plt.xlabel('Active Learning Iteration')
plt.ylabel('Mean Relative Error (%)')
plt.legend(prop={'size': 6})
