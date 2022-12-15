# -*- coding: utf-8 -*-
"""
Python script to test GPs on binary activity coefficient data.

Sections:
    . Imports
    . Configuration
    . Datasets Generation
    . Gaussian Process Regression
    . Publication Plots

Last edit: 2022-10-27
Author: Dinis Abranches
"""

# =============================================================================
# Imports
# =============================================================================

# Specific
import numpy

# Local
from lib import thermoAux as thermo
from lib import mlAux as ml

# =============================================================================
# Configuration
# =============================================================================

# System definition
components=['Acetone','Water']

# Component of interest for gamma calculation
targetComp=1 # 1 or 2

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
X_Train=ml.build_X_Train_Binary(trainGridType,targetComp,Tmin,Tmax)
__,Y_Train=ml.buildDataset_Binary(F_Truth,X=X_Train)
# Build Testing Dataset
X_Test,Y_Test=ml.buildDataset_Binary(F_Truth,x1_range=numpy.linspace(0,1,101),
                                     T_range=numpy.linspace(Tmin,Tmax,101))
# T Slices
T_Slices=numpy.array([265,340,430,535])

# =============================================================================
# Gaussian Process Regression
# =============================================================================

# Define feature normalization
__,X_Scaler=ml.normalize(X_Test,method='MinMax')
# Build GP
model=ml.buildGP(X_Train,Y_Train,X_Scaler=X_Scaler,gpConfig=gpConfig)
# Perform predictions
Y_Pred,Y_STD=ml.gpPredict(model,X_Test,X_Scaler=X_Scaler,gpConfig=gpConfig)
# Score predictions
scores=ml.evaluateModel(Y_Pred,X_Test,Y_Test,Y_STD=Y_STD,
                        plotHM_STD=True,plotHM_PE=True,
                        titlePrefix=titlePrefix,X_Highlight=X_Train)
# Print Scores
print(scores)

# =============================================================================
# Publication Plots
# =============================================================================

# HM of Predictions
ml.plotHM(X_Test,numpy.log(Y_Pred),None,
          '$\mathregular{log(\gamma)_{'+components[targetComp-1]+'}}$',
          xLabel='$\mathregular{x_{'+components[0]+'}}$',
          X_Highlight=X_Train,savePath=None)

# T Slices of Predictions vs. Ground Truth
ml.plot_T_Slices(T_Slices,X_Test,numpy.log(Y_Pred),None,Y_2=numpy.log(Y_Test),
                 xLabel='$\mathregular{x_{'+components[0]+'}}$',
           yLabel='$\mathregular{log(\gamma)_{'+components[targetComp-1]+'}}$',
                 X_Highlight=X_Train,savePath=None)



