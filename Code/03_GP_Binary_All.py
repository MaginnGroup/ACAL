# -*- coding: utf-8 -*-
"""
Python script to evaluate the performance of a given set of GP parameters on
the 9 acetone-based binary systems.

Sections:
    . Imports
    . Configuration
    . Datasets Generation
    . Main Loop

Last edit: 2022-10-27
Author: Dinis Abranches
"""

# =============================================================================
# Imports
# =============================================================================

# Specific
import numpy
from tqdm import tqdm

# Local
from lib import thermoAux as thermo
from lib import mlAux as ml

# =============================================================================
# Configuration
# =============================================================================

# Systems definition
comp1='Acetone'
comp2s=['Methanol','Benzene','Chloroform','Cyclohexane','Hexane','Heptane',
       'TCM','Toluene','Water']

# Component of interest for gamma calculation
targetComp=2 # 1 or 2

# Temperature range
Tmin=250
Tmax=550

# Training Grid type
trainGridType=2 # 1 for 5x5, 2 for 4x5+100

# GP Configuration
gpConfig={'kernel':'ArcCosine_2',
          'useWhiteKernel':False,
          'doLogY':True,
          'indepDim':False,
          'trainLikelihood':False
          }

# =============================================================================
# Datasets Generation
# =============================================================================

# Testing Set Range
x1_range=numpy.linspace(0,1,101)
T_range=numpy.linspace(Tmin,Tmax,101)
# Build Training Grid
X_Train=ml.build_X_Train_Binary(trainGridType,targetComp,Tmin,Tmax)

# =============================================================================
# Main Loop
# =============================================================================

# Initialize MPE
MPE=[]
for comp2 in tqdm(comp2s,'Component: '):
    # Get NRTL parameters
    parameters=thermo.NRTL_Parameters(comp1,comp2)
    # Define truth function
    def F_Truth(x1,T):
        gammas=thermo.NRTL_Binary_getGamma(parameters,x1,T)
        return gammas[targetComp-1]
    # Build Training Dataset
    X_Train,Y_Train=ml.buildDataset_Binary(F_Truth,X=X_Train)
    # Build Testing Dataset
    X_Test,Y_Test=ml.buildDataset_Binary(F_Truth,x1_range=x1_range,
                                         T_range=T_range)
    # Define feature normalization
    __,X_Scaler=ml.normalize(X_Test,method='MinMax')
    # Build GP
    model=ml.buildGP(X_Train,Y_Train,X_Scaler=X_Scaler,gpConfig=gpConfig)
    # Perform predictions
    Y_Pred,Y_STD=ml.gpPredict(model,X_Test,X_Scaler=X_Scaler,gpConfig=gpConfig)
    # Evaluate GP
    scores=ml.evaluateModel(Y_Pred,X_Test,Y_Test)    
    # Append
    MPE.append(scores['MPE'])
# Print
print('Results:\n\n')
for entry in MPE:
    print(entry)
