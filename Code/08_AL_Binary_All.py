# -*- coding: utf-8 -*-
"""
Python script to evaluate the performance of active learning on the 9
acetone-based binary systems.

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

# GP Configuration
gpConfig={'kernel':'RQ',
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
# Build Initial Grid
X_Init=ml.build_X_Train_Binary(3,targetComp,Tmin,Tmax)

# =============================================================================
# Main Loop
# =============================================================================

# Initialize MPE
GP_MPE=[]
MPE=[]
N=[]
for comp2 in tqdm(comp2s,'Component: '):
    # Get NRTL parameters
    parameters=thermo.NRTL_Parameters(comp1,comp2)
    # Define truth function
    def F_Truth(x1,T):
        gammas=thermo.NRTL_Binary_getGamma(parameters,x1,T)
        return gammas[targetComp-1]
    # Build Initial Dataset
    __,Y_Init=ml.buildDataset_Binary(F_Truth,X=X_Init)
    # Build Testing Dataset
    X_Test,Y_Test=ml.buildDataset_Binary(F_Truth,x1_range=x1_range,
                                         T_range=T_range)
    # Define feature normalization
    __,X_Scaler=ml.normalize(X_Test,method='MinMax')
    # Perform AL
    a,b,scores=ml.AL_Independent_Binary(targetComp,X_Init,Y_Init,X_Test,Y_Test,
                                        gpConfig,X_Scaler=X_Scaler,maxIter=100,
                                        min_MRE=0.5)
    # Append
    GP_MPE.append(scores['GP_MPE'][-1])
    MPE.append(scores['MPE'][-1])
    N.append(len(scores['MPE']))

# Print
print('GP_MPE:\n\n')
for entry in GP_MPE:
    print(entry)
print('\n\nMPE:\n\n')
for entry in MPE:
    print(entry)
print('\n\nN:\n\n')
for entry in N:
    print(entry)
