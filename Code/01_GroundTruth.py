# -*- coding: utf-8 -*-
"""
Python script to display the ground truth data obtained from the NRTL model
parameterized by Gmehling and co-authors in 10.1252/jcej.08we123.

Sections:
    . Imports
    . Configuration
    . Datasets Generation
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
targetComp=2 # 1 or 2

# Temperature range
Tmin=250
Tmax=550

# =============================================================================
# Datasets Generation
# =============================================================================

# Get NRTL parameters
parameters=thermo.NRTL_Parameters(components[0],components[1])
# Define truth function
def F_Truth(x1,T):
    gammas=thermo.NRTL_Binary_getGamma(parameters,x1,T)
    return gammas[targetComp-1]
# Build Testing Dataset
X_Test,Y_Test=ml.buildDataset_Binary(F_Truth,x1_range=numpy.linspace(0,1,101),
                                     T_range=numpy.linspace(Tmin,Tmax,101))
# T Slices
T_Slices=numpy.linspace(Tmin,Tmax,5)

# =============================================================================
# Publication Plots
# =============================================================================

# HM of ground truth
ml.plotHM(X_Test,numpy.log(Y_Test),None,
          '$\mathregular{log(\gamma)_{'+components[targetComp-1]+'}}$',
          xLabel='$\mathregular{x_{'+components[0]+'}}$',
          X_Highlight=None,savePath=None)

# T Slices of ground truth
ml.plot_T_Slices(T_Slices,X_Test,numpy.log(Y_Test),None,Y_2=None,
                 xLabel='$\mathregular{x_{'+components[0]+'}}$',
           yLabel='$\mathregular{log(\gamma)_{'+components[targetComp-1]+'}}$',
                 X_Highlight=None,savePath=None)
