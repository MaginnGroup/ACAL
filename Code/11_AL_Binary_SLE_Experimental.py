# -*- coding: utf-8 -*-
"""
Python script to test AL on experimental activity coefficient data by computing
SLE phase diagrams.

Sections:
    . Imports
    . Configuration
    . SLE Experimental Data
    . Active Learning
    . Fit NRTL
    . Publication Plots

Published: 2022-10-27
Author: Dinis Abranches

Edited: 2024-11-19
Editor: Alexander Glotov
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
#components=['ChCl','Urea']
#components=['Thymol','Menthol']
components=['Thymol','TOPO']

# GP Configuration
gpConfig={'kernel':'RQ',
          'useWhiteKernel':True,
          'doLogY':True,
          'indepDim':False,
          'trainLikelihood':True
          }

# AL Configuration
maxIter=10
min_AF=5

# =============================================================================
# Experimental Data
# =============================================================================

# Get melting properties
properties_1=thermo.meltingProperties(components[0])
properties_2=thermo.meltingProperties(components[1])
# System 1 (ChCl/Urea, 10.1039/C9CP03552D)
if components[0]=='ChCl' and components[1]=='Urea':
    x_exp=[1,0.890,0.796,0.698,0.596,0.493,0.391,0.307,0.200,0.101,0]
    T_exp=[properties_1[1],
           539.6,475.6,438.7,380.4,331.8,294.6,321.3,355.5,381.8,
           properties_2[1]]
    SLE_Exp=numpy.concatenate((numpy.array(x_exp).reshape(-1,1),
                               numpy.array(T_exp).reshape(-1,1)),axis=1)
# System 2 (Thymol/Menthol, 10.1039/C9CC04846D)
if components[0]=='Thymol' and components[1]=='Menthol':
    x_exp=[1,0.537,0.602,0.700,0.799,0.899,0.105,0.200,0.300,0.352,0]
    T_exp=[properties_1[1],
           265.4,278.3,281.3,308.3,317.9,304.4,295.5,272.4,263.9,
           properties_2[1]]
    SLE_Exp=numpy.concatenate((numpy.array(x_exp).reshape(-1,1),
                               numpy.array(T_exp).reshape(-1,1)),axis=1)
# System 3 (Thymol/TOPO, 10.1039/D0GC00793E)
if components[0]=='Thymol' and components[1]=='TOPO':
    x_exp=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    T_exp=[properties_2[1],323.92,321.98,313.77,303.33,279.62,264.87,260,
           276.24,312.55,properties_1[1]]
    SLE_Exp=numpy.concatenate((numpy.array(x_exp).reshape(-1,1),
                               numpy.array(T_exp).reshape(-1,1)),axis=1)
# =============================================================================
# Active Learning
# =============================================================================

# Testing Set Grid
x1_range=numpy.linspace(0,1,101)
# Initialize mean acquisition function vector
MAF_Vector=[]
# Compute Ideal SLE
SLE_ID=thermo.compute_Tx_SLE_Ideal_Binary(properties_1,properties_2,
                                          x1_range=x1_range)
X_Test=SLE_ID
# Find eutectic
eutectic=X_Test[:,1].argmin()
# Select midway point of largest liquidus curve
if X_Test[eutectic,0]<0.5:
    index=int((len(SLE_ID[:,0])-eutectic)/2)
    X_New=numpy.array([SLE_ID[index,:]]).reshape(-1,2)
else:
    index=int(eutectic/2)
    X_New=numpy.array([SLE_ID[index,:]]).reshape(-1,2)
# Define X_AL
X_AL=numpy.array([]).reshape(-1,2)
# Define a copy of SLE_Exp and remove pure-component data
SLE_Exp_=SLE_Exp.copy()
SLE_Exp=SLE_Exp[1:-1,:]
# Loop over iterations requested (zeroth iteration with midway eutectic)
for n in range(maxIter):
    # Find closest experimental match
    exp_index=numpy.abs(SLE_Exp[:,0]-X_New[0,0]).argmin()
    X_New_=SLE_Exp[exp_index,:].copy().reshape(-1,2)
    # Remove availability of experimental point
    SLE_Exp=numpy.delete(SLE_Exp,exp_index,axis=0)
    # Append to X_AL
    X_AL=numpy.append(X_AL,X_New_,axis=0)
    # Initialize X_AL_i
    X_AL_1=numpy.array([]).reshape(-1,2)
    X_AL_2=numpy.array([]).reshape(-1,2)
    # Sort X_AL such that the temperature values are decreasing towards the eutectic
    for entry in X_AL:
        if abs(entry[0]-X_Test[eutectic,0])<=0.01 and abs(entry[1]-min(X_AL[:,1]))<=1:
            X_AL_2=numpy.concatenate((X_AL_2,entry.reshape(-1,2)),axis=0)
            X_AL_1=numpy.concatenate((X_AL_1,entry.reshape(-1,2)),axis=0)
        elif entry[0]<X_Test[eutectic,0]:
            if not len(X_AL_2)==0 and entry[0] > X_AL_2[-1,0] and entry[1] >= X_AL_2[-1,1]:
                X_AL_1=numpy.concatenate((X_AL_1,entry.reshape(-1,2)),axis=0)
            elif not len(X_AL_2)==0 and entry[0] < X_AL_2[-1,0]:
                X_AL_1=numpy.concatenate((X_AL_1,X_AL_2[-1,:].reshape(-1,2)),axis=0)
                X_AL_2 = numpy.delete(X_AL_2,-1,axis=0)
                X_AL_2=numpy.concatenate((X_AL_2,entry.reshape(-1,2)),axis=0)
            else:
                X_AL_2=numpy.concatenate((X_AL_2,entry.reshape(-1,2)),axis=0)
        elif entry[0]>X_Test[eutectic,0]:
            if not len(X_AL_1)==0 and entry[0] < X_AL_1[-1,0] and entry[1] >= X_AL_1[-1,1]:
                X_AL_2=numpy.concatenate((X_AL_2,entry.reshape(-1,2)),axis=0)
            elif not len(X_AL_1)==0 and entry[0] > X_AL_1[-1,0]:
                X_AL_2=numpy.concatenate((X_AL_2,X_AL_1[-1,:].reshape(-1,2)),axis=0)
                X_AL_1 = numpy.delete(X_AL_1,-1,axis=0)
                X_AL_1=numpy.concatenate((X_AL_1,entry.reshape(-1,2)),axis=0)
            else:
                X_AL_1=numpy.concatenate((X_AL_1,entry.reshape(-1,2)),axis=0)
        else:
            continue
    # Get minimum and maximum temperature from SLE_ID
    Tmin=X_Test[:,1].min()
    Tmax=X_Test[:,1].max()
    # Generate virtual points on top of SLE_ID
    X_VP_1=numpy.meshgrid(numpy.ones(1),numpy.linspace(Tmin,Tmax,100))
    X_VP_1=numpy.array(X_VP_1).T.reshape(-1,2)
    X_VP_2=numpy.meshgrid(numpy.zeros(1),numpy.linspace(Tmin,Tmax,100))
    X_VP_2=numpy.array(X_VP_2).T.reshape(-1,2)
    # Define Y of VPs
    Y_VP=numpy.ones((100,1))
    # Compute gammas
    gammas_1=thermo.get_Gammas_from_SLE(X_AL_1,properties_1,1)
    gammas_2=thermo.get_Gammas_from_SLE(X_AL_2,properties_2,2)
    # Define X_Train
    X_Train_1=numpy.concatenate((X_VP_1,X_AL_1),axis=0)
    X_Train_2=numpy.concatenate((X_VP_2,X_AL_2),axis=0)
    # Define Y_Train
    Y_Train_1=numpy.concatenate((Y_VP,gammas_1[:,2].reshape(-1,1)),axis=0)
    Y_Train_2=numpy.concatenate((Y_VP,gammas_2[:,2].reshape(-1,1)),axis=0)
    # Define X_Scaler
    __,X_Scaler=ml.normalize(X_Test,method='MinMax')
    # Build GPs
    model_1=ml.buildGP(X_Train_1,Y_Train_1,X_Scaler=X_Scaler,gpConfig=gpConfig)
    model_2=ml.buildGP(X_Train_2,Y_Train_2,X_Scaler=X_Scaler,gpConfig=gpConfig)
    # Store posteriors to decrease computational cost
    model_1=model_1.posterior()
    model_2=model_2.posterior()
    # Define lagging curves
    curve_1_lagging=len(X_AL_1)==0
    curve_2_lagging=len(X_AL_2)==0
    # Define gamma functions for SLE calculation such that both curves have an assigned function
    if curve_1_lagging:
        def F_Gamma_1_GP(x1,T):
            X=numpy.array([1-x1,T]).reshape(1,2)
            pred,__=ml.gpPredict(model_2,X,X_Scaler=X_Scaler,gpConfig=gpConfig)
            return pred[0,0]
        def F_Gamma_2_GP(x1,T):
            X=numpy.array([x1,T]).reshape(1,2)
            pred,__=ml.gpPredict(model_2,X,X_Scaler=X_Scaler,gpConfig=gpConfig)
            return pred[0,0]
    elif curve_2_lagging:
        def F_Gamma_1_GP(x1,T):
            X=numpy.array([x1,T]).reshape(1,2)
            pred,__=ml.gpPredict(model_1,X,X_Scaler=X_Scaler,gpConfig=gpConfig)
            return pred[0,0]
        def F_Gamma_2_GP(x1,T):
            X=numpy.array([1-x1,T]).reshape(1,2)
            pred,__=ml.gpPredict(model_1,X,X_Scaler=X_Scaler,gpConfig=gpConfig)
            return pred[0,0]
    else:
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
    SLE_gp,gamma_gp=thermo.compute_Tx_SLE_Binary(Fs_gamma_GP,
                                           properties_1,properties_2,
                                           x1_range=x1_range)
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
    plt.plot(SLE_Exp[:,0],SLE_Exp[:,1],'sk',markersize=2,label='Exp. Data')
    plt.plot(SLE_gp[:,0],SLE_gp[:,1],'--r',linewidth=1,label='GP-Predicted')
    plt.plot(X_AL_1[:,0],X_AL_1[:,1],'or',markersize=2)
    plt.plot(X_AL_2[:,0],X_AL_2[:,1],'sr',markersize=2)
    plt.xlabel('x_1')
    plt.ylabel('T /K',fontsize=7)
    plt.title(components[0]+'/'+components[1]+', SLE')
    plt.legend(prop={'size': 6})
    plt.show()
    # Define new X_Test based on predicted SLE
    X_Test=SLE_gp
    # Find eutectic
    eutectic=X_Test[:,1].argmin()
    # Define X_Test_i
    X_Test_1=X_Test[eutectic:,:]
    X_Test_2=X_Test[:eutectic,:]
    # Get STDs on new grid
    Y_Pred_1,Y_STD_1=ml.gpPredict(model_1,X_Test_1,X_Scaler=X_Scaler,
                                  gpConfig=gpConfig)
    Y_Pred_2,Y_STD_2=ml.gpPredict(model_2,X_Test_2,X_Scaler=X_Scaler,
                                  gpConfig=gpConfig)
    # Compute acquisition functions
    AF_1=100*Y_STD_1/Y_Pred_1
    AF_2=100*Y_STD_2/Y_Pred_2
    # Compute mean AF
    MAF_1=AF_1.mean()
    MAF_2=AF_2.mean()            
    # Append metric
    MAF=(MAF_1+MAF_2)/2
    MAF_Vector.append(MAF)
    # Check GP_MPE
    lagging=(curve_1_lagging or curve_2_lagging)
    if len(MAF_Vector)>0 and MAF<min_AF and not lagging: break
    # Select next point based on lagging or curve with largest AF
    if curve_1_lagging:
        midway=int((eutectic+len(X_Test))/2)
        X_New=X_Test[midway,:].reshape(-1,2)
    elif curve_2_lagging:
        midway=int(eutectic/2)
        X_New=X_Test[midway,:].reshape(-1,2)
    elif AF_1.max()>AF_2.max():
        X_New=X_Test_1[AF_1.argmax(),:].reshape(-1,2)
    else:
        X_New=X_Test_2[AF_2.argmax(),:].reshape(-1,2)

# =============================================================================
# Fit NRTL
# =============================================================================

# Fit NRTL
parameters,results=thermo.NRTL_Binary_fit(X_AL_1,Y_Train_1[100:,:],
                                          X_AL_2,Y_Train_2[100:,:],
                                          components)
# Define gamma functions
def F_Gamma_1(x1,T):
    gammas=thermo.NRTL_Binary_getGamma(parameters,x1,T)
    return gammas[0]
def F_Gamma_2(x1,T):
    gammas=thermo.NRTL_Binary_getGamma(parameters,x1,T)
    return gammas[1]
Fs_gamma=[F_Gamma_1,F_Gamma_2]
# Compute SLE
SLE_NRTL,__=thermo.compute_Tx_SLE_Binary(Fs_gamma,properties_1,properties_2,
                                         x1_range=x1_range)
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
# Experimental Data
plt.plot(SLE_Exp_[:,0],SLE_Exp_[:,1],'Dk',markersize=2,label='Exp. Data')
# AL points
plt.plot(X_AL[0,0],X_AL[0,1],'ob',markersize=5,markerfacecolor='none',
         label='AL Training Data')
plt.text(X_AL[0,0]+0.03,X_AL[0,1],str(1),color="b",fontsize=8)
for n in range(len(X_AL)-1):
    plt.plot(X_AL[n+1,0],X_AL[n+1,1],'ob',markersize=5,markerfacecolor='none')
    plt.text(X_AL[n+1,0]+0.03,X_AL[n+1,1],str(n+2),color="b",fontsize=8)
# GP-Predicted Data
plt.plot(SLE_gp[:,0],SLE_gp[:,1],'--r',linewidth=1,label='GP-Predicted')
# NRTL
plt.plot(SLE_NRTL[:,0],SLE_NRTL[:,1],'--k',linewidth=1,label='NRTL Fit')
plt.xlabel('$\mathregular{x_{'+components[0]+'}}$')
#plt.xlabel('$\mathregular{x_{ChCl}}$')
plt.ylabel('T /K',fontsize = 7)
#plt.ylim(bottom=240)
# plt.text(0.18,0.05,'VLE (P = '+str(int(P/101325))+' atm)',
#          color='black',
#          horizontalalignment='center',
#          verticalalignment='center',
#          transform=plt.gca().transAxes)
plt.text(0.65,0.05,components[0]+'/'+components[1],
         color='black',
         horizontalalignment='left',
         verticalalignment='center',
         transform=plt.gca().transAxes)
plt.legend(prop={'size': 6})
