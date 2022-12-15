# -*- coding: utf-8 -*-
"""
Python script to test AL on experimental activity coefficient data by computing
VLE phase diagrams.

Sections:
    . Imports
    . Configuration
    . Experimental Data
    . Active Learning
    . Fit NRTL
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
components=['Thymol','Menthol']

# GP Configuration
gpConfig={'kernel':'RQ',
          'useWhiteKernel':True,
          'doLogY':True,
          'indepDim':False,
          'trainLikelihood':True
          }

P_VLE=100000/2

# Use SLE
useSLE=True

# AL Configuration
maxIter=10
min_AF=0.2

# =============================================================================
# Experimental Data
# =============================================================================

# SLE
# Get melting properties
properties_1=thermo.meltingProperties(components[0])
properties_2=thermo.meltingProperties(components[1])
# System 1 (Thymol/Menthol, 10.1039/C9CC04846D)
if components[0]=='Thymol' and components[1]=='Menthol':
    x_exp=[1,0.537,0.602,0.700,0.799,0.899,0.105,0.200,0.300,0.352,0]
    T_exp=[properties_1[1],
           265.4,278.3,281.3,308.3,317.9,304.4,295.5,272.4,263.9,
           properties_2[1]]
    SLE_Exp=numpy.concatenate((numpy.array(x_exp).reshape(-1,1),
                               numpy.array(T_exp).reshape(-1,1)),axis=1)

# VLE
# System 1 (Thymol/Menthol, 10.1039/C9CC04846D)
if components[0]=='Thymol' and components[1]=='Menthol':
    T_exp=numpy.array([467.1,467.6,468.5,468.8,469.1,469.4,470.0,470.3,470.7,
                       470.9,471.2,471.7,472.2,472.6,472.6,473.2,473.3,473.6])
    x_exp=numpy.array([0.272,0.309,0.386,0.390,0.402,0.413,0.467,0.468,0.474,
                       0.479,0.514,0.565,0.637,0.645,0.638,0.603,0.605,0.697])
    y_exp=numpy.array([0.152,0.184,0.257,0.260,0.270,0.280,0.336,0.336,0.341,
                       0.345,0.385,0.442,0.525,0.534,0.526,0.485,0.487,0.597])
    VLE_Exp=numpy.concatenate((numpy.array(T_exp).reshape(-1,1),
                               numpy.array(x_exp).reshape(-1,1),
                               numpy.array(y_exp).reshape(-1,1)),axis=1)

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

# =============================================================================
# Active Learning
# =============================================================================

# Testing Set Grid
x1_range=numpy.linspace(0,1,101)
# Define X_AL and Y_AL
X_AL=numpy.array([]).reshape(-1,2)
X_AL_1=numpy.array([]).reshape(-1,2)
X_AL_2=numpy.array([]).reshape(-1,2)
Y_AL=numpy.array([]).reshape(-1,2)
Y_AL_1=numpy.array([]).reshape(-1,1)
Y_AL_2=numpy.array([]).reshape(-1,1)
if useSLE:
    # Find experimental eutectic
    eutectic=SLE_Exp[:,1].argmin()
    # Sort X_AL
    for entry in SLE_Exp:
        if entry[0]<SLE_Exp[eutectic,0]:
            # Add to X_AL_2
            X_AL_2=numpy.concatenate((X_AL_2,entry.reshape(-1,2)),axis=0)
        else:
            # Add to X_AL_1
            X_AL_1=numpy.concatenate((X_AL_1,entry.reshape(-1,2)),axis=0)
    # Compute gammas
    Y_AL_1=thermo.get_Gammas_from_SLE(X_AL_1,properties_1,1)[:,2].reshape(-1,1)
    Y_AL_2=thermo.get_Gammas_from_SLE(X_AL_2,properties_2,2)[:,2].reshape(-1,1)
# Compute ideal VLE
bubble,__=thermo.compute_Tx_VLE_Ideal_Binary(Fs_VP,Fs_Inverse_VP,P_VLE,
                                             z1_range=x1_range)
# Get minimum and maximum temperature from bubble
Tmin=bubble[:,1].min()
Tmax=bubble[:,1].max()
# Generate virtual points on top of bubble
X_VP_1=numpy.meshgrid(numpy.ones(1),numpy.linspace(Tmin,Tmax,100))
X_VP_1=numpy.array(X_VP_1).T.reshape(-1,2)
X_VP_2=numpy.meshgrid(numpy.zeros(1),numpy.linspace(Tmin,Tmax,100))
X_VP_2=numpy.array(X_VP_2).T.reshape(-1,2)
# Define X_Train
X_Train_1=numpy.concatenate((X_VP_1,X_AL_1),axis=0)
X_Train_2=numpy.concatenate((X_VP_2,X_AL_2),axis=0)
# Define Y_Train
Y_VP=numpy.ones((100,1))
Y_Train_1=numpy.concatenate((Y_VP,Y_AL_1),axis=0)
Y_Train_2=numpy.concatenate((Y_VP,Y_AL_2),axis=0)
# Define X_Scaler
__,X_Scaler=ml.normalize(numpy.concatenate((X_Train_1,
                                            X_Train_2),axis=0),
                         method='MinMax')
# Define a copy of VLE_Exp and remove pure-component data
VLE_Exp_=VLE_Exp.copy()
# Initialize mean acquisition function vector
MAF_Vector=[]
# Loop over iterations requested (zeroth iteration without AL data)
for n in range(maxIter):
    # Build GPs
    model_1=ml.buildGP(X_Train_1,Y_Train_1,X_Scaler=X_Scaler,
                       gpConfig=gpConfig)
    model_2=ml.buildGP(X_Train_2,Y_Train_2,X_Scaler=X_Scaler,
                       gpConfig=gpConfig)
    # Store posteriors to decrease computational cost
    model_1=model_1.posterior()
    model_2=model_2.posterior()
    # Define gamma functions for VLE calculation
    def F_Gamma_1_GP(x1,T):
        X=numpy.array([x1,T]).reshape(1,2)
        pred,__=ml.gpPredict(model_1,X,X_Scaler=X_Scaler,gpConfig=gpConfig)
        return pred[0,0]
    def F_Gamma_2_GP(x1,T):
        X=numpy.array([x1,T]).reshape(1,2)
        pred,__=ml.gpPredict(model_2,X,X_Scaler=X_Scaler,gpConfig=gpConfig)
        return pred[0,0]
    Fs_gamma_GP=[F_Gamma_1_GP,F_Gamma_2_GP]
    # Compute VLE
    bubble_gp,dew_gp,gamma_gp=thermo.compute_Tx_VLE_Binary(Fs_gamma_GP,
                                                           Fs_VP,
                                                           Fs_Inverse_VP,
                                                           P_VLE,
                                                           z1_range=x1_range)
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
    plt.plot(VLE_Exp[:,1],VLE_Exp[:,0],'sk',markersize=2,label='Exp. Data')
    plt.plot(VLE_Exp[:,2],VLE_Exp[:,0],'sk',markersize=2)
    plt.plot(bubble_gp[:,0],bubble_gp[:,1],
             '--r',linewidth=1,label='GP-Predicted')
    plt.plot(dew_gp[:,0],dew_gp[:,1],
             '--r',linewidth=1)
    plt.plot(X_AL[:,0],X_AL[:,1],'or',markersize=2)
    plt.xlabel('x_1')
    plt.ylabel('T /K',fontsize=7)
    plt.legend(prop={'size': 6})
    plt.show()
    # Define new X_Test based on predicted VLE
    X_Test=bubble_gp.copy()
    # Get STDs on new grid
    Y_Pred_1,Y_STD_1=ml.gpPredict(model_1,X_Test,X_Scaler=X_Scaler,
                                  gpConfig=gpConfig)
    Y_Pred_2,Y_STD_2=ml.gpPredict(model_2,X_Test,X_Scaler=X_Scaler,
                                  gpConfig=gpConfig)
    # Compute vapor pressures
    vp1=[]
    vp2=[]
    for entry in X_Test:
        vp1.append(Fs_VP[0](entry[1]))
        vp2.append(Fs_VP[1](entry[1]))
    vp1=numpy.array(vp1).reshape(-1,1)
    vp2=numpy.array(vp2).reshape(-1,1)
    # Compute acquisition function
    x1=X_Test[:,0].reshape(-1,1)
    x2=1-x1
    AF=x1*(vp1/P_VLE)*Y_STD_1+x2*(vp2/P_VLE)*Y_STD_2
    # Append metric
    MAF=AF.mean()
    MAF_Vector.append(MAF)
    # Select next point
    X_New=X_Test[AF.argmax(),:].reshape(1,2)
    # Find closest experimental match
    exp_index=numpy.abs(VLE_Exp[:,1]-X_New[0,0]).argmin()
    X_New_=numpy.array([VLE_Exp[exp_index,1].copy(),
                        VLE_Exp[exp_index,0].copy()]).reshape(-1,2)
    # Append to X_AL
    X_AL=numpy.append(X_AL,X_New_,axis=0)
    # Compute gammas
    aux=numpy.array([VLE_Exp[exp_index,1].copy(),
                     1-VLE_Exp[exp_index,1].copy(),
                     VLE_Exp[exp_index,2].copy(),
                     1-VLE_Exp[exp_index,2].copy(),
                     VLE_Exp[exp_index,0].copy()]).reshape(-1,5)
    gammas=thermo.get_Gammas_from_VLE(aux,Fs_VP,P_VLE)
    # Remove availability of experimental point
    VLE_Exp=numpy.delete(VLE_Exp,exp_index,axis=0)
    # Append to Y_AL_i
    Y_AL_1=numpy.append(Y_AL_1,gammas[:,2].reshape(-1,1),axis=0)
    Y_AL_2=numpy.append(Y_AL_2,gammas[:,3].reshape(-1,1),axis=0)
    # Get minimum and maximum temperature from bubble
    Tmin=bubble[:,1].min()
    Tmax=bubble[:,1].max()
    # Generate virtual points on top of bubble
    X_VP_1=numpy.meshgrid(numpy.ones(1),numpy.linspace(Tmin,Tmax,100))
    X_VP_1=numpy.array(X_VP_1).T.reshape(-1,2)
    X_VP_2=numpy.meshgrid(numpy.zeros(1),numpy.linspace(Tmin,Tmax,100))
    X_VP_2=numpy.array(X_VP_2).T.reshape(-1,2)
    # Define X_Train
    X_Train_1=numpy.concatenate((X_VP_1,X_AL_1,X_AL),axis=0)
    X_Train_2=numpy.concatenate((X_VP_2,X_AL_2,X_AL),axis=0)
    # Define Y_Train
    Y_VP=numpy.ones((100,1))
    Y_Train_1=numpy.concatenate((Y_VP,Y_AL_1),axis=0)
    Y_Train_2=numpy.concatenate((Y_VP,Y_AL_2),axis=0)
    # Define X_Scaler
    __,X_Scaler=ml.normalize(numpy.concatenate((X_Train_1,
                                                X_Train_2),axis=0),
                             method='MinMax')
    # Check GP_MPE
    if len(MAF_Vector)>1 and MAF<min_AF: break
# Remove last row of X_AL (added in the last iteration but not used)
X_AL=numpy.delete(X_AL,-1,0)
Y_AL_1=numpy.delete(Y_AL_1,-1,0)
Y_AL_2=numpy.delete(Y_AL_2,-1,0)

# =============================================================================
# Fit NRTL
# =============================================================================

# Fit NRTL
parameters,__=thermo.NRTL_Binary_fit(numpy.concatenate((X_AL_1,X_AL),axis=0),
                                     Y_AL_1,
                                     numpy.concatenate((X_AL_2,X_AL),axis=0),
                                     Y_AL_2,
                                     components)
# Define gamma functions
def F_Gamma_1(x1,T):
    gammas=thermo.NRTL_Binary_getGamma(parameters,x1,T)
    return gammas[0]
def F_Gamma_2(x1,T):
    gammas=thermo.NRTL_Binary_getGamma(parameters,x1,T)
    return gammas[1]
Fs_gamma=[F_Gamma_1,F_Gamma_2]
# Compute VLE
bubble_NRTL,dew_NRTL,gamma_NRTÇ=thermo.compute_Tx_VLE_Binary(Fs_gamma,
                                                             Fs_VP,
                                                             Fs_Inverse_VP,
                                                             P_VLE,
                                                             z1_range=x1_range)

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
plt.plot(VLE_Exp_[:,1],VLE_Exp_[:,0],'Dk',markersize=2,label='Exp. Data')
plt.plot(VLE_Exp_[:,2],VLE_Exp_[:,0],'Dk',markersize=2)
# AL points
plt.plot(X_AL[0,0],X_AL[0,1],'ob',markersize=5,markerfacecolor='none',
         label='AL Training Data')
plt.text(X_AL[0,0]+0.03,X_AL[0,1]-1,str(1),color="b",fontsize=8)
for n in range(len(X_AL)-1):
    plt.plot(X_AL[n+1,0],X_AL[n+1,1],'ob',markersize=5,markerfacecolor='none')
    plt.text(X_AL[n+1,0]+0.03,X_AL[n+1,1]-1,str(n+2),color="b",fontsize=8)
# GP-Predicted Data
plt.plot(bubble_gp[:,0],bubble_gp[:,1],'--r',linewidth=1,label='GP-Predicted')
plt.plot(dew_gp[:,0],dew_gp[:,1],'--r',linewidth=1)
# NRTL
plt.plot(bubble_NRTL[:,0],bubble_NRTL[:,1],'--k',linewidth=1,label='NRTL Fit')
plt.plot(dew_NRTL[:,0],dew_NRTL[:,1],'--k',linewidth=1)
plt.xlabel('$\mathregular{x_{'+components[0]+'}}$')
#plt.xlabel('$\mathregular{x_{ChCl}}$')
plt.ylabel('T /K',fontsize = 7)
#plt.ylim(bottom=240)
# plt.text(0.18,0.05,'VLE (P = '+str(int(P/101325))+' atm)',
#          color='black',
#          horizontalalignment='center',
#          verticalalignment='center',
#          transform=plt.gca().transAxes)
plt.text(0.02,0.90,components[0]+'/'+components[1],
         color='black',
         horizontalalignment='left',
         verticalalignment='center',
         transform=plt.gca().transAxes)
plt.legend(prop={'size': 6})