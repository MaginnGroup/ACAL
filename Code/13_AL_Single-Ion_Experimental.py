# -*- coding: utf-8 -*-
"""
Python script to test AL on experimental single-ion activity coefficient data.

Sections:
    . Imports
    . Configuration
    . Experimental Data
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
from lib import mlAux as ml

# =============================================================================
# Configuration
# =============================================================================

# System definition
components=['Water','K','F']
targetIon='F'
# GP Configuration
gpConfig={'kernel':'RQ',
          'useWhiteKernel':True,
          'doLogY':True,
          'indepDim':False,
          'trainLikelihood':True
          }

# AL Configuration
maxIter=10
min_AF=10

# =============================================================================
# Experimental Data
# =============================================================================

# 10.1021/acs.jpcb.1c04019
if components[1]=='Na' and components[2]=='Cl':
    m_exp=[0.019,0.056,0.167,0.611,0.944,1.943,2.942,3.941,4.940,5.939]
    if targetIon=='Na':
        ln_gamma_exp=[-0.18,-0.12,-0.27,0.01,0.17,0.57,0.92,1.23,1.28,1.37]
    elif targetIon=='Cl':
        ln_gamma_exp=[-0.18,-0.31,-0.36,-0.72,-1.06,-1.02,-1.07,-0.92,-0.43,
                      0.02]
if components[1]=='K' and components[2]=='Cl':
    m_exp=[0.019,0.056,0.167,0.611,0.944,1.943,2.942,3.941,4.940]
    if targetIon=='K':
        ln_gamma_exp=[-0.18,-0.25,-0.40,-0.83,-1.14,-1.82,-2.67,-3.51,-4.42]
    elif targetIon=='Cl':
        ln_gamma_exp=[-0.18,-0.11,-0.17,0.05,0.14,0.83,1.83,2.62,3.73]
if components[1]=='Na' and components[2]=='F':
    m_exp=[0.019,0.056,0.167,0.611,0.944,1.943]
    if targetIon=='Na':
        ln_gamma_exp=[-0.18,-0.20,-0.32,-0.30,-0.34,-0.18]
    elif targetIon=='F':
        ln_gamma_exp=[-0.18,-0.32,-0.57,-1.18,-1.30,-1.69]
if components[1]=='K' and components[2]=='F':
    m_exp=[0.019,0.056,0.167,0.611,0.944,1.943,2.942,3.941,4.940]
    if targetIon=='K':
        ln_gamma_exp=[-0.18,-0.16,-0.41,-0.98,-1.47,-2.50,-3.70,-4.89,-6.11]
    elif targetIon=='F':
        ln_gamma_exp=[-0.18,-0.26,-0.41,-0.52,-0.46,0.04,0.91,1.90,2.80]
# Define Exp data
Exp=numpy.concatenate((numpy.sqrt(m_exp).reshape(-1,1),
                       numpy.exp(ln_gamma_exp).reshape(-1,1)),axis=1)
# Define copy of Exp data
Exp_=Exp.copy()

# =============================================================================
# Active Learning
# =============================================================================


# Find closest experimental match to m=1
exp_index=numpy.abs(Exp[:,0]-1).argmin()
# Append to X_AL and Y_AL
X_AL=Exp[exp_index,0].reshape(-1,1)
Y_AL=Exp[exp_index,1].reshape(-1,1)
# Remove initial point from exp data available
Exp=numpy.delete(Exp,exp_index,axis=0)
# Testing Set Grid
X_Test=numpy.sqrt(numpy.linspace(0,max(m_exp)*1.1,1000)).reshape(-1,1)
# Define X_Scaler
__,X_Scaler=ml.normalize(X_Test,method='MinMax')
# Initialize mean acquisition function vector
MAF_Vector=[]
# Loop over iterations requested
for n in range(maxIter):
    # Build GP
    model=ml.buildGP(X_AL,Y_AL,X_Scaler=X_Scaler,gpConfig=gpConfig)
    # Store posterior to decrease computational cost
    model=model.posterior()
    # Perform predictions
    Y_Pred,Y_STD=ml.gpPredict(model,X_Test,X_Scaler=X_Scaler,gpConfig=gpConfig)
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
    plt.plot(Exp[:,0],Exp[:,1],'sk',markersize=2,label='Exp. Data')
    plt.plot(X_Test,Y_Pred,'--r',linewidth=1,label='GP-Predicted')
    plt.plot(X_AL,Y_AL,'or',markersize=2)
    plt.xlabel('Salt Molality')
    plt.ylabel('$\mathregular{\gamma_{'+targetIon+'}}$',
               fontsize=7)
    plt.legend(prop={'size': 6})
    plt.show()
    # Compute acquisition function
    AF=100*Y_STD/Y_Pred
    # Append metric
    MAF=AF.mean()
    MAF_Vector.append(MAF)
    # Check GP_MPE
    if MAF<min_AF: break
    # Select next point
    X_New=X_Test[AF.argmax()]
    # Find closest experimental match
    exp_index=numpy.abs(Exp[:,0]-X_New).argmin()
    # Append to X_AL and Y_AL
    X_AL=numpy.append(X_AL,Exp[exp_index,0].reshape(-1,1),axis=0)
    Y_AL=numpy.append(Y_AL,Exp[exp_index,1].reshape(-1,1),axis=0)
    # Remove availability of experimental point
    Exp=numpy.delete(Exp,exp_index,axis=0)

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
# Convert to usual representation
Exp_[:,1]=numpy.log(Exp_[:,1])
Y_AL=numpy.log(Y_AL)
Y_Pred=numpy.log(Y_Pred)
# Experimental Data
plt.plot(Exp_[:,0],Exp_[:,1],'Dk',markersize=2,label='Exp. Data')
# AL points
plt.plot(X_AL[0],Y_AL[0],'ob',markersize=5,markerfacecolor='none',
         label='AL Training Data')
plt.text(X_AL[0]-0.05,Y_AL[0]+0.2,str(1),color="b",fontsize=8)
for n in range(len(X_AL)-1):
    plt.plot(X_AL[n+1],Y_AL[n+1],'ob',markersize=5,markerfacecolor='none')
    plt.text(X_AL[n+1]-0.05,Y_AL[n+1]+0.2,str(n+2),color="b",fontsize=8)
# GP-Predicted Data
plt.plot(X_Test,Y_Pred,'--r',linewidth=1,label='GP-Predicted')
plt.xlabel('$\mathregular{\sqrt{m}}$')
plt.ylabel('$\mathregular{log(\gamma_{'+targetIon+'})}$',
           fontsize=7)
plt.text(0.98,0.05,targetIon+' in Aq. '+components[1]+components[2],
         color='black',
         horizontalalignment='right',
         verticalalignment='center',
         transform=plt.gca().transAxes)
plt.legend(prop={'size': 6})