# -*- coding: utf-8 -*-
"""
Python script to test AL on experimental activity coefficient data by computing
ternary VLE phase diagrams.

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
from ternary_diagram import TernaryDiagram

# Local
from lib import thermoAux as thermo
from lib import mlAux as ml

# =============================================================================
# Configuration
# =============================================================================

# System definition
components=['Acetone','Benzene','Cyclohexane']

# GP Configuration
gpConfig={'kernel':'RQ',
          'useWhiteKernel':True,
          'doLogY':True,
          'indepDim':False,
          'trainLikelihood':True
          }

P_VLE=101330

# AL Configuration
maxIter=6
min_AF=0.2

# =============================================================================
# Experimental Data
# =============================================================================

# Rao V.K.K.; Krishnamurty V.V.G.; Rao C.V.: Vapour-Liquid Equilibria: System
# Acetone 1 Benzene 2 Cyclohexane 3. Recl.Trav.Chim.Pays-Bas 76 (1957) 769-778
if components==['Acetone','Benzene','Cyclohexane']:
    T_exp=[350.75,350.95,350.45,345.55,341.35,342.65,342.85,344.65,340.05,
           337.95,335.55,334.25,331.45,333.25,334.45,336.55,337.85,337.15,
           335.75,333.65,332.95,332.65,331.95,330.95,330.95,331.05,331.55,
           333.65,335.75,336.85,338.05,340.35,342.05,342.25,329.15,329.95,
           330.85,329.55,328.95,327.95,327.85]
    x1_exp=[0.05500,0.01500,0.03000,0.08500,0.18500,0.12500,0.11000,0.07500,
            0.16500,0.22000,0.31500,0.37000,0.49000,0.42000,0.40300,0.35500,
            0.30500,0.34500,0.40000,0.53500,0.59000,0.63500,0.58500,0.53500,
            0.48500,0.43000,0.36500,0.31000,0.27000,0.22500,0.17000,0.13100,
            0.10200,0.09500,0.91100,0.83500,0.76000,0.72000,0.68000,0.70000,
            0.74500]
    x2_exp=[0.93000,0.88500,0.83000,0.71500,0.65500,0.61500,0.57000,0.52500,
            0.46500,0.43000,0.37500,0.34000,0.27500,0.33500,0.38000,0.48500,
            0.54000,0.52000,0.47000,0.37500,0.32500,0.30500,0.28500,0.26000,
            0.23500,0.21500,0.19500,0.25500,0.34100,0.31500,0.28000,0.33600,
            0.41500,0.37000,0.06500,0.13000,0.20000,0.18000,0.17500,0.13500,
            0.11500]
    y1_exp=[0.13600,0.03900,0.08500,0.18600,0.36800,0.28200,0.26900,0.20100,
            0.35400,0.45900,0.55200,0.59100,0.66100,0.61800,0.60500,0.55900,
            0.51700,0.54600,0.59200,0.68100,0.70500,0.73800,0.70200,0.68700,
            0.66500,0.65000,0.62500,0.58500,0.52000,0.48200,0.42700,0.33600,
            0.28000,0.26200,0.88000,0.82600,0.80100,0.77200,0.74800,0.75400,
            0.76800]
    y2_exp=[0.81500,0.81400,0.73900,0.57400,0.46600,0.45700,0.43200,0.43000,
            0.31600,0.26200,0.20400,0.17700,0.13100,0.17600,0.20800,0.27600,
            0.33400,0.31000,0.28100,0.21500,0.17500,0.17400,0.13200,0.11600,
            0.10200,0.09600,0.08400,0.12200,0.19400,0.18300,0.17400,0.24500,
            0.31400,0.29100,0.02200,0.05700,0.10200,0.08400,0.07500,0.05500,
            0.04400]
# Kurihara K.; Hori H.; Kojima K.: Vapor-Liquid Equilibrium Data for Acetone +
# Methanol + Benzene, Chloroform + Methanol + Benzene, and Constituent Binary
# Systems at 101.3 kPa. J.Chem.Eng.Data 43 (1998) 264-268
elif components==['Acetone','Methanol','Benzene']:
    T_exp=[353.24,337.70,329.26,331.14,333.60,332.27,335.04,333.28,330.87,
           332.54,330.73,333.95,331.20,330.65,330.60,330.71,330.45,331.65,
           331.36,330.72,332.96,330.35,333.23,330.34,334.78,330.18,330.34,
           330.13,330.67,331.71,330.04,329.98,329.82,334.41,329.92,330.05,
           329.97,329.64,330.89,329.61,333.93,329.61,330.81,334.48,329.35,
           330.72,329.47,334.16,329.79,329.18,331.07,331.73,329.69,333.66,
           329.66,330.50,328.98,328.82,328.96,330.40,328.96,330.31,328.66,
           329.01,330.20,328.60,328.60,329.47,329.21,329.24]
    x1_exp=[0.00000,0.00000,1.00000,0.02300,0.02400,0.06000,0.06600,0.06600,
            0.09400,0.11800,0.13600,0.14600,0.14900,0.15400,0.17500,0.19100,
            0.20100,0.20900,0.21400,0.21800,0.21900,0.22900,0.25000,0.25400,
            0.27400,0.27400,0.27500,0.29700,0.31000,0.32000,0.32300,0.32700,
            0.35000,0.35100,0.36100,0.37200,0.39800,0.40000,0.40500,0.41300,
            0.42600,0.43900,0.44700,0.46700,0.47800,0.48900,0.49200,0.49800,
            0.50400,0.50800,0.52300,0.53200,0.53700,0.55300,0.56200,0.57900,
            0.58700,0.61000,0.61700,0.62100,0.64600,0.65800,0.66900,0.68200,
            0.70700,0.72800,0.75000,0.76500,0.81600,0.84400]
    x2_exp=[0.00000,1.00000,0.00000,0.51100,0.90400,0.83100,0.10300,0.87600,
            0.62300,0.21000,0.53300,0.13300,0.72700,0.58400,0.50800,0.43400,
            0.54600,0.25700,0.73500,0.66800,0.16400,0.56900,0.14200,0.45900,
            0.08500,0.55500,0.61900,0.45700,0.31600,0.20400,0.57800,0.46800,
            0.49900,0.07700,0.41500,0.37800,0.35200,0.51300,0.22400,0.43400,
            0.06700,0.38000,0.20700,0.04100,0.39500,0.19200,0.34200,0.03800,
            0.27500,0.42000,0.14800,0.10800,0.26100,0.03300,0.24500,0.15600,
            0.33100,0.35100,0.29700,0.14000,0.26300,0.12500,0.29200,0.22200,
            0.10500,0.24000,0.22000,0.12200,0.11000,0.09200]
    y1_exp=[0.00000,0.00000,1.00000,0.02300,0.04100,0.09100,0.09100,0.11300,
            0.10800,0.13200,0.14800,0.18200,0.19900,0.17400,0.19100,0.20300,
            0.22500,0.22500,0.30400,0.27800,0.25500,0.26400,0.29600,0.27200,
            0.35700,0.31400,0.33700,0.32100,0.32300,0.34800,0.38600,0.35500,
            0.38900,0.44400,0.38300,0.39100,0.41600,0.45500,0.42800,0.44200,
            0.52300,0.46100,0.47200,0.59100,0.50600,0.51400,0.51100,0.62000,
            0.51800,0.54800,0.55900,0.58600,0.55000,0.66900,0.57300,0.60500,
            0.60700,0.63700,0.63100,0.64600,0.65600,0.68100,0.68200,0.68600,
            0.73000,0.73100,0.74900,0.77000,0.81700,0.84400]
    y2_exp=[0.00000,1.00000,0.00000,0.58400,0.78400,0.70300,0.41900,0.75500,
            0.58500,0.45600,0.54300,0.38700,0.61600,0.55300,0.52100,0.49400,
            0.52400,0.42700,0.61200,0.56900,0.36800,0.52200,0.33300,0.47700,
            0.24900,0.50300,0.53200,0.46100,0.40700,0.34500,0.49900,0.45600,
            0.46000,0.20100,0.42500,0.40500,0.38800,0.45200,0.32800,0.41700,
            0.16000,0.38700,0.30100,0.10000,0.38200,0.27600,0.35500,0.08900,
            0.31900,0.38400,0.22800,0.18600,0.30200,0.07000,0.28700,0.21900,
            0.32400,0.32800,0.30000,0.19600,0.27400,0.17600,0.28600,0.24300,
            0.14700,0.24400,0.22800,0.15300,0.13200,0.11200]
x1_exp=numpy.array([x1_exp]).reshape(-1,1)
x2_exp=numpy.array([x2_exp]).reshape(-1,1)
x3_exp=1-x1_exp-x2_exp
y1_exp=numpy.array([y1_exp]).reshape(-1,1)
y2_exp=numpy.array([y2_exp]).reshape(-1,1)
y3_exp=1-y1_exp-y2_exp
T_exp=numpy.array([T_exp]).reshape(-1,1)
VLE_Exp=numpy.concatenate((x1_exp,x2_exp,x3_exp,y1_exp,y2_exp,y3_exp,T_exp),
                          axis=1)
# Remove binary points
indexes=[]
for n,entry in enumerate(VLE_Exp):
    if entry[0]<10**-3 or entry[1]<10**-3 or entry[2]<10**-3:
        indexes.append(n)
VLE_Exp=numpy.delete(VLE_Exp,indexes,axis=0)
# Get Antoine parameters
antoine1=thermo.antoineParameters(components[0])
antoine2=thermo.antoineParameters(components[1])
antoine3=thermo.antoineParameters(components[2])
# Define Vapor Pressure Functions
def F_VP_1(T):
    P=thermo.antoineEquation(antoine1,T,getVar='P')
    return P
def F_VP_2(T):
    P=thermo.antoineEquation(antoine2,T,getVar='P')
    return P
def F_VP_3(T):
    P=thermo.antoineEquation(antoine3,T,getVar='P')
    return P  
def F_Inverse_VP_1(P):
    T=thermo.antoineEquation(antoine1,P,getVar='T')
    return T
def F_Inverse_VP_2(P):
    T=thermo.antoineEquation(antoine2,P,getVar='T')
    return T
def F_Inverse_VP_3(P):
    T=thermo.antoineEquation(antoine3,P,getVar='T')
    return T
Fs_VP=[F_VP_1,F_VP_2,F_VP_3]
Fs_Inverse_VP=[F_Inverse_VP_1,F_Inverse_VP_2,F_Inverse_VP_3]
# Define a copy of VLE_Exp
VLE_Exp_=VLE_Exp.copy()

# =============================================================================
# Active Learning
# =============================================================================

# Define X_AL (x1,x2,T) and Y_AL_i (gamma_i)
X_AL=numpy.array([]).reshape(-1,3)
Y_AL_1=numpy.array([]).reshape(-1,1)
Y_AL_2=numpy.array([]).reshape(-1,1)
Y_AL_3=numpy.array([]).reshape(-1,1)
# Define VLE xGrid
x_range=numpy.linspace(0.1,0.8,6)
aux=numpy.array(numpy.meshgrid(x_range,x_range)).T.reshape(-1,2)
indexes=[]
for n,entry in enumerate(aux): # Remove invalid entries (inneficient)
    if entry.sum()>1: indexes.append(n)
aux=numpy.delete(aux,indexes,axis=0)
xGrid=numpy.concatenate((aux,(1-aux[:,0]-aux[:,1]).reshape(-1,1)),axis=1)
# Get number of data points
nData=xGrid.shape[0]
# Get number of components
nC=xGrid.shape[1]
# Define output arrays
bubble=numpy.zeros([nData,nC+1])
# Fill outputs
bubble[:,0:nC]=xGrid
# Iterate over
for n in range(nData):
    if n==0:
        T=300
    else:
        # Guess temperature from previous point
        T=bubble[n-1,-1]
    # T Loop
    while True:
        # Calculate Vapor Pressures
        VP1=Fs_VP[0](T)
        VP2=Fs_VP[1](T)
        VP3=Fs_VP[2](T)
        # Calculate vapor composition
        y1=xGrid[n,0]*VP1/P_VLE
        y2=xGrid[n,1]*VP2/P_VLE
        y3=xGrid[n,2]*VP3/P_VLE
        yT=y1+y2+y3
        yT_error=1-yT
        # Check yT
        if abs(yT_error)<10**-6:
            break
        else:
            # Select new temperature
            T=T+yT_error*T/100       
    # Update output
    bubble[n,-1]=T
# Get minimum and maximum temperature from bubble
Tmin=bubble[:,-1].min()
Tmax=bubble[:,-1].max()
# Generate virtual points on top of bubble
T_range=numpy.linspace(Tmin*0.9,Tmax*1.1,100).reshape(-1,1)
X_VP_1=numpy.concatenate((numpy.ones(100).reshape(-1,1), # Pure comp. 1
                          numpy.zeros(100).reshape(-1,1),
                          T_range),axis=1)
X_VP_2=numpy.concatenate((numpy.zeros(100).reshape(-1,1), # Pure comp. 2
                          numpy.ones(100).reshape(-1,1),
                          T_range),axis=1)
X_VP_3=numpy.concatenate((numpy.zeros(100).reshape(-1,1), # Pure comp. 3
                          numpy.zeros(100).reshape(-1,1),
                          T_range),axis=1)
X_VP=numpy.concatenate((X_VP_1,X_VP_2,X_VP_3),axis=0)
# Plot grid
plt.figure(figsize=(3,1.7))
fig,ax=plt.subplots(facecolor='w')
td=TernaryDiagram(components,ax=ax)
td.scatter(xGrid,marker='s')
td.scatter(numpy.concatenate((X_VP[:,0:2],
                              (1-X_VP[:,0]-X_VP[:,1]).reshape(-1,1)),
                             axis=1),marker='s')
plt.show()
# Define X_Train_i
X_Train_1=numpy.concatenate((X_VP_1,X_AL),axis=0)
X_Train_2=numpy.concatenate((X_VP_2,X_AL),axis=0)
X_Train_3=numpy.concatenate((X_VP_3,X_AL),axis=0)
# Define Y_Train_i
Y_VP=numpy.ones((100,1))
Y_Train_1=numpy.concatenate((Y_VP,Y_AL_1),axis=0)
Y_Train_2=numpy.concatenate((Y_VP,Y_AL_2),axis=0)
Y_Train_3=numpy.concatenate((Y_VP,Y_AL_3),axis=0)
# Define new X_Test based on predicted VLE
X_Test=bubble.copy()
X_Test=numpy.delete(X_Test,2,axis=1) # Remove x3 column
# Initialize mean acquisition function vector
MAF_Vector=[]
# Loop over iterations requested (zeroth iteration without AL data)
for n in range(maxIter):
    # Define X_Scaler
    __,X_Scaler=ml.normalize(X_Test,method='MinMax')
    # Build GPs
    model_1=ml.buildGP(X_Train_1,Y_Train_1,X_Scaler=X_Scaler,gpConfig=gpConfig)
    model_2=ml.buildGP(X_Train_2,Y_Train_2,X_Scaler=X_Scaler,gpConfig=gpConfig)
    model_3=ml.buildGP(X_Train_3,Y_Train_3,X_Scaler=X_Scaler,gpConfig=gpConfig)
    # Store posteriors to decrease computational cost
    model_1=model_1.posterior()
    model_2=model_2.posterior()
    model_3=model_3.posterior()
    # Define gamma functions for VLE calculation
    def F_Gamma_1_GP(x,T):
        X=numpy.array([x[0],x[1],T]).reshape(1,3)
        pred,__=ml.gpPredict(model_1,X,X_Scaler=X_Scaler,gpConfig=gpConfig)
        return pred[0,0]
    def F_Gamma_2_GP(x,T):
        X=numpy.array([x[0],x[1],T]).reshape(1,3)
        pred,__=ml.gpPredict(model_2,X,X_Scaler=X_Scaler,gpConfig=gpConfig)
        return pred[0,0]
    def F_Gamma_3_GP(x,T):
        X=numpy.array([x[0],x[1],T]).reshape(1,3)
        pred,__=ml.gpPredict(model_3,X,X_Scaler=X_Scaler,gpConfig=gpConfig)
        return pred[0,0]
    Fs_gamma=[F_Gamma_1_GP,F_Gamma_2_GP,F_Gamma_3_GP]
    # Compute VLE
    bubble,dew,gammas=thermo.compute_Tx_VLE_Multinary(Fs_gamma,
                                                      Fs_VP,
                                                      Fs_Inverse_VP,
                                                      P_VLE,
                                                      zGrid=xGrid,
                                                      do_Bubble_Only=True)
    # Define new X_Test based on predicted VLE
    X_Test=bubble.copy()
    X_Test=numpy.delete(X_Test,2,axis=1) # Remove x3 column
    # Get STDs on new grid
    Y_Pred_1,Y_STD_1=ml.gpPredict(model_1,X_Test,X_Scaler=X_Scaler,
                                  gpConfig=gpConfig)
    Y_Pred_2,Y_STD_2=ml.gpPredict(model_2,X_Test,X_Scaler=X_Scaler,
                                  gpConfig=gpConfig)
    Y_Pred_3,Y_STD_3=ml.gpPredict(model_3,X_Test,X_Scaler=X_Scaler,
                                  gpConfig=gpConfig)
    # Compute vapor pressures
    vp1=[]
    vp2=[]
    vp3=[]
    for entry in X_Test:
        vp1.append(Fs_VP[0](entry[-1]))
        vp2.append(Fs_VP[1](entry[-1]))
        vp3.append(Fs_VP[2](entry[-1]))
    vp1=numpy.array(vp1).reshape(-1,1)
    vp2=numpy.array(vp2).reshape(-1,1)
    vp3=numpy.array(vp2).reshape(-1,1)
    # Compute acquisition function
    x1=X_Test[:,0].reshape(-1,1)
    x2=X_Test[:,1].reshape(-1,1)
    x3=(1-x1-x2).reshape(-1,1)
    AF=x1*(vp1/P_VLE)*Y_STD_1+x2*(vp2/P_VLE)*Y_STD_2+x3*(vp3/P_VLE)*Y_STD_3
    # Append metric
    MAF=AF.mean()
    MAF_Vector.append(MAF)
    # Select next point
    X_New=X_Test[AF.argmax(),:].reshape(1,3)
    # Find closest experimental match
    distance=numpy.sqrt((VLE_Exp[:,0]-X_New[0,0])**2+ \
                        (VLE_Exp[:,1]-X_New[0,1])**2+ \
                        (VLE_Exp[:,2]-(1-X_New[0,0]-X_New[0,1]))**2)
    exp_index=distance.argmin()
    X_New_=numpy.array([VLE_Exp[exp_index,0].copy(),
                        VLE_Exp[exp_index,1].copy(),
                        VLE_Exp[exp_index,-1].copy()]).reshape(-1,3)
    # Append to X_AL
    X_AL=numpy.append(X_AL,X_New_,axis=0)
    # Compute gammas
    gammas=thermo.get_Gammas_from_VLE(VLE_Exp[exp_index,:].reshape(1,-1),
                                      Fs_VP,P_VLE)
    # Remove availability of experimental point
    VLE_Exp=numpy.delete(VLE_Exp,exp_index,axis=0)
    # Append to Y_AL_i
    Y_AL_1=numpy.append(Y_AL_1,gammas[:,3].reshape(-1,1),axis=0)
    Y_AL_2=numpy.append(Y_AL_2,gammas[:,4].reshape(-1,1),axis=0)
    Y_AL_3=numpy.append(Y_AL_3,gammas[:,5].reshape(-1,1),axis=0)
    # Get minimum and maximum temperature from bubble
    Tmin=bubble[:,-1].min()
    Tmax=bubble[:,-1].max()
    # Generate virtual points on top of bubble
    T_range=numpy.linspace(Tmin,Tmax,100).reshape(-1,1)
    X_VP_1=numpy.concatenate((numpy.ones(100).reshape(-1,1),
                              numpy.zeros(100).reshape(-1,1),
                              T_range),axis=1)
    X_VP_2=numpy.concatenate((numpy.zeros(100).reshape(-1,1),
                              numpy.ones(100).reshape(-1,1),
                              T_range),axis=1)
    X_VP_3=numpy.concatenate((numpy.zeros(100).reshape(-1,1),
                              numpy.zeros(100).reshape(-1,1),
                              T_range),axis=1)
    X_VP=numpy.concatenate((X_VP_1,X_VP_2,X_VP_3),axis=0)
    # Define X_Train
    X_Train_1=numpy.concatenate((X_VP_1,X_AL),axis=0)
    X_Train_2=numpy.concatenate((X_VP_2,X_AL),axis=0)
    X_Train_3=numpy.concatenate((X_VP_3,X_AL),axis=0)
    # Define Y_Train_i
    Y_VP=numpy.ones((100,1))
    Y_Train_1=numpy.concatenate((Y_VP,Y_AL_1),axis=0)
    Y_Train_2=numpy.concatenate((Y_VP,Y_AL_2),axis=0)
    Y_Train_3=numpy.concatenate((Y_VP,Y_AL_3),axis=0)
    # Plot grid
    plt.figure(figsize=(3,1.7))
    fig,ax=plt.subplots(facecolor='w')
    td=TernaryDiagram(components,ax=ax)
    td.scatter(xGrid,z=AF,marker='s')
    td.scatter(xGrid[AF.argmax(),:].reshape(1,-1),marker='d',color='black')
    td.scatter(numpy.concatenate((X_AL[:,0:2],
                                  (1-X_AL[:,0]-X_AL[:,1]).reshape(-1,1)),
                                 axis=1),marker='o')
    plt.show()
    # Check GP_MPE
    if len(MAF_Vector)>3 and MAF<min_AF: break
# Remove last row of X_AL (added in the last iteration but not used)
X_AL=numpy.delete(X_AL,-1,0)
Y_AL_1=numpy.delete(Y_AL_1,-1,0)
Y_AL_2=numpy.delete(Y_AL_2,-1,0)
Y_AL_3=numpy.delete(Y_AL_3,-1,0)

# =============================================================================
# Pulbication Plots
# =============================================================================

# Define gamma functions for VLE calculation
def F_Gamma_1_GP(x,T):
    X=numpy.array([x[0],x[1],T]).reshape(1,3)
    pred,__=ml.gpPredict(model_1,X,X_Scaler=X_Scaler,gpConfig=gpConfig)
    return pred[0,0]
def F_Gamma_2_GP(x,T):
    X=numpy.array([x[0],x[1],T]).reshape(1,3)
    pred,__=ml.gpPredict(model_2,X,X_Scaler=X_Scaler,gpConfig=gpConfig)
    return pred[0,0]
def F_Gamma_3_GP(x,T):
    X=numpy.array([x[0],x[1],T]).reshape(1,3)
    pred,__=ml.gpPredict(model_3,X,X_Scaler=X_Scaler,gpConfig=gpConfig)
    return pred[0,0]
# Compute VLE
bubble,dew,gammas=thermo.compute_Tx_VLE_Multinary(Fs_gamma,
                                                  Fs_VP,
                                                  Fs_Inverse_VP,
                                                  P_VLE,
                                                  zGrid=VLE_Exp_[:,0:3],
                                                  do_Bubble_Only=True)
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
plt.figure(figsize=(3.33,2))
fig,ax=plt.subplots(facecolor='w')
td=TernaryDiagram(components,ax=ax)
# Experimental Data
td.scatter(VLE_Exp_[:,0:3],z=VLE_Exp_[:,-1],marker='s',
           label='Exp. Bubble')
td.scatter(VLE_Exp_[:,3:6],marker='d',color='black',
           label='Exp. Dew')
td.plot([VLE_Exp_[0,0:3],VLE_Exp_[0,3:6]],color='black',linewidth=2,
        label='Exp. Tie Line')
for entry in VLE_Exp_:
    td.plot([entry[0:3],entry[3:6]],color='black',linewidth=2)
# GP-Predicted
td.plot([bubble[0,:3],dew[0,:3]],linestyle='dashed',color='red',
        label='GP-Predicted Tie Line')
for n in range(bubble.shape[0]):
    td.plot([bubble[n,:3],dew[n,:3]],linestyle='dashed',color='red')
# AL points
X_AL_=numpy.concatenate((X_AL[:,0].reshape(-1,1),
                         X_AL[:,1].reshape(-1,1),
                         (1-X_AL[:,0]-X_AL[:,1]).reshape(-1,1)),axis=1)
td.scatter(X_AL_,marker='x',color='black',s=100,
           label='AL Training Data')
plt.legend(loc=(-0.1,0.7))
plt.show()