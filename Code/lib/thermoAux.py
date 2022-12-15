# -*- coding: utf-8 -*-
"""
Python library containing thermodynamics-related functions.

Sections:
    . Imports
    . NRTL Model
        . NRTL_Binary_getGamma()
        . NRTL_Parameters()
        . NRTL_Binary_fit()
    . Vapor-Liquid Equilibrium
        . antoineParameters()
        . antoineEquation()
        . compute_Tx_VLE_Binary()
        . compute_Tx_VLE_Ideal_Binary()
        . get_Gammas_from_VLE()
        . compute_Tx_VLE_Multinary()
    . Liquid-Liquid Equilibrium
        . compute_Tx_LLE_Binary()
    . Solid-Liquid Equilibrium
        . meltingProperties()
        . compute_Tx_SLE_Binary()
        . compute_Tx_SLE_Ideal_Binary()
        . get_Gammas_from_SLE()
        
Last edit: 2022-10-27
Author: Dinis Abranches
"""

# =============================================================================
# Imports
# =============================================================================

# General
import warnings

# Specific
import pandas
import numpy
import scipy
from matplotlib import pyplot as plt
from tqdm import tqdm

# =============================================================================
# NRTL Model
# =============================================================================

def NRTL_Binary_getGamma(parameters,x1,T):
    """
    NRTL_Binary_getGamma() calculates the activity coefficient of both
    components in a binary mixture using the NRTL model as implemented by
    Gmehling and co-authors in 10.1252/jcej.08we123.
    
    The NRTL parameters ("parameters") can be obtained using the function
    NRTL_Parameters().
    
    Parameters
    ----------
    parameters : pandas DataFrame
        Dataframe containing the NRTL parameters for the mixture.
        Columns:
            . Comp. 1
            . Comp. 2
            . a12
            . b12
            . c12
            . a21
            . b21
            . c21
            . alpha
    x1 : float
        Mole fraction of component 1.
    T : float
        Temperature /K.

    Returns
    -------
    gammas : list of floats
        List containing the activity coefficient of components 1 and 2
        ([gamma1,gamma2])

    """
    # Definition of ideal gas constant
    R=8.31446
    # Calculate x2
    x2=1-x1
    # Define parameters from input
    a_12=parameters.iloc[0,2]
    a_21=parameters.iloc[0,5]
    b_12=parameters.iloc[0,3]
    b_21=parameters.iloc[0,6]
    c_12=parameters.iloc[0,4]
    c_21=parameters.iloc[0,7]
    alpha=parameters.iloc[0,8]
    # Calculate delta gs
    dg_12=a_12+b_12*(10**-3)*T+c_12*(10**-5)*T**2
    dg_21=a_21+b_21*(10**-3)*T+c_21*(10**-5)*T**2
    # Calculate taus
    tau_12=dg_12/(R*T)
    tau_21=dg_21/(R*T)
    # Calculate Gs
    G_12=numpy.exp(-alpha*tau_12)
    G_21=numpy.exp(-alpha*tau_21)
    # Calculate gamma1
    A1=(G_21/(x1+x2*G_21))**2
    B1=tau_12*G_12/((x2+x1*G_12)**2)
    gamma1=numpy.exp((x2**2)*(tau_21*A1+B1))
    # Calculate gamma2
    A2=(G_12/(x2+x1*G_12))**2
    B2=tau_21*G_21/((x1+x2*G_21)**2)
    gamma2=numpy.exp((x1**2)*(tau_12*A2+B2))
    # Output
    gammas=[gamma1,gamma2]
    return gammas

def NRTL_Parameters(comp1,comp2):
    """
    NRTL_Parameters() returns the NRTl parameters for the mixture defined by
    "comp1" and "comp2". The order of the components matter, e.g., if the
    database entry lists acetone/chloroform, searching chloroform/acetone does
    not return a match.
    Assumes the NRTL implementation described by Gmehling and co-authors in
    10.1252/jcej.08we123.
    
    Parameters
    ----------
    comp1 : string
        Name of component 1.
    comp2 : string
        Name of component 2.

    Raises
    ------
    ValueError
        If the database does not contain an entry for comp1/comp2, an exception
        is raised.
    
    Returns
    -------
    lineResult : pandas DataFrame
        Dataframe containing the NRTL parameters for the mixture.
        Columns:
            . Comp. 1
            . Comp. 2
            . a12
            . b12
            . c12
            . a21
            . b21
            . c21
            . alpha
            
    """
    # Build database
    columns=['Comp. 1','Comp. 2',
             'a_12','b_12','c_12','a_21','b_21','c_21','alpha']
    NRTL_Database=pandas.DataFrame(data=[],columns=columns)
    # Ref: 10.1252/jcej.08we123
    line1=['Acetone','Methanol',1048.9,6707.1,-2426.4,1249.7,-4600.1,1336.3,
           0.4378]
    line2=['Acetone','Benzene',4293.5,-27787.7,4139.2,-4437.1,34895.8,-4949,
           0.1748]
    line3=['Acetone','Chloroform',14064.6,-13859.1,-52.3,-9032.8,11762.6,
           -1472.5,0.2]
    line4=['Acetone','Cyclohexane',5957.1,-12363,337.8,12049.7,-42410.6,4950.4,
           0.4212]
    line5=['Acetone','Hexane',5194.7,-8306.5,10.5,2919.7,-2254.34,9.3,0.2372]
    line6=['Acetone','Heptane',-1185.9,16010.7,-132.5,8533.6,-21951.6,-252.2,
           0.2078]
    line7=['Acetone','TCM',-4021.5,27311.8,-3912.8,3072.5,-4256.7,485.8,0.71]
    line8=['Acetone','Toluene',1016.2,-2153.1,573.1,-166.7,7155.4,-1101.2,0.71]
    line9=['Acetone','Water',3489.3,3477.2,-1582.2,-13165.1,73040.1,-5699.4,
           0.5466]
    ref1=[line1,line2,line3,line4,line5,line6,line7,line8,line9]
    ref1=pandas.DataFrame(data=ref1,columns=columns)
    NRTL_Database=pandas.concat([NRTL_Database,ref1])
    # Search components and return list
    lineResult=NRTL_Database[(NRTL_Database['Comp. 1']==comp1)]
    lineResult=lineResult[(lineResult['Comp. 2']==comp2)]
    # Check if empty
    if lineResult.empty: raise ValueError('Mixture not found in database.')
    # Output
    return lineResult

def NRTL_Binary_fit(X_1,Y_1,X_2,Y_2,components):
    """
    NRTL_Binary_fit() fits an NRTL model (as implemented by Gmehling and
    co-authors in 10.1252/jcej.08we123) to activity coefficient data.

    Parameters
    ----------
    X_1 : numpy array (N1,2)
        Array containing the features (x1,T) associated with Y_1.
    Y_1 : (N1,1)
        Activity coefficients of component 1 associated with the (x1,T) data
        points contained in X_1.
    X_2 : numpy array (N2,2)
        Array containing the features (x1,T) associated with Y_2.
    Y_2 : (N2,1)
        Activity coefficients of component 2 associated with the (x2,T) data
        points contained in X_2.
    components : list of strings
        List of size 2 containing the names of the components in the binary
        system.

    Returns
    -------
    parameters : pandas DataFrame
        Dataframe containing the NRTL parameters for the mixture.
        Columns:
            . Comp. 1
            . Comp. 2
            . a12
            . b12
            . c12
            . a21
            . b21
            . c21
            . alpha

    """
    # Define objective function to be minimized
    def objective(parameters,X_1,Y_1,X_2,Y_2):
        # Convert parameters array to input pandas DataFrame
        columns=['Comp. 1','Comp. 2',
                 'a_12','b_12','c_12','a_21','b_21','c_21','alpha']
        line=[components[0],components[1],parameters[0],parameters[1],
              parameters[2],parameters[3],parameters[4],parameters[5],
              parameters[6]]
        inputParameters=pandas.DataFrame(data=[line],columns=columns)
        # Initialize square error
        SE=0
        # Loop over data points
        for n in range(X_1.shape[0]):
            # Get gamma for data point
            gammas=NRTL_Binary_getGamma(inputParameters,X_1[n,0],X_1[n,1])
            # Compute SE
            SE+=(Y_1[n]-gammas[0])**2
        SE_1=numpy.array(SE).mean()
        # Initialize square error
        SE=0
        # Loop over data points
        for n in range(X_2.shape[0]):
            # Get gamma for data point
            gammas=NRTL_Binary_getGamma(inputParameters,X_2[n,0],X_2[n,1])
            # Compute SE
            SE+=(Y_2[n]-gammas[1])**2
        SE_2=numpy.array(SE).mean()
        # Compute RMSE
        #RMSE=max(SE_1,SE_2)
        RMSE=(SE_1+SE_2)/2
        # Output
        return RMSE
    aux=(-10**6,10**6)
    init=(numpy.random.rand()*2-0.5)*(10**3)
    # Perform minimization
    results=scipy.optimize.minimize(objective,
                                    [init,init,init,init,init,init,0],
                                    args=(X_1,Y_1,X_2,Y_2),
                                    bounds=(aux,aux,aux,aux,aux,aux,(0.1,0.6)),
                                    method='Nelder-Mead',
                                    options={'maxiter':10**9})
    # Get optimized parameters
    parameters=results.x
    # Convert parameters array to input pandas DataFrame
    columns=['Comp. 1','Comp. 2',
             'a_12','b_12','c_12','a_21','b_21','c_21','alpha']
    line=[components[0],components[1],parameters[0],parameters[1],
          parameters[2],parameters[3],parameters[4],parameters[5],
          parameters[6]]
    parameters=pandas.DataFrame(data=[line],columns=columns)
    # Output
    return parameters,results

# =============================================================================
# Vapor-Liquid Equilibrium
# =============================================================================

def antoineParameters(compound,T=None):
    """
    antoineParameters() returns the Antoine Equation parameters for compound
    "compound" at all temperature ranges available, unless a temperature "T" is
    specified. If no parameters are available for temperature "T", a warning
    is issued and the parameter set with the closest temperature range is
    returned.
    Assumes:
        . Equation of the form: log10(P) = A − (B / (T + C))
        . P = vapor pressure (bar)
        . T = temperature (K)

    Parameters
    ----------
    compound : string
        Name of the compound.
    T : float, optional
        Temperature (/K).
        If None, all "compound" entries are returned.

    Raises
    ------
    ValueError
        Error raised if the compound is not found in the database.
    UserWarning
        Warning raised if T is outside the range of the database entry for
        the specific compound.

    Returns
    -------
    lineResult : Pandas DataFrame
        Antoine parameters for the compound in all T ranges or at specified T.
        Columns: 'Compound','T_min','T_max','A','B','C'

    """    
    # Create empty pandas dataframe
    columns=['Compound','T_min','T_max','A','B','C']
    database=pandas.DataFrame(data=[],columns=columns)
    # Acetone (NIST)
    line=['Acetone',259.16,507.60,4.42448,1312.253,-32.445]
    database.loc[len(database.index)]=line
    # Methanol (NIST)
    line=['Methanol',288.1,356.83,5.20409,1581.341,-33.50]
    database.loc[len(database.index)]=line
    line=['Methanol',353.5,512.63,5.15853,1569.613,-34.846]
    database.loc[len(database.index)]=line
    # Benzene (NIST)
    line=['Benzene',287.70,354.07,4.01814,1203.835,-53.226]
    database.loc[len(database.index)]=line
    line=['Benzene',333.4,373.5,4.72583,1660.652,-1.461]
    database.loc[len(database.index)]=line
    line=['Benzene',421.56,554.8,4.60362,1701.073,20.806]
    database.loc[len(database.index)]=line
    # Chloroform (NIST)
    line=['Chloroform',215,334.4,4.20772,1233.129,-40.953]
    database.loc[len(database.index)]=line
    # Cyclohexane (NIST)
    line=['Cyclohexane',293.06,354.73,3.96988,1203.526,-50.287]
    database.loc[len(database.index)]=line
    line=['Cyclohexane',323,523,4.13983,1316.554,-35.581]
    database.loc[len(database.index)]=line
    # Hexane (NIST)
    line=['Hexane',177.70,264.93,3.45604,1044.038,-53.893]
    database.loc[len(database.index)]=line
    line=['Hexane',286.18,342.69,4.00266,1171.53,-48.784]
    database.loc[len(database.index)]=line
    # Heptane (NIST)
    line=['Heptane',185.29,295.60,4.81803,1635.409,-27.338]
    database.loc[len(database.index)]=line
    line=['Heptane',299.07,372.43,4.02832,1268.636,-56.199]
    database.loc[len(database.index)]=line
    # Tetrachloromethane (NIST)
    line=['TCM',293.03,350.86,4.01720,1221.781,-45.739]
    database.loc[len(database.index)]=line
    # Toluene (NIST)
    line=['Toluene',273,323,4.14157,1377.578,-50.507]
    database.loc[len(database.index)]=line
    line=['Toluene',303,343,4.08245,1346.382,-53.508]
    database.loc[len(database.index)]=line
    #line=['Toluene',420,580,4.54436,1738.123,0.394]
    database.loc[len(database.index)]=line
    # Water (NIST)
    line=['Water',255.9,373,4.6543,1435.264,-64.848]
    database.loc[len(database.index)]=line
    line=['Water',379,573,3.55959,643.748,-198.043]
    database.loc[len(database.index)]=line
    # Thymol (NIST)
    line=['Thymol',337.5,505,5.29395,2522.332,-28.575]
    database.loc[len(database.index)]=line
    # Menthol (NIST)
    line=['Menthol',329,485,5.38347,2405.946,-37.853]
    database.loc[len(database.index)]=line
    # Search compound
    compoundEntries=database[database['Compound']==compound]
    # Raise exception if compound not found
    if compoundEntries.empty:
        raise ValueError('No Antoine parameters for compound '+compound+'.')
    # Select parameters based on temperature requested
    if T is not None:
        found=False
        for n in range(compoundEntries.shape[0]):
            if T>compoundEntries.iloc[n,1] and T<compoundEntries.iloc[n,2]:
                lineResult=compoundEntries.iloc[n,:]
                found=True
                break
        # If T is outside all ranges, issue warning and choose closest range
        if not found:
            warnings.warn('No suitable T range found for the Antoine '
                          +'parameters of compound '+compound+'. '
                          +'Selecting closest set.')
            distance=[]
            for n in range(compoundEntries.shape[0]):
                distance1=abs(T-compoundEntries.iloc[n,1])
                distance2=abs(T-compoundEntries.iloc[n,2])
                distance.append(min(distance1,distance2))
            lineResult=compoundEntries.iloc[distance.index(min(distance)),:]
    else:
        lineResult=compoundEntries
    # Output
    return lineResult

def antoineEquation(parameters,var,getVar='P'):
    """
    antoineEquation() calculates vapor pressure given temperature or
    temperature given vapor pressure.
    Pressure must be inputted in Pascal; it is converted to bar internally.
    
    Assumes:
        . Equation of the form: log10(P) = A − (B / (T + C))
        . P = vapor pressure (bar but function input must be Pascal)
        . T = temperature (K)
    
    Equation parameters ("parameters") should be obtained from
    antoineParameters().
    
    Parameters
    ----------
    parameters : Pandas DataFrame
        Antoine parameters for the compound in all T ranges.
        Columns: 'Compound','T_min','T_max','A','B','C'
    var : float
        Temperature (/K) or Pressure (/Pa).
    getVar : string, optional
        Whether to calculate pressure ('P', expects temperature as var input)
        or temperature ('T', expects pressure as var input).
        Options:
            . 'P'
            . 'T'
        The default is 'P'.

    Raises
    ------
    ValueError
        Error raised if internal consistency test fails.
    UserWarning
        Warning raised if T is outside the range of the database entry for
        the specific compound.

    Returns
    -------
    output : float
        Pressure (/Pa) or Temperature (/K).

    """
    # Define issueWarning
    issueWarning=False
    # Check input
    if getVar=='P':
        T_input=var
        # Get parameters for T_input
        found=False
        for n in range(parameters.shape[0]):
            cond1=T_input>parameters.iloc[n,1]
            cond2=T_input<parameters.iloc[n,2]
            if cond1 and cond2:
                lineResult=parameters.iloc[n,:]
                found=True
                break
        # If T is outside all ranges, issue warning and choose closest range
        if not found:
            issueWarning=True
            distance=[]
            for n in range(parameters.shape[0]):
                distance1=abs(T_input-parameters.iloc[n,1])
                distance2=abs(T_input-parameters.iloc[n,2])
                distance.append(min(distance1,distance2))
            lineResult=parameters.iloc[distance.index(min(distance)),:]
        A=lineResult[3]
        B=lineResult[4]
        C=lineResult[5]
        # Calculate pressure
        P=10**(A-B/(T_input+C))
        # Check consistency
        T=-C+B/(A-numpy.log10(P))
        if abs(T_input-T)>10**-6:
            raise ValueError('Antoine Equation: consistency test failed.')
        # Convert bar to pascal
        output=P*10**5
    elif getVar=='T':
        # Convert pascal to bar
        P_input=var*10**-5
        # Get parameters for initial guess temperature
        T=100
        found=False
        for n in range(parameters.shape[0]):
            cond1=T>parameters.iloc[n,1]
            cond2=T<parameters.iloc[n,2]
            if cond1 and cond2:
                lineResult=parameters.iloc[n,:]
                found=True
                break
        # If T is outside all ranges, issue warning and choose closest range
        if not found:
            issueWarning=True
            distance=[]
            for n in range(parameters.shape[0]):
                distance1=abs(T-parameters.iloc[n,1])
                distance2=abs(T-parameters.iloc[n,2])
                distance.append(min(distance1,distance2))
            lineResult=parameters.iloc[distance.index(min(distance)),:]
        A=lineResult[3]
        B=lineResult[4]
        C=lineResult[5]
        # Calculate new temperature
        T_OLD=-C+B/(A-numpy.log10(P_input))
        # Loop
        while True:
            # Get parameters for T_OLD
            found=False
            for n in range(parameters.shape[0]):
                cond1=T_OLD>parameters.iloc[n,1]
                cond2=T_OLD<parameters.iloc[n,2]
                if cond1 and cond2:
                    lineResult=parameters.iloc[n,:]
                    found=True
                    break
            # If T is outside all ranges,issue warning and choose closest range
            if not found:
                issueWarning=True
                distance=[]
                for n in range(parameters.shape[0]):
                    distance1=abs(T_OLD-parameters.iloc[n,1])
                    distance2=abs(T_OLD-parameters.iloc[n,2])
                    distance.append(min(distance1,distance2))
                lineResult=parameters.iloc[distance.index(min(distance)),:]
            A=lineResult[3]
            B=lineResult[4]
            C=lineResult[5]
            # Calculate new temperature
            T_NEW=-C+B/(A-numpy.log10(P_input))
            # Check convergence
            if abs(T_NEW-T_OLD)<10**-6: break
            else: T_OLD=T_NEW
        # Check consistency
        P=10**(A-B/(T_NEW+C))
        if abs(P_input-P)>10**-6:
            raise ValueError('Antoine Equation: consistency test failed.')
        output=T_NEW
    # Issue warning
    if issueWarning:
        warnings.warn('No suitable T range found for the Antoine '
                      +'parameters of compound '+parameters.iloc[0,0]+'. '
                      +'Selecting closest set.')
    # Output
    return output


def compute_Tx_VLE_Binary(Fs_gamma,Fs_VP,Fs_Inverse_VP,P,
                          z1_range=numpy.linspace(0,1,101),do_Bubble_Only=True,
                          plot=False,title=None):
    """
    compute_Tx_VLE_Binary() computes the VLE phase diagram of the mixture defined
    by the input functions (Fs) using a gamma-phi apprach. Assumptions:
        . Binary Mixture (will change)
        . Ideal vapor phase (y_i * P = x_i * gamma_i * P_i^sat)

    Parameters
    ----------
    Fs_gamma : list of function handlers
        List where each entry "i" is the funciton that returns gamma for
        component "i". Must be of the type gamma=F_gamma(x1,T). T in K.
    Fs_VP : list of function handlers
        List where each entry "i" is the funciton that returns the vapor
        pressure for component "i". Must be of the type vp=F_VP(T). T in K, vp
        in Pa.
    Fs_Inverse_VP : list of function handlers
        List where each entry "i" is the funciton that returns the temperature
        corresponding to a given vapor pressure of component "i". Must be of
        the type T=F_VP(vp). T in K, vp in Pa.
    P : float
        Pressure of the system (/Pa).
    z1_range : numpy array, optional
        Range of z1 to perform VLE calculations.
        The default is numpy.linspace(0,1,101).
    do_Bubble_Only : boolean, optional
        If True, only the bubble curve is explicitly calculated, with the dew
        curve taken as the composition in equilibrium with the bubble curve.
        The default is True.
    plot : boolean, optional
        Whether to plot the VLE.
        The default is False.
    title : string, optional
        Title of the plot. Must be given if plot=True.
        The default is None.

    Returns
    -------
    bubble : numpy array (N,2)
        Array with the results for the bubble curve. N is the number of grid
        points, N=len(z1_range).
        Columns: "x1","T_bubble"
    dew : numpy array (N,2)
        Array with the results for the dew curve.  N is the number of grid
        points, N=len(z1_range).
        Columns: "y1","T_dew"
    gammas : numpy array (N,3)
        Array with the activity coefficients of each component at the VLE 
        equilibrium temperature for each x composition point.
        Columns: "x1","gamma_1","gamma_2"

    """
    # Define output arrays
    bubble=numpy.zeros((len(z1_range),2))
    dew=numpy.zeros((len(z1_range),2))
    gammas=numpy.zeros((len(z1_range),3))
    # Fill first column of outputs
    bubble[:,0]=z1_range
    gammas[:,0]=z1_range
    if not do_Bubble_Only:
        dew[:,0]=z1_range
    else:
        dew[0,0]=0
        dew[-1,0]=1
    # Calculate boiling points of components
    T1=Fs_Inverse_VP[0](P)
    T2=Fs_Inverse_VP[1](P)
    # Add to outputs
    bubble[0,1]=T2
    bubble[-1,1]=T1
    dew[0,1]=T2
    dew[-1,1]=T1
    gammas[-1,1]=1
    gammas[0,2]=1
    gammas[0,1]=gamma1=Fs_gamma[0](0,T2)
    gammas[-1,2]=gamma2=Fs_gamma[1](1,T1)
    # Select z range without pure components
    z1=z1_range[1:-1]
    # Iterate over new z1 range (Bubble Point Algorithm)
    for n in tqdm(range(len(z1)),'Computing Bubble Point: '):
        # Guess temperature from previous point
        T=bubble[n,1]
        # T Loop
        while True:
            # Calculate Gammas
            gamma1=Fs_gamma[0](z1[n],T)
            gamma2=Fs_gamma[1](z1[n],T)
            # Calculate Vapor Pressures
            VP1=Fs_VP[0](T)
            VP2=Fs_VP[1](T)
            # Calculate vapor composition
            y1=z1[n]*gamma1*VP1/P
            y2=(1-z1[n])*gamma2*VP2/P
            yT=y1+y2
            yT_error=1-yT
            # Check yT
            if abs(yT_error)<10**-6:
                break
            else:
                # Select new temperature
                T=T+yT_error*T/100       
        # Update output
        bubble[n+1,1]=T
        gammas[n+1,1]=gamma1
        gammas[n+1,2]=gamma2
        if do_Bubble_Only:
            dew[n+1,0]=y1
            dew[n+1,1]=T
    # Dew Point Algorithm
    if not do_Bubble_Only:
        # Iterate over new z1 range
        for n in range(len(z1)):
            # Guess temperature from previous point
            T=dew[n,1]
            # T Loop
            while True:
                # Calculate new Vapor Pressures
                VP1=Fs_VP[0](T)
                VP2=Fs_VP[1](T)
                # Assume ideal liquid phase to estimate liquid composition
                x1=z1[n]*P/VP1
                x2=(1-z1[n])*P/VP2
                xT_OLD=x1+x2
                # Liquid composition loop
                while True:
                    # Calculate Gammas
                    gamma1=Fs_gamma[0](x1,T)
                    gamma2=Fs_gamma[1](x1,T)
                    # Calculate liquid composition
                    x1=z1[n]*P/(gamma1*VP1)
                    x2=(1-z1[n])*P/(gamma2*VP2)
                    xT_NEW=x1+x2
                    if abs(xT_NEW-xT_OLD)<10**-6:
                        xT=xT_NEW
                        break
                    else:
                        x1=x1/xT_NEW
                        x2=x2/xT_NEW
                        xT_OLD=xT_NEW
                xT_error=xT-1
                # Check xT
                if abs(xT_error)<10**-6:
                    break
                else:
                    # Select new temperature
                    T=T+xT_error*T/100
            # Update output
            dew[n+1,1]=T
    # Plot
    if plot:
        plt.rcParams['figure.dpi'] = 600
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['mathtext.rm'] = 'serif'
        plt.rcParams['mathtext.it'] = 'serif:italic'
        plt.rcParams['mathtext.bf'] = 'serif:bold'
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams["savefig.pad_inches"] = 0.02
        plt.plot(bubble[:,0],bubble[:,1],'--k')
        plt.plot(dew[:,0],dew[:,1],'--k')
        plt.title(title)
        plt.xlabel('$\mathregular{z_1}$')
        plt.ylabel('T /K')
    # Output
    return bubble,dew,gammas

def compute_Tx_VLE_Ideal_Binary(Fs_VP,Fs_Inverse_VP,P,
                                z1_range=numpy.linspace(0,1,101),
                                plot=False,title=None):
    """
    compute_Tx_VLE_Ideal_Binary() computes the ideal VLE phase diagram of the
    mixture defined by the input functions (Fs).

    Parameters
    ----------
    Fs_VP : list of function handlers
        List where each entry "i" is the funciton that returns the vapor
        pressure for component "i". Must be of the type vp=F_VP(T). T in K, vp
        in Pa.
    Fs_Inverse_VP : list of function handlers
        List where each entry "i" is the funciton that returns the temperature
        corresponding to a given vapor pressure of component "i". Must be of
        the type T=F_VP(vp). T in K, vp in Pa.
    P : float
        Pressure of the system (/Pa).
    z1_range : numpy array, optional
        Range of z1 to perform VLE calculations.
        The default is numpy.linspace(0,1,101).
    plot : boolean, optional
        Whether to plot the VLE.
        The default is False.
    title : string, optional
        Title of the plot. Must be given if plot=True.
        The default is None.

    Returns
    -------
    bubble : numpy array (N,2)
        Array with the results for the bubble curve. N is the number of grid
        points, N=len(z1_range).
        Columns: "x1","T_bubble"
    dew : numpy array (N,2)
        Array with the results for the dew curve.  N is the number of grid
        points, N=len(z1_range).
        Columns: "y1","T_dew"
    gammas : numpy array (N,3)
        Array with the activity coefficients of each component at the VLE 
        equilibrium temperature for each x composition point.
        Columns: "x1","gamma_1","gamma_2"

    """
    # Define output arrays
    bubble=numpy.zeros((len(z1_range),2))
    dew=numpy.zeros((len(z1_range),2))
    # Fill first column of outputs
    bubble[:,0]=z1_range
    dew[0,0]=0
    dew[-1,0]=1
    # Calculate boiling points of components
    T1=Fs_Inverse_VP[0](P)
    T2=Fs_Inverse_VP[1](P)
    # Add to outputs
    bubble[0,1]=T2
    bubble[-1,1]=T1
    dew[0,1]=T2
    dew[-1,1]=T1
    # Select z range without pure components
    z1=z1_range[1:-1]
    # Iterate over new z1 range (Bubble Point Algorithm)
    for n in tqdm(range(len(z1)),'Computing Bubble Point: '):
        # Guess temperature from previous point
        T=bubble[n,1]
        # T Loop
        while True:
            # Calculate Vapor Pressures
            VP1=Fs_VP[0](T)
            VP2=Fs_VP[1](T)
            # Calculate vapor composition
            y1=z1[n]*VP1/P
            y2=(1-z1[n])*VP2/P
            yT=y1+y2
            yT_error=1-yT
            # Check yT
            if abs(yT_error)<10**-6:
                break
            else:
                # Select new temperature
                T=T+yT_error*T/100       
        # Update output
        bubble[n+1,1]=T
        dew[n+1,0]=y1
        dew[n+1,1]=T
    # Plot
    if plot:
        plt.rcParams['figure.dpi'] = 600
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['mathtext.rm'] = 'serif'
        plt.rcParams['mathtext.it'] = 'serif:italic'
        plt.rcParams['mathtext.bf'] = 'serif:bold'
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams["savefig.pad_inches"] = 0.02
        plt.plot(bubble[:,0],bubble[:,1],'--k')
        plt.plot(dew[:,0],dew[:,1],'--k')
        plt.title(title)
        plt.xlabel('$\mathregular{z_1}$')
        plt.ylabel('T /K')
    # Output
    return bubble,dew

def get_Gammas_from_VLE(vleData,Fs_VP,P):
    """
    get_Gammas_from_VLE() computes activity coefficients from VLE data.

    Parameters
    ----------
    vleData : numpy array (K,2*N+1)
        For a mixture of N components, vleData contains the composition of the
        liquid phase (first N columnds), the composition of the vapor phase
        (N+1 to N+N columns), and the equilibrium temperature (last column).
    Fs_VP : list of function handlers
        List where each entry "i" is the funciton that returns the vapor
        pressure for component "i". Must be of the type vp=F_VP(T). T in K, vp
        in Pa.
    P : float
        Pressure of the system (/Pa).

    Raises
    ------
    ValueError
        If size of vleData is not consistent, an exception is raised.

    Returns
    -------
    gammas : numpy array (K,2*N)
        Array containing the activity coefficients of each component (last N
        columns) for the liquid composition specified by the first N columns.

    """
    # Get number of data points
    nData=vleData.shape[0]
    # Get number of components
    nC=len(Fs_VP)
    # Check input dimensions
    if vleData.shape[1]!=2*nC+1:
        raise ValueError('Check number of components in get_Gammas_from_VLE()')
    # Initialize gammas container
    gammas=numpy.zeros([nData,nC*2])
    # Loop over data
    for n in range(nData):
        # Get temperature
        T=vleData[n,-1]
        # Loop over components
        for k in range(nC):
            # Compute vapor pressure
            VP=Fs_VP[k](T)
            # Get compositions
            x=vleData[n,k]
            y=vleData[n,nC+k]
            # Add liquid composition to gammas
            gammas[n,k]=vleData[n,k]
            # Compute gamma
            gamma=P*y/(x*VP)
            # Add gamma to gammas
            gammas[n,nC+k]=gamma
    # Output
    return gammas

def compute_Tx_VLE_Multinary(Fs_gamma,Fs_VP,Fs_Inverse_VP,P,
                   zGrid=numpy.linspace(0,1,101),
                   do_Bubble_Only=True,
                   plot=False,title=None):
    """
    compute_Tx_VLE() computes the VLE phase diagram of the mixture defined
    by the input functions (Fs) using a gamma-phi apprach. 
    The number of components is inferred from the shape of zGrid.

    Assumptions:
        . Ideal vapor phase (y_i * P = x_i * gamma_i * P_i^sat)

    Parameters
    ----------
    Fs_gamma : list of function handlers
        List where each entry "i" is the funciton that returns gamma for
        component "i". Must be of the type gamma=F_gamma(x1,T). T in K.
    Fs_VP : list of function handlers
        List where each entry "i" is the funciton that returns the vapor
        pressure for component "i". Must be of the type vp=F_VP(T). T in K, vp
        in Pa.
    Fs_Inverse_VP : list of function handlers
        List where each entry "i" is the funciton that returns the temperature
        corresponding to a given vapor pressure of component "i". Must be of
        the type T=F_VP(vp). T in K, vp in Pa.
    P : float
        Pressure of the system (/Pa).
    z1_range : numpy array, optional
        Range of z1 to perform VLE calculations.
        The default is numpy.linspace(0,1,101).
    do_Bubble_Only : boolean, optional
        If True, only the bubble curve is explicitly calculated, with the dew
        curve taken as the composition in equilibrium with the bubble curve.
        The default is True.
    plot : boolean, optional
        Whether to plot the VLE.
        The default is False.
    title : string, optional
        Title of the plot. Must be given if plot=True.
        The default is None.

    Returns
    -------
    bubble : numpy array (N,2)
        Array with the results for the bubble curve. N is the number of grid
        points, N=len(z1_range).
        Columns: "x1","T_bubble"
    dew : numpy array (N,2)
        Array with the results for the dew curve.  N is the number of grid
        points, N=len(z1_range).
        Columns: "y1","T_dew"
    gammas : numpy array (N,3)
        Array with the activity coefficients of each component at the VLE 
        equilibrium temperature for each x composition point.
        Columns: "x1","gamma_1","gamma_2"

    """
    # Get number of data ponts
    nData=zGrid.shape[0]
    # Get number of components
    nC=zGrid.shape[1]
    # Check dimensions of remaining inputs
    if len(Fs_VP)!=nC or len(Fs_Inverse_VP)!=nC or zGrid.shape[1]!=nC:
        raise ValueError('Check number of components in compute_Tx_VLE().')
    # Define output arrays
    bubble=numpy.zeros([nData,nC+1])
    dew=numpy.zeros([nData,nC+1])
    gammas=numpy.zeros([nData,nC*2])
    # Fill compositon columns of outputs
    bubble[:,0:nC]=zGrid
    gammas[:,0:nC]=zGrid
    if not do_Bubble_Only:
        dew[:,0:nC]=zGrid
    # Conpute pure component boiling points
    bPs=numpy.zeros(nC)
    for n in range(nC): bPs[n]=Fs_Inverse_VP[n](P)
    # Bubble Point Algorithm
    for n in tqdm(range(nData),'Computing Bubble Point: '):
        # If first iteration, use pure component boiling temperature as guess
        if n==0:
            # Get dominant component
            comp=numpy.argmax(bubble[0,0:nC])
            T=bPs[comp]
        else:
            # Guess temperature from previous point
            T=bubble[n-1,nC]
        # T Loop
        while True:
            # Update gammas, vapor pressures, and vapor compositions
            gammas_it=numpy.zeros(nC)
            VPs_it=numpy.zeros(nC)
            y_it=numpy.zeros(nC)
            for k in range(nC):
                # Calculate gammas
                gammas_it[k]=Fs_gamma[k](zGrid[n,0:nC],T)
                # Calculate Vapor Pressures
                VPs_it[k]=Fs_VP[k](T)
                # Calculate vapor compositions
                y_it[k]=zGrid[n,k]*gammas_it[k]*VPs_it[k]/P
            # Get total vapor phase compositions
            yT=y_it.sum()
            # Get deviation from unity
            yT_error=1-yT
            # Check yT
            if abs(yT_error)<10**-6:
                # Converged; break loop
                break
            else:
                # Select new temperature
                T=T+yT_error*T/200
        # Update output
        bubble[n,nC]=T
        gammas[n,nC:]=gammas_it
        if do_Bubble_Only:
            dew[n,:nC]=y_it
            dew[n,nC]=T
    # Output
    return bubble,dew,gammas

# =============================================================================
# Liquid-Liquid Equilibrium
# =============================================================================

def compute_Tx_LLE_Binary(Fs_gamma,T_range):
    """
    compute_Tx_LLE_Binary() computes the LLE phase diagram of the binary
    mixture defined by "Fs_gamma" in the temperature range requested.

    Parameters
    ----------
    Fs_gamma : list of function handlers
        List where each entry "i" is the funciton that returns gamma for
        component "i". Must be of the type gamma=F_gamma(x1,T). T in K.
    T_range : numpy array
        Temperatures, in K, at which LLE is calculated.

    Returns
    -------
    LLE : numpy array (K,3)
        Column #1: Temperature at which LLE occurs
        Column #2: Composition (x1) of the first phase
        Column #3: Composition (x1) of the second phase
    gammas : numpy array (K,5)
        Column #1: Temperature at which LLE occurs
        Column #2: Gamma of the first component in the first phase
        Column #3: Gamma of the second component in the first phase
        Column #4: Gamma of the first component in the second phase
        Column #5: Gamma of the second component in the second phase
        
    """
    # Initialize LLE containers
    LLE=[]
    gammas=[]
    # Define cond5 (LLE was found in previous iteration)
    cond5=False
    # Loop over temperatura range
    for T in tqdm(T_range,'Computing LLE: '):
        if not cond5:
            # Guess initial phase compositions
            x_alpha_1_OLD=0.999999
            x_beta_1_OLD=0.0000001
        # Update gammas
        gamma_alpha_1=Fs_gamma[0](x_alpha_1_OLD,T)
        gamma_alpha_2=Fs_gamma[1](x_alpha_1_OLD,T)
        gamma_beta_1=Fs_gamma[0](x_beta_1_OLD,T)
        gamma_beta_2=Fs_gamma[1](x_beta_1_OLD,T)
        # Loop to obtain equilibrium compositions
        while True:
            # Update K-values
            K_1=gamma_alpha_1/gamma_beta_1
            K_2=gamma_alpha_2/gamma_beta_2
            # Update compositions
            x_alpha_1_New=(1-K_2)/(K_1-K_2)
            x_beta_1_New=x_alpha_1_New*K_1
            # Update gammas
            gamma_alpha_1=Fs_gamma[0](x_alpha_1_New,T)
            gamma_alpha_2=Fs_gamma[1](x_alpha_1_New,T)
            gamma_beta_1=Fs_gamma[0](x_beta_1_New,T)
            gamma_beta_2=Fs_gamma[1](x_beta_1_New,T)
            # Check convergence or divergence
            cond1=abs(x_alpha_1_New-x_alpha_1_OLD)<10**-6
            cond2=abs(x_beta_1_New-x_beta_1_OLD)<10**-6
            cond3=x_alpha_1_New>0 and x_alpha_1_New<1
            cond4=x_beta_1_New>0 and x_beta_1_New<1
            if (cond1 and cond2) or not cond3 or not cond4:
                # Check if LLE exists
                cond5=abs(x_alpha_1_New-x_beta_1_New)>10**-4
                if cond5 and cond3 and cond4:
                    # Update LLE container
                    LLE.append([T,x_alpha_1_New,x_beta_1_New])
                    gammas.append([T,gamma_alpha_1,gamma_alpha_2,gamma_beta_1,
                                   gamma_beta_2])
                break
            else:
                # Update OLD compositions
                x_alpha_1_OLD=x_alpha_1_New
                x_beta_1_OLD=x_beta_1_New
    LLE=numpy.array(LLE)
    gammas=numpy.array(gammas)
    # Output
    return LLE,gammas

# =============================================================================
# Solid-Liquid Equilibrium
# =============================================================================

def meltingProperties(compound):
    """
    meltingProperties() returns the melting properties of compound "compound".
    The properties include, also, one solid-solid transition. If the compound
    has no SS transition, a value of zero is set for the T_SS. No compounds
    with more than one SS transition are included.

    Parameters
    ----------
    compound : string
        Name of the compound.

    Raises
    ------
    ValueError
        Error raised if the compound is not found in the database.

    Returns
    -------
    entry : list
        Melting properties of compound "compound".
        Columns: 'Compound','T_m (K)','deta_m_h (J/mol)',
                 'T_SS (K)','deta_SS_h (J/mol)'

    """
    # Create empty pandas dataframe
    columns=['Compound','T_m (K)','deta_m_h (J/mol)',
             'T_SS (K)','deta_SS_h (J/mol)']
    database=pandas.DataFrame(data=[],columns=columns)
    # Acetone (NIST)
    line=['Acetone',176.6,5720,0,0]
    database.loc[len(database.index)]=line
    # Methanol (NIST)
    line=['Methanol',175.59,3215.4,157.34,636]
    database.loc[len(database.index)]=line
    # Benzene (NIST)
    line=['Benzene',278.7,9870,0,0]
    database.loc[len(database.index)]=line
    # Chloroform (NIST)
    line=['Chloroform',209.6,8800,0,0]
    database.loc[len(database.index)]=line
    # Cyclohexane (NIST)
    line=['Cyclohexane',279.84,2628,186.09,6686]
    database.loc[len(database.index)]=line
    # Hexane (NIST)
    line=['Hexane',177.8,13080,0,0]
    database.loc[len(database.index)]=line
    # Heptane (NIST)
    line=['Heptane',182.6,14040,0,0]
    database.loc[len(database.index)]=line
    # Tetrachloromethane (NIST)
    line=['TCM',250.53,2562,225.7,4631]
    database.loc[len(database.index)]=line
    # Toluene (NIST)
    line=['Toluene',178,6610,0,0]
    database.loc[len(database.index)]=line
    # Water (No Reference)
    line=['Water',273.15,6009,0,0]
    database.loc[len(database.index)]=line
    # ChCl (10.1016/j.fluid.2017.03.015)
    line=['ChCl',597,4300,0,0]
    database.loc[len(database.index)]=line
    # Urea (NIST)
    line=['Urea',407.2,14600,0,0]
    database.loc[len(database.index)]=line
    # Thymol (10.1021/acssuschemeng.8b01203)
    line=['Thymol',323.59,19650,0,0]
    database.loc[len(database.index)]=line
    # Menthol (10.1021/acssuschemeng.8b01203)
    line=['Menthol',315.7,12890,0,0]
    database.loc[len(database.index)]=line
    # TOPO (10.1039/D0GC00793E)
    line=['TOPO',325.9,58020,0,0]
    database.loc[len(database.index)]=line
    # Search compound
    entry=database[database['Compound']==compound].values.tolist()[0]
    # Raise exception if compound not found
    if not entry:
        raise ValueError('No melting properties for compound '+compound+'.')
    # Output
    return entry

def compute_Tx_SLE_Binary(Fs_gamma,properties_1,properties_2,
                          x1_range=numpy.linspace(0,1,101),plot=False):
    """
    compute_Tx_SLE_Binary() computes the SLE phase diagram of the binary
    mixture defined by properties_1 and properties_2. Assumptions:
        . Eutectic-type behavior
        . Full solid-solid immiscibility
        . Delta Cp impact is negligible

    Parameters
    ----------
    Fs_gamma : list of function handlers
        List where each entry "i" is the funciton that returns gamma for
        component "i". Must be of the type gamma=F_gamma(x1,T). T in K.
    properties_1 : list
        SLE properties of compound 1, as obtained using meltingProperties().
        Columns: 'Compound','T_m (K)','deta_m_h (J/mol)',
                 'T_SS (K)','deta_SS_h (J/mol)'
    properties_2 : list
        SLE properties of compound 1, as obtained using meltingProperties().
        Columns: 'Compound','T_m (K)','deta_m_h (J/mol)',
                 'T_SS (K)','deta_SS_h (J/mol)'
    x1_range : numpy array, optional
        Range of x1 to perform SLE calculations. Must include 0 and 1, and
        must be sorted in ascending order.
        The default is numpy.linspace(0,1,101).
    plot : boolean, optional
        Whether to plot the SLE.
        The default is False.
    title : string, optional
        Title of the plot.
        The default is ''.

    Returns
    -------
    SLE : numpy array (N,2)
        Array with the SLE results. N is the number of grid points,
        N=len(x1_range).
        Columns: "x1","T_SLE"
    gammas : numpy array (N,2)
        Array with the activity coefficients of each component at the SLE 
        equilibrium temperature for each x composition point.
        Columns: "x1","gamma"

    """
    # Definition of ideal gas constant
    R=8.31446
    # Get N
    N=len(x1_range)
    # Retrieve pure-component properties
    Tms=[properties_1[1],properties_2[1]]
    Hms=[properties_1[2],properties_2[2]]
    Tss=[properties_1[3],properties_2[3]]
    Hss=[properties_1[4],properties_2[4]]
    # Define containers for independent liquidus curves
    SLE_1=numpy.zeros((N,))
    gammas_1=numpy.zeros((N,))
    SLE_2=numpy.zeros((N,))
    gammas_2=numpy.zeros((N,))
    # Fill pure-component props
    SLE_1[-1]=Tms[0]
    gammas_1[-1]=1
    SLE_2[0]=Tms[1]
    gammas_2[0]=1
    # Define aborted flags (if T gets too low or gamma too high)
    aborted_1=False
    aborted_2=False
    # Loop over x1_range without pure components
    for n in tqdm(range(N-2),'Computing SLE: '):
        # SLE curve for component 1
        if not aborted_1:
            x1=x1_range[-2-n]
            # Guess temperature from previous point
            T=SLE_1[-1-n]*1.05
            # T Loop
            k=0
            while True:
                k+=1
                # Calculate gamma
                gamma1=Fs_gamma[0](x1,T)
                # Calculate SLE composition
                A=(Hms[0]/R)*(1/Tms[0]-1/T)
                if T<Tss[0]:
                    B=(Hss[0]/R)*(1/Tss[0]-1/T)
                    x=numpy.exp(A+B)/gamma1
                else:
                    x=numpy.exp(A)/gamma1
                # x error
                x_error=x-x1
                # Check x
                if abs(x_error)<10**-6:
                    break
                else:
                    # Select new temperature
                    if T<Tss[0]:
                        A=-Tms[0]*Hms[0]*Tss[0]-Tms[0]*Hss[0]*Tss[0]
                        B=Tms[0]*Tss[0]*R*numpy.log(x1*gamma1)
                        C=-Hms[0]*Tss[0]-Hss[0]*Tms[0]
                        T_=A/(B+C)
                        T=(T+T_)/2
                    else:
                        T_=Hms[0]/(Hms[0]/Tms[0]-R*numpy.log(x1*gamma1))
                        T=(T+T_)/2
                # Abort if T is unphisically low or gamma too high
                if T<100 or gamma1>20 or k>1000:
                    aborted_1=True
                    break
            # Update arrays
            if not aborted_1:
                SLE_1[-2-n]=T
                gammas_1[-2-n]=gamma1
        # SLE curve for component 2
        if not aborted_2:
            x2=x1_range[-2-n]
            x1=1-x2
            # Guess temperature from previous point
            T=SLE_2[n]*1.05
            # T Loop
            k=0
            while True:
                k+=1
                # Calculate gamma
                gamma2=Fs_gamma[1](x1,T)
                # Calculate SLE composition
                A=(Hms[1]/R)*(1/Tms[1]-1/T)
                if T<Tss[1]:
                    B=(Hss[1]/R)*(1/Tss[1]-1/T)
                    x=numpy.exp(A+B)/gamma2
                else:
                    x=numpy.exp(A)/gamma2
                # x error
                x_error=x-x2
                # Check x
                if abs(x_error)<10**-6:
                    break
                else:
                    # Select new temperature
                    if T<Tss[1]:
                        A=-Tms[1]*Hms[1]*Tss[1]-Tms[1]*Hss[1]*Tss[1]
                        B=Tms[1]*Tss[1]*R*numpy.log(x2*gamma2)
                        C=-Hms[1]*Tss[1]-Hss[1]*Tms[1]
                        T_=A/(B+C)
                        T=(T+T_)/2
                    else:
                        T_=Hms[1]/(Hms[1]/Tms[1]-R*numpy.log(x2*gamma2))
                        T=(T+T_)/2
                # Abort if T is unphisically low or gamma too high
                if T<100 or gamma2>20 or k>1000:
                    aborted_2=True
                    break
            # Update arrays
            if not aborted_2:
                SLE_2[n+1]=T
                gammas_2[n+1]=gamma2
        # Get closest intersection between SLE_1 and SLE_2
        T_diff=abs(SLE_1-SLE_2).min()
        index=abs(SLE_1-SLE_2).argmin()
        # Check if curves have intersepted or both curves have aborted
        if (T_diff<0.1 and SLE_1[index]>0 and SLE_2[index]>0) or \
            (aborted_1 and aborted_2):
            # Break for loop
            break
    # Define output arrays
    SLE=numpy.zeros((N,2))
    gammas=numpy.zeros((N,2))
    SLE[:,0]=x1_range
    gammas[:,0]=x1_range
    SLE[:,1]=numpy.concatenate((SLE_2[:index],SLE_1[index:]))
    gammas[:,1]=numpy.concatenate((gammas_2[:index],gammas_1[index:]))
    # Plot
    if plot:
        plt.plot(SLE[:,0],SLE[:,1],'--k')
        plt.title(properties_1[0]+'/'+properties_2[0]+', SLE')
        plt.xlabel('$\mathregular{x_1}$')
        plt.ylabel('T /K')
    # Output
    return SLE,gammas

def compute_Tx_SLE_Ideal_Binary(properties_1,properties_2,
                                x1_range=numpy.linspace(0,1,101),plot=False):
    """
    compute_Tx_SLE_Ideal_Binary() computes the ideal SLE phase diagram of the
    binary mixture defined by properties_1 and properties_2. Assumptions:
        . Eutectic-type behavior
        . Full solid-solid immiscibility
        . Delta Cp impact is negligible

    Parameters
    ----------
    properties_1 : list
        SLE properties of compound 1, as obtained using meltingProperties().
        Columns: 'Compound','T_m (K)','deta_m_h (J/mol)',
                 'T_SS (K)','deta_SS_h (J/mol)'
    properties_2 : list
        SLE properties of compound 1, as obtained using meltingProperties().
        Columns: 'Compound','T_m (K)','deta_m_h (J/mol)',
                 'T_SS (K)','deta_SS_h (J/mol)'
    x1_range : numpy array, optional
        Range of x1 to perform SLE calculations. Must include 0 and 1, and
        must be sorted in ascending order.
        The default is numpy.linspace(0,1,101).
    plot : boolean, optional
        Whether to plot the SLE.
        The default is False.
    title : string, optional
        Title of the plot.
        The default is ''.

    Returns
    -------
    SLE : numpy array (N,2)
        Array with the SLE results. N is the number of grid points,
        N=len(x1_range).
        Columns: "x1","T_SLE"

    """
    # Definition of ideal gas constant
    R=8.31446
    # Get N
    N=len(x1_range)
    # Retrieve pure-component properties
    Tms=[properties_1[1],properties_2[1]]
    Hms=[properties_1[2],properties_2[2]]
    Tss=[properties_1[3],properties_2[3]]
    Hss=[properties_1[4],properties_2[4]]
    # Define containers for independent liquidus curves
    SLE_1=numpy.zeros((N,))
    SLE_2=numpy.zeros((N,))
    # Fill pure-component props
    SLE_1[-1]=Tms[0]
    SLE_2[0]=Tms[1]
    # Loop over x1_range without pure components
    for n in tqdm(range(N-2),'Computing SLE: '):
        # SLE curve for component 1
        x1=x1_range[-2-n]
        # Guess temperature from previous point
        T=SLE_1[-1-n]
        # T Loop
        for __ in range(2):
            # Select new temperature
            if T<Tss[0]:
                A=-Tms[0]*Hms[0]*Tss[0]-Tms[0]*Hss[0]*Tss[0]
                B=Tms[0]*Tss[0]*R*numpy.log(x1)
                C=-Hms[0]*Tss[0]-Hss[0]*Tms[0]
                T=A/(B+C)
            else:
                T=Hms[0]/(Hms[0]/Tms[0]-R*numpy.log(x1))
        # Update array
        SLE_1[-2-n]=T
        # SLE curve for component 2
        x2=x1_range[-2-n]
        x1=1-x2
        # Guess temperature from previous point
        T=SLE_2[n]
        # T Loop
        for __ in range(2):
            # Select new temperature
            if T<Tss[1]:
                A=-Tms[1]*Hms[1]*Tss[1]-Tms[1]*Hss[1]*Tss[1]
                B=Tms[1]*Tss[1]*R*numpy.log(x2)
                C=-Hms[1]*Tss[1]-Hss[1]*Tms[1]
                T=A/(B+C)
            else:
                T=Hms[1]/(Hms[1]/Tms[1]-R*numpy.log(x2))
        # Update array
        SLE_2[n+1]=T
        # Get closest intersection between SLE_1 and SLE_2
        T_diff=abs(SLE_1-SLE_2).min()
        index=abs(SLE_1-SLE_2).argmin()
        # Check if curves have intersepted or both curves have aborted
        if (T_diff<0.1 and SLE_1[index]>0):
            # Break for loop
            break
    # Define output arrays
    SLE=numpy.zeros((N,2))
    SLE[:,0]=x1_range
    SLE[:,1]=numpy.concatenate((SLE_2[:index],SLE_1[index:]))
    # Plot
    if plot:
        plt.plot(SLE[:,0],SLE[:,1],'--k')
        plt.title(properties_1[0]+'/'+properties_2[0]+', SLE')
        plt.xlabel('$\mathregular{x_1}$')
        plt.ylabel('T /K')
    # Output
    return SLE

def get_Gammas_from_SLE(SLE_Exp,properties,targetComp):
    """
    get_Gammas_from_SLE() computes activity coefficients from SLE data.

    Parameters
    ----------
    SLE_Exp : numpy array (K,2)
        Array containing x1,T SLE values.
    properties : list
        SLE properties of the target compound, as obtained using
        meltingProperties().
        Columns: 'Compound','T_m (K)','deta_m_h (J/mol)',
                 'T_SS (K)','deta_SS_h (J/mol)'
    targetComp : int
        Target compound (1 or 2).

    Returns
    -------
    gammas : numpy array (K,3)
        Array containing x1,T,gamma values.

    """
    # Definition of ideal gas constant
    R=8.31446
    # Retrieve pure-component properties
    Tm=properties[1]
    Hm=properties[2]
    Tss=properties[3]
    Hss=properties[4]
    # Initialize gammas
    gammas=numpy.array([]).reshape(-1,3)
    # Loop over SLE_Exp
    for entry in SLE_Exp:
        if targetComp==1:
            x=entry[0]
        else:
            x=1-entry[0]
        T=entry[1]
        # Compute x_ID
        if T<Tss:
            x_ID=numpy.exp((Hm/R)*(1/Tm-1/T)+(Hss/R)*(1/Tss-1/T))
        else:
            x_ID=numpy.exp((Hm/R)*(1/Tm-1/T))
        # Compute gamma
        gamma=x_ID/x
        # Set new
        new=numpy.array([entry[0],entry[1],gamma]).reshape(-1,3)
        # Append gamma
        gammas=numpy.append(gammas,new,axis=0)
    # Output
    return gammas