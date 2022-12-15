# -*- coding: utf-8 -*-
"""
Python library containing auxiliary functions to perform gaussian process
regression and active learning on activity coefficient data.

Sections:
    . Imports
    . Data Preprocessing
        . build_X_Train_Binary()
        . buildDataset_Binary()
        . normalize()
    . Data Visualization
        . plotHM()
        . plot_T_Slices()
        . generateGIF()
    . Gaussian Process Regression
        . buildGP()
        . evaluateGP()
        . evaluateModel()
    . Active Learning
        . AL_Independent_Binary()
        . AL_VLE_Binary_Type1()
        . AL_SLE_Binary_Type1()

Last edit: 2022-10-27
Author: Dinis Abranches
"""

# =============================================================================
# Imports
# =============================================================================

# Generic
import warnings
import os

# Specific
import gpflow
gpflow.config.set_default_summary_fmt("notebook")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # No CUDA
import numpy
import imageio
from sklearn import preprocessing
from matplotlib import pyplot as plt
from tqdm import tqdm

# Local
from . import thermoAux as thermo

# =============================================================================
# Data Preprocessing
# =============================================================================

def build_X_Train_Binary(trainGridType,targetComp,Tmin,Tmax):
    """
    build_X_Train_Binary() builds an X_Train composition/temperature grid for
    binary mixtures.

    Parameters
    ----------
    trainGridType : int
        The type of grid to be generated. One of:
            . 1 - 5x5 grid
            . 2 - 4x5 grid  plus 100 pure-component virtual points
            . 3 - 100 pure-component virtual points + 1 infinite dilution point
                  (useful initial X for active learning)
    targetComp : int
        The component of interest. Necessary if trainGridType is 2 or 3, to
        choose the location of the virtual points representing gamma=1. One of:
            . 1 - Component #1
            . 2 - Component #2
    Tmin : float
        Minimum temperature of the grid.
    Tmax : float
        Maximum temperature of the grid.

    Returns
    -------
    X_Train : numpy array
        Training features representing the grid requested.

    """
    if trainGridType==1:
        X_Train=numpy.meshgrid(numpy.linspace(0,1,5),
                               numpy.linspace(Tmin,Tmax,5))
        X_Train=numpy.array(X_Train).T.reshape(-1,2)
    elif trainGridType==2:
        if targetComp==1:
            X_Train1=numpy.meshgrid(numpy.ones(1),
                                    numpy.linspace(Tmin,Tmax,100))
            X_Train1=numpy.array(X_Train1).T.reshape(-1,2)
            X_Train2=numpy.meshgrid(numpy.linspace(0,0.75,4),
                                    numpy.linspace(Tmin,Tmax,5))
            X_Train2=numpy.array(X_Train2).T.reshape(-1,2)
            X_Train=numpy.concatenate((X_Train1,X_Train2))
        elif targetComp==2:
            X_Train1=numpy.meshgrid(numpy.zeros(1),
                                    numpy.linspace(Tmin,Tmax,100))
            X_Train1=numpy.array(X_Train1).T.reshape(-1,2)
            X_Train2=numpy.meshgrid(numpy.linspace(0.25,1,4),
                                    numpy.linspace(Tmin,Tmax,5))
            X_Train2=numpy.array(X_Train2).T.reshape(-1,2)
            X_Train=numpy.concatenate((X_Train1,X_Train2))
    elif trainGridType==3:
        if targetComp==1:
            X_Train1=numpy.meshgrid(numpy.ones(1),
                                    numpy.linspace(Tmin,Tmax,100))
            X_Train1=numpy.array(X_Train1).T.reshape(-1,2)
            X_Train2=numpy.array([[0,Tmin]])
            X_Train=numpy.concatenate((X_Train1,X_Train2))
        elif targetComp==2:
            X_Train1=numpy.meshgrid(numpy.zeros(1),
                                    numpy.linspace(Tmin,Tmax,100))
            X_Train1=numpy.array(X_Train1).T.reshape(-1,2)
            X_Train2=numpy.array([[1,Tmin]])
            X_Train=numpy.concatenate((X_Train1,X_Train2))
    # Output
    return X_Train
    
def buildDataset_Binary(F_Truth,x1_range=None,T_range=None,X=None):
    """
    buildDataset_Binary() builds an activity coefficient dataset for binary
    mixtures on an x1,T grid. The grid is constructed using the range of each
    dimension (x1_range and T_range). Alternatively, the X data can be provided
    in the input.

    Parameters
    ----------
    F_Truth : function Handler
        Truth function of the type gamma=F_Truth(x1,T).
    x1_range : numpy array (N1,1), optional
        Array representing the composition axis of the grid.
        The default is None.
    T_range : numpy array (N2,1), optional
        Array representing the temperature axis of the grid.
        The default is None.
    X : numpy array (N,2), optional
        X data (x1,T) used to compute Y. Alternative to specifying
        (x1_range,T_range) and building X inside this function.
        The default is None.

    Returns
    -------
    X : numpy array (N1*N2 or N,2)
        Testing features generated from the x1,T grid provided or X as
        inputted.
    Y : numpy array (N1*N2 or N,1)
        Testing labels generated frm the x1,T grid (or X) and truth function
        provided.

    """
    # Build X
    if X is None:
        X=numpy.array(numpy.meshgrid(x1_range,T_range)).T.reshape(-1,2)
    # Initialize Y
    Y=numpy.zeros((X.shape[0],1))
    # Initialize counter
    counter=0
    # Loop over X (inefficient)
    for x in X:
        # Get Y value
        Y[counter,0]=F_Truth(x[0],x[1])
        # Update counter
        counter+=1
    # Output
    return X,Y

def normalize(inputArray,skScaler=None,method='MinMax',reverse=False):
    """
    normalize() normalizes (or unnormalizes) inputArray using the skScaler
    provided or trains a new skScaler for the first time.

    Parameters
    ----------
    inputArray : numpy array (dim=2)
        Array to be normalized column-wise. Do not input dim=1 arrays, i.e.,
        arrays with shape=(N,).
    skScaler : scikit-learn preprocessing object or None
        Scikit-learn preprocessing object previosly fitted to data. If None,
        the object is fitted to inputArray.
        Default: None
    method : string, optional
        Normalization method to be used. Only required when training an 
        skScaler for the first time.
        Methods available:
            . Standardization - classic standardization, (x-mean(x))/std(x)
            . MinMax - scale to range (-1,+1)
            . MinMax_2 - scale to range (0,+1)
        Defalt: 'MinMax'
    reverse : bool
        Wether to normalize (False) or unnormalize (True) inputArray.
        Defalt: False

    Returns
    -------
    inputArray : numpy array
        Normalized (or unnormalized) version of inputArray.
    skScaler : scikit-learn preprocessing object
        Scikit-learn preprocessing object fitted to inputArray. It is the same
        as the inputted skScaler, if it was provided.

    """
    # If skScaler is None, train for the first time
    if skScaler is None:
        # Check method
        if method=='Standardization':
            skScaler=preprocessing.StandardScaler()
        elif method=='MinMax':
            skScaler=preprocessing.MinMaxScaler(feature_range=(-1,1))
        elif method=='MinMax_2':
            skScaler=preprocessing.MinMaxScaler(feature_range=(0,1))
        skScaler=skScaler.fit(inputArray)
    # Do main operation (normalize or unnormalize)
    if reverse:
        # Rescale the data back to its original distribution
        inputArray=skScaler.inverse_transform(inputArray)
    elif not reverse:
        inputArray=skScaler.transform(inputArray)
    # Return
    return inputArray,skScaler

# =============================================================================
# Data Visualization
# =============================================================================
    
def plotHM(X,Y,title,cblabel,xLabel='x1',X_Highlight=None,savePath=None):
    """
    plotHM() generates a heat map of the data provided on a
    composition-temperature grid.
    
    Parameters
    ----------
    X : numpy array (N,2)
        Features (x,T grid). Note that x and T must be slow and fast changing,
        respectively, to correctly broadcast into a sqrt(N) x sqrt(N) grid.
    Y : numpy array (N,1)
        Labels (activity coefficients, standard deviation, etc). 
    title : string
        Title of the plot. If None, no title is used.
    cblabel : string
        Label of the color bar.
    xLabel : string, optional
        Label for the x-axis of each plot.
        The default is 'x_1'.
    X_Highlight : numpy array (K,2), optional
        Array containing x,T data to be highlighted in the plot.
        The default is None.
    savePath : string, optional
        Path where the plot is to be saved.
        If None, the plot is displayed but not saved.
        The default is None.

    Returns
    -------
    None.

    """
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
    x1_range=numpy.unique(X[:,0])
    T_range=X[:len(x1_range),1]
    # Reshape Y
    ZMatrix=Y.reshape(len(T_range),len(x1_range),order='F')
    # Create figure
    plt.figure(figsize=(3,1.7))
    # Heatmap
    hm=plt.pcolormesh(x1_range,T_range,ZMatrix)
    cb=plt.colorbar(hm)
    cb.set_label(cblabel)
    if X_Highlight is not None:
        plt.plot(X_Highlight[:,0],X_Highlight[:,1],'or',markersize=2)
    # Labels
    plt.xlabel(xLabel)
    plt.ylabel('T /K')
    if title is not None: plt.title(title)
    # Display or save plot
    if savePath is not None: 
        plt.savefig(savePath,dpi=600,bbox_inches='tight')
        plt.close()
    else: plt.show()
    # Output
    return None

def plot_T_Slices(T_Slices,X,Y,title,Y_2=None,xLabel='x1',yLabel='Gamma_1',
                  X_Highlight=None,savePath=None):
    """
    plot_T_Slices() plots T slices on a gamma vs. x plot.

    Parameters
    ----------
    T_Slices : numpy array (N,1)
        Temperature slices to be plotted (N < 7). X must contain these values.
    X : numpy array (N,2)
        Features (x,T grid). Note that x and T must be slow and fast changing,
        respectively, to correctly broadcast into a sqrt(N) x sqrt(N)
        grid.
    Y : numpy array (N,1)
        Labels (activity coefficients).
    title : string
        Title of the plot. If None, no title is used.
    Y_2 : numpy array (N,1), optional
        Second labels (activity coefficients) to be plotted and compared
        against Y.
        The default is None.
    xLabel : string, optional
        Label of the x-axis of the plots generated.
        The default is 'x_1'.
    yLabel : string, optional
        Label of the y-axis of the plots generated.
        The default is 'Gamma_1'.
    X_Highlight : numpy array (K,2), optional
        Array containing x,T data to be highlighted in the plot.
        The default is None.
    savePath : string, optional
        Path where the plot is to be saved.
        If None, the plot is displayed but not saved.
        The default is None.

    Returns
    -------
    None.

    """
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
    colors=['k','b','r','g','m','y','c']
    # Get axis ranges from X
    x1_range=numpy.unique(X[:,0])
    # Create figure
    plt.figure(figsize=(3,1.7))
    # Loop over temperature slices
    for n in range(len(T_Slices)):
        # Get Y values
        plotY=Y[X[:,1]==T_Slices[n]]
        # Plot curve
        plt.plot(x1_range,plotY,'-'+colors[n],linewidth=1,
                 label=str(int(T_Slices[n]))+' K')
        # Check if points must be highlighted
        if X_Highlight is not None:
            for entry in X_Highlight:
                if entry[1]==T_Slices[n]:
                    mask=numpy.array([numpy.all(i) for i in X==entry])
                    plt.plot(entry[0],Y[mask],'xk',markersize=5)
    # Check if second Y was provided
    if Y_2 is not None:
        for n in range(len(T_Slices)):
            # Get Y values
            plotY=Y_2[X[:,1]==T_Slices[n]]
            plt.plot(x1_range,plotY,':'+colors[n],linewidth=2)
    # Labels
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    if title is not None: plt.title(title)
    plt.legend(prop={'size': 6})
    # Display or save plot
    if savePath is not None: 
        plt.savefig(savePath,dpi=600,bbox_inches='tight')
        plt.close()
    else: plt.show()
    # Output
    return None

def generateGIF(gifPath,frameList,duration=1):
    """
    generateGIF() generates a GIF from the image paths provided in frameList.

    Parameters
    ----------
    gifPath : string
        Path where the GIF should be saved.
    frameList : list of strings
        List containing the path of each frame (properly ordered).
    duration : int, optional
        Duration of each frame (seconds) in the final GIF.
        The default is 1.

    Returns
    -------
    None.

    """
    # Initialize list of frames
    frames=[] 
    # Iterate over individual snapshots
    for file in tqdm(frameList,'Generating GIF'):
        # Read image and append to frames
        frames.append(imageio.v2.imread(file,format='png'))
    # Save list of frames as GIF
    imageio.mimsave(gifPath,frames,duration=duration)
    # Output
    return None

# =============================================================================
# Gaussian Process Regression
# =============================================================================

def buildGP(X_Train,Y_Train,X_Scaler=None,gpConfig={}):
    """
    buildGP() builds and fits a GP model using the training data provided.

    Parameters
    ----------
    X_Train : numpy array (N,K)
        Training features, where N is the number of data points and K is the
        number of independent features (e.g., composition and temperature).
    Y_Train : numpy array (N,1)
        Training labels (e.g., activity coefficient of a given component).
    X_Scaler : scikit-learn preprocessing object, optional
        Scikit-learn preprocessing object to normalize X_Train by calling
        normalize(). If None, no normalization is performed.
        The default is None.
    gpConfig : dictionary, optional
        Dictionary containing the configuration of the GP. If a key is not
        present in the dictionary, its default value is used.
        Keys:
            . kernel : string
                Kernel to be used. One of:
                    . 'RBF' - gpflow.kernels.RBF()
                    . 'RQ' - gpflow.kernels.RationalQuadratic()
                    . 'Matern52' - gpflow.kernels.Matern52()
                    . 'ArcCosine_0' - gpflow.kernels.ArcCosine(order=0)
                    . 'ArcCosine_1' - gpflow.kernels.ArcCosine(order=1)
                    . 'ArcCosine_2' - gpflow.kernels.ArcCosine(order=2)
                The default is 'RQ'.
            . useWhiteKernel : boolean
                Whether to use a White kernel (gpflow.kernels.White).
                The default is False.
            . indepDim : boolean
                Whether to use different kernel lengthscales for each
                independent feature.
                The default is False.
            . doLogY : boolean
                If True, log(Y) is used to train the model.
                The default is True.
            . trainLikelihood : boolean
                Whether to treat the variance of the likelihood of the modeal
                as a trainable (or fitting) parameter. If False, this value is
                fixed at 10^-5.
                The default is False.
        The default is {}.
    Raises
    ------
    UserWarning
        Warning raised if the optimization (fitting) fails to converge.

    Returns
    -------
    model : gpflow.models.gpr.GPR object
        GP model.

    """
    # Normalize X_Train
    if X_Scaler is not None:
        X_Train_N,__=normalize(X_Train,skScaler=X_Scaler)
    else:
        X_Train_N=X_Train
    # Unpack gpConfig
    kernel=gpConfig.get('kernel','RQ')
    useWhiteKernel=gpConfig.get('useWhiteKernel','False')
    indepDim=gpConfig.get('indepDim','False')
    doLogY=gpConfig.get('doLogY','True')
    trainLikelihood=gpConfig.get('trainLikelihood','False')
    # Do Log Y
    if doLogY: Y_Train=numpy.log(Y_Train)
    # Check indepDim input
    if indepDim: l=numpy.ones(X_Train.shape[1])
    else: l=1
    # Select and initialize kernel
    if kernel=='RBF':
        gpKernel=gpflow.kernels.SquaredExponential(lengthscales=l)
    if kernel=='RQ':
        gpKernel=gpflow.kernels.RationalQuadratic(lengthscales=l)
    if kernel=='Matern52':
        gpKernel=gpflow.kernels.Matern52(lengthscales=l)
    if kernel=='ArcCosine_0':
        gpKernel=gpflow.kernels.ArcCosine(order=0,weight_variances=l)
    if kernel=='ArcCosine_1':
        gpKernel=gpflow.kernels.ArcCosine(order=1,weight_variances=l)
    if kernel=='ArcCosine_2':
        gpKernel=gpflow.kernels.ArcCosine(order=2,weight_variances=l)
    # Add White kernel
    if useWhiteKernel: gpKernel=gpKernel+gpflow.kernels.White()
    # Build GP model    
    model=gpflow.models.GPR((X_Train_N,Y_Train),gpKernel,noise_variance=10**-5)
    # Select whether the likelihood variance is trained
    gpflow.utilities.set_trainable(model.likelihood.variance,trainLikelihood)
    # Build optimizer
    optimizer=gpflow.optimizers.Scipy()
    # Fit GP to training data
    aux=optimizer.minimize(model.training_loss,
                           model.trainable_variables,
                           method='L-BFGS-B')
    # Check convergence
    if aux.success==False:
        warnings.warn('GP optimizer failed to converge.')
    # Output
    return model

def gpPredict(model,X,X_Scaler=None,gpConfig={}):
    """
    gpPredict() returns the prediction and standard deviation of the GP model
    on the X data provided.

    Parameters
    ----------
    model : gpflow.models.gpr.GPR object
        GP model.
    X : numpy array (N,2)
        Features to perform prediction (x,T).
    X_Scaler : scikit-learn preprocessing object
        Scikit-learn preprocessing object to normalize X. If None, no
        normalization is done.
        The default is None.
    gpConfig : dictionary, optional
        Dictionary containing the configuration of the GP; see buildGP().
        Only used here to obtain key "doLogY". If True, predicted means and
        standard deviations are corrected. The default if the key is not
        present is True.
        The default is {}.

    Returns
    -------
    Y : numpy array (N,1)
        GP predictions.
    STD : numpy array (N,1)
        GP standard deviations.

    """
    # Scale X
    if X_Scaler is not None: X,__=normalize(X,skScaler=X_Scaler)
    # Unpack gpConfig
    Pred_is_log=gpConfig.get('doLogY','True')
    # Do GP prediction, obtaining mean and variance
    GP_Mean,GP_Var=model.predict_f(X)
    # Convert to numpy
    GP_Mean=GP_Mean.numpy()
    GP_Var=GP_Var.numpy()
    # Check if prediction is in log form
    if Pred_is_log:
        Y=numpy.exp(GP_Mean)
        STD=Y*numpy.sqrt(GP_Var)
        STD=numpy.nan_to_num(STD)
    else:
        Y=GP_Mean
        STD=numpy.sqrt(GP_Var)
    # Output
    return Y,STD

def evaluateModel(Y_Pred,X_Test,Y_Test,Y_STD=None,
                  plotHM_Pred=False,plotHM_STD=False,plotHM_AE=False,
                  plotHM_PE=False,plotT_Slices=False,T_Slices=None,
                  titlePrefix=None,X_Highlight=None,
                  gammaLabel='Gamma_1',
                  xLabel='x1'):
    """
    evaluateModel() computes a series of performance-related metrics for the
    model-predicted data inputted.

    Parameters
    ----------
    Y_Pred : numpy array (N,1)
        Predicted labels (gamma).
    X_Test : numpy array (N,2)
        Testing features (x1,T).
    Y_Test : numpy array (N,1)
        Testing (true) labels (gamma).
    Y_STD : numpy array (N,1), optional
        Model-derived standard deviation of each data point. Only used to plot
        the STD heat map, if provided.
        The default is None.
    plotHM_Pred : boolean or string, optional
        Whether to plot a heatmap of GP predictions.
        If this is a string, it must be a path where the plot is to be saved.
        The default is True.
    plotHM_STD : boolean or string, optional
        Whether to plot a heatmap of GP STD.
        If this is a string, it must be a path where the plot is to be saved.
        The default is True.
    plotHM_AE : boolean or string, optional
        Whether to plot a heatmap of GP absolute error predictions.
        If this is a string, it must be a path where the plot is to be saved.
        The default is True.
    plotHM_PE : boolean or string, optional
        Whether to plot a heatmap of GP percentage error predictions.
        If this is a string, it must be a path where the plot is to be saved.
        The default is True.
    plotT_Slices : boolean or string, optional
        Whether to plot T slices of F_Truth in the grid provided.
        If plot_T_Slices is a string, it must be a path where the plot is to be
        saved.
        The default is False.
    T_Slices : numpy array (L,), optional
        Temperature slices to be plotted.
        The default is None.
    titlePrefix : string, optional
        Title prefix for the plots requested.
        The default is None.
    X_Highlight : numpy array (K,2), optional
        Array containing x1,T data to be highlighted in the plots. Rows
        represent different points and the two columns represent x1 and T.
        The default is None.
    gammaLabel : string, optional
        The label of the color bar of the pred. heat map plot or of the y-axis
        of the temperature slice plots.
        The default is 'Gamma_1'.
    xLabel : string, optional
        Label of the x-axis of the plots generated.
        The default is 'x_1'.

    Returns
    -------
    scores : dictionary
        Performance-related scores (R2, MAE, MPE, RMSE, RMSLE).

    """
    # Initialize empty scores dict
    scores={}
    # Compute scores
    mean=Y_Pred.mean()
    u=((Y_Test-Y_Pred)**2).sum()
    v=((Y_Test-mean)**2).sum()
    scores['R2']=1-u/v
    scores['MAE']=(abs(Y_Test-Y_Pred)).sum()/len(Y_Test)
    scores['MPE']=100*(abs(Y_Test-Y_Pred)/Y_Test).sum()/len(Y_Test)
    scores['RMSE']=numpy.sqrt(((Y_Test-Y_Pred)**2).sum()/len(Y_Test))
    aux1=numpy.log(Y_Test+1)
    aux2=numpy.log(Y_Pred+1)
    scores['RMSLE']=numpy.sqrt(((aux1-aux2)**2).sum()/len(Y_Test))
    # Check plot requests
    Pred_savePath=None
    STD_savePath=None
    AE_savePath=None
    PE_savePath=None
    T_Slices_savePath=None
    if isinstance(plotHM_Pred,str):
        Pred_savePath=plotHM_Pred
        plotHM_Pred=True
    if isinstance(plotHM_STD,str):
        STD_savePath=plotHM_STD
        plotHM_STD=True
    if isinstance(plotHM_AE,str):
        AE_savePath=plotHM_AE
        plotHM_AE=True
    if isinstance(plotHM_PE,str):
        PE_savePath=plotHM_PE
        plotHM_PE=True
    if isinstance(plotT_Slices,str):
        T_Slices_savePath=plotT_Slices
        plotT_Slices=True
    # Build plots requested
    if plotHM_Pred:
        title=titlePrefix+', GP Prediction'
        cblabel='$\mathregular{\gamma}$'
        plotHM(X_Test,Y_Pred,title,cblabel,X_Highlight=X_Highlight,
               savePath=Pred_savePath)
    if plotHM_STD:
        title=titlePrefix+', GP STD'
        cblabel='GP STD'
        plotHM(X_Test,Y_STD,title,cblabel,X_Highlight=X_Highlight,
               savePath=STD_savePath)
    if plotHM_AE:
        Y_ABS=numpy.abs(Y_Test-Y_Pred)
        title=titlePrefix+', AE'
        cblabel='AE'
        plotHM(X_Test,Y_ABS,title,cblabel,X_Highlight=X_Highlight,
               savePath=AE_savePath)
    if plotHM_PE:
        Y_PE=100*numpy.abs(Y_Test-Y_Pred)/Y_Test
        title=titlePrefix+', PE'
        cblabel='PE (%)'
        plotHM(X_Test,Y_PE,title,cblabel,X_Highlight=X_Highlight,
               savePath=PE_savePath)
    if plotT_Slices:
        title=titlePrefix+', GP Prediction'
        plot_T_Slices(T_Slices,X_Test,Y_Test,title,Y_2=Y_Pred,
                      X_Highlight=X_Highlight,savePath=T_Slices_savePath)
    # Output
    return scores

# =============================================================================
# Active Learning
# =============================================================================

def AL_Independent_Binary(targetComp,X_Init,Y_Init,X_Test,Y_Test,
                          gpConfig,X_Scaler=None,maxIter=100,min_MRE=0.1,
                          plot_T_Slices_GIF=None,T_Slices=None,
                          plotHM_Pred_GIF=None,plotHM_STD_GIF=None,
                          plotHM_AE_GIF=None,plotHM_PE_GIF=None,
                          plotHM_AF_GIF=None,titlePrefix=''):
    """
    AL_Independent_Binary() performs active learning on the activity
    coefficient of a single component in a mixture. The algorithm stops when
    the number of iterations reaches maxIter or when the GP-predicted mean 
    relative error reaches min_MRE.

    Parameters
    ----------
    targetComp : int
        The component of interest. Necessary to add new virtual points
        representing gamma=1. One of:
            . 1 - Component #1
            . 2 - Component #2
    X_Init : numpy array (K,2)
        Initial data points (features).
    Y_Init : numpy array (K,1)
        Initial data points (labels).
    X_Test : numpy array (N,2)
        Testing features (x1,T).
    Y_Test : numpy array (N,1)
        Testing (true) labels (gamma).
    gpConfig : dictionary
        Dictionary containing GP-related inputs, as defined in buildGP():
            . X_Scaler
            . kernel
            . indepDim
            . doLogY
            . trainLikelihood
    X_Scaler : scikit-learn preprocessing object
        Scikit-learn preprocessing object to normalize X. If None, no
        normalization is done.
        The default is None.
    maxIter : int, optional
        Maximum number of active learning iterations.
        The default is 100.
    min_MRE : float, optional
        Desired target value for the GP-predicted mean relative error (%).
        Once this alue is reached, active learning stops.
        The default is 0.1%.
    plot_T_Slices_GIF : string, optional
        Path to save a GIF of T slices plots.
        If None, no GIF is generated.
        The default is None.
    T_Slices : numpy array (L,), optional
        Temperature slices to be plotted.
        The default is None.
    plotHM_Pred_GIF : string, optional
        Path to save a GIF of GP prediction heatmaps.
        If None, no GIF is generated.
        The default is None.
    plotHM_STD_GIF : string, optional
        Path to save a GIF of GP STD heatmaps.
        If None, no GIF is generated.
        The default is None.
    plotHM_AE_GIF : string, optional
        Path to save a GIF of GP absolute error heatmaps.
        If None, no GIF is generated.
        The default is None.
    plotHM_PE_GIF : string, optional
        Path to save a GIF of GP percentage error heatmaps.
        If None, no GIF is generated.
        The default is None.
    plotHM_AF_GIF : string, optional
        Path to save a GIF of acquistion function (AF) heatmaps.
        If None, no GIF is generated.
        The default is None.
    titlePrefix : string, optional
        Title prefix for the plots requested.
        The default is ''.

    Returns
    -------
    model : gpflow.models.gpr.GPR object
        Final GP model after all AL iterations.
    X_Train : numpy array (N,2)
        Final training features after all AL iterations, where N is the number
        of initial data points + AL iterations.
    scores : dictionary
        Performance-related scores (R2, MAE, MPE, RMSE, RMSLE, GP_MPE) as a
        function of AL iteration.

    """
    # Get path to temp folder
    tempFolder=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..',
                            '_temp')
    # Initialize metrics
    R2_Vector=[]
    MAE_Vector=[]
    MPE_Vector=[]
    RMSE_Vector=[]
    RMSLE_Vector=[]
    GP_MPE_Vector=[]
    # Check GIF requests
    if plot_T_Slices_GIF is not None: plot_T_Slices_GIF_frames=[]
    if plotHM_Pred_GIF is not None: plotHM_Pred_GIF_frames=[]
    if plotHM_STD_GIF is not None: plotHM_STD_GIF_frames=[]
    if plotHM_AE_GIF is not None: plotHM_AE_GIF_frames=[]
    if plotHM_PE_GIF is not None: plotHM_PE_GIF_frames=[]
    if plotHM_AF_GIF is not None: plotHM_AF_GIF_frames=[]
    # Initialize X_Train and Y_Train
    X_Train=X_Init
    Y_Train=Y_Init
    # Loop over iterations requested
    for n in tqdm(range(maxIter),'Active Learning Iteration: '):
        # Build GP
        model=buildGP(X_Train,Y_Train,X_Scaler=X_Scaler,gpConfig=gpConfig)
        # Check GIF requests and build temporary paths for GIF frames
        if plot_T_Slices_GIF is not None:
            plot_T=os.path.join(tempFolder,'plot_T_Slices_'+str(n)+'.png')
            plot_T_Slices_GIF_frames.append(plot_T)
        else:
            plot_T=False
        if plotHM_Pred_GIF is not None:
            plotHM_Pred=os.path.join(tempFolder,'plotHM_Pred_'+str(n)+'.png')
            plotHM_Pred_GIF_frames.append(plotHM_Pred)
        else:
            plotHM_Pred=False
        if plotHM_STD_GIF is not None:
            plotHM_STD=os.path.join(tempFolder,'plotHM_STD_'+str(n)+'.png')
            plotHM_STD_GIF_frames.append(plotHM_STD)
        else:
            plotHM_STD=False
        if plotHM_AE_GIF is not None:
            plotHM_AE=os.path.join(tempFolder,'plotHM_AE_'+str(n)+'.png')
            plotHM_AE_GIF_frames.append(plotHM_AE)
        else:
            plotHM_AE=False
        if plotHM_PE_GIF is not None:
            plotHM_PE=os.path.join(tempFolder,'plotHM_PE_'+str(n)+'.png')
            plotHM_PE_GIF_frames.append(plotHM_PE)
        else:
            plotHM_PE=False
        if plotHM_AF_GIF is not None:
            plotHM_AF=os.path.join(tempFolder,'plotHM_AF_'+str(n)+'.png')
            plotHM_AF_GIF_frames.append(plotHM_AF)
        else:
            plotHM_AF=False
        # Perform predictions
        Y_Pred,Y_STD=gpPredict(model,X_Test,X_Scaler=X_Scaler,
                               gpConfig=gpConfig)
        # Score predictions
        scores=evaluateModel(Y_Pred,X_Test,Y_Test,Y_STD=Y_STD,
                             plotHM_Pred=plotHM_Pred,plotHM_STD=plotHM_STD,
                             plotHM_AE=plotHM_AE,plotHM_PE=plotHM_PE,
                             plotT_Slices=plot_T,T_Slices=T_Slices,
                             titlePrefix=titlePrefix+', It='+str(n),
                             X_Highlight=X_Train)
        # Compute acquisition function
        AF=100*Y_STD/Y_Pred
        # Compute GP_MPE
        GP_MPE=AF.mean()
        # Append scores
        R2_Vector.append(scores['R2'])
        MAE_Vector.append(scores['MAE'])
        MPE_Vector.append(scores['MPE'])
        RMSE_Vector.append(scores['RMSE'])
        RMSLE_Vector.append(scores['RMSLE'])
        GP_MPE_Vector.append(GP_MPE)
        # Check request for AF GIF
        if plotHM_AF_GIF is not None:
            plotHM(X_Test,AF,titlePrefix+',It='+str(n)+', AF','GP_RE (%)',
                   X_Highlight=X_Train,savePath=plotHM_AF)
        # Select next point
        X_New=X_Test[AF.argmax(),:].copy().reshape(1,2)
        Y_New=Y_Test[AF.argmax(),:].copy().reshape(-1,1)
        # Append to Training Dataset
        X_Train=numpy.append(X_Train,X_New,axis=0)
        Y_Train=numpy.append(Y_Train,Y_New,axis=0)
        # Check GP_MPE
        if len(GP_MPE_Vector)>2 and GP_MPE<min_MRE: break
    # Remove last two row of X (added in the last iteration but not used)
    X_Train=numpy.delete(X_Train,-1,0)
    # Generate scores
    scores={}
    scores['R2']=R2_Vector
    scores['MAE']=MAE_Vector
    scores['MPE']=MPE_Vector
    scores['RMSE']=RMSE_Vector
    scores['RMSLE']=RMSLE_Vector
    scores['GP_MPE']=GP_MPE_Vector
    # Generate GIFs
    if plot_T_Slices_GIF is not None:
        generateGIF(plot_T_Slices_GIF,plot_T_Slices_GIF_frames)
        for file in plot_T_Slices_GIF_frames: os.remove(file)
    if plotHM_Pred_GIF is not None:
        generateGIF(plotHM_Pred_GIF,plotHM_Pred_GIF_frames)
        for file in plotHM_Pred_GIF_frames: os.remove(file)
    if plotHM_STD_GIF is not None:
        generateGIF(plotHM_STD_GIF,plotHM_STD_GIF_frames)
        for file in plotHM_STD_GIF_frames: os.remove(file)
    if plotHM_AE_GIF is not None:
        generateGIF(plotHM_AE_GIF,plotHM_AE_GIF_frames)
        for file in plotHM_AE_GIF_frames: os.remove(file)
    if plotHM_PE_GIF is not None:
        generateGIF(plotHM_PE_GIF,plotHM_PE_GIF_frames)
        for file in plotHM_PE_GIF_frames: os.remove(file)
    if plotHM_AF_GIF is not None:
        generateGIF(plotHM_AF_GIF,plotHM_AF_GIF_frames)
        for file in plotHM_AF_GIF_frames: os.remove(file)
    #Output
    return model,X_Train,scores

def AL_VLE_Binary_Type1(F_Truth_1,F_Truth_2,gpConfig,P_VLE,Fs_VP,Fs_Inverse_VP,
                        z1_range,maxIter=100,min_AF=0.1,plot_VLE_GIF=None,
                        title=None,bubbleTruth=None,dewTruth=None):
    """
    AL_VLE_Binary_Type1() uses a VLE-specific acquisition function to perform
    active learning and build the requested VLE phase diagram.

    Parameters
    ----------
    F_Truth_1 : function handler
        Funciton that returns gamma for component 1. Must be of the type
        gamma=F_gamma(x1,T). T in K.
    F_Truth_2 : function handler
        Funciton that returns gamma for component 2. Must be of the type
        gamma=F_gamma(x1,T). T in K.
    gpConfig : dictionary
        Dictionary containing GP-related inputs, as defined in buildGP():
            . X_Scaler
            . kernel
            . indepDim
            . doLogY
            . trainLikelihood
    P_VLE : float
        Pressure of the system (/Pa).
    Fs_VP : list of function handlers
        List where each entry "i" is the funciton that returns the vapor
        pressure for component "i". Must be of the type vp=F_VP(T). T in K, vp
        in Pa.
    Fs_Inverse_VP : list of function handlers
        List where each entry "i" is the funciton that returns the temperature
        corresponding to a given vapor pressure of component "i". Must be of
        the type T=F_VP(vp). T in K, vp in Pa.
    z1_range : numpy array, optional
        Range of z1 to perform VLE calculations.
        The default is numpy.linspace(0,1,101).
    maxIter : int, optional
        Maximum number of active learning iterations.
        The default is 100.
    min_AF : float, optional
        Desired target value for the VLE-specific acquisition function
        (mean uncertainty of the vapor phase composition).
        Once this alue is reached, active learning stops.
        The default is 0.1.
    plot_VLE_GIF : string, optional
        Path to save a GIF of the VLE at each iteration.
        If None, no GIF is generated.
        The default is None.
    title : string, optional
        Title for the VLE GIF.
        The default is None.
    bubbleTruth : numpy array, optional
        Array with the ground turth bubble curve, to be plotted in the VLE GIF.
        Columns: "x1","T_bubble"
        The default is None.
    dewTruth : numpy array, optional
        Array with the ground truth dew curve, to be plotted in the VLE GIF.
        Columns: "x1","T_dew"
        The default is None.

    Returns
    -------
    bubble_gp : numpy array (N,2)
        Array with the GP-Predicted bubble curve after active learning.
        Columns: "x1","T_bubble"
    dew_gp : numpy array (N,2)
        Array with the GP-Predicted dew curve after active learning.
        Columns: "x1","T_dew"
    gammas : numpy array (N,3)
        Array with the activity coefficients of each component at the VLE 
        equilibrium temperature for each x composition point.
        Columns: "x1","gamma_1","gamma_2"
    MAF_Vector : numpy array (K,)
        Array containing the mean value of the acquisition function at each
        active learning iteration.
    X_AL : numpy array (K,2)
        Array containing the composition/temperature points requested during
        the active learning algorithm.

    """
    # Get path to temp folder
    tempFolder=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..',
                            '_temp')
    # Initialize mean acquisition function vector
    MAF_Vector=[]
    # Check GIF request
    if plot_VLE_GIF is not None: plot_VLE_GIF_frames=[]
    # Compute Ideal VLE
    bubble,__=thermo.compute_Tx_VLE_Ideal_Binary(Fs_VP,Fs_Inverse_VP,P_VLE,
                                                 z1_range=z1_range)
    # Get minimum and maximum temperature from bubble
    Tmin=bubble[:,1].min()
    Tmax=bubble[:,1].max()
    # Generate virtual points on top of bubble
    X_VP_1=numpy.meshgrid(numpy.ones(1),numpy.linspace(Tmin,Tmax,100))
    X_VP_1=numpy.array(X_VP_1).T.reshape(-1,2)
    X_VP_2=numpy.meshgrid(numpy.zeros(1),numpy.linspace(Tmin,Tmax,100))
    X_VP_2=numpy.array(X_VP_2).T.reshape(-1,2)
    # Define X_Train
    X_Train_1=X_VP_1
    X_Train_2=X_VP_2
    # Compute Y_Train
    __,Y_Train_1=buildDataset_Binary(F_Truth_1,X=X_Train_1)
    __,Y_Train_2=buildDataset_Binary(F_Truth_2,X=X_Train_2)
    # Define X_Scaler
    __,X_Scaler=normalize(X_Train_1,method='MinMax')
    # Initialize X_AL
    X_AL=numpy.array([]).reshape(-1,2)
    # Loop over iterations requested (zeroth iteration without AL data)
    for n in range(maxIter):
        # Build GPs
        model_1=buildGP(X_Train_1,Y_Train_1,X_Scaler=X_Scaler,
                        gpConfig=gpConfig)
        model_2=buildGP(X_Train_2,Y_Train_2,X_Scaler=X_Scaler,
                        gpConfig=gpConfig)
        # Store posteriors to decrease computational cost
        model_1=model_1.posterior()
        model_2=model_2.posterior()
        # Define gamma functions for VLE calculation
        def F_Gamma_1_GP(x1,T):
            X=numpy.array([x1,T]).reshape(1,2)
            pred,__=gpPredict(model_1,X,X_Scaler=X_Scaler,gpConfig=gpConfig)
            return pred[0,0]
        def F_Gamma_2_GP(x1,T):
            X=numpy.array([x1,T]).reshape(1,2)
            pred,__=gpPredict(model_2,X,X_Scaler=X_Scaler,gpConfig=gpConfig)
            return pred[0,0]
        Fs_gamma_GP=[F_Gamma_1_GP,F_Gamma_2_GP]
        # Compute VLE
        bubble_gp,dew_gp,gamma_gp=thermo.compute_Tx_VLE_Binary(Fs_gamma_GP,
                                                               Fs_VP,
                                                               Fs_Inverse_VP,
                                                               P_VLE,
                                                             z1_range=z1_range)
        # Check plot request
        if plot_VLE_GIF is not None:
            plot_VLE=os.path.join(tempFolder,'plot_VLE_'+str(n)+'.png')
            plot_VLE_GIF_frames.append(plot_VLE)
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
            if bubbleTruth is not None and dewTruth is not None:
                plt.plot(bubbleTruth[:,0],bubbleTruth[:,1],'-k',linewidth=1,
                         label='Ground Truth')
                plt.plot(dewTruth[:,0],dewTruth[:,1],'-k',linewidth=1)
            plt.plot(bubble_gp[:,0],bubble_gp[:,1],'--r',linewidth=1,
                     label='GP-Predicted')
            plt.plot(dew_gp[:,0],dew_gp[:,1],'--r',linewidth=1)
            plt.plot(X_AL[:,0],X_AL[:,1],'or',markersize=2)
            plt.xlabel('x1')
            plt.ylabel('T /K',fontsize=7)
            if title is not None: plt.title(title)
            plt.legend(prop={'size': 6})
            plt.savefig(plot_VLE,dpi=600,bbox_inches='tight')
            plt.close()
        # Define new X_Test based on predicted VLE
        X_Test=bubble_gp.copy()
        # Get STDs on new grid
        Y_Pred_1,Y_STD_1=gpPredict(model_1,X_Test,X_Scaler=X_Scaler,
                                   gpConfig=gpConfig)
        Y_Pred_2,Y_STD_2=gpPredict(model_2,X_Test,X_Scaler=X_Scaler,
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
        # Append to X_AL
        X_AL=numpy.append(X_AL,X_New,axis=0)
        # Get minimum and maximum temperature from X_Test
        Tmin=X_Test[:,1].min()
        Tmax=X_Test[:,1].max()
        # Generate virtual points on top of X_Test
        X_VP_1=numpy.meshgrid(numpy.ones(1),numpy.linspace(Tmin,Tmax,100))
        X_VP_1=numpy.array(X_VP_1).T.reshape(-1,2)
        X_VP_2=numpy.meshgrid(numpy.zeros(1),numpy.linspace(Tmin,Tmax,100))
        X_VP_2=numpy.array(X_VP_2).T.reshape(-1,2)
        # Define X_Train
        X_Train_1=numpy.concatenate((X_VP_1,X_AL),axis=0)
        X_Train_2=numpy.concatenate((X_VP_2,X_AL),axis=0)
        # Compute Y_Train
        __,Y_Train_1=buildDataset_Binary(F_Truth_1,X=X_Train_1)
        __,Y_Train_2=buildDataset_Binary(F_Truth_2,X=X_Train_2)
        # Define X_Scaler
        __,X_Scaler=normalize(X_Train_1,method='MinMax')
        # Check GP_MPE
        if len(MAF_Vector)>1 and MAF<min_AF: break
    # Remove last row of X_AL (added in the last iteration but not used)
    X_AL=numpy.delete(X_AL,-1,0)
    # Generate GIFs
    if plot_VLE_GIF is not None:
        generateGIF(plot_VLE_GIF,plot_VLE_GIF_frames)
        for file in plot_VLE_GIF_frames: os.remove(file)
    #Output
    return bubble_gp,dew_gp,gamma_gp,MAF_Vector,X_AL

def AL_SLE_Binary_Type1(F_Truth_1,F_Truth_2,gpConfig,properties_1,properties_2,
                        x1_range,maxIter=100,min_AF=0.5,plot_SLE_GIF=None,
                        title=None,SLE_Truth=None):
    """
    AL_SLE_Binary_Type1() uses an SLE-specific algorithm to perform
    active learning and build the requested SLE phase diagram.

    Parameters
    ----------
    F_Truth_1 : function handler
        Funciton that returns gamma for component 1. Must be of the type
        gamma=F_gamma(x1,T). T in K.
    F_Truth_2 : function handler
        Funciton that returns gamma for component 2. Must be of the type
        gamma=F_gamma(x1,T). T in K.
    gpConfig : dictionary
        Dictionary containing GP-related inputs, as defined in buildGP():
            . X_Scaler
            . kernel
            . indepDim
            . doLogY
            . trainLikelihood
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
    maxIter : int, optional
        Maximum number of active learning iterations.
        The default is 100.
    min_AF : float, optional
        Desired target value for the SLE-specific acquisition function
        (GP-predicted MAE).
        Once this alue is reached, active learning stops.
        The default is 0.5%.
    plot_SLE_GIF : string, optional
        Path to save a GIF of the SLE at each iteration.
        If None, no GIF is generated.
        The default is None.
    title : string, optional
        Title for the VLE GIF.
        The default is None.
    SLE_Truth : numpy array, optional
        Array with the ground truth SLE diagram, to be plotted in the SLE GIF.
        Columns: "x1","T_SLE"
        The default is None.

    Returns
    -------
    bubble_gp : numpy array (N,2)
        Array with the GP-Predicted bubble curve after active learning.
        Columns: "x1","T_bubble"
    dew_gp : numpy array (N,2)
        Array with the GP-Predicted dew curve after active learning.
        Columns: "x1","T_dew"
    gammas : numpy array (N,3)
        Array with the activity coefficients of each component at the VLE 
        equilibrium temperature for each x composition point.
        Columns: "x1","gamma_1","gamma_2"
    MAF_Vector : numpy array (K,)
        Array containing the mean value of the acquisition function at each
        active learning iteration.
    X_AL : numpy array (K,2)
        Array containing the composition/temperature points requested during
        the active learning algorithm.

    """
    # Get path to temp folder
    tempFolder=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..',
                            '_temp')
    # Initialize mean acquisition function vector
    MAF_Vector=[]
    # Check GIF request
    if plot_SLE_GIF is not None: plot_SLE_GIF_frames=[]
    # Compute Ideal SLE
    SLE_ID=thermo.compute_Tx_SLE_Ideal_Binary(properties_1,properties_2,
                                              x1_range=x1_range)
    X_Test=SLE_ID
    # Find eutectic
    eutectic=X_Test[:,1].argmin()
    # Select midway point of largest liquidus curve
    if X_Test[eutectic,0]<0.5:
        index=int((len(SLE_ID[:,0])-eutectic)/2)
        X_AL=numpy.array([SLE_ID[index,:]]).reshape(-1,2)
    else:
        index=int(eutectic/2)
        X_AL=numpy.array([SLE_ID[index,:]]).reshape(-1,2)
    # Loop over iterations requested (zeroth iteration with midway eutectic)
    for n in range(maxIter):
        # Get minimum and maximum temperature from SLE_ID
        Tmin=X_Test[:,1].min()
        Tmax=X_Test[:,1].max()
        # Generate virtual points on top of SLE_ID
        X_VP_1=numpy.meshgrid(numpy.ones(1),numpy.linspace(Tmin,Tmax,100))
        X_VP_1=numpy.array(X_VP_1).T.reshape(-1,2)
        X_VP_2=numpy.meshgrid(numpy.zeros(1),numpy.linspace(Tmin,Tmax,100))
        X_VP_2=numpy.array(X_VP_2).T.reshape(-1,2)
        # Define X_Train
        X_Train_1=numpy.concatenate((X_VP_1,X_AL),axis=0)
        X_Train_2=numpy.concatenate((X_VP_2,X_AL),axis=0)
        # Compute Y_Train
        __,Y_Train_1=buildDataset_Binary(F_Truth_1,X=X_Train_1)
        __,Y_Train_2=buildDataset_Binary(F_Truth_2,X=X_Train_2)
        # Define X_Scaler
        __,X_Scaler=normalize(X_Test,method='MinMax')
        # Build GPs
        model_1=buildGP(X_Train_1,Y_Train_1,X_Scaler=X_Scaler,
                        gpConfig=gpConfig)
        model_2=buildGP(X_Train_2,Y_Train_2,X_Scaler=X_Scaler,
                        gpConfig=gpConfig)
        # Store posteriors to decrease computational cost
        model_1=model_1.posterior()
        model_2=model_2.posterior()
        # Define gamma functions for SLE calculation
        def F_Gamma_1_GP(x1,T):
            X=numpy.array([x1,T]).reshape(1,2)
            pred,__=gpPredict(model_1,X,X_Scaler=X_Scaler,gpConfig=gpConfig)
            return pred[0,0]
        def F_Gamma_2_GP(x1,T):
            X=numpy.array([x1,T]).reshape(1,2)
            pred,__=gpPredict(model_2,X,X_Scaler=X_Scaler,gpConfig=gpConfig)
            return pred[0,0]
        Fs_gamma_GP=[F_Gamma_1_GP,F_Gamma_2_GP]
        # Compute SLE
        SLE_gp,gamma_gp=thermo.compute_Tx_SLE_Binary(Fs_gamma_GP,
                                               properties_1,properties_2,
                                               x1_range=x1_range)
        # Check plot request
        if plot_SLE_GIF is not None:
            plot_SLE=os.path.join(tempFolder,'plot_SLE_'+str(n)+'.png')
            plot_SLE_GIF_frames.append(plot_SLE)
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
            if SLE_Truth is not None:
                plt.plot(SLE_Truth[:,0],SLE_Truth[:,1],'-k',linewidth=1,
                         label='Ground Truth')
            plt.plot(SLE_gp[:,0],SLE_gp[:,1],'--r',linewidth=1,
                     label='GP-Predicted')
            plt.plot(X_AL[:,0],X_AL[:,1],'or',markersize=2)
            plt.xlabel('x1')
            plt.ylabel('T /K',fontsize=7)
            if title is not None: plt.title(title)
            plt.legend(prop={'size': 6})
            plt.savefig(plot_SLE,dpi=600,bbox_inches='tight')
            plt.close()
        # Define new X_Test based on predicted SLE
        X_Test=SLE_gp
        # Find eutectic
        eutectic=X_Test[:,1].argmin()
        # Define X_Test_i
        X_Test_1=X_Test[eutectic:,:]
        X_Test_2=X_Test[:eutectic,:]
        # Get STDs on new grid
        Y_Pred_1,Y_STD_1=gpPredict(model_1,X_Test_1,X_Scaler=X_Scaler,
                                   gpConfig=gpConfig)
        Y_Pred_2,Y_STD_2=gpPredict(model_2,X_Test_2,X_Scaler=X_Scaler,
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
        # Check MAF
        if len(MAF_Vector)>0 and MAF<min_AF: break
        # Select next point based on largest AF
        if AF_1.max()>AF_2.max():
            X_New=X_Test_1[AF_1.argmax(),:].reshape(-1,2)
        else:
            X_New=X_Test_2[AF_2.argmax(),:].reshape(-1,2)
        # Append to X_AL
        X_AL=numpy.append(X_AL,X_New,axis=0)
    # Generate GIFs
    if plot_SLE_GIF is not None:
        generateGIF(plot_SLE_GIF,plot_SLE_GIF_frames)
        for file in plot_SLE_GIF_frames: os.remove(file)
    #Output
    return SLE_gp,gamma_gp,MAF_Vector,X_AL
