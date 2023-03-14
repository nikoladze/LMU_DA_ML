import warnings
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve

# 
#





def visualize_classifier(model, X, y, ax=None, cmap='RdBu', plot_proba=False):
    """
    helper function to visualize 2D model output

    Args:
        model     : classifier model
        X         : feature matrix (numpy array, pandas df)
        y         : target (numpy array, pandas df)
        ax        : The :class:`matplotlib.axes.Axes` to plot on. If not given automatically scale.
        cmap      : Color map to use
        plot_proba: Boolean flag -- optional plot probabilities instead classification
 
    Returns:
        None

    """
 

    ax = ax or plt.gca()
    
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap, edgecolors="black",
               clim=(y.min(), y.max()), zorder=3)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
        
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    X = np.stack([xx.ravel(), yy.ravel()], axis=1)
    
    if not plot_proba:
        Z = model.predict(X).reshape(xx.shape)
    else:
        Z = model.predict_proba(X)[:,1].reshape(xx.shape)
    
    # Create a color plot with the results
    n_classes = len(np.unique(y))
    if not plot_proba:
        ax.contourf(xx, yy, Z, alpha=0.3,
                    levels=np.arange(n_classes + 1) - 0.5,
                    cmap=cmap, zorder=1)
    else:
        ax.pcolormesh(xx, yy, Z, cmap=cmap, shading="auto")

    ax.set(xlim=xlim, ylim=ylim)
    
    
    
def plot_feature_importance( model, feature_names, figsize=(6,8), sort=False ):
    """
    helper function to visualize feature importance

    Args:
        model         : classifier model
        feature_names : list with names of features
        figsize       : tuple with size of figure 
        sort          : Boolean flag -- sort before plot
  
    Returns:
        None

    """
    n_features = len(feature_names)
    if sort:
        # sort feature importance and keys accdg to feature importance
        fks = [ (f, k) for f,k in sorted(zip(model.feature_importances_, feature_names ))]
    else:
        fks = [ (f, k) for f,k in zip(model.feature_importances_, feature_names )]
        
    fis = [ f for f,k in fks]
    ks = [ k for f,k in fks]
    plt.figure(figsize = (6,8))
    plt.barh(range(n_features), fis, align='center')
    plt.yticks(np.arange(n_features), ks)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features);

    
def plot_proba(df, model, x):
    """
    helper function to plot probability for signal and background

    Args:
        df            : dataframe with features and s/b labels
        model         : classifier model
        x             : feature matrix (numpy array, pandas df)
  
    Returns:
        None

    """
    df['Prob']=model.predict_proba(x)[:, 1]
    kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
    df[df.Label==0].Prob.hist(label='Background',**kwargs)
    df[df.Label==1].Prob.hist(label='Signal',**kwargs)
    plt.legend()
    

def plot_roc_test_train(model, ytest, xtest, wgttest=None, ytrain=None, xtrain=None, wgttrain=None, pcut=None):
    """
    helper function to plot roc curve

    Args:
        model      : classifier model
        xtest      : feature matrix (numpy array, pandas df)
        ytest      : target (numpy array, pandas df)
        wgttest    : weights (numpy array, pandas df)
        xtrain     : feature matrix (numpy array, pandas df)
        ytrain     : target (numpy array, pandas df)
        wgttrain   : weights (numpy array, pandas df)
        pcut       : optional pcut value to plot
 
    Returns:
        None

    """
    fpr, tpr, thresholds = roc_curve(ytest, model.predict_proba(xtest)[:, 1], sample_weight = wgttest)
    plt.plot(fpr, tpr, label="ROC Curve test")
    if ytrain is not None:
        fpr_tr, tpr_tr, thresholds_tr = roc_curve(ytrain, model.predict_proba(xtrain)[:, 1], sample_weight = wgttrain)
        plt.plot(fpr_tr, tpr_tr, label="ROC Curve train")
    plt.xlabel("FPR")
    plt.ylabel("TPR (recall)")
    if pcut is not None:
        mark_threshold = pcut # mark selected threshold
        idx = np.argmin(np.abs(thresholds - mark_threshold))
        plt.plot(fpr[idx], tpr[idx], 'o', markersize=10, label=f"threshold {mark_threshold:7.4f}", fillstyle="none", mew=2)
    plt.legend(loc=4);



    
# compute approximate median significance (AMS) (Higgs challenge)
def ams(s,b):
    # The number 10, added to the background yield, is a regularization term to decrease the variance of the AMS.
    return np.sqrt(2*((s+b+10)*np.log(1+s/(b+10))-s))



# Run the AMS scan
from sklearn.metrics import roc_curve
def ams_scan(y, y_prob, weights=None, sigall=1., backall=1.):
    """
    helper function to calculate ams values along roc curve

    Args:
        y          : true y
        x_prob     : predicted y score
        weights    : weights 
        sigall     : total weight signal
        backall    : total weight background
 
    Returns:
        tuple(pcut-array, ams-array)

    """
    fpr, tpr, thr = roc_curve(y, y_prob, sample_weight=weights)
    ams_vals = ams(tpr * sigall, fpr * backall)
    return ( thr, ams_vals) 


