# imports
import matplotlib.pyplot as plt
import numpy as np
from os.path import join


# module graphics_utilities
# a module containing methods for graphics utilities


#---------------------------------------------------
# Methods
#---------------------------------------------------

  
def fit_single_variable(actual, fitted, dates, name, path):
    
    """
    fit_single_variable(actual, fitted, dates, name, path)
    creates a plot for fit vs. actual, for a single variable
    
    parameters:
    actual : ndarray of dimension (n,)
        vector of actual (observed) values
    fitted : array of dimension (n,)
        vector of fitted (in-sample predictions) values
    dates : DateTime index
        index of insample datetime entries
    name : str
        name of variable being plotted as actual
    path : str
        full path to folder where plot is going to be saved as image
    """
    
    # create figure    
    fig = plt.figure(figsize = (6.60, 4.70), dpi = 500, tight_layout = True)
    min_YLim, max_YLim = set_min_and_max(np.hstack((actual,fitted)), 0.07, 0.1)
    # plot actual and fitted
    plt.plot(dates, actual, linewidth = 1.3, color = (0.1, 0.3, 0.8))
    plt.plot(dates, fitted, linewidth = 1.3, color = (0, 0.6, 0))
    # set graphic limits, background and font size for ticks
    plt.xlim(dates[0], dates[-1])
    plt.ylim(min_YLim, max_YLim)
    plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
    plt.grid(True, color = (0.8, 0.8, 0.8))
    plt.title('Actual and fitted: ' + name, fontdict = {'fontsize' : 14, \
              'fontname': 'Serif', 'fontweight': 'semibold'}) 
    plt.tick_params(direction = 'in', labelsize = 12)         
    # save in folder as image
    full_path = join(path, 'graphics', 'fit_' + name + '.png')
    plt.savefig(full_path)  
    plt.close(fig) 
    
    
def residual_single_variable(residuals, dates, name, path):
    
    """
    residual_single_variable(residuals, dates, name, path)
    creates a plot for the residuals, for a single variable
    
    parameters:
    residuals : ndarray of dimension (n,)
        vector of actual (observed) values
    dates : DateTime index
        index of insample datetime entries
    name : str
        name of variable being plotted as residuals
    path : str
        full path to folder where plot is going to be saved as image
    """
    
    # create figure    
    fig = plt.figure(figsize = (6.60, 4.70), dpi = 500, tight_layout = True)
    min_YLim, max_YLim = set_min_and_max(residuals, 0.07, 0.1)
    # plot actual and fitted
    plt.plot(dates, residuals, linewidth = 1.3, color = (0.1, 0.3, 0.8))
    plt.plot([dates[0],dates[-1]], [0,0], linewidth = 0.5, color = (0, 0, 0), linestyle = (0,(12,8)))
    # set graphic limits, background and font size for ticks
    plt.xlim(dates[0], dates[-1])
    plt.ylim(min_YLim, max_YLim)
    plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
    plt.grid(True, color = (0.8, 0.8, 0.8))
    plt.title('Residuals: ' + name, fontdict = {'fontsize' : 14, \
              'fontname': 'Serif', 'fontweight': 'semibold'})
    plt.tick_params(direction = 'in', labelsize = 12)         
    # save in folder as image
    full_path = join(path, 'graphics', 'residuals_' + name + '.png')
    plt.savefig(full_path)  
    plt.close(fig)    
    
    
def ols_forecasts_single_variable(forecasts, y_p, dates, name, path):
    
    """
    ols_forecasts_single_variable(forecasts, dates, name, path)
    creates a plot for the forecasts of an OLS model
    
    parameters:
    forecasts : ndarray
        array of forecast estimates
    y_p : ndarray
        array (possibly empty) of actual values for the forecasts
    dates : DateTime index
        index of forecast datetime entries
    name : str
        name of variable being plotted as forecast
    path : str
        full path to folder where plot is going to be saved as image
    """    

    # create figure   
    fig = plt.figure(figsize = (6.60, 4.70), dpi = 500, tight_layout = True)
    ax = fig.add_subplot(1,1,1)
    min_YLim, max_YLim = set_min_and_max(forecasts, 0.15, 0.15)    
    # plot forecasts, with patched credibility intervals
    patch_dates = np.hstack((dates, np.flipud(dates)))
    patch_data = np.hstack((forecasts[:,0], np.flipud(forecasts[:,2])))
    plt.fill(patch_dates, patch_data, facecolor = (0.3, 0.7, 0.2), alpha = 0.4)
    plt.plot(dates, forecasts[:,1], linewidth = 1.3, color = (0, 0.6, 0))
    if len(y_p) != 0:
        plt.plot(dates, y_p, linewidth = 1.3, color = (0.1, 0.3, 0.8))
    # set graphic limits, background and font size for ticks
    plt.xlim(dates[0], dates[-1])
    plt.ylim(min_YLim, max_YLim)
    plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
    plt.grid(True, color = (0.8, 0.8, 0.8))
    plt.gcf().autofmt_xdate(rotation = 0, ha = 'center')
    plt.setp(ax.get_xticklabels()[1::2], visible=False);
    # title
    plt.title('Forecasts: ' + name, fontdict = {'fontsize' : 14, \
              'fontname': 'Serif', 'fontweight': 'semibold'})  
    plt.tick_params(direction = 'in', labelsize = 12) 
    # save in folder as image
    full_path = join(path, 'graphics', 'forecasts_' + name + '.png')
    plt.savefig(full_path)  
    plt.close(fig)  
    
    
def set_min_and_max(data, min_space, max_space):
    
    """
    set_min_and_max(data)
    returns the min and max Y values for a graphic, given data
    
    parameters:
    data : ndarray
        matrix of plotted values
    min_space : float
        scale to define lower space
    max_space : float
        scale to define upper space
            
    returns:
    min_YLim : scalar
        min Y value for the plot
    max_YLim : scalar
        max Y value for the plot
    """
    
    # get min, max, and compute window width
    min_value = np.amin(data)
    max_value = np.amax(data)
    width = max_value - min_value
    min_YLim = min_value - min_space * width
    max_YLim = max_value + max_space * width
    return min_YLim, max_YLim  
    
    
    
    
    
    
    
    
    
    