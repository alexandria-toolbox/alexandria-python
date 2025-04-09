# imports
import matplotlib.pyplot as plt
import numpy as np
from os.path import join


# module graphics_utilities
# a module containing methods for graphics utilities


#---------------------------------------------------
# Methods
#---------------------------------------------------


def fit_single_variable(actual, fitted, dates, name):
    
    """
    fit_single_variable(actual, fitted, dates, name)
    produces fitted figure for regression model, single variable
    
    parameters:     
    actual: numpy ndarray of size (n,)
        actual sample values
    fitted: numpy ndarray of size (n,)
        fitted values
    dates: datetime index of size (n)
        index of in-sample dates
    name: str
        name of variable for which figure is produced
        
    returns:
    fig: matplotlib figure
        fitted figure
    """  
    
    # create figure    
    fig = plt.figure(figsize = (6.60, 4.70), dpi = 500, tight_layout = True)
    min_YLim, max_YLim = set_min_and_max(np.hstack((actual,fitted)), 0.07, 0.1)
    # plot actual and fitted
    plt.plot(dates, actual, linewidth = 1.5, color = (0.1, 0.3, 0.8))
    plt.plot(dates, fitted, linewidth = 1.5, color = (0, 0.6, 0))
    # set graphic limits, background and font size for ticks
    plt.xlim(dates[0], dates[-1])
    plt.ylim(min_YLim, max_YLim)
    plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
    plt.grid(True, color = (0.8, 0.8, 0.8))
    plt.title('Actual and fitted: ' + name, fontdict = {'fontsize' : 14, \
              'fontname': 'Serif', 'fontweight': 'semibold'}) 
    plt.tick_params(axis = 'x', direction = 'in', labelsize = 12, labelrotation=20)      
    return fig


def residual_single_variable(residuals, dates, name):

    """
    residual_single_variable(residuals, dates, name)
    produces residual figure for regression model, single variable
    
    residuals: numpy ndarray of size (n,)
        residual values
    dates: datetime index of size (n)
        index of in-sample dates
    name: str
        name of variable for which figure is produced
        
    returns:
    fig: matplotlib figure
        residual figure
    """  
        
    # create figure    
    fig = plt.figure(figsize = (6.60, 4.70), dpi = 500, tight_layout = True)
    min_YLim, max_YLim = set_min_and_max(residuals, 0.07, 0.1)
    # plot residuals
    plt.plot([dates[0],dates[-1]], [0,0], linewidth = 1, color = (0, 0, 0), linestyle='--', dashes=(5, 5))
    plt.plot(dates, residuals, linewidth = 1.5, color = (0.1, 0.3, 0.8))
    # set graphic limits, background and font size for ticks
    plt.xlim(dates[0], dates[-1])
    plt.ylim(min_YLim, max_YLim)
    plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
    plt.grid(True, color = (0.8, 0.8, 0.8))
    plt.title('Residuals: ' + name, fontdict = {'fontsize' : 14, \
              'fontname': 'Serif', 'fontweight': 'semibold'})
    plt.tick_params(axis = 'x', direction = 'in', labelsize = 12, labelrotation=20)         
    return fig


def ols_forecasts_single_variable(forecasts, y_p, dates, name):  

    """
    ols_forecasts_single_variable(forecasts, y_p, dates, name)
    produces forecast figure for regression model, single variable
    
    forecasts: ndarray of size (n_forecast,3)
        forecast values, median, lower and upper bound
    y_p: ndarray of size (n_forecast,) or empty list
        actual values for forecast evaluation
    dates: datetime index of size (n)
        index of in-sample dates
    name: str
        name of variable for which figure is produced
        
    returns:
    fig: matplotlib figure
        forecast figure
    """  
    
    # create figure   
    fig = plt.figure(figsize = (6.60, 4.70), dpi = 500, tight_layout = True)
    ax = fig.add_subplot(1,1,1)
    min_YLim, max_YLim = set_min_and_max(forecasts, 0.15, 0.15)    
    # plot forecasts, with patched credibility intervals
    patch_dates = np.hstack((dates, np.flipud(dates)))
    patch_data = np.hstack((forecasts[:,1], np.flipud(forecasts[:,2])))
    plt.fill(patch_dates, patch_data, facecolor = (0.3, 0.7, 0.2), alpha = 0.4)
    if min_YLim < 0 and max_YLim > 0:
        plt.plot([dates[0],dates[-1]], [0,0], linewidth = 1, color = (0, 0, 0), linestyle='--', dashes=(5, 5))
    plt.plot(dates, forecasts[:,0], linewidth = 1.5, color = (0, 0.6, 0))
    if len(y_p) != 0:
        plt.plot(dates, y_p, linewidth = 1.5, linestyle='--', color = (0.1, 0.3, 0.8))
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
    plt.tick_params(axis = 'x', direction = 'in', labelsize = 12, labelrotation=20)   
    return fig


def var_fit_single_variable(actual, fitted, dates, name):

    """
    var_fit_single_variable(actual, fitted, dates, name)
    produces fitted figure for var model, single variable
    
    parameters:     
    actual: numpy ndarray of size (T,1)
        actual sample values
    fitted: numpy ndarray of size (T,3)
        fitted values, median, lower and upper bounds
    dates: datetime index of size (T)
        index of in-sample dates
    name: str
        name of variable for which figure is produced
        
    returns:
    fig: matplotlib figure
        fitted figure
    """  
    
    # create figure    
    fig = plt.figure(figsize = (6.60, 4.70), dpi = 500, tight_layout = True)
    min_YLim, max_YLim = set_min_and_max(np.hstack((actual,fitted)), 0.07, 0.1)
    # plot actual, and fitted with patched credibility intervals
    patch_dates = np.hstack((dates, np.flipud(dates)))
    patch_data = np.hstack((fitted[:,1], np.flipud(fitted[:,2])))
    plt.fill(patch_dates, patch_data, facecolor = (0.3, 0.7, 0.2), alpha = 0.4)
    plt.plot(dates, fitted[:,0], linewidth = 1.5, color = (0, 0.5, 0))
    plt.plot(dates, actual, linewidth = 1.5, color = (0.1, 0.3, 0.8))
    # set graphic limits, background and font size for ticks
    plt.xlim(dates[0], dates[-1])
    plt.ylim(min_YLim, max_YLim)
    plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
    plt.grid(True, color = (0.8, 0.8, 0.8))
    plt.title('Actual and fitted: ' + name, fontdict = {'fontsize' : 14, \
              'fontname': 'Serif', 'fontweight': 'semibold'}) 
    plt.tick_params(axis = 'x', direction = 'in', labelsize = 12, labelrotation=20)        
    return fig


def var_fit_all(actual, fitted, dates, endogenous, n):
    
    """
    var_fit_all(actual, fitted, dates, endogenous, n)
    produces fitted figure for var model, all variables
    
    parameters:     
    actual: numpy ndarray of size (T,n)
        actual sample values
    fitted: numpy ndarray of size (T,n,3)
        fitted values, median, lower and upper bounds
    dates: datetime index of size (T)
        index of in-sample dates
    endogenous: str list
        list of endogenous variables for which figure is produced
    n: int
        number of endogenous variables
        
    returns:
    fig: matplotlib figure
        fitted figure
    """     

    # get plot dimensions
    columns = int(np.ceil(n ** 0.5))
    rows = columns
    # create figure    
    fig = plt.figure(figsize = (6.60*columns, 4.70*rows), dpi = 500, tight_layout = True)
    patch_dates = np.hstack((dates, np.flipud(dates)))
    # loop over variables
    for i in range(n):
        # initiate subplot
        plt.subplot(rows, columns, i+1)        
        # get min and max for subplot
        min_YLim, max_YLim = set_min_and_max(np.hstack((actual[:,[i]],fitted[:,i,:])), 0.07, 0.1)
        # plot actual, and fitted with patched credibility intervals
        patch_data = np.hstack((fitted[:,i,1], np.flipud(fitted[:,i,2])))
        plt.fill(patch_dates, patch_data, facecolor = (0.3, 0.7, 0.2), alpha = 0.4)
        plt.plot(dates, fitted[:,i,0], linewidth = 1.5, color = (0, 0.5, 0))
        plt.plot(dates, actual[:,i], linewidth = 1.5, color = (0.1, 0.3, 0.8))        
        # set graphic limits, background and font size for ticks
        plt.xlim(dates[0], dates[-1])
        plt.ylim(min_YLim, max_YLim)
        plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
        plt.grid(True, color = (0.8, 0.8, 0.8))
        name = endogenous[i]
        plt.title('Actual and fitted: ' + name, fontdict = {'fontsize' : 14, \
                  'fontname': 'Serif', 'fontweight': 'semibold'}) 
        plt.tick_params(axis = 'x', direction = 'in', labelsize = 12, labelrotation=20)   
    return fig

  
def var_residual_single_variable(residuals, dates, name):

    """
    var_residual_single_variable(residuals, dates, name)
    produces residual figure for var model, single variable
    
    parameters:     
    residuals: numpy ndarray of size (T,3)
        residual values, median, lower and upper bounds
    dates: datetime index of size (T)
        index of in-sample dates
    name: str
        name of variable for which figure is produced
        
    returns:
    fig: matplotlib figure
        residual figure
    """  

    # create figure    
    fig = plt.figure(figsize = (6.60, 4.70), dpi = 500, tight_layout = True)
    min_YLim, max_YLim = set_min_and_max(residuals, 0.07, 0.1)
    # plot residuals with patched credibility intervals
    patch_dates = np.hstack((dates, np.flipud(dates)))
    patch_data = np.hstack((residuals[:,1], np.flipud(residuals[:,2])))
    plt.fill(patch_dates, patch_data, facecolor = (0.3, 0.7, 0.2), alpha = 0.4)
    plt.plot(dates, np.zeros(len(dates)), linewidth = 1, color = (0, 0, 0), linestyle='--', dashes=(5, 5))
    plt.plot(dates, residuals[:,0], linewidth = 1.5, color = (0, 0.5, 0))
    # set graphic limits, background and font size for ticks
    plt.xlim(dates[0], dates[-1])
    plt.ylim(min_YLim, max_YLim)
    plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
    plt.grid(True, color = (0.8, 0.8, 0.8))
    plt.title('Residuals: ' + name, fontdict = {'fontsize' : 14, \
              'fontname': 'Serif', 'fontweight': 'semibold'}) 
    plt.tick_params(axis = 'x', direction = 'in', labelsize = 12, labelrotation=20)    
    return fig
 
    
def var_residual_all(residuals, dates, endogenous, n):
    
    """
    var_residual_all(residuals, dates, endogenous, n)
    produces residual figure for var model, all variables
    
    parameters:     
    residuals: numpy ndarray of size (T,n,3)
        residual values, median, lower and upper bounds
    dates: datetime index of size (T)
        index of in-sample dates
    endogenous: str list
        list of endogenous variables for which figure is produced
    n: int
        number of endogenous variables
        
    returns:
    fig: matplotlib figure
        residual figure
    """     

    # get plot dimensions
    columns = int(np.ceil(n ** 0.5))
    rows = columns
    # create figure    
    fig = plt.figure(figsize = (6.60*columns, 4.70*rows), dpi = 500, tight_layout = True)
    # loop over variables
    for i in range(n):
        # initiate subplot
        plt.subplot(rows, columns, i+1)        
        # get min and max for subplot
        min_YLim, max_YLim = set_min_and_max(residuals[:,i,:], 0.07, 0.1)
        # plot residuals with patched credibility intervals
        patch_dates = np.hstack((dates, np.flipud(dates)))
        patch_data = np.hstack((residuals[:,i,1], np.flipud(residuals[:,i,2])))
        plt.fill(patch_dates, patch_data, facecolor = (0.3, 0.7, 0.2), alpha = 0.4)
        plt.plot(dates, np.zeros(len(dates)), linewidth = 1, color = (0, 0, 0), linestyle='--', dashes=(5, 5))  
        plt.plot(dates, residuals[:,i,0], linewidth = 1.5, color = (0, 0.5, 0))   
        # set graphic limits, background and font size for ticks
        plt.xlim(dates[0], dates[-1])
        plt.ylim(min_YLim, max_YLim)
        plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
        plt.grid(True, color = (0.8, 0.8, 0.8))
        name = endogenous[i]
        plt.title('Residuals: ' + name, fontdict = {'fontsize' : 14, \
                  'fontname': 'Serif', 'fontweight': 'semibold'}) 
        plt.tick_params(axis = 'x', direction = 'in', labelsize = 12, labelrotation=20)  
    return fig


def var_shocks_single_variable(shocks, dates, name):

    """
    var_shocks_single_variable(residuals, dates, name)
    produces structural shocks figure for var model, single variable
    
    parameters:     
    shocks: numpy ndarray of size (T,3)
        shock values, median, lower and upper bounds
    dates: datetime index of size (T)
        index of in-sample dates
    name: str
        name of variable for which figure is produced
        
    returns:
    fig: matplotlib figure
        shock figure
    """  
    
    # create figure    
    fig = plt.figure(figsize = (6.60, 4.70), dpi = 500, tight_layout = True)
    min_YLim, max_YLim = set_min_and_max(shocks, 0.07, 0.1)
    # plot shocks with patched credibility intervals
    patch_dates = np.hstack((dates, np.flipud(dates)))
    patch_data = np.hstack((shocks[:,1], np.flipud(shocks[:,2])))
    plt.fill(patch_dates, patch_data, facecolor = (0.3, 0.7, 0.2), alpha = 0.4)
    plt.plot(dates, np.zeros(len(dates)), linewidth = 1, color = (0, 0, 0), linestyle='--', dashes=(5, 5))
    plt.plot(dates, shocks[:,0], linewidth = 1.5, color = (0, 0.5, 0))
    # set graphic limits, background and font size for ticks
    plt.xlim(dates[0], dates[-1])
    plt.ylim(min_YLim, max_YLim)
    plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
    plt.grid(True, color = (0.8, 0.8, 0.8))
    plt.title('Shocks: ' + name, fontdict = {'fontsize' : 14, \
              'fontname': 'Serif', 'fontweight': 'semibold'}) 
    plt.tick_params(axis = 'x', direction = 'in', labelsize = 12, labelrotation=20)       
    return fig


def var_shocks_all(shocks, dates, endogenous, n):
    
    """
    var_shocks_all(shocks, dates, endogenous, n)
    produces shock figure for var model, all variables
    
    parameters:     
    shocks: numpy ndarray of size (T,n,3)
        shock values, median, lower and upper bounds
    dates: datetime index of size (T)
        index of in-sample dates
    endogenous: str list
        list of endogenous variables for which figure is produced
    n: int
        number of endogenous variables
        
    returns:
    fig: matplotlib figure
        shock figure
    """     

    # get plot dimensions
    columns = int(np.ceil(n ** 0.5))
    rows = columns
    # create figure    
    fig = plt.figure(figsize = (6.60*columns, 4.70*rows), dpi = 500, tight_layout = True)
    # loop over variables
    for i in range(n):
        # initiate subplot
        plt.subplot(rows, columns, i+1)        
        # get min and max for subplot
        min_YLim, max_YLim = set_min_and_max(shocks[:,i,:], 0.07, 0.1)
        # plot shocks with patched credibility intervals
        patch_dates = np.hstack((dates, np.flipud(dates)))
        patch_data = np.hstack((shocks[:,i,1], np.flipud(shocks[:,i,2])))
        plt.fill(patch_dates, patch_data, facecolor = (0.3, 0.7, 0.2), alpha = 0.4)
        if min_YLim < 0 and max_YLim > 0:
            plt.plot(dates, np.zeros(len(dates)), linewidth = 1, color = (0, 0, 0), linestyle='--', dashes=(5, 5))  
        plt.plot(dates, shocks[:,i,0], linewidth = 1.5, color = (0, 0.5, 0))    
        # set graphic limits, background and font size for ticks
        plt.xlim(dates[0], dates[-1])
        plt.ylim(min_YLim, max_YLim)
        plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
        plt.grid(True, color = (0.8, 0.8, 0.8))
        name = endogenous[i]
        plt.title('Shocks: ' + name, fontdict = {'fontsize' : 14, \
                  'fontname': 'Serif', 'fontweight': 'semibold'}) 
        plt.tick_params(axis = 'x', direction = 'in', labelsize = 12, labelrotation=20)
    return fig


def var_steady_state_single_variable(actual, steady_state, dates, name):

    """
    var_steady_state_single_variable(actual, steady_state, dates, name)
    produces steady-state figure for var model, single variable
    
    parameters:     
    actual: numpy ndarray of size (T,1)
        actual sample values
    steady_state: numpy ndarray of size (T,3)
        steady-state values, median, lower and upper bounds
    dates: datetime index of size (T)
        index of in-sample dates
    name: str
        name of variable for which figure is produced
        
    returns:
    fig: matplotlib figure
        steady-state figure
    """  
    
    # create figure    
    fig = plt.figure(figsize = (6.60, 4.70), dpi = 500, tight_layout = True)
    min_YLim, max_YLim = set_min_and_max(np.hstack((actual,steady_state)), 0.07, 0.1)
    # plot actual, and steady-state with patched credibility intervals
    patch_dates = np.hstack((dates, np.flipud(dates)))
    patch_data = np.hstack((steady_state[:,1], np.flipud(steady_state[:,2])))
    plt.fill(patch_dates, patch_data, facecolor = (0.3, 0.7, 0.2), alpha = 0.4)
    plt.plot(dates, steady_state[:,0], linewidth = 1.3, color = (0, 0.5, 0))
    plt.plot(dates, actual, linewidth = 1.5, color = (0.1, 0.3, 0.8))
    # set graphic limits, background and font size for ticks
    plt.xlim(dates[0], dates[-1])
    plt.ylim(min_YLim, max_YLim)
    plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
    plt.grid(True, color = (0.8, 0.8, 0.8))
    plt.title('Steady-state: ' + name, fontdict = {'fontsize' : 14, \
              'fontname': 'Serif', 'fontweight': 'semibold'}) 
    plt.tick_params(axis = 'x', direction = 'in', labelsize = 12, labelrotation=20)       
    return fig


def var_steady_state_all(actual, steady_state, dates, endogenous, n):
    
    """
    var_steady_state_all(actual, steady_state, dates, endogenous, n)
    produces steady-state figure for var model, all variables
    
    parameters:     
    actual: numpy ndarray of size (T,n)
        actual sample values
    steady_state: numpy ndarray of size (T,n,3)
        steady-state values, median, lower and upper bounds
    dates: datetime index of size (T)
        index of in-sample dates
    endogenous: str list
        list of endogenous variables for which figure is produced
    n: int
        number of endogenous variables
        
    returns:
    fig: matplotlib figure
        steady-state figure
    """     

    # get plot dimensions
    columns = int(np.ceil(n ** 0.5))
    rows = columns
    # create figure    
    fig = plt.figure(figsize = (6.60*columns, 4.70*rows), dpi = 500, tight_layout = True)
    # loop over variables
    for i in range(n):
        # initiate subplot
        plt.subplot(rows, columns, i+1)        
        # get min and max for subplot
        min_YLim, max_YLim = set_min_and_max(np.hstack((actual[:,[i]],steady_state[:,i,:])), 0.07, 0.1)
        # plot actual, and steady-state with patched credibility intervals
        patch_dates = np.hstack((dates, np.flipud(dates)))
        patch_data = np.hstack((steady_state[:,i,1], np.flipud(steady_state[:,i,2])))
        plt.fill(patch_dates, patch_data, facecolor = (0.3, 0.7, 0.2), alpha = 0.4)
        plt.plot(dates, steady_state[:,i,0], linewidth = 1.5, color = (0, 0.5, 0))
        plt.plot(dates, actual[:,i], linewidth = 1.5, color = (0.1, 0.3, 0.8))        
        # set graphic limits, background and font size for ticks
        plt.xlim(dates[0], dates[-1])
        plt.ylim(min_YLim, max_YLim)
        plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
        plt.grid(True, color = (0.8, 0.8, 0.8))
        name = endogenous[i]
        plt.title('Steady-state: ' + name, fontdict = {'fontsize' : 14, \
                  'fontname': 'Serif', 'fontweight': 'semibold'}) 
        plt.tick_params(axis = 'x', direction = 'in', labelsize = 12, labelrotation=20)
    return fig


def var_forecasts_single_variable(actual, forecasts, Y_p, dates, forecast_dates, name):
    
    """
    var_forecasts_single_variable(actual, forecasts, dates, forecast_dates, name)
    produces forecast figure for var model, single variable
    
    parameters:     
    actual: numpy ndarray of size (T,1)
        actual sample values
    forecasts: numpy ndarray of size (f_periods,3)
        forecast values, median, lower and upper bounds
    Y_p: numpy ndarray of size (f_periods,1)
        actual out-of-sample values    
    dates: datetime index of size (T)
        index of in-sample dates
    forecast_dates: datetime index of size (f_periods)
        index of forecast dates        
    name: str
        name of variable for which figure is produced
        
    returns:
    fig: matplotlib figure
        forecast figure
    """      
    
    # periods to plot
    T = max(10, 2 * forecasts.shape[0])
    sample_data = actual[-T:]
    sample_dates = dates[-T:]
    if len(Y_p) == 0:
        plot_data = np.vstack([np.tile(sample_data,[1,3]),forecasts])
    else:
        plot_data = np.vstack([np.tile(sample_data,[1,4]),np.hstack([forecasts,Y_p])])
    plot_dates = np.hstack([sample_dates,forecast_dates])
    prediction_data = plot_data[T-1:]
    prediction_dates = plot_dates[T-1:]
    # create figure    
    fig = plt.figure(figsize = (6.60, 4.70), dpi = 500, tight_layout = True)
    min_YLim, max_YLim = set_min_and_max(plot_data, 0.07, 0.1)
    # plot actual, and forecasts with patched credibility intervals
    patch_dates = np.hstack((prediction_dates, np.flipud(prediction_dates)))
    patch_data = np.hstack((prediction_data[:,1], np.flipud(prediction_data[:,2])))
    plt.fill(patch_dates, patch_data, facecolor = (0.3, 0.7, 0.2), alpha = 0.4)
    plt.plot(prediction_dates, prediction_data[:,0], linewidth = 1.5, color = (0, 0.5, 0))
    if len(Y_p) != 0:
        plt.plot(prediction_dates, prediction_data[:,3], linewidth = 1.5, linestyle='--', color = (0.1, 0.3, 0.8))
    plt.plot(sample_dates, sample_data, linewidth = 1.5, color = (0.1, 0.3, 0.8))
    # set graphic limits, background and font size for ticks
    plt.xlim(plot_dates[0], plot_dates[-1])
    plt.ylim(min_YLim, max_YLim)
    plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
    plt.grid(True, color = (0.8, 0.8, 0.8))
    plt.title('Forecasts: ' + name, fontdict = {'fontsize' : 14, \
              'fontname': 'Serif', 'fontweight': 'semibold'}) 
    plt.tick_params(axis = 'x', direction = 'in', labelsize = 12, labelrotation=20)    
    return fig


def var_forecasts_all(actual, forecasts, Y_p, dates, forecast_dates, endogenous, n):

    """
    var_forecasts_all(actual, forecasts, dates, forecast_dates, endogenous, n)
    produces forecast figure for var model, all variables
    
    parameters:     
    actual: numpy ndarray of size (T,n)
        actual sample values
    forecasts: numpy ndarray of size (f_periods,n,3)
        forecast values, median, lower and upper bounds
    Y_p: numpy ndarray of size (f_periods,n)
        actual out-of-sample values
    dates: datetime index of size (T)
        index of in-sample dates
    forecast_dates: datetime index of size (f_periods)
        index of in-sample dates            
    endogenous: str list
        list of endogenous variables for which figure is produced
    n: int
        number of endogenous variables
        
    returns:
    fig: matplotlib figure
        forecast figure
    """      

    # get plot dimensions
    columns = int(np.ceil(n ** 0.5))
    rows = columns
    # periods to plot
    T = max(10, 2 * forecasts.shape[0])   
    # create figure    
    fig = plt.figure(figsize = (6.60*columns, 4.70*rows), dpi = 500, tight_layout = True)
    # loop over variables
    for i in range(n):
        sample_data = actual[-T:,[i]]
        sample_dates = dates[-T:]
        if len(Y_p) == 0:
            plot_data = np.vstack([np.tile(sample_data,[1,3]),forecasts[:,i,:]])
        else:
            plot_data = np.vstack([np.tile(sample_data,[1,4]),np.hstack([forecasts[:,i,:],Y_p[:,[i]]])])        
        plot_dates = np.hstack([sample_dates,forecast_dates])
        prediction_data = plot_data[T-1:]
        prediction_dates = plot_dates[T-1:]         
        # initiate subplot
        plt.subplot(rows, columns, i+1)        
        # get min and max for subplot
        min_YLim, max_YLim = set_min_and_max(plot_data, 0.07, 0.1)
        # plot actual, and steady-state with patched credibility intervals
        patch_dates = np.hstack((prediction_dates, np.flipud(prediction_dates)))
        patch_data = np.hstack((prediction_data[:,1], np.flipud(prediction_data[:,2])))
        plt.fill(patch_dates, patch_data, facecolor = (0.3, 0.7, 0.2), alpha = 0.4)
        plt.plot(prediction_dates, prediction_data[:,0], linewidth = 1.5, color = (0, 0.5, 0))
        if len(Y_p) != 0:
            plt.plot(prediction_dates, prediction_data[:,3], linewidth = 1.5, linestyle='--', color = (0.1, 0.3, 0.8))        
        plt.plot(sample_dates, sample_data, linewidth = 1.5, color = (0.1, 0.3, 0.8))     
        # set graphic limits, background and font size for ticks
        plt.xlim(plot_dates[0], plot_dates[-1])
        plt.ylim(min_YLim, max_YLim)
        plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
        plt.grid(True, color = (0.8, 0.8, 0.8))
        name = endogenous[i]
        plt.title('Forecasts: ' + name, fontdict = {'fontsize' : 14, \
                  'fontname': 'Serif', 'fontweight': 'semibold'}) 
        plt.tick_params(axis = 'x', direction = 'in', labelsize = 12, labelrotation=20)   
    return fig


def var_conditional_forecasts_single_variable(actual, forecasts, Y_p, dates, forecast_dates, name):
    
    """
    var_conditional_forecasts_single_variable(actual, forecasts, Y_p, dates, forecast_dates, name)
    produces conditional forecast figure for var model, single variable
    
    parameters:     
    actual: numpy ndarray of size (T,1)
        actual sample values
    forecasts: numpy ndarray of size (f_periods,3)
        forecast values, median, lower and upper bounds
    Y_p: numpy ndarray of size (f_periods,1)
        actual out-of-sample values    
    dates: datetime index of size (T)
        index of in-sample dates
    forecast_dates: datetime index of size (f_periods)
        index of in-sample dates        
    name: str
        name of variable for which figure is produced
        
    returns:
    fig: matplotlib figure
        forecast figure
    """      
    
    # periods to plot
    T = max(10, 2 * forecasts.shape[0])
    sample_data = actual[-T:]
    sample_dates = dates[-T:]
    if len(Y_p) == 0:
        plot_data = np.vstack([np.tile(sample_data,[1,3]),forecasts])
    else:
        plot_data = np.vstack([np.tile(sample_data,[1,4]),np.hstack([forecasts,Y_p])])
    plot_dates = np.hstack([sample_dates,forecast_dates])
    prediction_data = plot_data[T-1:]
    prediction_dates = plot_dates[T-1:]
    # create figure    
    fig = plt.figure(figsize = (6.60, 4.70), dpi = 500, tight_layout = True)
    min_YLim, max_YLim = set_min_and_max(plot_data, 0.07, 0.1)
    # plot actual, and forecasts with patched credibility intervals
    patch_dates = np.hstack((prediction_dates, np.flipud(prediction_dates)))
    patch_data = np.hstack((prediction_data[:,1], np.flipud(prediction_data[:,2])))
    plt.fill(patch_dates, patch_data, facecolor = (0.3, 0.7, 0.2), alpha = 0.4)
    plt.plot(prediction_dates, prediction_data[:,0], linewidth = 1.5, color = (0, 0.5, 0))
    if len(Y_p) != 0:
        plt.plot(prediction_dates, prediction_data[:,3], linewidth = 1.5, linestyle='--', color = (0.1, 0.3, 0.8))
    plt.plot(sample_dates, sample_data, linewidth = 1.5, color = (0.1, 0.3, 0.8))
    # set graphic limits, background and font size for ticks
    plt.xlim(plot_dates[0], plot_dates[-1])
    plt.ylim(min_YLim, max_YLim)
    plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
    plt.grid(True, color = (0.8, 0.8, 0.8))
    plt.title('Conditional forecasts: ' + name, fontdict = {'fontsize' : 14, \
              'fontname': 'Serif', 'fontweight': 'semibold'}) 
    plt.tick_params(axis = 'x', direction = 'in', labelsize = 12, labelrotation=20)    
    return fig


def var_conditional_forecasts_all(actual, forecasts, Y_p, dates, forecast_dates, endogenous, n):

    """
    var_conditional_forecasts_all(actual, forecasts, dates, forecast_dates, endogenous, n)
    produces conditional forecast figure for var model, all variables
    
    parameters:     
    actual: numpy ndarray of size (T,n)
        actual sample values
    forecasts: numpy ndarray of size (f_periods,n,3)
        forecast values, median, lower and upper bounds
    Y_p: numpy ndarray of size (f_periods,n)
        actual out-of-sample values
    dates: datetime index of size (T)
        index of in-sample dates
    forecast_dates: datetime index of size (f_periods)
        index of in-sample dates            
    endogenous: str list
        list of endogenous variables for which figure is produced
    n: int
        number of endogenous variables
        
    returns:
    fig: matplotlib figure
        forecast figure
    """      

    # get plot dimensions
    columns = int(np.ceil(n ** 0.5))
    rows = columns
    # periods to plot
    T = max(10, 2 * forecasts.shape[0])   
    # create figure    
    fig = plt.figure(figsize = (6.60*columns, 4.70*rows), dpi = 500, tight_layout = True)
    # loop over variables
    for i in range(n):
        sample_data = actual[-T:,[i]]
        sample_dates = dates[-T:]
        if len(Y_p) == 0:
            plot_data = np.vstack([np.tile(sample_data,[1,3]),forecasts[:,i,:]])
        else:
            plot_data = np.vstack([np.tile(sample_data,[1,4]),np.hstack([forecasts[:,i,:],Y_p[:,[i]]])])        
        # plot_data = np.vstack([np.tile(sample_data,[1,3]),forecasts[:,i,:]])
        plot_dates = np.hstack([sample_dates,forecast_dates])
        prediction_data = plot_data[T-1:]
        prediction_dates = plot_dates[T-1:]         
        # initiate subplot
        plt.subplot(rows, columns, i+1)        
        # get min and max for subplot
        min_YLim, max_YLim = set_min_and_max(plot_data, 0.07, 0.1)
        # plot actual, and steady-state with patched credibility intervals
        patch_dates = np.hstack((prediction_dates, np.flipud(prediction_dates)))
        patch_data = np.hstack((prediction_data[:,1], np.flipud(prediction_data[:,2])))
        plt.fill(patch_dates, patch_data, facecolor = (0.3, 0.7, 0.2), alpha = 0.4)
        plt.plot(prediction_dates, prediction_data[:,0], linewidth = 1.5, color = (0, 0.5, 0))
        if len(Y_p) != 0:
            plt.plot(prediction_dates, prediction_data[:,3], linewidth = 1.5, linestyle='--', color = (0.1, 0.3, 0.8))        
        plt.plot(sample_dates, sample_data, linewidth = 1.5, color = (0.1, 0.3, 0.8))     
        # set graphic limits, background and font size for ticks
        plt.xlim(plot_dates[0], plot_dates[-1])
        plt.ylim(min_YLim, max_YLim)
        plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
        plt.grid(True, color = (0.8, 0.8, 0.8))
        name = endogenous[i]
        plt.title('Conditional forecasts: ' + name, fontdict = {'fontsize' : 14, \
                  'fontname': 'Serif', 'fontweight': 'semibold'}) 
        plt.tick_params(axis = 'x', direction = 'in', labelsize = 12, labelrotation=20)   
    return fig


def var_irf_single_variable(irf, name, shock):
    
    """
    var_irf_single_variable(irf, variable, shock)
    produces IRF figure for var model, single variable
    
    parameters:     
    irf: numpy ndarray of size (irf_periods,n,3)
        IRF values, median, lower and upper bounds
    name: str
        name of variable for which figure is produced
    shock: str
        name of shock for which figure is produced        
        
    returns:
    fig: matplotlib figure
        IRF figure
    """       
    
    # periods to plot
    irf_periods = np.arange(1,irf.shape[0]+1)
    # create figure    
    fig = plt.figure(figsize = (6.60, 4.70), dpi = 500, tight_layout = True)
    min_YLim, max_YLim = set_min_and_max(irf, 0.07, 0.1)
    # plot irf with patched credibility intervals
    patch_periods = np.hstack((irf_periods, np.flipud(irf_periods)))
    patch_data = np.hstack((irf[:,1], np.flipud(irf[:,2])))
    plt.fill(patch_periods, patch_data, facecolor = (0.3, 0.7, 0.2), alpha = 0.4)
    plt.plot(irf_periods, irf[:,0], linewidth = 1.5, color = (0, 0.5, 0))
    if min_YLim < 0 and max_YLim > 0:
        plt.plot(irf_periods, np.zeros(len(irf_periods)), linewidth = 1, color = (0, 0, 0), linestyle='--', dashes=(5, 5))
    # set graphic limits, background and font size for ticks
    plt.xlim(1, irf_periods[-1])
    plt.ylim(min_YLim, max_YLim)
    plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
    plt.grid(True, color = (0.8, 0.8, 0.8))
    plt.title('IRF: ' + name + '_' + shock, fontdict = {'fontsize' : 14, \
              'fontname': 'Serif', 'fontweight': 'semibold'}) 
    plt.tick_params(axis = 'x', direction = 'in', labelsize = 12)    
    return fig


def var_irf_all(irf, variables, shocks, n_endo, n_shocks):
    
    """
    var_irf_all(irf, variables, shocks, n_endo, n_shocks)
    produces forecast figure for var model, all variables
    
    parameters:     
    irf: numpy ndarray of size (n,n,irf_periods,3)
        IRF values, median, lower and upper bounds
    variables: str list
        list of endogenous variables for which figure is produced
    shocks: str list
        list of structural shocks for which figure is produced    
    n_endo: int
        number of endogenous variables
    n_exo: int
        number of structural shocks and exogenous variables    
        
    returns:
    fig: matplotlib figure
        IRF figure
    """    

    # periods to plot
    irf_periods = np.arange(1,irf.shape[2]+1)
    # create figure    
    fig = plt.figure(figsize = (6.60*n_shocks, 4.70*n_shocks), dpi = 500, tight_layout = True)
    # loop over variables and shocks
    k = 0
    for i in range(n_endo):
        for j in range(n_shocks):
            # initiate subplot
            k += 1
            plt.subplot(n_shocks, n_shocks, k)        
            # get min and max for subplot
            min_YLim, max_YLim = set_min_and_max(irf[i,j,:,:], 0.07, 0.1)
            # plot IRF with patched credibility intervals
            patch_dates = np.hstack((irf_periods, np.flipud(irf_periods)))
            patch_data = np.hstack((irf[i,j,:,1], np.flipud(irf[i,j,:,2])))
            plt.fill(patch_dates, patch_data, facecolor = (0.3, 0.7, 0.2), alpha = 0.4)
            if min_YLim < 0 and max_YLim > 0:
                plt.plot(irf_periods, np.zeros(len(irf_periods)), linewidth = 1, color = (0, 0, 0), linestyle='--', dashes=(5, 5)) 
            plt.plot(irf_periods, irf[i,j,:,0], linewidth = 1.5, color = (0, 0.5, 0))   
            # set graphic limits, background and font size for ticks
            plt.xlim(1, irf_periods[-1])
            plt.ylim(min_YLim, max_YLim)
            plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
            plt.grid(True, color = (0.8, 0.8, 0.8))
            plt.title('IRF: ' + variables[i] + '_' + shocks[j], fontdict = {'fontsize' : 14, \
                'fontname': 'Serif', 'fontweight': 'semibold'}) 
            plt.tick_params(axis = 'x', direction = 'in', labelsize = 12)  
    return fig


def var_fevd_single_variable(fevd, name, shock):
    
    """
    var_fevd_single_variable(fevd, name, shock)
    produces FEVD figure for var model, single variable
    
    parameters:     
    fevd: numpy ndarray of size (fevd_periods,3)
        fevd values, median, lower and upper bounds
    name: str
        name of variable for which figure is produced
    shock: str
        name of shock for which figure is produced        
        
    returns:
    fig: matplotlib figure
        FEVD figure
    """       
    
    # periods to plot
    fevd_periods = np.arange(1,fevd.shape[0]+1)
    # create figure    
    fig = plt.figure(figsize = (6.60, 4.70), dpi = 500, tight_layout = True)
    for period in fevd_periods:
        # create custom bar plot
        left_x_edge = period - 0.4
        right_x_edge = period + 0.4
        median = fevd[period-1,0]
        lower = fevd[period-1,1]
        upper = fevd[period-1,2]
        patch_periods = np.hstack([left_x_edge, right_x_edge, right_x_edge, left_x_edge])
        patch_data = np.hstack([upper, upper, lower, lower])
        plt.fill(patch_periods, patch_data, facecolor = (0.3, 0.7, 0.2), alpha = 0.4)
        plt.plot([left_x_edge, right_x_edge], [median, median], linewidth = 2, color = (0, 0.5, 0))
        plt.plot([left_x_edge, left_x_edge], [lower, upper], linewidth = 0.5, color = (0, 0, 0))
        plt.plot([right_x_edge, right_x_edge], [lower, upper], linewidth = 0.5, color = (0, 0, 0))        
        plt.plot([left_x_edge, right_x_edge], [upper, upper], linewidth = 0.5, color = (0, 0, 0))
        plt.plot([left_x_edge, right_x_edge], [lower, lower], linewidth = 0.5, color = (0, 0, 0))
    # set graphic limits, background and font size for ticks
    plt.xlim(0, fevd_periods[-1]+1)
    plt.ylim(0, 1)
    plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
    plt.grid(True, color = (0.8, 0.8, 0.8))        
    plt.title('FEVD: ' + name + '_' + shock, fontdict = {'fontsize' : 14, \
              'fontname': 'Serif', 'fontweight': 'semibold'}) 
    plt.tick_params(axis = 'x', direction = 'in', labelsize = 12) 
    return fig     
    

def var_fevd_joint(fevd, name, shocks, n):
    
    """
    var_fevd_joint(fevd, name, shocks, n)
    produces FEVD figure for var model, single variable to all shocks
    
    parameters:     
    fevd: numpy ndarray of size (fevd_periods,n)
        fevd values for all shocks
    name: str
        name of variable for which figure is produced
    shocks: str
        name of shocks for which figure is produced        
    n: int
        number of endogenous variables
        
    returns:
    fig: matplotlib figure
        FEVD figure
    """      
    
    # periods to plot
    fevd_periods = np.arange(1,fevd.shape[0]+1)
    cum_fevd = np.cumsum(fevd,1)
    # create figure    
    fig = plt.figure(figsize = (6.60, 4.70), dpi = 500, tight_layout = True)
    colors = make_colors(n)
    for period in fevd_periods:
        # create custom bar plot
        left_x_edge = period - 0.4
        right_x_edge = period + 0.4
        patch_periods = np.hstack([left_x_edge, right_x_edge, right_x_edge, left_x_edge])
        for i in range(n):
            if i == 0:
                lower = 0
            else:
                lower = cum_fevd[period-1,i-1]
            upper = cum_fevd[period-1,i]
            patch_data = np.hstack([upper, upper, lower, lower])
            if period == 1 and i < 16:
                plt.fill(patch_periods, patch_data, facecolor = colors[i], alpha = 0.4, label=shocks[i])
            else:
                plt.fill(patch_periods, patch_data, facecolor = colors[i], alpha = 0.4)
            plt.plot([left_x_edge, right_x_edge], [upper, upper], linewidth = 0.8, color = (0, 0, 0))
        plt.plot([left_x_edge, left_x_edge], [0, 1], linewidth = 0.8, color = (0, 0, 0))
        plt.plot([right_x_edge, right_x_edge], [0, 1], linewidth = 0.8, color = (0, 0, 0))
    # set graphic limits, background and font size for ticks
    plt.xlim(0, fevd_periods[-1]+1)
    plt.ylim(0, 1)
    plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
    plt.grid(True, color = (0.8, 0.8, 0.8))        
    plt.title('FEVD: ' + name, fontdict = {'fontsize' : 14, \
              'fontname': 'Serif', 'fontweight': 'semibold'}) 
    plt.tick_params(axis = 'x', direction = 'in', labelsize = 12)     
    plt.legend(loc='upper center', bbox_to_anchor = (0.5, -0.05), ncol = 5, edgecolor = (0, 0, 0))
    return fig


def var_fevd_all(fevd, variables, shocks, n):
    
    """
    var_fevd_joint(fevd, name, shocks, n)
    produces FEVD figure for var model, all variables to all shocks
    
    parameters:     
    fevd: numpy ndarray of size (n,n,fevd_periods,3)
        fevd values for all shocks
    variables: str list
        name of variables for which figure is produced
    shocks: str
        name of shocks for which figure is produced        
    n: int
        number of endogenous variables
        
    returns:
    fig: matplotlib figure
        FEVD figure
    """      

    # periods to plot
    fevd_periods = np.arange(1,fevd.shape[2]+1)
    # get plot dimensions
    columns = int(np.ceil(n ** 0.5))
    rows = columns
    # create figure    
    fig = plt.figure(figsize = (6.60*columns, 4.70*rows), dpi = 500, tight_layout = True)
    colors = make_colors(n)
    # loop over variables
    for i in range(n):
        # initiate subplot
        plt.subplot(rows, columns, i+1)   
        # recover FEVD for this variable
        cum_fevd = np.cumsum(fevd[i,:,:,0].T,1)
        for period in fevd_periods:
            # create custom bar plot
            left_x_edge = period - 0.4
            right_x_edge = period + 0.4
            patch_periods = np.hstack([left_x_edge, right_x_edge, right_x_edge, left_x_edge])
            for j in range(n):
                if j == 0:
                    lower = 0
                else:
                    lower = cum_fevd[period-1,j-1]
                upper = cum_fevd[period-1,j]
                patch_data = np.hstack([upper, upper, lower, lower])
                if period == 1 and j < 16:
                    plt.fill(patch_periods, patch_data, facecolor = colors[j], alpha = 0.4, label=shocks[j])
                else:
                    plt.fill(patch_periods, patch_data, facecolor = colors[j], alpha = 0.4)
                plt.plot([left_x_edge, right_x_edge], [upper, upper], linewidth = 0.8, color = (0, 0, 0))
            plt.plot([left_x_edge, left_x_edge], [0, 1], linewidth = 0.8, color = (0, 0, 0))
            plt.plot([right_x_edge, right_x_edge], [0, 1], linewidth = 0.8, color = (0, 0, 0))
        # set graphic limits, background and font size for ticks
        plt.xlim(0, fevd_periods[-1]+1)
        plt.ylim(0, 1)
        plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
        plt.grid(True, color = (0.8, 0.8, 0.8))        
        plt.title('FEVD: ' + variables[i], fontdict = {'fontsize' : 14, \
                  'fontname': 'Serif', 'fontweight': 'semibold'}) 
        plt.tick_params(axis = 'x', direction = 'in', labelsize = 12)     
        plt.legend(loc='upper center', bbox_to_anchor = (0.5, -0.05), ncol = 5, edgecolor = (0, 0, 0))
    return fig            
        

def var_hd_single_variable(hd, name, shock, dates):
    
    """
    var_hd_single_variable(fevd, name, shock)
    produces HD figure for var model, single variable
    
    parameters:     
    hd: numpy ndarray of size (T,3)
        hd values, median, lower and upper bounds
    name: str
        name of variable for which figure is produced
    shock: str
        name of shock for which figure is produced        
    dates: datetime index of size (T)
        index of in-sample dates
        
    returns:
    fig: matplotlib figure
        HD figure
    """       

    # create figure    
    fig = plt.figure(figsize = (6.60, 4.70), dpi = 500, tight_layout = True)
    min_YLim, max_YLim = set_min_and_max(hd, 0.07, 0.1)
    # plot actual, and fitted with patched credibility intervals
    patch_dates = np.hstack((dates, np.flipud(dates)))
    patch_data = np.hstack((hd[:,1], np.flipud(hd[:,2])))
    plt.fill(patch_dates, patch_data, facecolor = (0.3, 0.7, 0.2), alpha = 0.4)
    plt.plot(dates, np.zeros(len(dates)), linewidth = 1, color = (0, 0, 0), linestyle='--', dashes=(5, 5))
    plt.plot(dates, hd[:,0], linewidth = 1.5, color = (0, 0.5, 0))
    # set graphic limits, background and font size for ticks
    plt.xlim(dates[0], dates[-1])
    plt.ylim(min_YLim, max_YLim)
    plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
    plt.grid(True, color = (0.8, 0.8, 0.8))
    plt.title('HD: ' + name + '_' + shock, fontdict = {'fontsize' : 14, \
              'fontname': 'Serif', 'fontweight': 'semibold'}) 
    plt.tick_params(axis = 'x', direction = 'in', labelsize = 12, labelrotation=20)   
    return fig
    

def var_hd_joint(hd, name, shocks, dates, n, T):
    
    """
    var_hd_joint(hd, name, shocks, dates, n, T)
    produces HD figure for var model, single variable
    
    parameters:     
    hd: numpy ndarray of size (T,n)
        hd valuesfor all shocks
    name: str
        name of variable for which figure is produced
    shocks: str list
        name of shocks for which figure is produced         
    dates: datetime index of size (T)
        index of in-sample dates
    n: int
        number of endogenous variables
    T: int
        number of sample periods
        
    returns:
    fig: matplotlib figure
        HD figure
    """     
    
    # create figure    
    fig = plt.figure(figsize = (6.60, 4.70), dpi = 500, tight_layout = True)    
    patch_dates = np.hstack((dates, np.flipud(dates)))
    colors = make_colors(n)   
    # positive contributions
    positive_hd = hd.copy()
    positive_hd[positive_hd<0] = 0
    cum_positive_hd = np.cumsum(positive_hd,1)
    cum_positive_hd = np.hstack([np.zeros((T,1)),cum_positive_hd])    
    for i in range(n):
        patch_data = np.hstack((cum_positive_hd[:,i+1], np.flipud(cum_positive_hd[:,i])))
        if i < 15:
            plt.fill(patch_dates, patch_data, facecolor = colors[i], alpha = 0.4, label=shocks[i])
        else:
            plt.fill(patch_dates, patch_data, facecolor = colors[i], alpha = 0.4)
        plt.plot(dates, cum_positive_hd[:,i+1], linewidth = 0.2, color = (0, 0, 0))
    # negative contributions
    negative_hd = hd.copy()
    negative_hd[negative_hd>0] = 0
    cum_negative_hd = np.cumsum(negative_hd,1)
    cum_negative_hd = np.hstack([np.zeros((T,1)),cum_negative_hd])    
    for i in range(n):
        patch_data = np.hstack((cum_negative_hd[:,i+1], np.flipud(cum_negative_hd[:,i])))
        plt.fill(patch_dates, patch_data, facecolor = colors[i], alpha = 0.4)
        plt.plot(dates, cum_negative_hd[:,i+1], linewidth = 0.2, color = (0, 0, 0))        
    # trend
    cum_hd = np.cumsum(hd,1)
    plt.plot(dates,cum_hd[:,-1], linewidth = 1.5, color = (0, 0, 0), label='trend')        
    # set graphic limits, background and font size for ticks
    plt.xlim(dates[0], dates[-1])
    min_YLim, max_YLim = set_min_and_max(np.hstack([cum_positive_hd,cum_negative_hd]), 0.07, 0.1)
    plt.ylim(min_YLim, max_YLim)
    plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
    plt.grid(True, color = (0.8, 0.8, 0.8))        
    plt.title('HD: ' + name, fontdict = {'fontsize' : 14, 'fontname': 'Serif', 'fontweight': 'semibold'}) 
    plt.tick_params(axis = 'x', direction = 'in', labelsize = 12)                
    plt.legend(loc='upper center', bbox_to_anchor = (0.5, -0.05), ncol = 5, edgecolor = (0, 0, 0))
    return fig  

    
def var_hd_all(hd, variables, shocks, dates, n, T):
    
    """
    var_hd_all(hd, variables, shocks, dates, n, T)
    produces HD figure for var model, all variables to all shocks
    
    parameters:     
    hd: numpy ndarray of size (n,n,T,3)
        hd values for all shocks
    variables: str list
        name of variables for which figure is produced
    shocks: str list
        name of shocks for which figure is produced         
    dates: datetime index of size (T)
        index of in-sample dates
    n: int
        number of endogenous variables
    T: int
        number of sample periods
        
    returns:
    fig: matplotlib figure
        HD figure
    """      

    # get plot dimensions
    columns = int(np.ceil(n ** 0.5))
    rows = columns
    # create figure    
    fig = plt.figure(figsize = (6.60*columns, 4.70*rows), dpi = 500, tight_layout = True)
    patch_dates = np.hstack((dates, np.flipud(dates)))
    colors = make_colors(n)   
    # loop over variables
    for i in range(n):
        # initiate subplot
        plt.subplot(rows, columns, i+1)  
        # recover HD for this variable
        variable_hd = hd[i,:,:,0].T
        # positive contributions
        positive_hd = variable_hd.copy()
        positive_hd[positive_hd<0] = 0
        cum_positive_hd = np.cumsum(positive_hd,1)
        cum_positive_hd = np.hstack([np.zeros((T,1)),cum_positive_hd])    
        for j in range(n):
            patch_data = np.hstack((cum_positive_hd[:,j+1], np.flipud(cum_positive_hd[:,j])))
            if j < 15:
                plt.fill(patch_dates, patch_data, facecolor = colors[j], alpha = 0.4, label=shocks[j])
            else:
                plt.fill(patch_dates, patch_data, facecolor = colors[j], alpha = 0.4)
            plt.plot(dates, cum_positive_hd[:,j+1], linewidth = 0.2, color = (0, 0, 0))
        # negative contributions
        negative_hd = variable_hd.copy()
        negative_hd[negative_hd>0] = 0
        cum_negative_hd = np.cumsum(negative_hd,1)
        cum_negative_hd = np.hstack([np.zeros((T,1)),cum_negative_hd])    
        for j in range(n):
            patch_data = np.hstack((cum_negative_hd[:,j+1], np.flipud(cum_negative_hd[:,j])))
            plt.fill(patch_dates, patch_data, facecolor = colors[j], alpha = 0.4)
            plt.plot(dates, cum_negative_hd[:,j+1], linewidth = 0.2, color = (0, 0, 0))        
        # trend
        cum_hd = np.cumsum(variable_hd,1)
        plt.plot(dates,cum_hd[:,-1], linewidth = 1.5, color = (0, 0, 0), label='trend')           
        # set graphic limits, background and font size for ticks
        plt.xlim(dates[0], dates[-1])
        min_YLim, max_YLim = set_min_and_max(np.hstack([cum_positive_hd,cum_negative_hd]), 0.07, 0.1)
        plt.ylim(min_YLim, max_YLim)
        plt.gca().set_facecolor(color = (0.9, 0.9, 0.9)) 
        plt.grid(True, color = (0.8, 0.8, 0.8))        
        plt.title('HD: ' + variables[i], fontdict = {'fontsize' : 14, 'fontname': 'Serif', 'fontweight': 'semibold'}) 
        plt.tick_params(axis = 'x', direction = 'in', labelsize = 12)                
        plt.legend(loc='upper center', bbox_to_anchor = (0.5, -0.05), ncol = 5, edgecolor = (0, 0, 0))
    return fig

    
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
    
    
def show_and_save(current_figure, show, save, path, file_name):
    
    """
    show_and_save(current_figure, show, save, path, file_name)
    display and save a given figure, if requested
    
    parameters:
    current_figure : matplotlib figure
        figure to display and save
    show : bool
        if True, display the figure
    save : bool
        if True, save figure as image
    path : str
        path to folder where figure is saved
    file_name : str
        name of file to save the figure
            
    returns:
    none
    """
    
    # save in folder as image if requested
    if save:
        full_path = join(path, file_name)
        plt.savefig(full_path)    
    # display figure if requested
    if show:
        plt.show()
    plt.close(current_figure)
    

def make_colors(n):
    
    """
    make_colors(n)
    return an array of n color triplets, in a determined order
    
    parameters:
    n : int
        number of color triplets to return
            
    returns:
    colors: ndarray of size(n,3)
        array of column triplets
    """    
    
    # define fixed colors
    fixed_colors = np.array([
        [0.196, 0.804, 0.196],  # 'limegreen'
        [1.000, 0.647, 0.000],  # 'orange'
        [0.000, 1.000, 1.000],  # 'cyan'
        [1.000, 0.000, 0.000],  # 'red'
        [0.580, 0.000, 0.827],  # 'darkviolet'
        [0.647, 0.165, 0.165],  # 'brown'        
        [1.000, 0.714, 0.757],  # 'lightpink' 
        [0.753, 0.753, 0.753],  # 'silver'    
        [0.502, 0.502, 0.000],  # 'olive'    
        [0.000, 0.000, 1.000],  # 'blue'   
        [0.804, 0.521, 0.247],  # 'peru'   
        [0.541, 0.169, 0.886],  # 'blueviolet'   
        [1.000, 1.000, 0.000],  # 'yellow' 
        [0.498, 1.000, 0.000],  # 'chartreuse' 
        [0.863, 0.078, 0.235],  # 'crimson' 
        [0.981, 0.686, 0.650],  #  random colors from here on
        [0.688, 0.389, 0.135],
        [0.721, 0.525, 0.310],
        [0.486, 0.889, 0.934],
        [0.358, 0.572, 0.322],
        [0.594, 0.338, 0.392],
        [0.890, 0.227, 0.623],
        [0.084, 0.833, 0.787],
        [0.239, 0.876, 0.059],
        [0.336, 0.150, 0.450],
        [0.796, 0.231, 0.052],
        [0.405, 0.199, 0.091],
        [0.580, 0.299, 0.672],
        [0.200, 0.942, 0.365],
        [0.105, 0.629, 0.927],
        [0.440, 0.955, 0.500],
        [0.425, 0.620, 0.995],
        [0.949, 0.460, 0.758],
        [0.497, 0.529, 0.786],
        [0.415, 0.734, 0.711],
        [0.932, 0.115, 0.729],
        [0.927, 0.968, 0.015],
        [0.864, 0.981, 0.957],
        [0.149, 0.973, 0.890],
        [0.822, 0.480, 0.232],
        [0.802, 0.924, 0.266],
        [0.539, 0.443, 0.931],
        [0.041, 0.732, 0.614],
        [0.028, 0.719, 0.016],
        [0.758, 0.513, 0.929],
        [0.066, 0.841, 0.067],
        [0.344, 0.430, 0.966],
        [0.562, 0.259, 0.242],
        [0.888, 0.226, 0.125],
        [0.288, 0.586, 0.554],
        [0.810, 0.560, 0.288],
        [0.413, 0.818, 0.627],
        [0.959, 0.369, 0.553],
        [0.594, 0.848, 0.145],
        [0.407, 0.910, 0.043],
        [0.823, 0.415, 0.830],
        [0.010, 0.365, 0.079],
        [0.653, 0.274, 0.703],
        [0.944, 0.127, 0.865],
        [0.059, 0.381, 0.430],
        [0.489, 0.976, 0.776],
        [0.309, 0.270, 0.863],
        [0.881, 0.511, 0.344],
        [0.995, 0.316, 0.183],
        [0.880, 0.812, 0.668],
        [0.958, 0.926, 0.748],
        [0.861, 0.247, 0.141],
        [0.670, 0.715, 0.167],
        [0.396, 0.910, 0.561],
        [0.578, 0.194, 0.526],
        [0.523, 0.089, 0.982],
        [0.571, 0.006, 0.773],
        [0.978, 0.590, 0.320],
        [0.188, 0.673, 0.195],
        [0.578, 0.602, 0.962],
        [0.072, 0.500, 0.744],
        [0.177, 0.388, 0.063],
        [0.726, 0.088, 0.395],
        [0.874, 0.472, 0.913],
        [0.766, 0.915, 0.127],
        [0.074, 0.070, 0.869],
        [0.634, 0.497, 0.164],
        [0.674, 0.318, 0.711],
        [0.460, 0.507, 0.790],
        [0.093, 0.579, 0.197],
        [0.808, 0.489, 0.989],
        [0.183, 0.963, 0.801],
        [0.481, 0.814, 0.603],
        [0.655, 0.914, 0.065],
        [0.835, 0.382, 0.326],
        [0.994, 0.781, 0.486],
        [0.423, 0.878, 0.087],
        [0.708, 0.789, 0.799],
        [0.322, 0.797, 0.225],
        [0.362, 0.417, 0.541],
        [0.113, 0.407, 0.000],
        [0.744, 0.852, 0.139],
        [0.704, 0.821, 0.982],
        [0.844, 0.424, 0.980],
        [0.974, 0.504, 0.753]       
        ]) 
    if n > 100:
        rng = np.random.default_rng(0)
        random_colors = rng.uniform(size=(n-100,3))
        colors = np.vstack([fixed_colors, random_colors])
    else:
        colors = fixed_colors[:n]
    return colors


