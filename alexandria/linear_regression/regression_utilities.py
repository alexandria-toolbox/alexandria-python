# imports
import numpy as np
import numpy.linalg as nla


# module regression_utilities
# a module containing methods for linear regression utilities


#---------------------------------------------------
# Methods
#---------------------------------------------------


def add_intercept_and_trends(exogenous, constant, trend, quadratic_trend, shift):
    
    
    """
    add_intercept_and_trends(X, constant, trend, quadratic_trend, shift)
    add intercept and trends to regressor matrix X
    
    parameters:
    exogenous : ndarray of shape (periods,n_exogenous)
        matrix of regressors  
    constant : bool
        if True, a constant is added to the VAR model
    trend : bool
        if True, a linear trend is added to the VAR model
    quadratic_trend : bool
        if True, a quadratic trend is added to the VAR model
    shift : int
        sample period shift: 0 for in-sample data, sample periods otherwise
        
    returns:
    X : ndarray of shape (periods,n_exogenous_total)
        matrix of regressors 
    """      

    # sample periods
    X = exogenous
    periods = X.shape[0]
    # quadratic trend
    if quadratic_trend:
        quadratic_trend_column = (shift + np.arange(1,periods+1).reshape(-1,1)) ** 2
        X = np.hstack((quadratic_trend_column, X))
    # trend
    if trend:
        trend_column = shift + np.arange(1,periods+1).reshape(-1,1)
        X = np.hstack((trend_column, X))
    # intercept
    if constant:
        constant_column = np.ones((periods,1))
        X = np.hstack((constant_column, X))
    return X        


def ols_regression(y, X, XX, Xy, n):
    
    """
    ols_regression(y, X, XX, Xy, n)
    maximum likelihood estimates for beta and sigma, from (3.9.7)
    
    parameters:
    y : ndarray of shape (n,)
        endogenous (explained) variable
    X : ndarray of shape (n,k)
        exogenous (explanatory) variables    
    XX : ndarray of shape (k,k)
        regressors variance matrix     
    Xy : ndarray of shape (k,)
        regressors covariance matrix  
    n : int
        number of sample observations
        
    returns:
    beta_hat : ndarray of shape (k,)
        sample estimate for beta
    sigma_hat : float
        sample estimate for sigma
    """  

    beta_hat = nla.solve(XX, Xy)
    res = y - X @ beta_hat
    sigma_hat = res @ res / n
    return beta_hat, sigma_hat 


def generate_b(b_exogenous, b_constant, b_trend, b_quadratic_trend, constant, trend, quadratic_trend, n_exogenous):
    
    """
    generate_b(b_exogenous, b_constant, b_trend, b_quadratic_trend, constant, trend, quadratic_trend, n_exogenous)
    generates prior mean vector b
    
    parameters:
    b_exogenous : float or ndarray of shape (n_exogenous,)
        prior mean for exogenous variables
    b_constant : float
        prior mean for constant       
    b_trend : float
        prior mean for linear trend     
    b_quadratic_trend : float
        prior mean for quadratic trend
    constant : bool
        if True, a constant is added to the VAR model
    trend : bool
        if True, a linear trend is added to the VAR model
    quadratic_trend : bool
        if True, a quadratic trend is added to the VAR model
    n_exogenous : int
        number of exogenous regressors, excluding constant, trend and quadratic trend
        
    returns:
    b : ndarray of shape (n_exogenous_total,)
        prior mean for exogenous variables, comprising exogenous regressors, constant, trend and quadratic trend
    """  
    
    # if b_exogenous is a scalar, turn it into a vector replicating the value
    if isinstance(b_exogenous, (int,float)):
        b_exogenous = b_exogenous * np.ones(n_exogenous)
    b = b_exogenous
    # if quadratic trend is included, add to prior mean
    if quadratic_trend:
        b = np.hstack((b_quadratic_trend, b))
    # if trend is included, add to prior mean
    if trend:
        b = np.hstack((b_trend, b))
    # if constant is included, add to prior mean
    if constant:
        b = np.hstack((b_constant, b))
    return b
            
 
def generate_V(V_exogenous, V_constant, V_trend, V_quadratic_trend, constant, trend, quadratic_trend, n_exogenous):
    
    """
    generate_V(V_exogenous, V_constant, V_trend, V_quadratic_trend, constant, trend, quadratic_trend, n_exogenous)
    generates prior variance vector V
    
    parameters:
    V_exogenous : float or ndarray of shape (n_exogenous,)
        prior variance for exogenous variables
    V_constant : float
        prior variance for constant       
    V_trend : float
        prior variance for linear trend     
    V_quadratic_trend : float
        prior variance for quadratic trend
    constant : bool
        if True, a constant is added to the VAR model
    trend : bool
        if True, a linear trend is added to the VAR model
    quadratic_trend : bool
        if True, a quadratic trend is added to the VAR model
    n_exogenous : int
        number of exogenous regressors, excluding constant, trend and quadratic trend
        
    returns:
    V : ndarray of shape (n_exogenous_total,)
        prior variance for exogenous variables, comprising exogenous regressors, constant, trend and quadratic trend
    """  
    
    # if V_exogenous is a scalar, turn it into a vector replicating the value
    if isinstance(V_exogenous, (int,float)):
        V_exogenous = V_exogenous * np.ones(n_exogenous)
    V = V_exogenous
    # if quadratic trend is included, add to prior mean
    if quadratic_trend:
        V = np.hstack((V_quadratic_trend, V))
    # if trend is included, add to prior mean
    if trend:
        V = np.hstack((V_trend, V))
    # if constant is included, add to prior mean
    if constant:
        V = np.hstack((V_constant, V))  
    return V
        

def generate_b_and_V_arrays(b, V):
    
    """
    generate_b_and_V_arrays(b, V)
    generates arrays related to b and V
    
    parameters:
    b : ndarray of shape (k,)
        prior mean for exogenous variables
    V : ndarray of shape (k,)
        prior variance for exogenous variables
        
    returns:
    V : ndarray of shape (k,k)
        prior variance matrix for exogenous variables  
    inv_V : ndarray of shape (k,k)
        inverse of prior variance matrix for exogenous variables
    inv_V_b : ndarray of shape (k,)
        product inv(V) * b 
    """
          
    # convert the vector V into an array
    inv_V_b = b / V
    inv_V = np.diag(1/V)
    V = np.diag(V)
    return V, inv_V, inv_V_b


def fitted_and_residual(y, X, beta):
    
    """
    fitted_and_residual(y, X, beta)
    generates in-sample fitted and residuals
    
    parameters:
    y : ndarray of shape (n,)
        endogenous (explained) variable
    X : ndarray of shape (n,k)
        exogenous (explanatory) variables
    beta : ndarray of shape (k,)
        regression coefficients
        
    returns:
    fitted : ndarray of shape (n,)
        in-sample fitted values
    residual : ndarray of shape (n,)
        in-sample residual
    """    
    
    fitted = X @ beta    
    residual = y - X @ beta
    return fitted, residual


def insample_evaluation_criteria(y, res, n, k):
    
    """
    insample_evaluation_criteria(y, res, n, k)
    generates in-sample evaluation criteria
    
    parameters:
    y : ndarray of shape (n,)
        endogenous (explained) variable
    res : ndarray of shape (n,)
        in-sample residual
    n : int
        number of sample observations
    k : int
        dimension of VAR coefficients        
        
    returns:
    insample_evaluation : dict
        in-sample evaluation criteria
    """
    
    ssr = res @ res
    tss = (y - np.mean(y)) @ (y - np.mean(y))
    r2 = 1 - ssr / tss
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)
    insample_evaluation = {}
    insample_evaluation['ssr'] = ssr
    insample_evaluation['r2'] = r2
    insample_evaluation['adj_r2'] = adj_r2
    return insample_evaluation


def ml_insample_evaluation_criteria(y, res, n, k, sigma):
    
    """
    ml_insample_evaluation_criteria(y, res, n, k, sigma)
    generates in-sample evaluation criteria for maximum lilekihood regression
    
    parameters:
    y : ndarray of shape (n,)
        endogenous (explained) variable
    res : ndarray of shape (n,)
        in-sample residual
    n : int
        number of sample observations
    k : int
        dimension of VAR coefficients  
    sigma : float
        residual variance
        
    returns:
    insample_evaluation : dict
        in-sample evaluation criteria
    """
    
    ssr = res @ res
    tss = (y - np.mean(y)) @ (y - np.mean(y))
    r2 = 1 - ssr / tss
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)
    aic = 2 * k / n + np.log(sigma)
    bic = k * np.log(n) / n + np.log(sigma)
    insample_evaluation = {}
    insample_evaluation['ssr'] = ssr
    insample_evaluation['r2'] = r2
    insample_evaluation['adj_r2'] = adj_r2
    insample_evaluation['aic'] = aic
    insample_evaluation['bic'] = bic
    return insample_evaluation


def forecast_evaluation_criteria(y_hat, y):

    """
    forecast_evaluation_criteria(y_hat, y)
    forecast evaluation criteria from equations (3.10.11) and (3.10.12)
    
    parameters:
    y_hat : ndarray of shape (m,)
        array of forecast values for forecast evaluation        
    y : ndarray of shape (m,)
        array of realised values for forecast evaluation

    returns:
    forecast_evaluation_criteria : dict
        forecast evaluation criteria
    """
    
    err = y - y_hat
    m = y_hat.shape[0]
    rmse = np.sqrt(err @ err / m)
    mae = np.sum(np.abs(err)) / m
    mape = 100 * np.sum(np.abs(err / y)) / m
    theil_u = np.sqrt(err @ err) / (np.sqrt(y @ y) + np.sqrt(y_hat @ y_hat))
    bias = np.sum(err) / np.sum(np.abs(err))              
    forecast_evaluation_criteria = {}
    forecast_evaluation_criteria['rmse'] = rmse
    forecast_evaluation_criteria['mae'] = mae
    forecast_evaluation_criteria['mape'] = mape
    forecast_evaluation_criteria['theil_u'] = theil_u
    forecast_evaluation_criteria['bias'] = bias
    return forecast_evaluation_criteria

