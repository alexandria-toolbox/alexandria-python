# imports
import numpy as np
import math
from scipy.special import multigammaln


# module math_utilities
# a module containing methods for mathematics utilities


#---------------------------------------------------
# Methods
#---------------------------------------------------

  
def eps():
    
    """
    eps()
    replicates Matlab eps, producing a tiny float of roughly 2e-16
    
    parameters:
    none
    
    returns:
    epsilon : float
        tiny float
    """
    
    epsilon = 2.220446049250313e-16
    return epsilon

    
def log_sum_exp(log_x):
    
    """
    log_sum_exp(log_x)
    log-sum-exp function that computes log(sum(x)) as a function of sum(log(x))
    using equation (3.10.30)
    
    parameters:
    log_x : ndarray of shape (n,)
        vector containing log(x) values
    
    returns:
    log_sum_x : float
        log of the sum of the x's
    """
    
    log_x_bar = np.max(log_x)
    diff = log_x - log_x_bar
    log_sum_exp = log_x_bar + np.log(np.sum(np.exp(diff)))
    return log_sum_exp

     
def gamma(x):
    
    """
    gamma(x)
    Gamma function for scalars
    
    parameters:
    x : float
        argument of Gamma function
    
    returns:
    y : float
        value returned by the gamma function
    """
    
    y = math.gamma(x)
    return y    


def log_multivariate_gamma(x, n):
    
    """
    log_multivariate_gamma(x, n)
    log of multivariate Gamme function
    
    parameters:
    x : float
        argument of Gamma function
    n : int
        dimension of function
    
    returns:
    y : float
        log value returned by the multivariate gamma function
    """    
    
    y = multigammaln(x, n)
    return y
    
    
    
    
    

