# imports
import numpy as np
import numpy.random as nrd
import scipy.linalg as sla
import scipy.stats as sst
import scipy.special as ssp
import alexandria.math.linear_algebra as la

    
# module stat_utilities
# a module containing methods for statistics utilities


#---------------------------------------------------
# Methods
#---------------------------------------------------

        
def normal_pdf(x, mu, sigma):
    
    """
    normal_pdf(x)
    log-pdf and pdf for the normal distribution
    
    parameters:
    x : float
        x value at which the pdf must be calculated
    mu : float
        mean parameter
    sigma : float
        variance parameter (positive)

    returns:
    log_val : float
        log-density of normal distribution at x
    val : float
        density of normal distribution at x
    """
    
    log_val = - 0.5 * (np.log(2 * np.pi) + np.log(sigma) + (x - mu)**2 / sigma)
    val = np.exp(log_val)
    return log_val, val

         
def normal_cdf(x, mu, sigma):
    
    """
    normal_cdf(x, mu, sigma)
    cumulative distribution function for the normal distribution
    
    parameters:
    x : float
        x value at which the cdf must be calculated
    mu : float
        mean parameter
    sigma : float
        variance parameter (positive)

    returns:
    val : float
        cdf of normal distribution at x
    """
    
    val = 0.5 * (1 + ssp.erf((x - mu) / np.sqrt(2 * sigma)));
    return val

        
def normal_icdf(p):
    
    """
    normal_icdf(p)
    inverse cdf for the normal distribution
    
    parameters:
    p : float
        probability of the cdf (between 0 and 1)

    returns:
    val : float
        inverse cdf value
    """
    
    val = ssp.ndtri(p)
    return val

       
def student_pdf(x, mu, sigma, v):
    
    """
    student_pdf(x, mu, sigma, v)
    log-pdf and pdf for the student distribution
    
    parameters:
    x : float
        x value at which the pdf must be calculated
    mu : float
        location parameter
    sigma : float
        scale parameter (positive)
    v : float
        degrees of freedom (positive)

    returns:
    log_val : float
        log-density of student distribution at x
    val : float
        density of student distribution at x
    """
    
    term_1 = ssp.loggamma((v + 1) / 2) - ssp.loggamma(v / 2)
    term_2 = - 0.5 * (np.log(v) + np.log(np.pi) + np.log(sigma))
    term_3 = - 0.5 * (v + 1) * np.log(1 + (x - mu)**2 / (v * sigma))
    log_val = term_1 +  term_2 + term_3
    val = np.exp(log_val)
    return log_val, val

        
def student_cdf(x, mu, sigma, v):
    
    """
    student_cdf(x, mu, sigma, v)
    cumulative distribution function for the student distribution
    
    parameters:
    x : float
        x value at which the cdf must be calculated
    mu : float
        location parameter
    sigma : float
        scale parameter (positive)
    v : float
        degrees of freedom (positive)

    returns:
    val : float
        cdf value
    """
    
    val = sst.t.cdf(x, v, mu, sigma)
    return val

        
def student_icdf(p, v):
    
    """
    student_icdf(p, v)
    inverse cdf for the Student distribution
    
    parameters:
    p : float
        probability of the cdf (between 0 and 1)
    v : float
        degrees of freedom (positive)

    returns:
    val : float
        inverse cdf value
    """
    
    val = sst.t.ppf(p, v)
    return val



