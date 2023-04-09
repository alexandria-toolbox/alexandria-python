# imports
import numpy as np
import numpy.random as nrd
import numpy.linalg as nla
import scipy.linalg as sla
import random as rd
import math as m
import alexandria.math.linear_algebra as la


# module random_number_generators
# a module containing methods for random number generation


#---------------------------------------------------
# Methods
#---------------------------------------------------

  
def normal(mu, sigma):
    
    """
    normal(mu, sigma)
    random number generator for the normal distribution
    
    parameters:
    mu : float
        mean
    sigma : float
        variance (positive)

    returns:
    x : float 
        pseudo-random number from the normal distribution
    """
    
    x = rd.gauss(mu, m.sqrt(sigma))
    return x

  
def multivariate_normal(mu, Sigma):
    
    """
    multivariate_normal(mu, Sigma)
    random number generator for the multivariate normal distribution
    based on algorithm d.13
    
    parameters:
    mu : ndarray of shape (n,)
        mean
    Sigma : ndarray of shape (n,n)
        variance-covariance (symmetric positive definite)

    returns:
    x : ndarray of shape (n,)
        pseudo-random number from the multivariate normal distribution
    """
    
    x = mu + la.cholesky_nspd(Sigma) @ nrd.randn(Sigma.shape[0])
    return x


def efficient_multivariate_normal(m, inv_Sigma):
    
    """
    efficient_multivariate_normal(m, inv_Sigma)
    efficient random number generator for the multivariate normal distribution, using algorithm 9.4
    
    parameters:
    m: ndarray of shape (n,)
        partial term for the distribution mean        
    inv_Sigma: ndarray of shape (n,n)
        inverse of variance matrix of the distribution

    returns:
    x: ndarray of shape (n,)
        pseudo-random vector from the multivariate normal distribution
    """
    
    G = la.cholesky_nspd(inv_Sigma)
    zeta = nrd.randn(G.shape[0])
    temp = sla.solve_triangular(G, m, lower = True, check_finite = False) + zeta
    x = sla.solve_triangular(G.T, temp, check_finite = False)
    return x

   
def matrix_normal(M, Sigma, Omega):
    
    """
    matrix_normal(M, Sigma, Omega)
    random number generator for the matrix normal distribution
    based on algorithm d.15
    
    parameters:
    M : ndarray of shape (n,m)
        location
    Sigma : ndarray of shape (n,n)
        row scale (symmetric positive definite)
    Omega : ndarray of shape (m,m)
    
    returns:
    X : ndarray of shape (n,m)
        pseudo-random number from the matrix normal distribution
    """
    
    X = M + la.cholesky_nspd(Sigma) @ nrd.randn(M.shape[0], M.shape[1]) \
        @ la.cholesky_nspd(Omega).T
    return X

   
def gamma(a, b):
    
    """
    gamma(a, b)
    random number generator for the Gamma distribution
    based on algorithms d.25 and d.26
    
    parameters:
    a : float
        shape (positive)
    b : float
        scale (positive)
        
    returns:
    x : float
        pseudo-random number from the Gamma distribution
    """
    
    x = rd.gammavariate(a, b)
    return x
    
      
def inverse_gamma(a, b):
    
    """
    inverse_gamma(a,b)
    random number generator for the inverse Gamma distribution
    based on algorithm d.29
    
    parameters:
    a : float
        shape (positive)
    b : float
        scale (positive)
    
    returns:
    x : float
        pseudo-random number from the inverse Gamma distribution
    """
    
    x = 1 / rd.gammavariate(a, 1 / b)
    return x

     
def chi2(nu):
    
    """
    chi2(nu)
    random number generator for the chi2 distribution
    based on property in Table d.15
    
    parameters:
    nu : float
        degrees of freedom (positive)
    
    returns:
    x : float 
        pseudo-random number from the chi2 distribution
    """

    x = rd.gammavariate(0.5 * nu, 2)
    return x    

   
def vector_chi2(nu):
    
    """
    vector_chi2(nu)
    random number generator for the chi2 distribution
    based on property in Table d.15
    
    parameters:
    nu : ndarray of shape (n,)
        degrees of freedom (all positive)
    
    returns:
    x : ndarray of shape (n,)
        pseudo-random number from the chi2 distribution
    """

    x = nrd.gamma(0.5 * nu, 2)
    return x  

 
def wishart(nu, S):
    
    """
    wishart(nu, S)
    random number generator for the Wishart distribution
    based on algorithms d.27 and d.28
    
    parameters:
    nu : float
        degrees of freedom (positive, nu >= n)
    S : ndarray of shape (n,n)
        scale (symmetric positive definite)
    
    returns:
    X : ndarray of shape (n,n)
        pseudo-random number from the Wishart distribution
    """
    
    dimension = S.shape[0]
    if nu == np.floor(nu) and nu <= 80 + dimension:
        A = nrd.randn(dimension, nu)
    elif dimension < 25:
        A = np.zeros((dimension, dimension))
        for i in range(dimension):
            A[i, i] = m.sqrt(chi2(nu - i))
            A[i,:i] = nrd.randn(i)
    else:
        degree_freedom = nu - np.arange(dimension)
        A = np.diag(np.sqrt(vector_chi2(degree_freedom)))
        index = np.tril_indices(dimension, -1)
        tril_size = int(0.5 * dimension * (dimension - 1))
        A[index] = nrd.randn(tril_size)
    G = la.cholesky_nspd(S)
    Z = G @ A
    X = Z @ Z.T
    return X    

    
def inverse_wishart(nu, S):
    
    """
    inverse_wishart(nu, S)
    random number generator for the inverse Wishart distribution
    based on algorithms d.30 and d.31
    
    parameters:
    nu : float
        degrees of freedom (positive, nu >= n)
    S : ndarray of shape (n,n)
        scale (symmetric positive definite)
    
    returns:
    X : ndarray of shape (n,n)
        pseudo-random number from the inverse Wishart distribution
    """

    dimension = S.shape[0]
    G = la.cholesky_nspd(S)    
    if nu == np.floor(nu) and nu <= 80 + dimension:
        A = nrd.randn(dimension, nu)
        X = G @ nla.solve(A @ A.T, G.T)
        return X
    elif dimension < 25:
        A = np.zeros((dimension, dimension))
        for i in range(dimension):
            A[i, i] = m.sqrt(chi2(nu - i))
            A[i, :i] = nrd.randn(i)
    else:
        degree_freedom = nu - np.arange(dimension)
        A = np.diag(np.sqrt(vector_chi2(degree_freedom)))
        index = np.tril_indices(dimension, -1)
        tril_size = int(0.5 * dimension * (dimension - 1))
        A[index] = nrd.randn(tril_size)
    Z = sla.solve_triangular(A, G.T, lower=True, check_finite=False).T
    X = Z @ Z.T
    return X    

    
def student(mu, sigma, nu):
    
    """
    student(mu, sigma, nu)
    random number generator for the student distribution
    based on algorithm d.17
    
    parameters:
    mu : float
        location
    sigma : float
        scale (positive)
    nu : float
        degrees of freedom
    
    returns:
    x : float
        pseudo-random number from the student distribution
    """
    
    s = inverse_gamma(0.5 * nu, 0.5 * nu)
    z = m.sqrt(s) * nrd.randn()
    x = m.sqrt(sigma) * z + mu
    return x

      
def multivariate_student(mu, Sigma, nu):
    
    """
    multivariate_student(mu, Sigma, nu)
    random number generator for the multivariate student distribution
    based on algorithm d.19
    
    parameters:
    mu : ndarray of shape (n,)
        location
    Sigma : ndarray of shape (n,n)
        scale (symmetric positive definite)
    nu : float
        degrees of freedom
    
    returns:
    x : ndarray of shape (n,)
        pseudo-random number from the multivariate student distribution
    """

    s = inverse_gamma(0.5 * nu, 0.5 * nu)
    z = m.sqrt(s) * nrd.randn(Sigma.shape[0])
    x = mu + la.cholesky_nspd(Sigma) @ z    
    return x

     
def matrix_student(M, Sigma, Omega, nu):
 
    """
    matrix_student(M, Sigma, Omega, nu)
    random number generator for the matrix student distribution
    based on algorithm d.21
    
    parameters:
    M : ndarray of shape (n,m)
        location
    Sigma : ndarray of shape (n,n)
        row scale (symmetric positive definite)
    Omega : ndarray of shape (m,m)
        column scale (symmetric positive definite)
    nu : float
        degrees of freedom
    
    returns:
    X: ndarray of shape (n,m)
        pseudo-random number from the matrix student distribution
    """
    
    rows, columns = M.shape[0], M.shape[1]
    Z = nrd.randn(rows, columns)
    if columns <= rows:
        Phi = inverse_wishart(nu + columns - 1, nu * Omega)
        G = la.cholesky_nspd(Sigma)
        H = la.cholesky_nspd(Phi)
    else:
        Phi = inverse_wishart(nu + rows - 1, nu * Sigma)
        G = la.cholesky_nspd(Phi)
        H = la.cholesky_nspd(Omega)
    X = M + G @ Z @ H.T
    return X   

    
def truncated_normal(mu, sigma, a, b):
    
    """
    truncated_normal(mu, sigma, a, b)
    random number generator for the truncated normal distribution
    based on algorithm d.23
    
    parameters:
    mu: float
        mean of untruncated distribution
    sigma : float
        variance of untruncated distribution (positive)
    a : float
        lower bound of truncation
    b : float
        upper bound of truncation (b > a)
    
    returns:
    x : float
        pseudo-random number from the truncated normal distribution
    """
    
    def standard_truncated_normal(a, b):
        while True:
            x = a + (b - a) * nrd.rand()
            if a <= 0 and b >= 0:
                w = m.exp(- x * x / 2)
            elif b < 0:
                w = m.exp((b * b - x * x) / 2)
            else:
                w = m.exp((a * a - x * x) / 2)
            u = nrd.rand()
            if u <= w:
                return  x
            
    standard_deviation = np.sqrt(sigma)
    a_bar = (a - mu) / standard_deviation
    b_bar = (b - mu) / standard_deviation
    z = standard_truncated_normal(a_bar, b_bar)
    x = mu + standard_deviation * z
    return x    
    
    
def beta(a, b):
    
    """
    beta(a, b)
    random number generator for the beta distribution
    based on algorithm d.32
    
    parameters:
    a : float
        shape (positive)
    b : float
        shape (positive)
    
    returns:
    x : float
        pseudo-random number from the beta distribution
    """
    
    x = nrd.beta(a, b)
    return x    
    
      
def dirichlet(a):
    
    """
    dirichlet(a)
    random number generator for the Dirichlet distribution
    based on algorithm d.33
    
    parameters:
    a : ndarray of shape (n,)
        concentration (all positive)
    
    returns:
    x : ndarray of shape (n,)
        pseudo-random number from the Dirichlet distribution
    """
    
    x = nrd.dirichlet(a)
    return x    

   
def bernoulli(p):
    
    """
    bernoulli(p)
    random number generator for the Bernoulli distribution
    based on algorithm d.2
    
    parameters:
    p : float
        probability of success (0 <= p <= 1)
    
    returns:
    x : int
        pseudo-random number from the Bernoulli distribution
    """
    
    x = int(nrd.rand() <= p)
    return x  


def binomial(n, p):
    
    """
    binomial(n, p)
    random number generator for the binomial distribution
    based on algorithm d.4
    
    parameters:
    n : int 
        number of trials (positive)
    p : float
        probability of success (0 <= p <= 1)
    
    returns:
    x : int
        pseudo-random number from the binomial distribution
    """
    
    x = nrd.binomial(n, p)
    return x 


def categorical(p):
    
    """
    categorical(p)
    random number generator for the categorical distribution
    based on algorithm d.3
    
    parameters:
    p : ndarray of shape (k,)
        probability of success for each category (sum = 1)
    
    returns:
    x : int
        pseudo-random number from the categorical distribution
    """

    x = np.where(nrd.rand() <= np.cumsum(p))[0][0] + 1
    return x

   
def multinomial(n, p):
    
    """
    multinomial(n, p)
    random number generator for the multinomial distribution
    based on algorithm d.5
    
    parameters:
    n : int
        number of trials (positive)
    p : float
        probability of success for each category (sum = 1)
    
    returns:
    x : ndarray of shape (k,)
        pseudo-random number from the multinomial distribution
    """
    
    x = nrd.multinomial(n, p)
    return x


def discrete_uniform(a, b):
    
    """
    discrete_uniform(a, b)
    random number generator for the discrete uniform distribution
    based on algorithm d.1
    
    parameters:
    a : int 
        lower bound
    b : int
        upper bound
    
    returns:
    x : int
        pseudo-random numbers from the discrete uniform distribution
    """
    
    x = np.floor(a + (b + 1 - a) * nrd.rand())
    return x

   
def poisson(lamda):
    
    """
    poisson(lambda)
    random number generator for the Poisson distribution
    based on algorithms d.6 and d.7
    
    parameters:
    lambda : float
        intensity (positive)
    
    returns:
    x : float
        pseudo-random number from the Poisson distribution
    """
        
    x = nrd.poisson(lamda)
    return x    



    
    
    
