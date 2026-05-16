# imports
from math import sqrt
import numpy as np
import numpy.random as nrd


# module nowcasting_utilities
# a module containing methods for nowcasting utilities


#---------------------------------------------------
# Methods
#---------------------------------------------------


def check_data_integrity(data):
    
    """
    check_data_integrity(data)
    verifies that data is valid, i.e. does not contain all-NaN rows or columns
    
    parameters:
    data : ndarray of shape (n_periods,n_exogenous)
        array of endogenous data to verify
        
    returns:
    none
    """
    
    # check for rows that are all-NaN
    all_nan_rows = np.all(np.isnan(data), axis=1).nonzero()[0]
    if len(all_nan_rows) > 0:
        row = all_nan_rows[0]
        raise Exception('Data error: row ' + str(row+1) + ' (and possibly others) of endogenous data is all-NaN')
    # check for columns that are all-NaN
    all_nan_columns = np.all(np.isnan(data), axis=0).nonzero()[0]
    if len(all_nan_columns) > 0:
        column = all_nan_columns[0]
        raise Exception('Data error: column ' + str(column+1) + ' (and possibly others) of endogenous data is all-NaN')
            
        
def standardize_data(endogenous):       
        
    """
    standardize_data(endogenous)
    standardize data, making it zero mean and unit standard deviation
    
    parameters:
    endogenous: ndarray of shape (n_periods,n_endogenous)
        array of endogenous data to process 
        
    returns:
    y: ndarray of shape (n_periods,n_endogenous)
        array of standardized endogenous data 
    c: ndarray of shape (n,)
        vector of empirical mean     
    S: ndarray of shape (n,)
        vector of empirical standard deviation        
    """    
    
    c = np.nanmean(endogenous, axis=0)
    S = np.nanstd(endogenous, axis=0)
    y = (endogenous - c) / S
    return y, c, S
        

def make_dfm_regressors(data):
    
    """
    make_dfm_regressors(data)
    get dynamic factor model regressors and dimensions
    
    parameters:
    data : ndarray of shape (n_periods,n_exogenous)
        array of endogenous data to process 
        
    returns:
    T: int
        total number of sample periods, defined in (6.18.1)
    n: int
        number of endogenous variables, defined in (6.18.1)
    t_: list of size (n)
        list of non-NaN periods for each endogenous variable, defined in (6.18.9)
    T_: list of size (n)
        list of non-NaN dimension for each endogenous variable, defined in (6.18.9)        
    x: list of size (n)
        list of non-NaN data for each endogenous variable, defined in (6.18.9)  
    x_: list of size (T)
        list of non-NaN data for each sample period, defined in (6.18.10)         
    J: list of size (T)
        list of non-NaN variables for each sample period, defined in (6.18.3)
    d: list of size (T)
        list of non-NaN dimension for each sample period, defined in (6.18.3)            
    """
    
    T = data.shape[0]
    n = data.shape[1]
    t_ = [None] * n
    T_ = [None] * n
    x = [None] * n
    J = [None] * T
    d = [None] * T
    x_ = [None] * T
    for i in range(n):
        data_i = data[:,i]
        non_nan_entries = np.logical_not(np.isnan(data_i)).nonzero()[0]
        t_[i] = non_nan_entries
        T_[i] = len(non_nan_entries)
        x[i] = data_i[non_nan_entries]
    for t in range(T):
        data_t = data[t,:]
        non_nan_entries = np.logical_not(np.isnan(data_t)).nonzero()[0]
        J[t] = non_nan_entries
        d[t] = len(non_nan_entries)
        x_[t] = data_t[non_nan_entries]
    return T, n, t_, T_, x, x_, J, d
    
    
def make_dfm_dimensions(factors, loadings_lags, factor_lags, residual_lags):
    
    """
    make_dfm_dimensions(factors, loadings_lags, factor_lags)
    generate additional dimensions for dynamic factor model
    
    parameters:
    factors : int
        number of fundamental factors in the model
    loadings_lags : int
        number of lags in the loadings equation
    factor_lags : int
        number of lags in the dynamic equation        
        
    returns:
    m: int
        number of fundamental factors in the model, defined in (6.18.1)        
    q: int
        number of lags in the loadings equation, defined in (6.18.2)        
    p: int
        number of lags in the VAR model, defined in (6.18.6)  
    r: int
        number of lags in the residual AR process, defined in (6.18.7) 
    s: int
        maximum number of factor lags, defined in (6.18.38)
    k: int
        total number of VAR regressors, defined in (6.18.15) 
    l: int
        total number of loadings regressors, defined in (6.18.11)    
    """    
    
    m = factors
    q = loadings_lags
    p = factor_lags
    r = residual_lags
    s = max(q,p)
    k = m * p
    l = m * (q + 1)
    return m, q, p, r, s, k, l


def make_lambda_prior(n, m, q, l, delta1, delta2):
    
    """ 
    make_lambda_prior(n, m, q, l, delta1, delta2)
    generate prior terms for loadings lambda

    parameters:

    n: int
        number of endogenous variables, defined in (6.18.1)        
    m: int
        number of fundamental factors in the model, defined in (6.18.1)        
    q: int
        number of lags in the loadings equation, defined in (6.18.2)           
    l: int
        total number of loadings regressors, defined in (6.18.11)         
        
    factors : int
        number of fundamental factors in the model
    loadings_lags : int
        number of lags in the loadings equation
    factor_lags : int
        number of lags in the dynamic equation        
    delta1: float
        overall tightness hyperparameter, defined in (6.18.21)     
    delta2: float
        hyperparameter for identification coefficients, defined in (6.18.23)
        
    returns:
    inv_U: list of size (n)
        list of inverse prior variances, defined in (6.18.30) 
    inv_U_h: list of size (n)
        list of inverse prior means, defined in (6.18.30)   
    """    

    # prior mean g
    h = np.zeros((n,l))
    # prior variance U
    U = np.zeros((n,m))
    # variance for Lambda0
    U[:,:m] = (2 * delta1) ** 2
    # variance for lagged values
    for j in range(1,q+1):
        temp = np.ones((n,m)) * (delta1 / j) ** 2
        U = np.hstack([U,temp])
    inv_U = [None] * n
    inv_U_h = [None] * n
    for i in range(n):
        inv_u_i = 1 / U[i,:]
        inv_U[i] = np.diag(inv_u_i)
        h_i = h[i,:]
        inv_U_h[i] = inv_u_i * h_i
    return inv_U, inv_U_h


def make_beta_prior(m, p, pi1, pi2, pi3):

    """
    make_beta_prior(q, p, pi1, pi2, pi3)
    generate prior terms for VAR coefficients beta
    
    parameters:
    m: int
        number of fundamental factors in the model, defined in (6.18.1) 
    p: int
        number of lags in the VAR model, defined in (6.18.6)  
    pi1: float
        overall tightness hyperparameter, defined in (6.18.24)     
    pi2: float
        cross-variable shrinkage hyperparameter, defined in (6.18.24)
    pi3: float
        lag decay hyperparameter, defined in (6.18.24)
        
    returns:
    inv_V: list of size (m)
        list of inverse prior variance, defined in (6.18.33)   
    """     
    
    V = (pi1 * np.kron(1 / (np.arange(p) + 1).reshape(-1,1) ** pi3, \
        pi2 * np.ones((m,m)) + (1 - pi2) * np.eye(m))).T ** 2
    inv_V = [None] * m
    for j in range(m):
        inv_v_j = 1 / V[j,:]
        inv_V[j] = np.diag(inv_v_j)
    return inv_V 


def make_gamma_prior(r, omega1):

    """
    make_gamma_prior(r, omega1)
    generate prior terms for residual AR coefficients gamma
    
    parameters:
    r: int
        number of lags in the residual AR process, defined in (6.18.7) 
    omega1: float
        overall tightness hyperparameter, defined in (6.18.26)  
        
    returns:
    inv_Q: ndarray of shape (r,r)
        inverse prior variance, defined in (6.18.36)   
    """ 
    
    q = (omega1 / np.arange(1,r+1)) ** 2
    inv_q = 1 / q
    inv_Q = np.diag(inv_q)
    return inv_Q


def make_state_regressors(beta, gamma, z, m, q, p, s, n, r, T):
    
    """
    make_state_regressors(beta, gamma, z, m, q, p, s, n, r, T)
    make factor and residual regressors for dynamic factor models
    
    parameters:
    beta: ndarray of size (k,m)
        matrix of coefficients beta_j, defined in (6.18.16)
    gamma: ndarray of size (n,r)
        matrix of residual AR coefficients gamma_i, defined in (6.18.13)        
    z: ndarray of size (T,m(s+1)+n(r+1))
        array of state variables, defined in (6.18.39)
    m: int
        number of fundamental factors in the model, defined in (6.18.1) 
    q: int
        number of lags in the loadings equation, defined in (6.18.2)          
    p: int
        number of lags in the VAR model, defined in (6.18.6)  
    s: int
        maximum number of factor lags, defined in (6.18.38)
    n: int
        number of endogenous variables, defined in (6.18.1)  
    r: int
        number of lags in the residual AR process, defined in (6.18.7) 
    T: int
        total number of sample periods, defined in (6.18.1)
        
    returns:
    W: ndarray of shape (T,m(q+1))
        factor regressors for loadings, defined in (6.18.11)   
    F: ndarray of shape (T,m)
        factor regressors for VAR, defined in (6.18.15)
    Z: ndarray of shape (T,m(p+1))
        lagged factor regressors for VAR, defined in (6.18.15)
    eps_: list of shape (n)
        list of residuals, defined in (6.18.13)   
    E_: list of shape (n)
        list of lagged residual regressors, defined in (6.18.13) 
    E: ndarray of shape (T,n)
        current period residuals, defined in (6.18.13)    
    xi: ndarray of shape (T,m)
        factor VAR residuals, defined in (6.18.15) 
    e: ndarray of shape (T,n)
        residual AR disturbances, defined in (6.18.13)          
    """     

    W = z[:,:m*(q+1)]
    F = z[:,:m]
    Z = z[:,m:m*(p+1)]
    xi = F - Z @ beta
    eps = z[:,m*(s+1):]
    E = eps[:,:n]
    eps_ = [None] * n
    E_ = [None] * n
    e = np.zeros(((T,n)))
    index = n * np.arange(1,r+1)    
    for i in range(n):
        eps_[i] = eps[:,i]
        E_[i] = eps[:,index+i]
        e[:,i] = eps_[i] - E_[i] @ gamma[i,:]
    return W, F, Z, eps_, E_, E, xi, e


def dfm_state_space_representation(lamda, beta, gamma, sigma, omega, n, m, q, p, s, r):
    
    """
    dfm_state_space_representation(lamda, beta, gamma, sigma, omega, n, m, q, p, s, r)
    create state-space representation matrices
    
    parameters:
    lamda: ndarray of size (n,l)
        matrix of coefficients lambda_i, defined in (6.18.10)
    beta: ndarray of size (k,m)
        matrix of coefficients beta_j, defined in (6.18.16)
    gamma: ndarray of size (n,r)
        matrix of residual AR coefficients gamma_i, defined in (6.18.13) 
    sigma: float
        variance on residuals, defined in (6.18.7)
    omega: float
        variance on factors, defined in (6.18.6)         
    n: int
        number of endogenous variables, defined in (6.18.1)          
    m: int
        number of fundamental factors in the model, defined in (6.18.1)        
    q: int
        number of lags in the loadings equation, defined in (6.18.2)          
    p: int
        number of lags in the VAR model, defined in (6.18.6)  
    s: int
        maximum number of factor lags, defined in (6.18.38)        
    r: int
        number of lags in the residual AR process, defined in (6.18.7) 
        
    returns:
    Lambda: ndarray of shape (n,m(s+1)+n(r+1))
        equivalent of A_t matrix, defined in (6.18.39)
    B: ndarray of shape (m(s+1)+n(r+1),m(s+1)+n(r+1))
        companion form matrix for VAR dynamics, defined in (6.18.39) 
    Upsilon: ndarray of shape (m(s+1)+n(r+1),m(s+1)+n(r+1))
        covariance matrix for VAR dynamics, defined in (6.18.39)         
    """        

    temp_1 = np.hstack([lamda, np.zeros((n,m*(s-q)))])
    temp_2 = np.hstack([np.eye(n), np.zeros((n,n*r))])
    Lambda = np.hstack([temp_1,temp_2])
    temp_1 = np.hstack([beta.T, np.zeros((m,(s-p+1)*m)), np.zeros((m,(r+1)*n))])
    temp_2 = np.hstack([np.eye(s*m), np.zeros((s*m,m)), np.zeros((s*m,(r+1)*n))])
    temp_3 = [np.zeros((n,(s+1)*m))]
    for j in range(r):
        temp_3.append(np.diag(gamma[:,j]))
    temp_3.append(np.zeros((n,n)))
    temp_3 = np.hstack(temp_3)
    temp_4 = np.hstack([np.zeros((n*r,(s+1)*m)), np.eye(r*n), np.zeros((n*r,n))])
    B = np.vstack([temp_1, temp_2, temp_3, temp_4])
    Upsilon = np.diag(np.hstack([np.ones(m) * omega, np.zeros(s*m), np.ones(n) * sigma, np.zeros(r*n)]))
    return Lambda, B, Upsilon
    
    
def dfm_kalman_initial_values(sigma, omega, s, r, m, n):
    
    """
    dfm_kalman_initial_values(sigma, omega, s, r, m, n)
    create initial values for state-space representation, defined in (6.18.39)
    
    parameters:
    sigma: float
        variance on residuals, defined in (6.18.7)
    omega: float
        variance on factors, defined in (6.18.6)         
    q: int
        number of fundamental factors in the model, defined in (6.18.1)
    s: int
        maximum number of factor lags, defined in (6.18.38)        
    r: int
        number of lags in the residual AR process, defined in (6.18.7) 
    m: int
        number of fundamental factors in the model, defined in (6.18.1)  
    n: int
        number of endogenous variables, defined in (6.18.1) 

    returns:
    z_00: ndarray of shape (m(s+1)+n(r+1),)
        vector of mean for initial factor value
    Upsilon_00: ndarray of shape (m(s+1)+n(r+1),m(s+1)+n(r+1))
        matrix of variance for initial factor value 
    """       

    z_00 = np.zeros((m*(s+1)+n*(r+1)))
    Upsilon_00 = np.diag(np.hstack([5*np.ones(m)*omega, np.zeros(s*m), \
                 np.ones(n)*sigma, np.zeros(r*n)])) + 1e-10 * np.eye(m*(s+1)+n*(r+1))
    return z_00, Upsilon_00


def epsilon_state_space_representation(gamma, sigma, l, r):
    
    """
    epsilon_state_space_representation(gamma, sigma, l, r)
    create state-space representation matrices for the residuals
        
    parameters:
    gamma: ndarray of size (l,r)
        matrix of residual AR coefficients gamma_i for first l variables, defined in (6.18.13)         
    sigma: float
        variance on residuals, defined in (6.18.7) 
    l: int
        total number of loadings regressors, defined in (6.18.11)         
    r: int
        number of lags in the residual AR process, defined in (6.18.7)      
        
    returns:
    A: ndarray of shape (l,l(r+1))
        equivalent of A_t matrix
    B: ndarray of shape (l(r+1),l(r+1))
        companion form matrix B for residual dynamics 
    Upsilon: ndarray of shape (l(r+1),l(r+1))
        covariance matrix for residual dynamics         
    """        

    A = np.hstack([np.eye(l), np.zeros((l,l*r))])
    if r == 0:
        B = np.zeros((l,l))
    elif r > 0:
        temp_1 = np.zeros((l,l*(r+1)))
        for j in range(r):
            temp_1[:,j*l:(j+1)*l] = np.diag(gamma[:,j])
        temp_2 = np.hstack([np.eye(r*l), np.zeros((l*r,l))])
        B = np.vstack([temp_1, temp_2])
    Upsilon = np.diag(np.hstack([np.ones(l) * sigma, np.zeros(r*l)]))       
    return A, B, Upsilon


def epsilon_kalman_initial_values(sigma, r, l):
    
    """
    eps_kalman_initial_values(sigma, r, l)
    create initial values for state-space representation  
    
    parameters:        
    sigma: float
        variance on residuals, defined in (6.18.7) 
    l: int
        total number of loadings regressors, defined in (6.18.11)         
    r: int
        number of lags in the residual AR process, defined in (6.18.7)   
        
    returns:
    z_00: ndarray of shape (l(r+1),)
        vector of mean for initial factor value
    Upsilon_00: ndarray of shape (l(r+1),l(r+1))
        matrix of variance for initial factor value          
    """       

    z_00 = np.zeros((l*(r+1)))
    Upsilon_00 = np.diag(np.hstack([3*np.ones(l)*sigma, np.zeros(r*l)]))
    return z_00, Upsilon_00


def update_epsilon_regressors(gamma, z, l, r, T):
    
    """
    update_epsilon_regressors(gamma, z, l, r, T)
    update residual regressors  
    
    parameters:        
    z : ndarray of shape (T,l(r+1))
        matrix of sampled values for the state variables   
    l: int
        total number of loadings regressors, defined in (6.18.11)         
    r: int
        number of lags in the residual AR process, defined in (6.18.7)     
    T: int
        total number of sample periods, defined in (6.18.1)

    returns:        
    eps_: list of shape (n)
        list of residuals, defined in (6.18.13)   
    E_: list of shape (n)
        list of lagged residual regressors, defined in (6.18.13) 
    E: ndarray of shape (T,n)
        current period residuals, defined in (6.18.13)     
    e: ndarray of shape (T,n)
        residual AR disturbances, defined in (6.18.13)           
    """     

    E = z[:,:l]
    eps_ = [None] * l
    E_ = [None] * l    
    e = np.zeros((T,l))
    index = l * np.arange(1,r+1) 
    for i in range(l):
        eps_[i] = z[:,i]
        E_[i] = z[:,index+i]
        e[:,i] = eps_[i] - E_[i] @ gamma[i,:]
    return eps_, E_, E, e


def posterior_estimates(X, credibility_level):
    
    """
    posterior_estimates(X, credibility_level)
    median, lower bound and upper bound of credibility interval
    
    parameters:
    X : ndarray of shape (n,m,iterations)
        matrix of MCMC draws
    credibility_level : float between 0 and 1
        credibility level for credibility interval
    
    returns:
    posterior_estimates : ndarray of shape (n,m,4)
        matrix of posterior estimates
    """

    posterior_estimates = np.zeros((X.shape[0],X.shape[1],4))
    posterior_estimates[:,:,0] = np.quantile(X,0.5,axis=2)
    posterior_estimates[:,:,1] = np.quantile(X,(1-credibility_level)/2,axis=2)
    posterior_estimates[:,:,2] = np.quantile(X,(1+credibility_level)/2,axis=2)
    posterior_estimates[:,:,3] = np.std(X,axis=2)
    return posterior_estimates


def dfm_forecast(lamda, beta, gamma, sigma, omega, f, eps, h, m, n, p, r, l):

    """
    dfm_forecast(lamda, beta, gamma, sigma, omega, f, eps, h, m, n, p, r, l)
    forecasts for the dynamic factor model, using algorithm 18.3
    
    parameters:
    lamda: ndarray of shape (n,l)
        matrix of coefficients lambda_i, defined in (6.18.10)
    beta: ndarray of shape (k,m)
        matrix of coefficients beta_j, defined in (6.18.16)
    gamma: ndarray of shape (n,r)
        matrix of residual AR coefficients gamma_i, defined in (6.18.13) 
    sigma: float
        variance on residuals, defined in (6.18.7)
    omega: float
        variance on factors, defined in (6.18.6)  
    f: ndarray of shape (l,)
        matrix of sample factors     
    eps: ndarray of shape (n*(1+r),)
        matrix of sample residuals
    h: int
        number of forecast periods        
    m: int
        number of fundamental factors in the model, defined in (6.18.1)        
    n: int
        number of endogenous variables, defined in (6.18.1)          
    p: int
        number of lags in the VAR model, defined in (6.18.6)  
    r: int
        number of lags in the residual AR process, defined in (6.18.7)  
    l: int
        total number of loadings regressors, defined in (6.18.11) 
        
    returns:
    posterior_estimates : ndarray of shape (n,m,4)
        matrix of posterior estimates
    """

    # initiate storage and shocks
    X_p = np.zeros((h,n))
    F_p = np.zeros((h,m))
    Xi = sqrt(omega) * nrd.randn(h,m)
    e = sqrt(sigma) * nrd.randn(h,n)
    for t in range(h):
        # predict factors using (6.18.15)
        Z = f[:p*m]
        F = Z @ beta + Xi[t,:]
        f = np.hstack([F,f])
        # predict residuals using (6.18.12)
        E = eps[:n*r].reshape(n,-1, order='F')
        eps_ = np.sum(E * gamma,axis=1) + e[t,:]
        eps = np.hstack([eps_,eps])
        # predict endogenous using (6.18.10)
        W = f[:l]
        x = W @ lamda.T + eps_
        X_p[t,:] = x
        F_p[t,:] = F
    return X_p, F_p
        

def dfm_impulse_response_function(lamda, beta, gamma, n, m, q, p, r, h):
    
    """
    dfm_impulse_response_function(lamda, beta, gamma, n, m, q, p, r, h)
    impulse response function for dfm, using algorithm 18.4
    
    parameters:
    lamda: ndarray of shape (n,l)
        matrix of coefficients lambda_i, defined in (6.18.10)
    beta: ndarray of shape (k,m)
        matrix of coefficients beta_j, defined in (6.18.16)
    gamma: ndarray of shape (n,r)
        matrix of residual AR coefficients gamma_i, defined in (6.18.13) 
    n: int
        number of endogenous variables, defined in (6.18.1)            
    m: int
        number of fundamental factors in the model, defined in (6.18.1)  
    q: int
        number of fundamental factors in the model, defined in (6.18.1)        
    p: int
        number of lags in the VAR model, defined in (6.18.6)  
    r: int
        number of lags in the residual AR process, defined in (6.18.7) 
    h: int
        number of irf periods 
        
    returns:
    irf : ndarray of shape (n,m+1,h)
        matrix of dfm impulse response function
    """

    irf = np.zeros((n,m+1,h))
    factor_irf = factor_impulse_response_function(beta, m, p, h)
    residual_irf = residual_impulse_response_function(gamma, n, r, h)
    Lambda_0 = lamda[:,:m]
    full_factor_irf = Lambda_0 @ factor_irf
    for i in range(1,q+1):
        Lambda_i = lamda[:,i*m:(i+1)*m]
        factor_irf_i = np.hstack([np.zeros((m,i*m)), factor_irf[:,:-m*i]])
        full_factor_irf += Lambda_i @ factor_irf_i
    for i in range(h):
        irf[:,:m,i] = full_factor_irf[:,i*m:(i+1)*m]
        irf[:,m,i] = residual_irf[:,i]
    return irf
        

def factor_impulse_response_function(beta, m, p, h):
    
    """
    factor_impulse_response_function(beta, m, p, h)
    impulse response function for factor VAR
    
    parameters:
    beta: ndarray of shape (k,m)
        matrix of coefficients beta_j, defined in (6.18.16)           
    m: int
        number of fundamental factors in the model, defined in (6.18.1)         
    p: int
        number of lags in the VAR model, defined in (6.18.6)  
    h: int
        number of irf periods 
        
    returns:
    irf : ndarray of shape (m,m,h)
        matrix of factor impulse response function
    """    
    
    Yh = np.eye(m)
    irf = [Yh]
    Xh = np.zeros((m,m*p))   
    for i in range(1,h):
        Xh = np.hstack([Yh,Xh[:,:-m]])
        Yh = Xh @ beta
        irf.append(Yh.T)
    irf = np.hstack(irf)
    return irf


def residual_impulse_response_function(gamma, n, r, h):

    """
    residual_impulse_response_function(gamma, n, r, h)
    impulse response function for residual AR models
    
    parameters:
    gamma: ndarray of shape (n,r)
        matrix of residual AR coefficients gamma_i, defined in (6.18.13) 
    n: int
        number of endogenous variables, defined in (6.18.1)            
    r: int
        number of lags in the residual AR process, defined in (6.18.7) 
    h: int
        number of irf periods 
        
    returns:
    irf : ndarray of shape (n,h)
        matrix of residual impulse response function
    """
    
    Yh = np.ones(n)
    irf = np.hstack([Yh.reshape(-1,1), np.zeros((n,h-1))])
    Xh = np.zeros((n,r))
    for i in range(1,h):
        Xh = np.hstack([Yh.reshape(-1,1),Xh[:,:r-1]])
        Yh = np.sum(Xh * gamma, axis=1)
        irf[:,i] = Yh
    return irf


def posterior_estimates_3d(X, credibility_level):
    
    """
    posterior_estimates(X, credibility_level)
    median, lower bound and upper bound of credibility interval
    
    parameters:
    X : ndarray of shape (n,m,h,iterations)
        matrix of MCMC draws
    credibility_level : float between 0 and 1
        credibility level for credibility interval
    
    returns:
    posterior_estimates : ndarray of shape (n,m,h,3)
        matrix of posterior estimates
    """

    posterior_estimates = np.zeros((X.shape[0],X.shape[1],X.shape[2],3))
    posterior_estimates[:,:,:,0] = np.quantile(X,0.5,axis=3)
    posterior_estimates[:,:,:,1] = np.quantile(X,(1-credibility_level)/2,axis=3)
    posterior_estimates[:,:,:,2] = np.quantile(X,(1+credibility_level)/2,axis=3)
    return posterior_estimates


def dfm_forecast_error_variance_decomposition(irf, sigma, omega, n, m, h):
    
    """
    dfm_forecast_error_variance_decomposition(irf, sigma, omega, n, m, h)
    products FEVD for the dynamic factor model, using (6.18.50)
    
    parameters:
    irf : ndarray of shape (n,m+1,h)
        matrix of impulse response functions
    sigma: float
        variance on residuals, defined in (6.18.7)
    omega: float
        variance on factors, defined in (6.18.6)      
    n : int
        number of endogenous variables  
    m: int
        number of fundamental factors in the model, defined in (6.18.1)         
    h : int
        number of forecast periods
    
    returns:
    fevd : ndarray of shape (n,m+1,h)
        matrix of forecast error variance decomposition
    """ 
    
    cum_squared_irf = np.cumsum(irf ** 2, axis=2)
    variances = np.dstack([np.hstack([omega*np.ones((n,m)),sigma*np.ones((n,1))])]*h)
    cum_squared_irf = variances * cum_squared_irf
    total_variance = np.hstack([np.sum(cum_squared_irf,axis=1, keepdims = True)] * (m+1))
    fevd = cum_squared_irf / total_variance
    return fevd


def normalize_fevd_estimates(fevd_estimates):

    """
    normalize_fevd_estimates(fevd_estimates)
    normalizes FEVD estimates so that they sum up to 1
    
    parameters:
    fevd_estimates : ndarray of shape (n,n,h,iterations)
        matrix of posterior FEVD estimates
    
    returns:
    normalized_fevd_estimates : ndarray of shape (n,m,h,iterations)
        matrix of normalized posterior FEVD estimates
    """  

    point_estimate_contribution = np.sum(fevd_estimates[:,:,:,0],axis=1, keepdims = True)
    total_contribution = np.hstack([point_estimate_contribution] * fevd_estimates.shape[1])
    estimates_contribution = np.stack([total_contribution] * 3, axis=3)
    normalized_fevd_estimates = fevd_estimates / estimates_contribution
    return normalized_fevd_estimates


def dfm_historical_decomposition(irf, xi, e, n, m, T):
    
    """
    dfm_historical_decomposition(irf, xi, e, n, m, T)
    products historical decomposition for the dynamic factor model, using (6.18.51)-(6.18.53)
    
    parameters:
    irf : ndarray of shape (n,m+1,h)
        matrix of impulse response functions
    xi : ndarray of shape (T,m)
        matrix of factor shocks    
    e : ndarray of shape (T,n)
        matrix of residual shocks              
    n : int
        number of endogenous variables  
    m: int
        number of fundamental factors in the model, defined in (6.18.1)         
    T: int
        total number of sample periods, defined in (6.18.1)
    
    returns:
    hd : ndarray of shape (n,m+1,T)
        matrix of historical decomposition
    """ 
    
    reshaped_xi = np.transpose(np.dstack([np.flip(xi,0)] * n), [2, 1, 0])
    reshaped_e = np.transpose(np.reshape(np.flip(e,0),[T,n,-1]), [1,2,0])
    reshaped_shocks = np.hstack([reshaped_xi,reshaped_e])
    hd = np.zeros((n,m+1,T))
    for i in range(T):
        hd[:,:,i] = np.sum(irf[:,:,:i+1] * reshaped_shocks[:,:,-(i+1):],axis=2)
    return hd
        
        
def dfm_insample_evaluation_criteria(Y, E, T, k):

    """
    dfm_insample_evaluation_criteria(Y, E, T, k)
    computes ssr, R2 and adjusted R2 for each VAR equation
    
    parameters:
    Y : ndarray of shape (T,n)
        matrix of endogenous variables
    E : ndarray of shape (T,n)
        matrix of residuals
    T : int
        number of sample periods
    k : int
        number of coefficients in each VAR equation
    
    returns:
    insample_evaluation : dict
        dictionary of insample evaluation criteria
    """  

    ssr = np.diag(E.T @ E)
    Z = Y - np.mean(Y,axis=0)
    tss = np.diag(Z.T @ Z)
    r2 = 1 - ssr / tss
    adj_r2 = 1 - (1 - r2) * (T - 1) / (T - k)
    insample_evaluation = {}
    insample_evaluation['ssr'] = ssr
    insample_evaluation['r2'] = r2
    insample_evaluation['adj_r2'] = adj_r2
    return insample_evaluation        


def make_mfbvar_state_regressors(Z_, B, Sigma, n, m, p, T_):
    
    """
    make_mfbvar_state_regressors(Z_, B, Sigma, n, m, p, T_)
    make state-space parameters for MF-BVAR model, defined in (6.17.20)-(6.17.22)
    
    parameters:
    Z_ : ndarray of shape (T_,m)
        matrix of endogenous variables, defined in (6.17.1), including initial conditions
    B : ndarray of shape (k,n)
        matrix of VAR coefficients, defined in (6.17.2)
    Sigma : ndarray of shape (n,n)
        variance-covariance matrix of residuals, defined in (6.17.1)
    n : int
        number of endogenous variables, defined in (6.17.1)
    m : int
        number of exogenous variables, defined in (6.17.1)
    p : int
        number of lags in the VAR model, defined in (6.17.1)
    T_ : int
        number of sample periods, including initial conditions
        
    returns:
    F : ndarray of shape (n*p,n*p)
        companion matrix for VAR coefficients
    mu_ : ndarray of shape (n*p,)
        exogenous vector of state equation
    Upsilon : ndarray of shape (n*p,n*p)
        variance covariance matrix of state equation
    """     

    F = np.eye(n*p,k=-n)
    F[:n,:] = B[m:,:].T
    C = B[:m,:]
    mu_ = np.hstack([Z_ @ C,np.zeros((T_,(p-1)*n))])
    Upsilon = 1e-12 * np.eye(n*p)
    Upsilon[:n,:n] = Sigma
    return F, mu_, Upsilon
    
  
def mfbvar_kalman_initial_values(Sigma, n, p):

    """
    mfbvar_kalman_initial_values(Sigma, n, p)
    create initial values for state-space representation, defined in (6.17.20)-(6.17.21)
    
    parameters:
    Sigma : ndarray of shape (n,n)
        variance-covariance matrix of residuals, defined in (6.17.1)
    n : int
        number of endogenous variables, defined in (6.17.1)
    p : int
        number of lags in the VAR model, defined in (6.17.1)
        
    returns:
    gamma_00 : ndarray of shape (n*p,)
        initial values for state vector
    Upsilon_00 : ndarray of shape (n*p,n*p)
        initial value for variance covariance matrix of state vector
    """      
    
    gamma_00 = np.zeros(n*p)    
    Upsilon_00 = 1e-12 * np.eye(n*p)
    Upsilon_00[:n,:n] = 10 * Sigma
    return gamma_00, Upsilon_00
    
  
def update_mfbvar_state_regressors(Z, Gamma, n, p):
    
    """
    update_mfbvar_state_regressors(Z, Gamma, n, p)
    define state regressors W and X from Gamma, as defined in (6.17.3)
    
    parameters:
    Z : ndarray of shape (T,m)
        matrix of endogenous variables, defined in (6.17.1)        
    Gamma : ndarray of shape (T_,n*r)
        full matrix of state vectors gamma, including initial conditions
    n : int
        number of endogenous variables, defined in (6.17.1)
    p : int
        number of lags in the VAR model, defined in (6.17.1)
        
    returns:
    W : ndarray of shape (T,n)
        current period states
    X : ndarray of shape (T,k)
        matrix of lagged and endogenous regressors, defined in (6.17.2)
    """ 
    
    W = Gamma[p:,:n]
    X = np.hstack([Z,Gamma[p:,n:p*n],Gamma[:-p,:n]])
    return W, X
    
    

    