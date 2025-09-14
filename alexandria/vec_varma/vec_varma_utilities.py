# imports
import numpy as np
import alexandria.vector_autoregression.var_utilities as vu
from alexandria.vector_autoregression.maximum_likelihood_var import MaximumLikelihoodVar
import alexandria.math.linear_algebra as la
import numpy.random as nrd


# module vec_varma_utilities
# a module containing methods for vec and varma utilities


#---------------------------------------------------
# Methods
#---------------------------------------------------


def make_var_regressors(endogenous, exogenous, lags, constant, trend, quadratic_trend):
    
    """
    make_var_regressors(endogenous, exogenous, lags, constant, trend, quadratic_trend)
    generates VAR regressors, defined in (4.11.3)
    
    parameters:
    endogenous : ndarray of shape (periods,n_endogenous)
        matrix of endogenous regressors        
    exogenous : ndarray of shape (periods,n_exogenous)
        matrix of exogenous regressors     
    lags: int
        number of lags in VAR model        
    constant : bool
        if True, a constant is added to the VAR model
    trend : bool
        if True, a linear trend is added to the VAR model
    quadratic_trend : bool
        if True, a quadratic trend is added to the VAR model
        
    returns:
    Y : ndarray of shape (T,n)
        matrix of endogenous variables 
    Z : ndarray of shape (T,m)
        matrix of exogenous variables         
    X : ndarray of shape (T,k)
        matrix of regressors
    """     
        
    # get Y matrix
    Y = endogenous[lags:]
    # get X matrix
    periods = endogenous.shape[0] - lags
    X_1 = vu.generate_intercept_and_trends(constant, trend, quadratic_trend, periods, 0)
    X_2 = vu.generate_exogenous_regressors(exogenous, lags, periods)
    X_3 = vu.generate_lagged_endogenous(endogenous, lags)
    Z = np.hstack([X_1,X_2])
    X = np.hstack([X_1,X_2,X_3])    
    return Y, Z, X


def make_vec_regressors(endogenous, exogenous, lags, constant, trend, quadratic_trend):
    
    """
    make_vec_regressors(endogenous, exogenous, lags, constant, trend, quadratic_trend)
    generates VEC regressors, defined in (5.15.8)
    
    parameters:
    endogenous : ndarray of shape (periods,n_endogenous)
        matrix of endogenous regressors        
    exogenous : ndarray of shape (periods,n_exogenous)
        matrix of exogenous regressors        
    lags: int
        number of lags in VAR model        
    constant : bool
        if True, a constant is added to the VAR model
    trend : bool
        if True, a linear trend is added to the VAR model
    quadratic_trend : bool
        if True, a quadratic trend is added to the VAR model
        
    returns:
    DY : ndarray of shape (T,n)
        matrix of differenced endogenous variables   
    Y_1 : ndarray of shape (T,n)
        matrix of lagged endogenous variables, one period           
    Z : ndarray of shape (T,k)
        matrix of endogenous and lagged regressors
    """    
    
    # get DY matrix
    diff_endogenous = endogenous[1:,:] - endogenous[:-1,:]
    DY = diff_endogenous[lags:]
    # get Y_1 matrix
    Y_1 = endogenous[lags:-1,:]
    # get Z matrix
    periods = endogenous.shape[0] - lags - 1
    Z_1 = vu.generate_intercept_and_trends(constant, trend, quadratic_trend, periods, 0)
    Z_2 = vu.generate_exogenous_regressors(exogenous, lags+1, periods)    
    Z_3 = vu.generate_lagged_endogenous(diff_endogenous, lags)
    Z = np.hstack([Z_1,Z_2,Z_3])
    return DY, Y_1, Z


def generate_dimensions(Y, DY, exogenous, lags, max_cointegration_rank, constant, trend, quadratic_trend):

    """
    generate_dimensions(DY, exogenous, lags, max_cointegration_rank, constant, trend, quadratic_trend)
    generate VEC dimensions, defined in (5.15.8)
    
    parameters:
    Y : ndarray of shape (T,n)
        matrix of endogenous variables 
    DY : ndarray of shape (T,n)
        matrix of differenced endogenous variables               
    exogenous : ndarray of shape (periods,n_exogenous)
        matrix of exogenous regressors        
    lags: int
        number of lags in VAR model     
    max_cointegration_rank: int
        maximum cointegration rank
    constant : bool
        if True, a constant is added to the VAR model
    trend : bool
        if True, a linear trend is added to the VAR model
    quadratic_trend : bool
        if True, a quadratic trend is added to the VAR model
        
    returns:
    n : int
        number of endogenous variables         
    m : int
        number of exogenous variables     
    p_vec : int
        number of lags of VEC model       
    p : int
        number of lags of equivalent VAR model
    T_vec : int
        number of sample periods of VEC model            
    T : int
        number of sample periods
    k_vec : int
        number of coefficients in each equation of VEC model       
    k : int
        number of coefficients in each equation of equivalent VAR model    
    q_vec : int
        total number of coefficients of VEC model
    q : int
        total number of coefficients of equivalent VAR model   
    r : int
        maximum cointegration rank
    """    

    T_vec = DY.shape[0]
    T = Y.shape[0]
    n = DY.shape[1]
    p_vec = lags
    p = lags + 1
    m = int(constant) + int(trend) + int(quadratic_trend)    
    if len(exogenous) != 0:
        m += exogenous.shape[1]
    k_vec = m + n * p_vec    
    k = m + n * p
    q_vec = n * k_vec
    q = n * k
    r = max_cointegration_rank
    return n, m, p_vec, p, T_vec, T, k_vec, k, q_vec, q, r


def individual_ar_variances(n, endogenous, lags):
    
    """ 
    individual_ar_variances(n, endogenous, lags)
    generates residual variance for each variable
    
    parameters:  
    n : int
        number of endogenous variables  
    endogenous : ndarray of shape (periods,n_endogenous)
        matrix of endogenous regressors               
    lags: int
        number of lags in VAR model   

    returns:        
    s : ndarray of shape (n,)
        vector of individual residual variances
    """
    
    diff_endogenous = endogenous[1:,:] - endogenous[:-1,:]
    s = np.zeros(n)
    for i in range(n):
        ar = MaximumLikelihoodVar(diff_endogenous[:,[i]], lags=lags)
        ar.estimate()
        s[i] = ar.Sigma[0,0]
    return s


def vec_to_var(Xi_T, Phi, n, m, p, k):
    
    """ 
    vec_to_var(Xi_T, Phi, n, m, p, k)
    converts VEC to VAR using equation (5.15.103)
    
    parameters:  
    Xi_T : ndarray of shape (n,n)
        transpose of error correction matrix Xi, defined in (5.15.3)
    Phi : ndarray of shape (k,n)
        matrix of VEC coefficients, defined in (5.15.8)          
    n : int
        number of endogenous variables         
    m : int
        number of exogenous variables         
    p : int
        number of lags
    k : int
        number of coefficients in each VEC equation 

    returns:        
    B : ndarray of shape (k+n,n)
        matrix of equivalent VAR coefficients, defined in (4.11.3) and (5.15.103)
    """
    
    # initiate storage
    A_T = np.zeros((n,n,p+1))
    B = np.zeros((k+n,n))
    # last VAR lag
    A_T[:,:,-1] = - Phi[-n:,:]
    B[-n:,:] = A_T[:,:,-1]
    # VAR lags from 2 to p-1
    for i in range(p-1,0,-1):
        A_T[:,:,i] = - Phi[m+(i-1)*n:m+i*n] - np.sum(A_T[:,:,i+1:],axis=2)
        B[m+i*n:m+(i+1)*n] = A_T[:,:,i]
    # first VAR lag
    A_T[:,:,0] = Xi_T + np.eye(n) - np.sum(A_T[:,:,1:],axis=2)
    B[m:m+n] = A_T[:,:,0]
    # exogenous regressors
    B[:m,:] = Phi[:m,:]
    return B
        
    
def vec_steady_state(Y,n,T):
    
    """
    vec_steady_state(Y);
    pseudo steady-state estimates for VEC model
    
    parameters:  
    Y : matrix of size (T,n)
        matrix of endogenous variables   
    n : int
        number of endogenous variables 
    T : int
        number of sample periods

    returns:            
    steady_state_estimates : matrix of size (T,n,3)
        matrix of steady-state estimates
    """

    steady_state_estimates = np.zeros((T,n,3))
    steady_state_mean = np.mean(Y,axis=0)
    steady_state_std = np.std(Y,axis=0)
    lower_bound = steady_state_mean - 0.1 * steady_state_std
    upper_bound = steady_state_mean + 0.1 * steady_state_std
    steady_state_estimates[:,:,0] = steady_state_mean
    steady_state_estimates[:,:,1] = lower_bound
    steady_state_estimates[:,:,2] = upper_bound
    return steady_state_estimates


def make_varma_restriction_irf(mcmc_beta, mcmc_kappa, mcmc_chol_Sigma, iterations, n, p, q, max_irf_period):

    """
    make_varma_restriction_irf(mcmc_beta, mcmc_kappa, mcmc_chol_Sigma, iterations, n, p, q, max_irf_period)
    creates preliminary orthogonalized IRFs for restriction algorithm
    
    parameters:
    mcmc_beta : ndarray of shape (k1, n, iterations)
        matrix of mcmc values for beta
    mcmc_kappa : ndarray of shape (k2, n, iterations)
        matrix of mcmc values for K        
    mcmc_chol_Sigma : ndarray of shape (n, n, iterations)
        matrix of mcmc values for h(Sigma)
    iterations: int
        number of MCMC iterations
    n : int
        number of endogenous variables       
    p : int
        number of lags
    q : int
        number of residual lags        
    max_irf_period : int
        maximum number of periods for which IRFs will have to be computed in later algorithms        
          
    returns:
    mcmc_irf : ndarray of shape (n, n, n_periods, iterations)
        matrix of mcmc values for preliminary orthogonalized IRFs 
    """

    if max_irf_period == 0:
        mcmc_irf = []
    else:
        mcmc_irf = np.zeros((n, n, max_irf_period, iterations))
        for i in range(iterations):
            irf = varma_impulse_response_function(mcmc_beta[:,:,i], mcmc_kappa[:,:,i], n, p, q, max_irf_period)
            structural_irf = vu.structural_impulse_response_function(irf, mcmc_chol_Sigma[:,:,i], n)            
            mcmc_irf[:,:,:,i] = structural_irf
    return mcmc_irf



def make_varma_restriction_shocks(mcmc_E, mcmc_chol_Sigma, T, n, iterations, restriction_matrices):

    """
    make_varma_restriction_shocks(mcmc_E, mcmc_chol_Sigma, T, n, iterations, restriction_matrices)
    creates preliminary structural shocks for restriction algorithm
    
    parameters:
    mcmc_E : ndarray of shape (T, n, iterations)
        matrix of mcmc values for residuals
    mcmc_chol_Sigma : ndarray of shape (n, n, iterations)
        matrix of mcmc values for h(Sigma)   
    T : int
        number of sample periods   
    n : int
        number of endogenous variables        
    iterations: int
        number of MCMC iterations        
    restriction_matrices : list of length 7
        each list entry stores matrices of restriction and coefficient values        
          
    returns:
    mcmc_shocks : ndarray of shape (T, n, iterations)
        matrix of mcmc values for preliminary structural shocks
    """    

    if len(restriction_matrices[3]) == 0 and len(restriction_matrices[4]) == 0:
        mcmc_shocks = []
    else:
        mcmc_shocks = np.zeros((T, n, iterations))
        for i in range(iterations):
            E = mcmc_E[:,:,i]
            Xi = la.slash_inversion(E, mcmc_chol_Sigma[:,:,i].T)
            mcmc_shocks[:,:,i] = Xi
    return mcmc_shocks


def varma_steady_state(X, B, n, m, p, T):
    
    """
    varma_steady_state(X, B, n, m, p, T)
    computes the steady-state of the VAR model, using equation (4.12.30)
    
    parameters:
    X : ndarray of shape (T,k1)
        matrix of exogenous variables
    B : ndarray of shape (k1,n)
        matrix of VAR coefficients
    n : int
        number of endogenous variables
    m : int
        number of exogenous variables        
    p : int
        number of lags
    T : int
        number of sample periods         
        
    returns:
    steady_state : ndarray of shape (T,n)
        matrix of steady-state values
    """      

    if m == 0:
        steady_state = np.zeros((T,n))
    else:
        Z = X[:,:m]
        C = B[:m,:]
        A = np.eye(n)
        for i in range(p):
            A -= B[m+i*n:m+(i+1)*n]
        steady_state = Z @ la.slash_inversion(C, A)
    return steady_state


def varma_fit_and_residuals(X, B, Z, K, E):

    """
    varma_fit_and_residuals(X, B, Z, K, E
    generates fit and residuals for a VARMA model, using (5.16.2)
    
    parameters:
    X : ndarray of shape (T,k1)
        matrix of regressors
    B : ndarray of shape (k1,n)
        matrix of VAR coefficients
    Z : ndarray of shape (T,k2)
        matrix of lagged residuals
    K : ndarray of shape (k2,n)
        matrix of residual coefficients        
    E : ndarray of shape (T,n)
        matrix of residuals 
        
    returns:
    Y_hat : ndarray of shape (T,n)
        matrix of fitted endogenous
    E : ndarray of shape (T,n)
        matrix of VAR residuals
    """    

    Y_hat = X @ B + Z @ K
    return E, Y_hat


def varma_forecast(B, K, chol_Sigma, h, Z_p, Y, E, n):

    """
    varma_forecast(B, K, chol_Sigma, h, Z_p, Y, E, n)
    products simulated forecasts
    
    parameters:
    B : ndarray of shape (k1,n)
        matrix of VAR coefficients
    K : ndarray of shape (k2,n)
        matrix of residual coefficients          
    chol_Sigma : ndarray of shape (n,n)
        Cholesky factor of residual variance-covariance matrix Sigma
    h : int
        number of forecast periods
    Z_p : ndarray of shape (h,m)
        matrix of exogenous regressors for forecasts
    Y : ndarray of shape (p,n)
        matrix of initial conditions for endogenous variables  
    E : ndarray of shape (q,n)
        matrix of residuals         
    n : int
        number of endogenous variables        
    
    returns:
    Y_p : ndarray of shape (h,n)
        matrix of simulated forecast values
    """

    Y_p = np.zeros((h,n))
    for i in range(h):
        # get lagged endogenous regressors
        X = la.vec(np.fliplr(Y.T)).reshape(1,-1)
        # add exogenous regressors, if any
        if len(Z_p) != 0:
            X = np.hstack([Z_p[[i],:],X])
        # get lagged residual regressors
        Z = la.vec(np.fliplr(E.T))
        # recover current residuals
        e = chol_Sigma @ nrd.randn(n)
        # generate forecasts
        y = X @ B + Z @ K + e
        # update Y, E and Y_p
        Y = np.vstack([Y[1:,:],y])
        E = np.vstack([E[1:,:],e])
        Y_p[i,:] = y
    return Y_p


def varma_impulse_response_function(B, K, n, p, q, h):
    
    """
    varma_impulse_response_function(B, K, n, p, q, h)
    generates impulse response function for given VARMA coefficients
    using equations (5.16.31)-(5.16.33)
    
    parameters:
    B : ndarray of shape (k1,n)
        matrix of VAR coefficients
    K : ndarray of shape (k2,n)
        matrix of residual coefficients  
    n : int
        number of endogenous variables
    p : int
        number of lags
    q : int
        number of residual lags        
    h : int
        number of irf periods (including impact period)
    
    returns:
    irf : ndarray of shape (n,n,h)
        matrix of impulse response functions
    """

    B = B[-n*p:]
    Yh = np.eye(n)
    irf = np.dstack([Yh, np.zeros((n,n,h-1))])
    Xh = np.zeros((n,n*p))
    Zh = np.eye(n,n*q)
    for i in range(1,h):
        Xh = np.hstack([Yh,Xh[:,:-n]])
        Yh = Xh @ B + Zh @ K
        Zh = np.hstack([np.zeros((n,n)),Zh[:,:-n]])
        irf[:,:,i] = Yh.T
    return irf


def varma_conditional_forecast_regressors_1(conditions, h, n, p, q):

    """
    varma_conditional_forecast_regressors_1(conditions, h, Y, n, p)
    first set of elements for conditional forecasts: iteration-invariant 
    
    parameters:
    conditions : ndarray of shape (nconditions,4)
        matrix of conditions (one row per condition: variable, period, mean, variance)
    h : int
        number of forecast periods         
    n : int
        number of endogenous variables 
    p : int
        number of lags 
    q : int
        number of residual lags         
    
    returns:
    y_bar : ndarray of shape (h,n)
        matrix of mean values for conditions
    Q : ndarray of shape (n,n*(p+q))
        selection matrix for conditional forecasts state-space representation        
    omega : ndarray of shape (h,n)
        matrix of variance values for conditions
    """
    
    y_bar = np.zeros((h,n))
    omega = 10e6 * np.ones((h,n))
    for i in range(conditions.shape[0]):
        variable = int(conditions[i,0] - 1)
        period = int(conditions[i,1] - 1)
        mean = conditions[i,2]
        variance = max(1e-10,conditions[i,3])
        y_bar[period,variable] = mean
        omega[period,variable] = variance
    Q = np.eye(n,n*(p+q))
    return y_bar, Q, omega


def varma_conditional_forecast_regressors_2(Y, E, B, Kappa, Sigma, conditions, Z_p, n, m, p, q, h):
    
    """
    varma_conditional_forecast_regressors_2(Y, E, B, Kappa, Sigma, conditions, Z_p, n, m, p, q, h)
    second set of elements for conditional forecasts: iteration-specific
    
    parameters:    
        
    Y : ndarray of shape (p,n)
        matrix of initial conditions for exogenous   
    E : ndarray of shape (q,n)
        matrix of initial conditions for residuals      
    B : ndarray of shape (k1,n)
        matrix of VAR coefficients
    Kappa : ndarray of shape (k2,n)
        matrix of residual coefficients 
    Sigma : ndarray of shape (n,n)
        variance-covariance matrix of VAR residuals   
    conditions : ndarray of shape (nconditions,4)
        matrix of conditions (one row per condition: variable, period, mean, variance)        
    Z_p : ndarray of shape (h,m)
        matrix of exogenous regressors for forecasts
    n : int
        number of endogenous variables 
    m : int
        number of exogenous variables          
    p : int
        number of lags 
    q : int
        number of residual lags         
    h : int
        number of forecast periods 
    
    returns:
    mu : ndarray of shape (h,n*(p+q))
        matrix of intercepts for state variables
    F : ndarray of shape (n*p,n*p)
        companion form matrix
    K : ndarray of shape (n*p,n*p,h)
        variance-covariance matrix for state errors
    gamma_00 : ndarray of shape (n*p,)
        initial conditions (mean) for the space vector gamma_hat        
    Upsilon_00 : ndarray of shape (n*p,)
        initial conditions (variance) for the space vector gamma_hat
    """    
    
    F = make_varma_companion_form(B, Kappa, n, m, p, q)
    mu = np.zeros((h,n*(p+q)))
    mu[:,:n] = Z_p @ B[:m]
    K = np.zeros((n*(p+q),n*(p+q),h))
    selection = np.zeros((p+q,p+q))
    selection[0,0] = 1
    selection[0,p] = 1
    selection[p,0] = 1
    selection[p,p] = 1
    for i in range(h):
        temp = Sigma.copy()
        condition_variables = conditions[conditions[:,1] == (i+1)][:,0]
        for j in range(len(condition_variables)):
            variable = int(condition_variables[j])-1
            temp[variable,variable] = 100
        K[:,:,i] = np.kron(selection, temp)
    gamma_00 = np.hstack([la.vec(np.fliplr(Y.T)), la.vec(np.fliplr(E.T))])
    Upsilon_00 = np.kron(selection, Sigma) + 1e-10 * np.identity(n*(p+q))
    return mu, F, K, gamma_00, Upsilon_00    


def make_varma_companion_form(B, K, n, m, p, q):

    """
    make_varma_companion_form(B, K, n, m, p, q)
    creates companion form matix F as defined in (5.16.35)
    
    parameters:
    B : ndarray of shape (k1,n)
        matrix of VAR coefficients
    K : ndarray of shape (k2,n)
        matrix of residual coefficients 
    n : int
        number of endogenous variables
    m : int
        number of exogenous variables        
    p : int
        number of lags
    q : int
        number of residual lags         
    
    returns:
    F : ndarray of shape (n*p,n*p)
        companion form matrix
    """    

    block_1 = np.hstack([B[m:].T, K.T])
    block_2 = np.eye(n*(p+q-1),n*(p+q))
    F = np.vstack([block_1, block_2])
    if q == 1:
        F[-q*n:,-(q+1)*n:-q*n] = np.zeros((n,n))
    else:
        F[-q*n:-(q-1)*n,-(q+1)*n:-q*n] = np.zeros((n,n))
    return F


def varma_linear_forecast(B, K, h, Z_p, Y, E, n):

    """
    varma_linear_forecast(B, K, h, Z_p, Y, E, n)
    best linear forecasts for VARMA model, absent shocks
    
    parameters:
    B : ndarray of shape (k1,n)
        matrix of VAR coefficients
    K : ndarray of shape (k2,n)
        matrix of residual coefficients 
    h : int
        number of forecast periods
    Z_p : ndarray of shape (h,m)
        matrix of exogenous regressors for forecasts
    Y : ndarray of shape (p,n)
        matrix of initial conditions for endogenous variables 
    E : ndarray of shape (q,n)
        matrix of initial conditions for residuals         
    n : int
        number of endogenous variables        
    
    returns:
    Y_p : ndarray of shape (h,n)
        matrix of simulated forecast values
    """

    Y_p = np.zeros((h,n))
    for i in range(h):
        # get lagged endogenous regressors
        X = la.vec(Y[::-1].T).reshape(1,-1)
        # add exogenous regressors, if any
        if len(Z_p) != 0:
            X = np.hstack([Z_p[[i],:],X]) 
        # get lagged residuals
        Z = la.vec(E[::-1].T).reshape(1,-1)
        # generate forecasts
        y = X @ B + Z @ K
        # update Y, E and Y_p
        Y = np.vstack([Y[1:,:],y])
        E = np.vstack([E[1:,:],np.zeros(n)])
        Y_p[i,:] = y
    return Y_p

