# imports
import numpy as np
import numpy.random as nrd
import alexandria.math.linear_algebra as la
import alexandria.math.random_number_generators as rng
import warnings; warnings.filterwarnings('ignore')


# module state_space_utilities
# a module containing methods for state-space utilities


#---------------------------------------------------
# Methods
#---------------------------------------------------


def kalman_filter(X, A, Omega, C, B, Upsilon, T, n, k):
    
    """
    kalman_filter(X, A, Omega, C, B, Upsilon, T, n, k)
    Kalman filter to estimate the state variables of a general state-space model
    
    parameters:
    X : ndarray of shape (T,n)
        matrix of observed variables
    A : ndarray of shape (n,k,T)
        matrix of coefficients on observation equation
    Omega : ndarray of shape (n,n,T)
        variance-covariance matrix of observation errors
    C : ndarray of shape (T,k)
        intercept on observation equation  
    B : ndarray of shape (k,k,T)
        matrix of coefficients on state equation        
    Upsilon : ndarray of shape (k,k,T)
        variance-covariance matrix of state errors      
    T : int
        number of sample periods          
    n : int
        dimension of observation vector 
    k : int
        dimension of state vector         
        
    returns:
    Z_tt : ndarray of shape (T,k)
        matrix of state values z_t|t
    Z_tt1 : ndarray of shape (T,k)
        matrix of state values z_t|t-1 
    Ups_tt : ndarray of shape (k,k,T)
        matrix of state variance Upsilon_t|t       
    Ups_tt1 : ndarray of shape (k,k,T)
        matrix of state variance Upsilon_t|t-1          
    """       
    
    # initiate values
    z_t1t1 = np.zeros(k)
    Upsilon_t1t1 = np.zeros((k,k))
    Z_tt = np.zeros((T,k))
    Z_tt1 = np.zeros((T,k))
    Ups_tt = np.zeros((k,k,T))
    Ups_tt1 = np.zeros((k,k,T))
    # Kalman recursions
    for t in range(T):
        # period-specific parameters
        x_t = X[t]
        A_t = A[:,:,t]
        Omega_t = Omega[:,:,t]
        c_t = C[t]
        B_t = B[:,:,t]
        Upsilon_t = Upsilon[:,:,t]
        # step 1
        z_tt1 = c_t + B_t @ z_t1t1
        # step 2
        Upsilon_tt1 = B_t @ Upsilon_t1t1 @ B_t.T + Upsilon_t
        # step 3
        x_tt1 = A_t @ z_tt1
        # step 4
        Omega_tt1 = A_t @ Upsilon_tt1 @ A_t.T + Omega_t
        # Phi_t computation
        Phi_t = Upsilon_tt1 @ la.slash_inversion(A_t.T, Omega_tt1)
        # step 5
        z_tt = z_tt1 + Phi_t @ (x_t - x_tt1)
        # step 6
        Upsilon_tt = Upsilon_tt1 - Phi_t @ Omega_tt1 @ Phi_t.T
        # record and update for incoming period
        Z_tt[t] = z_tt
        Z_tt1[t] = z_tt1
        z_t1t1 = z_tt
        Ups_tt[:,:,t] = Upsilon_tt
        Ups_tt1[:,:,t] = Upsilon_tt1
        Upsilon_t1t1 = Upsilon_tt
    return Z_tt, Z_tt1, Ups_tt, Ups_tt1


def backward_pass(Z_tt, Z_tt1, Ups_tt, Ups_tt1, B, T, k):

    """
    backward_pass(Z_tt, Z_tt1, Ups_tt, Ups_tt1, B, T, k)
    Backward pass of Carter-Kohn algorithm (algorithm k.2)
    
    parameters:
    Z_tt : ndarray of shape (T,k)
        matrix of state values z_t|t
    Z_tt1 : ndarray of shape (T,k)
        matrix of state values z_t|t-1 
    Ups_tt : ndarray of shape (k,k,T)
        matrix of state variance Upsilon_t|t       
    Ups_tt1 : ndarray of shape (k,k,T)
        matrix of state variance Upsilon_t|t-1  
    B : ndarray of shape (k,k,T)
        matrix of coefficients on state equation            
    T : int
        number of sample periods          
    k : int
        dimension of state vector         
        
    returns:
    Z : ndarray of shape (T,k)
        matrix of sampled values for the state variables       
    """  
    
    # initiate values
    Z = np.zeros((T,k))
    # final period sampling
    z_TT = Z_tt[-1]
    Upsilon_TT = Ups_tt[:,:,-1]
    Z[-1] = rng.multivariate_normal(z_TT, Upsilon_TT)
    # backward pass, other periods
    for t in range(T-2,-1,-1):
        # period-specific parameters
        B_t1 = B[:,:,t+1]
        z_tt = Z_tt[t]
        z_t1t = Z_tt1[t+1]
        Upsilon_tt = Ups_tt[:,:,t]
        Upsilon_t1t = Ups_tt1[:,:,t+1]
        z_t1 = Z[t+1]
        # Xi_t computation
        Xi_t = Upsilon_tt @ la.slash_inversion(B_t1.T, Upsilon_t1t)
        # step 1
        z_bar_tt1 = z_tt + Xi_t @ (z_t1 - z_t1t)
        # step 2
        Upsilon_bar_tt1 = Upsilon_tt - Xi_t @ B_t1 @ Upsilon_tt
        # step 3
        Z[t] = rng.multivariate_normal(z_bar_tt1, Upsilon_bar_tt1)
    return Z


def conditional_forecast_kalman_filter(X, A, Omega, C, B, Upsilon, z_00, Upsilon_00, T, n, k):
    
    """
    conditional_forecast_kalman_filter(X, A, Omega, C, B, Upsilon, z_00, Upsilon_00, T, n, k)
    Kalman filter to estimate the state variables of a conditional forecast state-space model
    
    parameters:
    X : ndarray of shape (T,n)
        matrix of observed variables
    A : ndarray of shape (n,k)
        matrix of coefficients on observation equation
    Omega : ndarray of shape (T,n)
        variance-covariance matrix of observation errors
    C : ndarray of shape (T,k)
        intercept on observation equation  
    B : ndarray of shape (k,k)
        matrix of coefficients on state equation        
    Upsilon : ndarray of shape (k,k,T)
        variance-covariance matrix of state errors   
    z_00 : ndarray of shape (k,)
        initial conditions for state variables (mean)      
    Upsilon_00 : ndarray of shape (k,k)
        initial conditions for state variables (variance-covariance)          
    T : int
        number of sample periods          
    n : int
        dimension of observation vector 
    k : int
        dimension of state vector         
        
    returns:
    Z_tt : ndarray of shape (T,k)
        matrix of state values z_t|t
    Z_tt1 : ndarray of shape (T,k)
        matrix of state values z_t|t-1 
    Ups_tt : ndarray of shape (k,k,T)
        matrix of state variance Upsilon_t|t       
    Ups_tt1 : ndarray of shape (k,k,T)
        matrix of state variance Upsilon_t|t-1          
    """  
    
    # initiate values
    z_t1t1 = z_00
    Upsilon_t1t1 = Upsilon_00
    Z_tt = np.zeros((T,k))
    Z_tt1 = np.zeros((T,k))
    Ups_tt = np.zeros((k,k,T))
    Ups_tt1 = np.zeros((k,k,T))
    # Kalman recursions
    for t in range(T):
        # period-specific parameters
        x_t = X[t]
        Omega_t = np.diag(Omega[t])
        c_t = C[t]
        Upsilon_t = Upsilon[:,:,t]
        # step 1
        z_tt1 = c_t + B @ z_t1t1
        # step 2
        Upsilon_tt1 = B @ Upsilon_t1t1 @ B.T + Upsilon_t
        # step 3
        x_tt1 = A @ z_tt1
        # step 4
        Omega_tt1 = A @ Upsilon_tt1 @ A.T + Omega_t
        # Phi_t computation
        Phi_t = Upsilon_tt1 @ la.slash_inversion(A.T, Omega_tt1)
        # step 5
        z_tt = z_tt1 + Phi_t @ (x_t - x_tt1)
        # step 6
        Upsilon_tt = Upsilon_tt1 - Phi_t @ Omega_tt1 @ Phi_t.T
        # record and update for incoming period
        Z_tt[t] = z_tt
        Z_tt1[t] = z_tt1
        z_t1t1 = z_tt
        Ups_tt[:,:,t] = Upsilon_tt
        Ups_tt1[:,:,t] = Upsilon_tt1
        Upsilon_t1t1 = Upsilon_tt
    return Z_tt, Z_tt1, Ups_tt, Ups_tt1


def static_backward_pass(Z_tt, Z_tt1, Ups_tt, Ups_tt1, B, T, k):

    """
    static_backward_pass(Z_tt, Z_tt1, Ups_tt, Ups_tt1, B, T, k)
    Backward pass of Carter-Kohn algorithm (algorithm k.2) with static B
    
    parameters:
    Z_tt : ndarray of shape (T,k)
        matrix of state values z_t|t
    Z_tt1 : ndarray of shape (T,k)
        matrix of state values z_t|t-1 
    Ups_tt : ndarray of shape (k,k,T)
        matrix of state variance Upsilon_t|t       
    Ups_tt1 : ndarray of shape (k,k,T)
        matrix of state variance Upsilon_t|t-1  
    B : ndarray of shape (k,k)
        matrix of coefficients on state equation            
    T : int
        number of sample periods          
    k : int
        dimension of state vector         
        
    returns:
    Z : ndarray of shape (T,k)
        matrix of sampled values for the state variables       
    """  

    # initiate values
    Z = np.zeros((T,k))
    # final period sampling
    z_TT = Z_tt[-1]
    Upsilon_TT = Ups_tt[:,:,-1]
    Z[-1] = rng.multivariate_normal(z_TT, Upsilon_TT)
    # backward pass, other periods
    for t in range(T-2,-1,-1):
        # period-specific parameters
        z_tt = Z_tt[t]
        z_t1t = Z_tt1[t+1]
        Upsilon_tt = Ups_tt[:,:,t]
        Upsilon_t1t = Ups_tt1[:,:,t+1] + 1e-10 * np.eye(k)
        z_t1 = Z[t+1]
        # Xi_t computation
        Xi_t = Upsilon_tt @ la.slash_inversion(B.T, Upsilon_t1t)
        # step 1
        z_bar_tt1 = z_tt + Xi_t @ (z_t1 - z_t1t)
        # step 2
        Upsilon_bar_tt1 = Upsilon_tt - Xi_t @ B @ Upsilon_tt
        # step 3
        Z[t] = rng.multivariate_normal(z_bar_tt1, Upsilon_bar_tt1)
    return Z


def varma_forward_pass(X, A, B, Upsilon, z_00, Upsilon_00, T, n, k):
    
    """
    varma_forward_pass(X, A, B, Upsilon, z_00, Upsilon_00, T, n, k)
    forward pass for the state variables of a varma model
    
    parameters:
    X : ndarray of shape (T,n)
        matrix of observed variables
    A : ndarray of shape (n,k)
        matrix of coefficients on observation equation 
    B : ndarray of shape (k,k)
        matrix of coefficients on state equation        
    Upsilon : ndarray of shape (k,k)
        variance-covariance matrix of state errors   
    z_00 : ndarray of shape (k,)
        initial conditions for state variables (mean)      
    Upsilon_00 : ndarray of shape (k,k)
        initial conditions for state variables (variance-covariance)          
    T : int
        number of sample periods          
    n : int
        dimension of observation vector 
    k : int
        dimension of state vector         
        
    returns:
    Z_tt : ndarray of shape (T,k)
        matrix of state values z_t|t
    Z_tt1 : ndarray of shape (T,k)
        matrix of state values z_t|t-1 
    Ups_tt : ndarray of shape (k,k,T)
        matrix of state variance Upsilon_t|t       
    Ups_tt1 : ndarray of shape (k,k,T)
        matrix of state variance Upsilon_t|t-1          
    """  

    # initiate values
    z_t1t1 = z_00
    Upsilon_t1t1 = Upsilon_00
    Z_tt = np.zeros((T,k))
    Z_tt1 = np.zeros((T,k))
    Ups_tt = np.zeros((k,k,T))
    Ups_tt1 = np.zeros((k,k,T))
    # Kalman recursions
    for t in range(T):
        # period-specific parameters
        x_t = X[t]
        # step 1
        z_tt1 = B @ z_t1t1
        # step 2
        Upsilon_tt1 = B @ Upsilon_t1t1 @ B.T + Upsilon
        # step 3
        x_tt1 = A @ z_tt1
        # step 4
        Omega_tt1 = A @ Upsilon_tt1 @ A.T
        # Phi_t computation
        Phi_t = Upsilon_tt1 @ la.slash_inversion(A.T, Omega_tt1)
        # step 5
        z_tt = z_tt1 + Phi_t @ (x_t - x_tt1)
        # step 6
        Upsilon_tt = Upsilon_tt1 - Phi_t @ Omega_tt1 @ Phi_t.T
        # record and update for incoming period
        Z_tt[t] = z_tt
        Z_tt1[t] = z_tt1
        z_t1t1 = z_tt
        Ups_tt[:,:,t] = Upsilon_tt
        Ups_tt1[:,:,t] = Upsilon_tt1
        Upsilon_t1t1 = Upsilon_tt
    return Z_tt, Z_tt1, Ups_tt, Ups_tt1


def varma_backward_pass(Z_tt, Z_tt1, Ups_tt, Ups_tt1, B, T, k):

    """
    varma_backward_pass(Z_tt, Z_tt1, Ups_tt, Ups_tt1, B, T, k)
    backward pass for the state variables of a varma model
    
    parameters:
    Z_tt : ndarray of shape (T,k)
        matrix of state values z_t|t
    Z_tt1 : ndarray of shape (T,k)
        matrix of state values z_t|t-1 
    Ups_tt : ndarray of shape (k,k,T)
        matrix of state variance Upsilon_t|t       
    Ups_tt1 : ndarray of shape (k,k,T)
        matrix of state variance Upsilon_t|t-1  
    B : ndarray of shape (k,k)
        matrix of coefficients on state equation            
    T : int
        number of sample periods          
    k : int
        dimension of state vector         
        
    returns:
    Z : ndarray of shape (T,k)
        matrix of sampled values for the state variables       
    """  

    # initiate values
    Z = np.zeros((T,k))
    # final period sampling
    z_TT = Z_tt[-1]
    Upsilon_TT = Ups_tt[:,:,-1]
    Z[-1] = rng.multivariate_normal(z_TT, Upsilon_TT)
    # backward pass, other periods
    for t in range(T-2,-1,-1):
        # period-specific parameters
        z_tt = Z_tt[t]
        z_t1t = Z_tt1[t+1]
        Upsilon_tt = Ups_tt[:,:,t]
        Upsilon_t1t = Ups_tt1[:,:,t+1] + 1e-10 * np.eye(k)
        z_t1 = Z[t+1]
        # Xi_t computation
        Xi_t = Upsilon_tt @ la.slash_inversion(B.T, Upsilon_t1t)
        # step 1
        z_bar_tt1 = z_tt + Xi_t @ (z_t1 - z_t1t)
        # step 2
        Upsilon_bar_tt1 = Upsilon_tt - Xi_t @ B @ Upsilon_tt
        # step 3
        Z[t] = rng.multivariate_normal(z_bar_tt1, Upsilon_bar_tt1)
    return Z


def dfm_forward_pass(x_, Lambda, B, Upsilon, z_00, Upsilon_00, T, m, n, s, r, J):
    
    """
    dfm_forward_pass(x_, Lambda, B, Upsilon, z_00, Upsilon_00, T, m, n, s, r, J)
    forward pass for the state variables of a dfm model
    
    parameters:
    x_: list of size (T)
        list of non-NaN data for each sample period, defined in (6.18.10)      
    Lambda: ndarray of shape (n,m(s+1)+n(r+1))
        time invariant counterpart of matrix A_t, defined in (6.18.39)
    B: ndarray of shape (m(s+1)+n(r+1),m(s+1)+n(r+1))
        companion matrix B, defined in (6.18.39)        
    z_00: ndarray of shape (m(s+1)+n(r+1),)
        vector of mean for initial factor value
    Upsilon_00: ndarray of shape (m(s+1)+n(r+1),m(s+1)+n(r+1))
        matrix of variance for initial factor value  
    T: int
        total number of sample periods, defined in (6.18.1)
    m: int
        number of fundamental factors in the model, defined in (6.18.1)   
    n: int
        number of endogenous variables, defined in (6.18.1)
    s: int
        maximum number of factor lags, defined in (6.18.38)
    r: int
        number of lags in the residual AR process, defined in (6.18.7) 
    J: list of size (T)
        list of non-NaN variables for each sample period, defined in (6.18.3)        
        
    returns:
    Z_tt : ndarray of shape (T,m(s+1)+n(r+1),)
        matrix of state values z_t|t
    Z_tt1 : ndarray of shape (T,m(s+1)+n(r+1))
        matrix of state values z_t|t-1 
    Ups_tt : ndarray of shape (m(s+1)+n(r+1),m(s+1)+n(r+1),T)
        matrix of state variance Upsilon_t|t       
    Ups_tt1 : ndarray of shape (m(s+1)+n(r+1),m(s+1)+n(r+1),T)
        matrix of state variance Upsilon_t|t-1       
    """      
   
    # initiate values
    f_dim = m * (s + 1) + n * (r + 1)
    z_t1t1 = z_00
    Upsilon_t1t1 = Upsilon_00
    Z_tt = np.zeros((T,f_dim))
    Z_tt1 = np.zeros((T,f_dim))
    Ups_tt = np.zeros((f_dim,f_dim,T))
    Ups_tt1 = np.zeros((f_dim,f_dim,T))
    # Kalman recursions
    for t in range(T):
        # period-specific parameters
        x_t = x_[t]
        A_t = Lambda[J[t]]
        # step 1
        z_tt1 = B @ z_t1t1        
        # step 2
        Upsilon_tt1 = B @ Upsilon_t1t1 @ B.T + Upsilon        
        # step 3
        x_tt1 = A_t @ z_tt1        
        # step 4
        Omega_tt1 = A_t @ Upsilon_tt1 @ A_t.T  
        # Phi_t computation
        Phi_t = Upsilon_tt1 @ la.slash_inversion(A_t.T, Omega_tt1)        
        # step 5
        z_tt = z_tt1 + Phi_t @ (x_t - x_tt1)
        # step 6
        Upsilon_tt = Upsilon_tt1 - Phi_t @ Omega_tt1 @ Phi_t.T
        # record and update for incoming period
        Z_tt[t] = z_tt
        Z_tt1[t] = z_tt1
        z_t1t1 = z_tt
        Ups_tt[:,:,t] = Upsilon_tt
        Ups_tt1[:,:,t] = Upsilon_tt1
        Upsilon_t1t1 = Upsilon_tt
    return Z_tt, Z_tt1, Ups_tt, Ups_tt1


def dfm_backward_pass(Z_tt, Z_tt1, Ups_tt, Ups_tt1, B, T, m, s, n, r):
    
    """
    dfm_backward_pass(Z_tt, Z_tt1, Ups_tt, Ups_tt1, B, T, m, s, n, r)
    backward pass for the state variables of a dfm model
    
    parameters:
    Z_tt : ndarray of shape (m(s+1)+n(r+1),)
        matrix of state values z_t|t
    Z_tt1 : ndarray of shape (T,q(r+1))
        matrix of state values z_t|t-1 
    Ups_tt : ndarray of shape (q(r+1),q(r+1),T)
        matrix of state variance Upsilon_t|t       
    Ups_tt1 : ndarray of shape (q(r+1),q(r+1),T)
        matrix of state variance Upsilon_t|t-1  
    B: ndarray of shape (m(s+1)+n(r+1),m(s+1)+n(r+1))
        companion matrix B, defined in (6.18.39) 
    T: int
        total number of sample periods, defined in (6.18.1)
    m: int
        number of fundamental factors in the model, defined in (6.18.1)  
    s: int
        maximum number of factor lags, defined in (6.18.38)
    n: int
        number of endogenous variables, defined in (6.18.1)        
    r: int
        number of lags in the residual AR process, defined in (6.18.7)       
        
    returns:
    Z : ndarray of shape (T,m(s+1)+n(r+1))
        matrix of sampled values for the state variables      
    """      
    
    # initiate values
    f_dim = m * (s + 1) + n * (r + 1)
    Z = np.zeros((T,f_dim))
    # final period sampling
    z_TT = Z_tt[-1]
    Upsilon_TT = Ups_tt[:,:,-1] + 1e-8 * np.eye(f_dim)
    Z[-1] = rng.multivariate_normal(z_TT, Upsilon_TT)
    # backward pass, other periods
    for t in range(T-2,-1,-1):
        # period-specific parameters
        z_tt = Z_tt[t]
        z_t1t = Z_tt1[t+1]
        Upsilon_tt = Ups_tt[:,:,t]
        Upsilon_t1t = Ups_tt1[:,:,t+1] + 1e-8 * np.eye(f_dim)
        z_t1 = Z[t+1]
        # Xi_t computation
        Xi_t = Upsilon_tt @ la.slash_inversion(B.T, Upsilon_t1t)
        # step 1
        z_bar_tt1 = z_tt + Xi_t @ (z_t1 - z_t1t)
        # step 2
        Upsilon_bar_tt1 = Upsilon_tt - Xi_t @ B @ Upsilon_tt + 1e-8 * np.eye(f_dim)
        # step 3
        Z[t] = rng.multivariate_normal(z_bar_tt1, Upsilon_bar_tt1)
    return Z


def epsilon_forward_pass(x_, lamda, W, A, B, Upsilon, z_00, Upsilon_00, T, l, r, J):
    
    """
    dfm_forward_pass(x_, Lambda, B, Upsilon, z_00, Upsilon_00, T, m, n, s, r, J)
    forward pass for the state variables of a dfm model
    
    parameters:
    x_: list of size (T)
        list of non-NaN data for each sample period, defined in (6.18.10)      
    lamda: ndarray of shape (l,m(s+1))
        matrix of loadings, restricted equations, defined in theorem 18.1
    W: ndarray of shape (T,m(s+1))
        lagged factor matrix, defined in (6.18.11)         
    A: ndarray of shape (l,l(r+1))
        loadings on residuals, observation equation
    B: ndarray of shape (m(s+1)+n(r+1),m(s+1)+n(r+1))
        companion matrix on residuals, state equation
    z_00: ndarray of shape (l(r+1),)
        vector of mean for initial factor value
    Upsilon_00: ndarray of shape (l(r+1),l(r+1))
        matrix of variance for initial factor value  
    T: int
        total number of sample periods, defined in (6.18.1)
    l: int
        total number of loadings regressors, defined in (6.18.11)  
    r: int
        number of lags in the residual AR process, defined in (6.18.7) 
    J: list of size (T)
        list of non-NaN variables for each sample period, defined in (6.18.3)        
        
    returns:
    Z_tt : ndarray of shape (T,l(r+1))
        matrix of state values z_t|t
    Z_tt1 : ndarray of shape (T,l(r+1))
        matrix of state values z_t|t-1 
    Ups_tt : ndarray of shape (l(r+1),l(r+1),T)
        matrix of state variance Upsilon_t|t       
    Ups_tt1 : ndarray of shape (l(r+1),l(r+1),T)
        matrix of state variance Upsilon_t|t-1 
    """      

    # initiate values
    f_dim = l * (r + 1)   
    z_t1t1 = z_00
    Upsilon_t1t1 = Upsilon_00
    Z_tt = np.zeros((T,f_dim))
    Z_tt1 = np.zeros((T,f_dim))
    Ups_tt = np.zeros((f_dim,f_dim,T))
    Ups_tt1 = np.zeros((f_dim,f_dim,T))    
    # Kalman recursions
    for t in range(T):
        # period-specific parameters   
        J_t = J[t][J[t] < l]
        x_t = x_[t][:len(J_t)]
        f_t = lamda[J_t] @ W[t,:]
        x_t = x_t - f_t
        A_t = A[J_t]
        # step 1
        z_tt1 = B @ z_t1t1      
        # step 2
        Upsilon_tt1 = B @ Upsilon_t1t1 @ B.T + Upsilon       
        # if there are observations, run normal kalman steps
        if len(J_t) > 0:
            # step 3
            x_tt1 = A_t @ z_tt1        
            # step 4
            Omega_tt1 = A_t @ Upsilon_tt1 @ A_t.T + 1e-10 * np.eye(len(J_t))      
            # Phi_t computation
            Phi_t = Upsilon_tt1 @ la.slash_inversion(A_t.T, Omega_tt1)         
            # step 5
            z_tt = z_tt1 + Phi_t @ (x_t - x_tt1)    
            # step 6
            Upsilon_tt = Upsilon_tt1 - Phi_t @ Omega_tt1 @ Phi_t.T        
        # if there are no observations, run naive forecasts
        else:
            # step 5
            z_tt = z_tt1           
            # step 6
            Upsilon_tt = Upsilon_tt1       
        # record and update for incoming period
        Z_tt[t] = z_tt
        Z_tt1[t] = z_tt1
        z_t1t1 = z_tt
        Ups_tt[:,:,t] = Upsilon_tt
        Ups_tt1[:,:,t] = Upsilon_tt1
        Upsilon_t1t1 = Upsilon_tt        
    return Z_tt, Z_tt1, Ups_tt, Ups_tt1         
        

def epsilon_backward_pass(Z_tt, Z_tt1, Ups_tt, Ups_tt1, B, T, l, r):
    
    """
    Z_tt : ndarray of shape (T,l(r+1))
        matrix of state values z_t|t
    Z_tt1 : ndarray of shape (T,l(r+1))
        matrix of state values z_t|t-1 
    Ups_tt : ndarray of shape (l(r+1),l(r+1),T)
        matrix of state variance Upsilon_t|t       
    Ups_tt1 : ndarray of shape (l(r+1),l(r+1),T)
        matrix of state variance Upsilon_t|t-1     
    B: ndarray of shape (m(s+1)+n(r+1),m(s+1)+n(r+1))
        companion matrix on residuals, state equation    
    T: int
        total number of sample periods, defined in (6.18.1)
    l: int
        total number of loadings regressors, defined in (6.18.11)  
    r: int
        number of lags in the residual AR process, defined in (6.18.7)  
        
    returns:
    Z : ndarray of shape (T,l(r+1))
        matrix of sampled values for the state variables         
    """      

    # initiate values
    f_dim = l * (r + 1)
    Z = np.zeros((T,f_dim))
    # final period sampling
    z_TT = Z_tt[-1]
    Upsilon_TT = Ups_tt[:,:,-1] + 1e-10 * np.eye(f_dim)
    Z[-1] = rng.multivariate_normal(z_TT, Upsilon_TT)
    # backward pass, other periods
    for t in range(T-2,-1,-1):
        # period-specific parameters
        z_tt = Z_tt[t]
        z_t1t = Z_tt1[t+1]
        Upsilon_tt = Ups_tt[:,:,t]
        Upsilon_t1t = Ups_tt1[:,:,t+1] + 1e-10 * np.eye(f_dim)
        z_t1 = Z[t+1]
        # Xi_t computation
        Xi_t = Upsilon_tt @ la.slash_inversion(B.T, Upsilon_t1t)
        # step 1
        z_bar_tt1 = z_tt + Xi_t @ (z_t1 - z_t1t)
        # step 2
        Upsilon_bar_tt1 = Upsilon_tt - Xi_t @ B @ Upsilon_tt + 1e-10 * np.eye(f_dim)
        # step 3
        Z[t] = rng.multivariate_normal(z_bar_tt1, Upsilon_bar_tt1)
    return Z


def mfbvar_forward_pass(yo_, L_, F, mu_, n, p, Upsilon, T_, gamma_00, Upsilon_00):
      
    """
    mfbvar_forward_pass(yo_, L_, F, mu_, n, p, Upsilon, T_, gamma_00, Upsilon_00)
    forward pass for the state variables of the MF-BVAR model
    
    parameters:
    yo_: list of size (T_)
        list of non-NaN data yo_t for each sample period, defined in (6.17.16) 
    L_: list of size (T_)
        list of selection matrices L_t for each sample period, defined in (6.17.21)         
    F: ndarray of shape (n*p,n*p)
        companion matrix of VAR coefficients, defined in (6.17.20)
    mu_: list of size (T_)
        list of endogenous vectors for each sample period, defined in (6.17.19) 
    n : int
        number of endogenous variables, defined in (6.17.1)
    p : int
        number of lags in the VAR model, defined in (6.17.1)
    Upsilon: ndarray of shape (n*p,n*p)
        variance-covariance matrix for state equation
    T_ : int
        number of sample periods, including initial conditions
    gamma_00: ndarray of shape (n*p,)
        vector of mean for initial factor value
    Upsilon_00: ndarray of shape (n*p,n*p)
        matrix of variance for initial factor value  
        
    returns:
    Gamma_tt : ndarray of shape (T_,n*p)
        matrix of state values gamma_t|t
    Gamma_tt1 : ndarray of shape (T_,n*p)
        matrix of state values gamma_t|t-1 
    Ups_tt : ndarray of shape (n*p,n*p,T_)
        matrix of state variance Upsilon_t|t       
    Ups_tt1 : ndarray of shape (n*p,n*p,T_)
        matrix of state variance Upsilon_t|t-1       
    """     
    
    # initiate values
    f_dim = n * p
    gamma_t1t1 = gamma_00
    Upsilon_t1t1 = Upsilon_00
    Gamma_tt = np.zeros((T_,f_dim))
    Gamma_tt1 = np.zeros((T_,f_dim))
    Ups_tt = np.zeros((f_dim,f_dim,T_))
    Ups_tt1 = np.zeros((f_dim,f_dim,T_))
    # Kalman recursions
    for t in range(T_):
        # period-specific parameters
        yo_t = yo_[t]
        L_t = L_[t]
        mu_t = mu_[t,:]
        # step 1
        gamma_tt1 = mu_t + F @ gamma_t1t1        
        # step 2
        Upsilon_tt1 = F @ Upsilon_t1t1 @ F.T + Upsilon        
        # step 3
        yo_tt1 = L_t @ gamma_tt1
        # step 4
        Omega_tt1 = L_t @ Upsilon_tt1 @ L_t.T          
        # Phi_t computation
        Phi_t = Upsilon_tt1 @ la.slash_inversion(L_t.T, Omega_tt1) 
        # step 5
        gamma_tt = gamma_tt1 + Phi_t @ (yo_t - yo_tt1)        
        # step 6
        Upsilon_tt = Upsilon_tt1 - Phi_t @ Omega_tt1 @ Phi_t.T        
        # record and update for incoming period
        Gamma_tt[t] = gamma_tt
        Gamma_tt1[t] = gamma_tt1
        gamma_t1t1 = gamma_tt
        Ups_tt[:,:,t] = Upsilon_tt
        Ups_tt1[:,:,t] = Upsilon_tt1
        Upsilon_t1t1 = Upsilon_tt
    return Gamma_tt, Gamma_tt1, Ups_tt, Ups_tt1


def mfbvar_backward_pass(Gamma_tt, Gamma_tt1, Ups_tt, Ups_tt1, F, T_, n, p):
      
    """
    mfbvar_forward_pass(yo_, L_, F, mu_, n, p, Upsilon, T_, gamma_00, Upsilon_00)
    forward pass for the state variables of the MF-BVAR model
    
    parameters:
    Gamma_tt : ndarray of shape (T_,n*p)
        matrix of state values gamma_t|t
    Gamma_tt1 : ndarray of shape (T_,n*p)
        matrix of state values gamma_t|t-1 
    Ups_tt : ndarray of shape (n*p,n*p,T_)
        matrix of state variance Upsilon_t|t       
    Ups_tt1 : ndarray of shape (n*p,n*p,T_)
        matrix of state variance Upsilon_t|t-1  
    F: ndarray of shape (n*p,n*p)
        companion matrix of VAR coefficients, defined in (6.17.20)        
    T_ : int
        number of sample periods, including initial conditions        
    n : int
        number of endogenous variables, defined in (6.17.1)
    p : int
        number of lags in the VAR model, defined in (6.17.1)        

    returns:
    Gamma : ndarray of shape (T_,n*p)
        matrix of sampled values for the state variables       
    """      

    # initiate values
    f_dim = n * p
    Gamma = np.zeros((T_,f_dim))
    # final period sampling
    gamma_TT = Gamma_tt[-1]
    Upsilon_TT = Ups_tt[:,:,-1] + 1e-8 * np.eye(f_dim)
    Gamma[-1] = rng.multivariate_normal(gamma_TT, Upsilon_TT)
    # backward pass, other periods
    for t in range(T_-2,-1,-1):
        # period-specific parameters
        gamma_tt = Gamma_tt[t]
        gamma_t1t = Gamma_tt1[t+1]
        Upsilon_tt = Ups_tt[:,:,t]
        Upsilon_t1t = Ups_tt1[:,:,t+1] + 1e-8 * np.eye(f_dim)
        gamma_t1 = Gamma[t+1]
        # Xi_t computation
        Xi_t = Upsilon_tt @ la.slash_inversion(F.T, Upsilon_t1t)
        # step 1
        gamma_bar_tt1 = gamma_tt + Xi_t @ (gamma_t1 - gamma_t1t)
        # step 2
        Upsilon_bar_tt1 = Upsilon_tt - Xi_t @ F @ Upsilon_tt + 1e-8 * np.eye(f_dim)
        # step 3
        Gamma[t] = rng.multivariate_normal(gamma_bar_tt1, Upsilon_bar_tt1)
    return Gamma
