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


