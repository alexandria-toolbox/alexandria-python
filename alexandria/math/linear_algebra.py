# imports
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import alexandria.math.math_utilities as mu


# module linear_algebra
# a module containing methods for linear algebra applications
    
    
#---------------------------------------------------
# Methods
#---------------------------------------------------


def cholesky_nspd(A):
    
    """
    cholesky_nspd(A)
    produces the Cholesky factor of either A, or the spd matrix nearest to A
    
    parameters:
    A : ndarray of shape (n,n)
        possibly symmetric positive definite
    
    returns:
    G : ndarray of shape (n,n)
        lower triangular Cholesky factor of A
    """
    
    # first test for Cholesky factorisation
    try:
        # if factorisation is succesful, don't go further
        G = nla.cholesky(A)
    except:
        # if factorisation fails, symmetrize
        B = (A + A.T) / 2
        # compute H, the symmetric polar factor of B
        _, S, V = nla.svd(B)
        H = V.T @ np.diag(S) @ V
        # try again to obtain an spd matrix
        C = (B + H) / 2
        C = (C + C.T) / 2
        # if matrix is only semi spd, modify it slightly until it becomes spd
        spd = False
        k = 0
        eye = np.identity(A.shape[0])
        while not spd:
            try:
                G = nla.cholesky(C)
                spd = True
            except:
                k += 1
                mineig = min(nla.eig(C)[0].real);
                C += (-mineig * k ** 2 + mu.eps()) * eye 
    return G


def invert_spd_matrix(A):
    
    """
    invert_spd_matrix(A)
    produces the inverse of A, where A is symmetric and positive definite
    based on property m.35
    
    parameters:
    A : ndarray of shape (n,n)
        invertible, symmetric and positive definite
        
    returns:
    A_inverse : ndarray of shape (n,n)
        inverse of A
    """
    
    # dimension of A
    dimension = A.shape[0]
    # take the Cholesky factor of A and invert it
    G = cholesky_nspd(A)
    G_inverse = sla.solve_triangular(G, np.identity(dimension), lower = True)
    # recover inverse of A
    A_inverse = G_inverse.T @ G_inverse
    return A_inverse
 
  
def triangular_factorization(X):
    
    """
    triangular_factorization(X)
    triangular factorization of spd matrix
    based on property m.34
    
    parameters:
    X: ndarray of shape (n,n)
        matrix to factor (symmetric and positive definite)
        
    returns:
    F: ndarray of shape (n,n)
        unit lower triangular factor
    L: ndarray of shape (n,)
        diagonal factor (only reports the diagonal)
    """

    G = cholesky_nspd(X)
    diagonal = np.diag(G)
    F = G / diagonal
    L = diagonal ** 2
    return F, L


def vec(X):
    
    """
    vec(X)
    vectorize a matrix, column by column
    
    parameters:
    X : ndarray of shape (n,m)
        matrix to vectorize
        
    returns:
    x : ndarray of shape (n*m,)
        vector resulting from vectorization
    """
    
    x = X.flatten('F')
    return x


def stable_determinant(A):
    
    """
    stable_determinant(A)
    stable (log) determinant of a matrix of the form I + A
    uses property m.59
    
    parameters:
    A : ndarray of shape (n,n)
        matrix A in the form I + A
        
    returns:
    det : float
        log determinant of matrix I + A
    """
    
    w, _ = nla.eig(A)
    det = np.real(np.sum(np.log(1 + w)))
    return det


def lag_matrix(X, lags):
    
    """
    calculates arrays of present and lagged values of X
    
    parameters:
    X : ndarray of shape (n,k) or (n,)
        array to be lagged
        
    lags : int
        number of lags to apply
    
    returns:
    X_present : ndarray of shape (n-lags,k) or (n-lags,)
        array containing present values
    
    X_lagged : ndarray of shape (n-lags,k*lags) or (n-lags,lags)
        array containing lagged values
    """
    
    # array of present values
    X_present = X[lags:].copy()
    # if X is 1-dimensional array, convert first to 2-d arrays
    if X.ndim == 1:
        X = X.reshape(-1,1)
    # first lag
    X_lagged = X[lags-1:-1].copy()
    # concatenate remaining lags
    for i in range(1, lags):
        X_lagged = np.hstack((X_lagged, X[lags-1-i:-1-i]))
    return X_present, X_lagged


def lag_polynomial(X, gamma):
    
    """
    Applies lag polynomial, defined in (3.9.56)

    parameters:
    X : ndarray of shape (n,k) or (n,)
        array on which lag polynomial applies
        
    gamma: ndarray of shape (lags,)
        coefficients of the lag polynomial
    
    returns:
    Y : ndarray of shape (n-lags,k)
        array obtained after applying the lag polynomial
    """
    
    # number of lags
    lags = gamma.shape[0]
    # create present and lagged arrays
    Y, Z = lag_matrix(X, lags)
    # if Y is 1d, apply lag polynomial directly
    if Y.ndim == 1:
        Y -= Z @ gamma
    # else if Y is 2d, looping over lags is necessary
    else:
        columns = X.shape[1]
        for i in range(lags):
            Y -= gamma[i] * Z[:,columns*i:columns*(i+1)]
    return Y






