# imports
import numpy as np
import numpy.linalg as nla
import numpy.random as nrd
import scipy.linalg as sla
import alexandria.math.linear_algebra as la
import alexandria.math.stat_utilities as su
import alexandria.math.random_number_generators as rng
import alexandria.state_space.state_space_utilities as ss



# module var_utilities
# a module containing methods for vector autoregression utilities


#---------------------------------------------------
# Methods
#---------------------------------------------------


def generate_intercept_and_trends(constant, trend, quadratic_trend, periods, shift):
    
    """
    generate_intercept_and_trends(constant, trend, quadratic_trend, periods, shift)
    generates automated exogenous regressors (constant, trend, quadratic trend)
    
    parameters:
    constant : bool
        if True, a constant is added to the VAR model
    trend : bool
        if True, a linear trend is added to the VAR model
    quadratic_trend : bool
        if True, a quadratic trend is added to the VAR model
    periods : int
        number of periods for which regressors are generated
    shift : int
        number of periods to add to regressors (e.g. for predictions)
        
    returns:
    X : ndarray of shape (periods,)
        matrix of automated exogenous regressors
    """        
    
    X = np.empty([periods,0])
    if constant:
        constant_column = np.ones((periods,1))
        X = np.hstack((X,constant_column))
    if trend:
        trend_column = np.arange(1,1+periods).reshape(-1,1) + shift
        X = np.hstack((X,trend_column))
    if quadratic_trend:
        quadratic_trend_column = (np.arange(1,1+periods).reshape(-1,1) + shift) ** 2
        X = np.hstack((X,quadratic_trend_column))
    return X        


def generate_exogenous_regressors(exogenous, lags, periods):
    
    """
    generate_exogenous_regressors(exogenous, lags, periods)
    generate matrix of in-sample exogenous regressors
    
    parameters:
    exogenous : ndarray of shape (periods,n_exogenous)
        matrix of exogenous regressors
    lags: int
        number of lags in VAR model
    periods : int
        number of periods for which regressors are generated
        
    returns:
    X : ndarray of shape (periods,n_exogenous)
        matrix of exogenous regressors
    """   

    if len(exogenous) == 0:
        X = np.empty([periods,0])
    else:
        X = exogenous[lags:]
    return X


def generate_lagged_endogenous(endogenous, lags):
    
    """
    generate_lagged_endogenous(endogenous, lags)
    generate in-sample matrix of lagged endogenous regressors
    
    parameters:
    endogenous : ndarray of shape (periods,n_endogenous)
        matrix of endogenous regressors
    lags: int
        number of lags in VAR model
        
    returns:
    X : ndarray of shape (periods,n_endogenous)
        matrix of endogenous regressors
    """   
    
    X = []
    for i in range(lags):
        X.append(endogenous[lags-i-1:-i-1,:])
    X = np.hstack(X)
    return X
    

def fit_and_residuals(Y, X, B):

    """
    fit_and_residuals(Y, X, B)
    generates fit and residuals for a VAR model, using (4.11.2)
    
    parameters:
    Y : ndarray of shape (T,n)
        matrix of endogenous variables
    X : ndarray of shape (T,k)
        matrix of VAR regressors
    B : ndarray of shape (k,n)
        matrix of VAR coefficients
    
    returns:
    Y_hat : ndarray of shape (T,n)
        matrix of fitted endogenous
    E : ndarray of shape (T,n)
        matrix of VAR residuals
    """    

    Y_hat = X @ B
    E = Y - Y_hat
    return E, Y_hat
    

def insample_evaluation_criteria(Y, E, T, k):

    """
    insample_evaluation_criteria(Y, E, T, k)
    computes ssr, R2 and adjusted R2 for each VAR equation
    
    parameters:
    Y : ndarray of shape (T,n)
        matrix of endogenous variables
    E : ndarray of shape (T,n)
        matrix of VAR residuals
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


def steady_state(Z, B, n, m, p, T):
    
    """
    steady_state(Z, B, n, m, p, T)
    computes the steady-state of the VAR model, using equation (4.12.30)
    
    parameters:
    Z : ndarray of shape (T,m)
        matrix of exogenous variables
    B : ndarray of shape (m+n*p,n)
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
        C = B[:m,:]
        A = np.eye(n)
        for i in range(p):
            A -= B[m+i*n:m+(i+1)*n]
        steady_state = Z @ la.slash_inversion(C, A)
    return steady_state


def structural_shocks(E, inv_H):

    """
    structural_shocks(E, inv_H)
    computes the structural shocks of the VAR model, using equation (4.13.9)
    
    parameters:
    E : ndarray of shape (T,n)
        matrix of VAR residuals
    inv_H : ndarray of shape (n,n)
        inverse of structural matrix H
        
    returns:
    Xi : ndarray of shape (T,n)
        matrix of structural shocks
    """    
    
    Xi = E @ inv_H.T
    return Xi


def make_forecast_regressors(Z_p, Y, h, p, T, exogenous, constant, trend, quadratic_trend):
    
    """
    make_forecast_regressors(Z_p, Y, h, p, T, constant, trend, quadratic_trend)
    create regressors for forecast estimation
    
    parameters:
    Z_p : ndarray of shape (h,m) or empty list
        matrix of predicted exogenous
    Y : ndarray of shape (T,n)
        matrix of endogenous variables
    h : int
        number of forecast periods
    p : int
        number of lags
    T : int
        number of sample periods   
    exogenous : ndarray of shape (T,n_exogenous)
        matrix of exogenous regressors
    constant : bool
        if True, a constant is added to the VAR model
    trend : bool
        if True, a linear trend is added to the VAR model
    quadratic_trend : bool
        if True, a quadratic trend is added to the VAR model
        
    returns:
    Z_p : ndarray of shape (periods,m)
        full matrix of endogenous regressors
    Y : ndarray of shape (lags,n)
        matrix of endogenous variables for initial conditions
    """  

    temp = generate_intercept_and_trends(constant, trend, quadratic_trend, h, T)
    # if no exogenous, return empty list
    if exogenous == []:
        Z_p = []
    elif Z_p == []:
        Z_p = np.tile(exogenous[-1],[h,1])
    if len(Z_p) != 0:
        Z_p = np.hstack([temp,Z_p])
    elif any([constant, trend, quadratic_trend]):
        Z_p = temp
    else:
        Z_p = []
    Y = Y[-p:,:]
    return Z_p, Y
        

def forecast_evaluation_criteria(Y_hat, Y):
    
    """
    forecast_evaluation_criteria(Y_hat, Y)
    estimates RMSE, MAE, MAPE, Theil-U and bias for forecasts
    
    parameters:
    Y_p : ndarray of shape (periods,n)
        matrix of predicted endogenous
    Y : ndarray of shape (periods,n)
        matrix of actual endogenous values
        
    returns:
    forecast_evaluation_criteria : dict
        dictionary storing forecast evaluation criteria
    """      
    
    # check dimensions
    if Y.shape != Y_hat.shape:
        raise TypeError('Cannot calculate forecast evaluation criteria. Forecasts and actual values have different dimensions.')
    h = Y_hat.shape[0]
    # calculate forecast error
    err = Y - Y_hat
    # calculate RMSE, MAE and MAPE from (4.13.18)
    rmse = np.sqrt(((err ** 2).sum(axis=0) / h))
    mae = (np.abs(err).sum(axis=0) / h)
    mape = 100 * np.abs(err / Y).sum(axis=0) / h
    # calculate Theil-U and bias from (4.13.19)
    theil_u = np.sqrt((err ** 2).sum(axis=0)) / (np.sqrt((Y ** 2).sum(axis=0)) + np.sqrt((Y_hat ** 2).sum(axis=0)))
    bias = err.sum(axis=0) / np.abs(err).sum(axis=0)
    # store in dictionary
    forecast_evaluation_criteria = {}
    forecast_evaluation_criteria['rmse'] = rmse
    forecast_evaluation_criteria['mae'] = mae
    forecast_evaluation_criteria['mape'] = mape
    forecast_evaluation_criteria['theil_u'] = theil_u
    forecast_evaluation_criteria['bias'] = bias
    return forecast_evaluation_criteria


def bayesian_forecast_evaluation_criteria(mcmc_forecast, Y):
    
    """
    bayesian_forecast_evaluation_criteria(mcmc_forecast, Y)
    estimates log scores and CRPS for forecasts
    
    parameters:
    mcmc_forecast : ndarray of shape (periods,n,iterations)
        matrix of Gibbs sampler forecast values
    Y : ndarray of shape (periods,n)
        matrix of actual endogenous values
        
    returns:
    bayesian forecast_evaluation_criteria : dict
        dictionary storing forecast evaluation criteria
    """      
    
    # check dimensions
    if Y.shape != mcmc_forecast.shape[:2]:
        raise TypeError('Cannot calculate forecast evaluation criteria. Forecasts and actual values have different dimensions.')
    h, n = mcmc_forecast.shape[0], mcmc_forecast.shape[1]
    # initiate storage
    log_pdf = np.zeros((h,n))
    crps = np.zeros((h,n))
    # log scores for individual periods
    mu_hat = np.mean(mcmc_forecast,axis=2)
    sigma_hat = np.var(mcmc_forecast,axis=2)
    for i in range(h):
        for j in range(n):
            log_pdf[i,j], _ = su.normal_pdf(Y[i,j], mu_hat[i,j], sigma_hat[i,j])
    log_score = - log_pdf
    # log scores for joint periods
    if h == 1:
        joint_log_score = log_score
    else:
        joint_log_pdf = np.zeros(n)
        for i in range(n):
            Sigma = np.cov(mcmc_forecast[:,i,:])
            joint_log_pdf[i], _ = su.multivariate_normal_pdf(Y[:,i], mu_hat[:,i], Sigma)
        joint_log_score = - joint_log_pdf
    # CRPS for individual periods
    for i in range(h):
        for j in range(n):
            crps[i,j] = make_crps(Y[i,j], mcmc_forecast[i,j,:]) 
    # CRPS for joint periods
    joint_crps = np.sum(crps,axis=0)
    # store in dictionary
    forecast_evaluation_criteria = {}
    forecast_evaluation_criteria['log_score'] = log_score
    forecast_evaluation_criteria['joint_log_score'] = joint_log_score
    forecast_evaluation_criteria['crps'] = crps
    forecast_evaluation_criteria['joint_crps'] = joint_crps    
    return forecast_evaluation_criteria


def make_crps(y, y_hat):
    
    """
    make_crps(y, y_hat)
    continuous rank probability score for prediction y
    
    parameters:
    y : float
        actual (observed) value for the foecast
    y_hat : ndarray of shape (iteration,)
        vector of MCMC simulated values for predictions
        
    returns:
    crps : float
        crps value
    """   
    
    J = y_hat.shape[0]
    term_1 = np.sum(np.abs(y_hat - y)) / J
    temp = np.tile(y_hat.reshape(-1,1), J)
    term_2 = np.sum(np.sum(np.abs(temp - temp.T))) / (2 * J * J)
    crps = term_1 - term_2
    return crps



def sums_of_coefficients_extension(sums_of_coefficients, pi5, Y, n, m, p):

    """
    sums_of_coefficients_extension(sums_of_coefficients, pi5, Y, n, m, p)
    generates dummy extension matrices Y_sum and X_sum as defined in (4.12.6)
    
    parameters:
    sums_of_coefficients : bool
        if True, the sums-of-coefficients extension is added to the model
    pi5 : float
        prior shrinkage for the sums-of-coefficients extension
    Y : ndarray of shape (T,n)
        matrix of endogenous variables
    n : int
        number of endogenous variables
    m : int
        number of exogenous variables        
    p : int
        number of lags
    
    returns:
    Y_sum : ndarray of shape (n,n) or (0,n)
        Y matrix for sums-of-coefficients extension
    X_sum : ndarray of shape (n,m+n*p) or (0,m+n*p)
        X matrix for sums-of-coefficients extension        
    """    

    if sums_of_coefficients:
        Y_sum = np.diag(np.mean(Y,axis=0) / pi5)
        X_sum = np.hstack([np.zeros([n,m]),np.kron(np.ones(p),Y_sum)])
    else:
        Y_sum = np.empty((0,n))
        X_sum = np.empty((0,m+p*n))
    return Y_sum, X_sum


def dummy_initial_observation_extension(dummy_initial_observation, pi6, Y, X, n, m, p):

    """
    dummy_initial_observation_extension(dummy_initial_observation, pi6, Y, X, n, m, p)
    generates dummy extension matrices Y_obs and X_obs as defined in (4.12.10)
    
    parameters:
    dummy_initial_observation : bool
        if True, the dummy initial observation extension is added to the model
    pi6 : float
        prior shrinkage for the dummy initial observation extension
    Y : ndarray of shape (T,n)
        matrix of endogenous variables
    X : ndarray of shape (T,k)
        matrix of VAR regressors        
    n : int
        number of endogenous variables
    m : int
        number of exogenous variables        
    p : int
        number of lags
    
    returns:
    Y_obs : ndarray of shape (1,n) or (0,n)
        Y matrix for dummy initial observation extension
    X_obs : ndarray of shape (1,m+n*p) or (0,m+n*p)
        X matrix for dummy initial observation extension        
    """       
    
    if dummy_initial_observation:
        Y_obs = np.mean(Y,axis=0) / pi6
        X_obs = np.hstack([np.mean(X[:,:m],axis=0) / pi6, np.kron(np.ones(p),Y_obs)])
    else:
        Y_obs = np.empty((0,n))
        X_obs = np.empty((0,m+p*n))
    return Y_obs, X_obs


def long_run_prior_extension(long_run_prior, pi7, J, Y, n, m, p):

    """
    long_run_prior_extension(long_run_prior, pi7, H, Y, n, m, p)
    generates dummy extension matrices Y_lrp and X_lrp as defined in (4.12.16)
    
    parameters:
    long_run_prior : bool
        if True, the long run prior extension is added to the model
    pi7 : float
        prior shrinkage for the long run prior extension
    J : ndarray of shape (n,n)
        matrix of long-run relations          
    Y : ndarray of shape (T,n)
        matrix of endogenous variables
    n : int
        number of endogenous variables
    m : int
        number of exogenous variables        
    p : int
        number of lags
    
    returns:
    Y_lrp : ndarray of shape (n,n) or (0,n)
        Y matrix for long run prior extension
    X_lrp : ndarray of shape (n,m+n*p) or (0,m+n*p)
        X matrix for long run prior extension        
    """    

    if long_run_prior:
        Y_lrp = la.slash_inversion(np.diag(J @ np.mean(Y,axis=0) / pi7) , J.T)
        X_lrp = np.hstack([np.zeros([n,m]),np.kron(np.ones(p),Y_lrp)])
    else:
        Y_lrp = np.empty((0,n))
        X_lrp = np.empty((0,m+p*n))
    return Y_lrp, X_lrp


def make_b(delta, n, m, p):
    
    """
    make_b(delta, n, m, p)
    generates Minnesota prior parameter b as defined in (4.11.16)
    
    parameters:
    delta : ndarray of shape (n,)
        array of prior AR coefficients
    n : int
        number of endogenous variables
    m : int
        number of exogenous variables        
    p : int
        number of lags
    
    returns:
    b : ndarray of shape (q,)
        prior mean for beta   
    """     
    
    b = la.vec(np.vstack((np.zeros((m,n)), np.diag(delta), np.zeros((n*(p-1),n)))))
    return b


def make_V(s, pi1, pi2, pi3, pi4, n, m, p):
    
    """
    make_V(s, pi1, pi2, pi3, pi4, n, m, p)
    generates Minnesota prior parameter V as defined in (4.11.17)-(4.11.20)
    
    parameters:
    s : ndarray of shape (n,)
        array of individual residual variances        
    pi1 : float
        overall tightness
    pi2 : float
        cross-variable shrinkage        
    pi3 : float
        lag decay
    pi4 : float
        exogenous slackness
    n : int
        number of endogenous variables
    m : int
        number of exogenous variables        
    p : int
        number of lags
    
    returns:
    V : ndarray of shape (q,)
        prior variance (diagonal term only) for beta    
    """      
    
    scale = np.vstack((np.tile(s,[m,1]), np.tile(np.tile(s,[n,1]) / np.tile(s.reshape(-1,1),[1,n]), [p,1])))
    shrinkage = (pi1 * np.vstack((pi4 * np.ones((m,n)), np.kron(1 / (np.arange(p) + 1) \
                .reshape(-1,1) ** pi3, pi2 * np.ones((n,n)) + (1 - pi2) * np.eye(n)) ))) ** 2
    V = la.vec(scale * shrinkage)
    return V


def make_V_b_inverse(b, V):

    """
    make_V_b_inverse(b, V)
    generates elements inv_V and inv_B * b used in (4.11.15) and (4.11.43)
    
    parameters:
    b : ndarray of shape (q,)
        prior mean for beta 
    V : ndarray of shape (q,)
        prior variance (diagonal term only) for beta          
        
    
    returns:
    inv_V : ndarray of shape (q,q)
        inverse prior variance for beta
    inv_V_b : ndarray of shape (q,)
        product inv_V * b
    """   
    
    inv_V = np.diag(1 / V)
    inv_V_b = b / V
    return inv_V, inv_V_b


def minnesota_posterior(inv_V, inv_V_b, XX, XY, inv_Sigma):    
    
    """
    minnesota_posterior(inv_V, inv_V_b, XX, XY, inv_Sigma)
    generates Minnesota posterior parameters b_bar and V_bar as defined in (4.11.15)
    
    parameters:
    inv_V : ndarray of shape (q,q)
        inverse prior variance for beta
    inv_V_b : ndarray of shape (q,)
        product inv_V * b
    XX : ndarray of shape (k,k)
        matrix product X' X     
    XY : ndarray of shape (k,n)
        matrix product X' Y         
    inv_Sigma : ndarray of shape (n,n)
        inverse of residual variance-covariance matrix Sigma     

    returns:
    b_bar : ndarray of shape (q,)
        posterior mean for beta  
    V_bar : ndarray of shape (q,q)
        posterior variance for beta 
    inv_V_bar : ndarray of shape (q,q)
        inverse posterior variance for beta 
    """

    inv_V_bar = inv_V + np.kron(inv_Sigma, XX)
    V_bar = la.invert_spd_matrix(inv_V_bar)
    b_bar = V_bar @ (inv_V_b + la.vec(XY @ inv_Sigma))
    return b_bar, V_bar, inv_V_bar


def make_B(delta, n, m, p):
    
    """
    make_B(delta, n, m, p)
    generates normal-Wishart prior parameter B as defined in (4.11.33)
    
    parameters:
    delta : ndarray of shape (n,)
        array of prior AR coefficients
    n : int
        number of endogenous variables
    m : int
        number of exogenous variables        
    p : int
        number of lags
    
    returns:
    B : ndarray of shape (k,n)
        prior mean for beta   
    """     
    
    B = np.vstack((np.zeros((m,n)), np.diag(delta), np.zeros((n*(p-1),n))))
    return B


def make_W(s, pi1, pi3, pi4, n, m, p):
    
    """
    make_W(s, pi1, pi3, pi4, n, m, p)
    generates normal-Wishart prior parameter W as defined in (4.11.27)
    
    parameters:
    s : ndarray of shape (n,)
        array of individual residual variances        
    pi1 : float
        overall tightness      
    pi3 : float
        lag decay
    pi4 : float
        exogenous slackness
    n : int
        number of endogenous variables
    m : int
        number of exogenous variables        
    p : int
        number of lags
    
    returns:
    W : ndarray of shape (k,)
        prior variance (diagonal term only) for beta    
    """      

    scale =  np.hstack([np.ones(m), np.tile(1/s,p)])
    shrinkage =  (pi1 * np.hstack([pi4 * np.ones(m), 1 / np.kron(np.arange(p)+1, np.ones(n)) ** pi3])) ** 2
    W = scale * shrinkage
    return W


def make_alpha(n):
    
    """
    make_alpha(n)
    generates normal-Wishart prior parameter alpha as defined in (4.11.30)
    
    parameters:
    n : int
        number of endogenous variables
    
    returns:
    alpha : float
        prior degrees of freedom for Sigma   
    """     
    
    alpha = n + 2
    return alpha


def make_S(s):
    
    """
    make_S(s)
    generates normal-Wishart prior parameter S as defined in (4.11.30)
    
    parameters:
    s : ndarray of shape (n,)
        array of individual residual variances
    
    returns:
    S : ndarray of shape (n,)
        prior scale (diagonal term only) for Sigma  
    """     
    
    S = s
    return S


def normal_wishart_posterior(B, W, alpha, S, n, T, XX, XY, YY):    
    
    """
    normal_wishart_posterior(B, W, alpha, S, n, T, XX, XY, YY)
    generates normal-Wishart posterior parameters as defined in (4.11.33) and (4.11.38)
    
    parameters:
    B : ndarray of shape (k,n)
        prior mean for beta   
    W : ndarray of shape (k,)
        prior variance (diagonal term only) for beta         
    alpha : float
        prior degrees of freedom for Sigma  
    S : ndarray of shape (n,)
        prior scale (diagonal term only) for Sigma         
    n : int
        number of endogenous variables        
    T : int
        number of sample periods  
    XX : ndarray of shape (k,k)
        matrix product X' X     
    XY : ndarray of shape (k,n)
        matrix product X' Y 
    YY : ndarray of shape (n,n)
        matrix product Y' Y 

    returns:
    B_bar : ndarray of shape (k,n)
        posterior mean for beta  
    W_bar : ndarray of shape (k,k)
        posterior variance for beta 
    alpha_bar : float
        posterior degrees of freedom for Sigma  
    S_bar : ndarray of shape (n,n)
        posterior scale for Sigma  
    alpha_hat : float
        posterior degrees of freedom for B
    S_hat : ndarray of shape (n,n)
        posterior scale for B
    """

    inv_W_bar = np.diag(1 / W) + XX
    W_bar = la.invert_spd_matrix(inv_W_bar)
    B_bar = W_bar @ (B / W.reshape(-1,1) + XY)
    alpha_bar = alpha + T
    S_bar = np.diag(S) + YY + B.T @ (B / W.reshape(-1,1)) - B_bar.T @ inv_W_bar @ B_bar
    alpha_hat = alpha + T - n + 1
    S_hat = S_bar / alpha_hat
    return B_bar, W_bar, alpha_bar, S_bar, alpha_hat, S_hat


def make_constrained_coefficients(B, V, n, m, k, lags, constant, trend, \
                                  quadratic_trend, constrained_coefficients_table):
    
    """
    make_constrained_coefficients(B, V, n, m, k, lags, constant, trend, \
                                  quadratic_trend, constrained_coefficients_table)
        
    parameters:
    B : ndarray of shape (k,n)
        prior mean for beta (reshaped as k * n)
    V : ndarray of shape (k,n)
        prior variance for beta (reshaped as k * n)      
    n : int
        number of endogenous variables
    m : int
        number of exogenous variables  
    k : int
        number of coefficients in each VAR equation        
    lags : int
        number of lags 
    constant : bool
        if True, a constant is added to the VAR model
    trend : bool
        if True, a linear trend is added to the VAR model
    quadratic_trend : bool
        if True, a quadratic trend is added to the VAR model
    constrained_coefficients_table : ndarray of shape (n_constraints,5)
        table defining the constraints on VAR coefficients
        
    returns:
    new_b : ndarray of shape (q,)
        prior mean for beta with constraints applied
    new_V : ndarray of shape (q,)
        prior variance for beta with constraints applied
    """    
    
    new_B = B.copy()
    new_V = V.copy()
    for i in range(constrained_coefficients_table.shape[0]):
        variable = int(constrained_coefficients_table[i,0]-1)
        responding = constrained_coefficients_table[i,1]
        lag = constrained_coefficients_table[i,2]
        mean = constrained_coefficients_table[i,3]
        variance = constrained_coefficients_table[i,4]
        if responding == 0.1:
            new_B[0,variable] = mean
            new_V[0,variable] = variance
        elif responding == 0.2:
            new_B[int(constant),variable] = mean
            new_V[int(constant),variable] = variance            
        elif responding == 0.3:
            new_B[int(constant)+int(trend),variable] = mean
            new_V[int(constant)+int(trend),variable] = variance
        elif responding < 0:
            responding = - int(responding) -1
            new_B[int(constant)+int(trend)+int(quadratic_trend)+responding,variable] = mean
            new_V[int(constant)+int(trend)+int(quadratic_trend)+responding,variable] = variance
        else:
            responding = int(responding) - 1
            row = int(m + (lag-1) * n + responding)
            new_B[row,variable] = mean
            new_V[row,variable] = variance           
    new_b = la.vec(new_B)
    new_V = la.vec(new_V)
    return new_b, new_V
    

def check_stationarity(B, n, m, p):
    
    """
    check_stationarity(B, n, m, p)
    check for stability of VAR model as in definition 12.1, using companion form (4.12.27)
    
    parameters:
    B : ndarray of shape (m+n*p,n)
        matrix of VAR coefficients
    n : int
        number of endogenous variables
    m : int
        number of exogenous variables        
    p : int
        number of lags
    
    returns:
    stationary : bool
        if True, the VAR model is stationary
    """    

    F = make_companion_form(B, n, m, p)
    eigenvalues = np.flip(np.sort(np.abs(np.real(nla.eig(F)[0]))))
    max_eigenvalue = eigenvalues[0]
    if max_eigenvalue < 0.999:
        stationary = True
    else:
        stationary = False
    return stationary


def make_companion_form(B, n, m, p):
    
    """
    make_companion_form(B, n, m, p)
    creates companion form matix F as defined in (4.12.27)-(4.12.28)
    
    parameters:
    B : ndarray of shape (m+n*p,n)
        matrix of VAR coefficients
    n : int
        number of endogenous variables
    m : int
        number of exogenous variables        
    p : int
        number of lags
    
    returns:
    F : ndarray of shape (n*p,n*p)
        companion form matrix
    """      
    
    block_1 = B[m:].T
    block_2 = np.eye(n*(p-1))
    block_3 = np.zeros((n*(p-1),n))
    F = np.vstack([block_1, np.hstack([block_2, block_3])])
    return F
    

def impulse_response_function(B, n, p, h):
    
    """
    impulse_response_function(B, n, p, h)
    generates impulse response function for a given matrix B of VAR coefficients
    using equations (4.13.2)-(4.13.4)
    
    parameters:
    B : ndarray of shape (m+n*p,n)
        matrix of VAR coefficients
    n : int
        number of endogenous variables
    p : int
        number of lags
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
    for i in range(1,h):
        Xh = np.hstack([Yh,Xh[:,:-n]])
        Yh = Xh @ B
        irf[:,:,i] = Yh.T
    return irf


def exogenous_impulse_response_function(B, n, m, r, p, h):
    
    """
    exogenous_impulse_response_function(B, n, m, r, p, h)
    generates exogenous impulse response function for a given matrix B of VAR coefficients
    using equations (4.13.2) with exogenous regressors
    
    parameters:
    B : ndarray of shape (m+n*p,n)
        matrix of VAR coefficients
    n : int
        number of endogenous variables
    m : int
        number of exogenous variables          
    r : int
        number of exogenous variables other than constant, trend and quadratic trends        
    p : int
        number of lags
    h : int
        number of irf periods (including impact period)
    
    returns:
    irf : ndarray of shape (n,r,h)
        matrix of impulse response functions
    """    
    
    C = B[-n*p-r:-n*p]
    B = B[-n*p:]
    Yh = C
    irf = np.dstack([Yh, np.zeros((r,n,h-1))])
    Xh = np.zeros((r,n*p))    
    for i in range(1,h):
        Xh = np.hstack([Yh,Xh[:,:-n]])
        Yh = Xh @ B
        irf[:,:,i] = Yh
    irf = np.swapaxes(irf,0,1)
    return irf


def structural_impulse_response_function(irf, H, n):
    
    """
    structural_impulse_response_function(irf, H)
    generates structural impulse response function using equation (4.13.9)
    uses a vectorized form of (4.13.9) on all IRF periods to gain efficiency
    
    parameters:
    irf : ndarray of shape (n,n,h)
        matrix of impulse response functions
    H : ndarray of shape (n,n)
        structural identification matrix
    
    returns:
    structural_irf : ndarray of shape (n,n,h)
        matrix of structural impulse response functions
    """    

    temp = irf.transpose(0,2,1).reshape(-1,n) @ H
    structural_irf = temp.reshape(n,-1,n).transpose(0,2,1)
    return structural_irf


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
    posterior_estimates : ndarray of shape (n,m,3)
        matrix of posterior estimates
    """

    posterior_estimates = np.zeros((X.shape[0],X.shape[1],3))
    posterior_estimates[:,:,0] = np.quantile(X,0.5,axis=2)
    posterior_estimates[:,:,1] = np.quantile(X,(1-credibility_level)/2,axis=2)
    posterior_estimates[:,:,2] = np.quantile(X,(1+credibility_level)/2,axis=2)
    return posterior_estimates


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


def forecast(B, chol_Sigma, h, Z_p, Y, n):

    """
    forecast(B, chol_Sigma, h, Z_p, Y, n)
    products simulated forecasts
    
    parameters:
    B : ndarray of shape (k,n)
        matrix of VAR coefficients
    chol_Sigma : ndarray of shape (n,n)
        Cholesky factor of residual variance-covariance matrix Sigma
    h : int
        number of forecast periods
    Z_p : ndarray of shape (h,m)
        matrix of exogenous regressors for forecasts
    Y : ndarray of shape (p,n)
        matrix of initial conditions for endogenous variables  
    n : int
        number of endogenous variables        
    
    returns:
    Y_p : ndarray of shape (h,n)
        matrix of simulated forecast values
    """

    E = (chol_Sigma @ nrd.randn(n,h)).T
    Y_p = np.zeros((h,n))
    for i in range(h):
        # get lagged endogenous regressors
        X = la.vec(np.fliplr(Y.T)).reshape(1,-1)
        # add exogenous regressors, if any
        if len(Z_p) != 0:
            X = np.hstack([Z_p[[i],:],X])
        # recover residuals
        e = E[[i],:]
        # generate forecasts
        y = X @ B + e
        # update Y and Y_p
        Y = np.vstack([Y[1:,:],y])
        Y_p[i,:] = y
    return Y_p
    

def forecast_error_variance_decomposition(structural_irf, Gamma, n, h):
    
    """
    forecast_error_variance_decomposition(structural_irf, Gamma, n, h)
    products forecast error variance decomposition from structural IRFs, using algorithm 13.5
    
    parameters:
    structural_irf : ndarray of shape (n,n,h)
        matrix of structural impulse response functions
    Gamma : empty list or ndarray of shape (n,)
        structural shock variance (empty list if variance is 1)
    n : int
        number of endogenous variables          
    h : int
        number of forecast periods
    
    returns:
    fevd : ndarray of shape (n,n,h)
        matrix of forecast error variance decomposition
    """    

    cum_squared_irf = np.cumsum(structural_irf ** 2, axis=2)
    if len(Gamma) != 0:
        reshaped_Gamma = np.dstack([np.tile(Gamma,(n,1))] * h)
        cum_squared_irf =  reshaped_Gamma * cum_squared_irf
    total_variance = np.hstack([np.sum(cum_squared_irf,axis=1, keepdims = True)] * n)
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
    normalized_fevd_estimates : ndarray of shape (n,n,h,iterations)
        matrix of normalized posterior FEVD estimates
    """  

    point_estimate_contribution = np.sum(fevd_estimates[:,:,:,0],axis=1, keepdims = True)
    total_contribution = np.hstack([point_estimate_contribution] * fevd_estimates.shape[1])
    estimates_contribution = np.stack([total_contribution] * 3, axis=3)
    normalized_fevd_estimates = fevd_estimates / estimates_contribution
    return normalized_fevd_estimates


def historical_decomposition(structural_irf, structural_shocks, n, T):

    """
    historical_decomposition(structural_irf, structural_shocks, n, T)
    products historical decomposition from structural shocks and IRFs, using algorithm 13.6
    
    parameters:
    structural_irf : ndarray of shape (n,n,T)
        matrix of structural impulse response functions
    structural_shocks : ndarray of shape (T,n)
        matrix of structural shocks        
    n : int
        number of endogenous variables          
    T : int
        number of sample periods  
    
    returns:
    hd : ndarray of shape (n,n,T)
        matrix of historical decomposition
    """      

    reshaped_shocks = np.flip(np.transpose(np.dstack([structural_shocks] * n), [2, 1, 0]), 2)
    hd = np.zeros((n,n,T))
    for i in range(T):
        hd[:,:,i] = np.sum(structural_irf[:,:,:i+1] * reshaped_shocks[:,:,-(i+1):],axis=2)
    return hd
        
    
def conditional_forecast_regressors_1(conditions, h, Y, n, p):

    """
    conditional_forecast_regressors_1(conditions, h, Y, n, p)
    first set of elements for conditional forecasts: iteration-invariant 
    
    parameters:
    conditions : ndarray of shape (nconditions,4)
        matrix of conditions (one row per condition: variable, period, mean, variance)
    h : int
        number of forecast periods       
    Y : ndarray of shape (p,n)
        matrix of initial conditions for exogenous   
    n : int
        number of endogenous variables 
    p : int
        number of lags        
    
    returns:
    y_bar : ndarray of shape (h,n)
        matrix of mean values for conditions
    Q : ndarray of shape (n,n*p)
        selection matrix for conditional forecasts state-space representation        
    omega : ndarray of shape (h,n)
        matrix of variance values for conditions
    gamma_00 : ndarray of shape (n*p,)
        initial conditions (mean) for the space vector gamma_hat
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
    Q = np.hstack((np.eye(n),np.zeros((n,n*(p-1)))))
    gamma_00 = la.vec(np.fliplr(Y.T))
    return y_bar, Q, omega, gamma_00


def conditional_forecast_regressors_2(B, Sigma, conditions, Z_p, n, m, p, h):
    
    """
    conditional_forecast_regressors_2(B, Sigma, conditions, Z_p, n, m, p, h)
    second set of elements for conditional forecasts: iteration-specific
    
    parameters:    
    B : ndarray of shape (m+n*p,n)
        matrix of VAR coefficients
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
    h : int
        number of forecast periods 
    
    returns:
    mu : ndarray of shape (h,n*p)
        matrix of intercepts for state variables
    F : ndarray of shape (n*p,n*p)
        companion form matrix
    K : ndarray of shape (n*p,n*p,h)
        variance-covariance matrix for state errors
    Upsilon_00 : ndarray of shape (n*p,)
        initial conditions (variance) for the space vector gamma_hat
    """    

    F = make_companion_form(B, n, m, p)
    mu = np.zeros((h,n*p))
    mu[:,:n] = Z_p @ B[:m]
    K = np.zeros((n*p,n*p,h))
    for i in range(h):
        K[:n,:n,i] = Sigma
        condition_variables = conditions[conditions[:,1] == (i+1)][:,0]
        for j in range(len(condition_variables)):
            variable = int(condition_variables[j])-1
            K[variable,variable,i] = 100
    Upsilon_00 = 1e-10 * np.identity(n*p)
    Upsilon_00[:n,:n] = Sigma
    return mu, F, K, Upsilon_00    


def conditional_forecast_regressors_3(conditions, h, n):

    """
    conditional_forecast_regressors_3(conditions, h, n)
    first set of elements for structural conditional forecasts: iteration-invariant 
    
    parameters:
    conditions : ndarray of shape (nconditions,4)
        matrix of conditions (one row per condition: variable, period, mean, variance)
    h : int
        number of forecast periods 
    n : int
        number of endogenous variables       
        
    returns:
    R : ndarray of shape (n_conditions,n*h)
        selection matrix for conditional forecasts
    y_bar : ndarray of shape (n_conditions,)
        vector of mean values for conditions
    omega : ndarray of shape (n_conditions,)
        vector of variance values for conditions
    """
    
    k = conditions.shape[0]
    R = np.zeros((k,n*h))
    y_bar = np.zeros(k)
    omega = np.zeros(k)
    for i in range(k):
        variable = int(conditions[i,0] - 1)
        period = int(conditions[i,1] - 1)
        mean = conditions[i,2]
        variance = max(1e-10,conditions[i,3])
        y_bar[i] = mean
        omega[i] = variance
        R[i,n*period+variable] = 1
    return R, y_bar, omega


def conditional_forecast_regressors_4(structural_irf, n, h):

    """
    conditional_forecast_regressors_4(structural_irf, n, h)
    second set of elements for structural conditional forecasts, as in (4.14.14)
    
    parameters:
    structural_irf : ndarray of shape (n,n,h)
        matrix of structural impulse response functions
    n : int
        number of endogenous variables         
    h : int
        number of forecast periods 
      
        
    returns:
    M : ndarray of shape (n*h,n*h)
        matrix of stacked impulse response function 
    """
    
    temp = np.zeros((n,n*h))
    M = np.zeros((n*h,n*h))
    temp[:,-n:] = structural_irf[:,:,0]
    M[:n,:n] = structural_irf[:,:,0]
    for i in range(1,h):
        temp[:,-(i+1)*n:-i*n] = structural_irf[:,:,i]
        M[i*n:(i+1)*n,:(i+1)*n] = temp[:,-(i+1)*n:]
    return M


def conditional_forecast_regressors_5(shocks, h, n):

    """
    conditional_forecast_regressors_5(shocks, h, n)
    first set of elements for shock-specific structural conditional forecasts: iteration-invariant 
    
    parameters:
    shocks : ndarray of shape (n,)
        vector of generating shocks: 1 is generating, 0 is non-generating
    h : int
        number of forecast periods 
    n : int
        number of endogenous variables       
        
    returns:
    P : ndarray of shape (m,n*h)
        selection matrix for the m non-generating shocks
    non_generating : ndarray of shape (n,)
        vector of non-generating shocks: 1 is non-generating, 0 is generating
    """
    
    non_generating = np.ones(n) - shocks
    P = np.diag(np.tile(non_generating,h))
    P = P[~np.all(P == 0, axis=1)]
    return P, non_generating


def conditional_forecast_regressors_6(gamma, non_generating, h):

    """
    conditional_forecast_regressors_6(gamma, non_generating, h)
    second set of elements for structural conditional forecasts, as in (4.14.21)
    
    parameters:
    gamma : ndarray of shape (n,)
        vector of variance values for structural shocks
    non_generating : ndarray of shape (n,)
        vector of non-generating shocks: 1 is non-generating, 0 is generating        
    h : int
        number of forecast periods      
        
    returns:
    Gamma_nd : ndarray of shape (m,)
        variance vector for the m non-generating shocks
    """
    
    non_generating_variances = gamma * non_generating
    non_generating_variances = non_generating_variances[non_generating_variances != 0]
    Gamma_nd = np.tile(non_generating_variances, h)
    return Gamma_nd


def linear_forecast(B, h, Z_p, Y, n):

    """
    linear_forecast(B, h, Z_p, Y, n)
    best linear forecasts, absent shocks
    
    parameters:
    B : ndarray of shape (k,n)
        matrix of VAR coefficients
    h : int
        number of forecast periods
    Z_p : ndarray of shape (h,m)
        matrix of exogenous regressors for forecasts
    Y : ndarray of shape (p,n)
        matrix of initial conditions for endogenous variables  
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
        # generate forecasts
        y = X @ B
        # update Y and Y_p
        Y = np.vstack([Y[1:,:],y])
        Y_p[i,:] = y
    return Y_p


def conditional_forecast_posterior(y_bar, f, M, R, gamma, omega, n, h):
    
    """
    conditional_forecast_posterior(y_bar, f, M, R, gamma, omega, n, h)
    posterior parameters for the structural conditional forecasts, as in (4.14.20)
    
    parameters:
    y_bar : ndarray of shape (n_conditions,)
        vector of mean values for conditions
    f : ndarray of shape (h,n)
        matrix of simulated forecast values   
    M : ndarray of shape (n*h,n*h)
        matrix of stacked impulse response function         
    R : ndarray of shape (n_conditions,n*h)
        selection matrix for conditional forecasts
    gamma : ndarray of shape (n,)
        vector of variance values for structural shocks
    omega : ndarray of shape (n_conditions,)
        vector of variance values for conditions
    n : int
        number of endogenous variables        
    h : int
        number of forecast periods
          
    returns:
    mu_hat : ndarray of shape (n*h,)
        vector of posterior mean values
    Omega_hat : ndarray of shape (n*h,n*h)
        matrix of posterior variance-covariance values        
    """  

    f_T = la.vec(f.T)
    D = R @ M
    D_star = la.slash_inversion(D.T, D @ D.T)
    mu_hat = f_T + M @ D_star @ (y_bar - R @ f_T)
    temp = np.eye(n*h) - D_star @ D
    Omega_hat = M @ (D_star * omega @ D_star.T + temp * np.tile(gamma,h) @ temp) @ M.T
    return mu_hat, Omega_hat



def shock_specific_conditional_forecast_posterior(y_bar, f, M, R, P, gamma, Gamma_nd, omega, n, h):
    
    """
    shock_specific_conditional_forecast_posterior(y_bar, f, M, R, P, gamma, Gamma_nd, omega, n, h)
    posterior parameters for the structural conditional forecasts, as in (4.14.20) and (4.14.26)
    
    parameters:
    y_bar : ndarray of shape (n_conditions,)
        vector of mean values for conditions
    f : ndarray of shape (h,n)
        matrix of simulated forecast values   
    M : ndarray of shape (n*h,n*h)
        matrix of stacked impulse response function         
    R : ndarray of shape (n_conditions,n*h)
        selection matrix for conditional forecasts
    P : ndarray of shape (m,n*h)
        selection matrix for the m non-generating shocks        
    gamma : ndarray of shape (n,)
        vector of variance values for structural shocks
    Gamma_nd : ndarray of shape (m,)
        variance vector for the m non-generating shocks        
    omega : ndarray of shape (n_conditions,)
        vector of variance values for conditions
    n : int
        number of endogenous variables        
    h : int
        number of forecast periods
          
    returns:
    mu_hat : ndarray of shape (n*h,)
        vector of posterior mean values
    Omega_hat : ndarray of shape (n*h,n*h)
        matrix of posterior variance-covariance values        
    """    
    
    Q = la.slash_inversion(P, M)
    Z = np.vstack([R,Q])
    f_T = la.vec(f.T)
    g_T = np.hstack([y_bar,Q @ f_T])
    xi = np.hstack([omega,Gamma_nd])
    D = Z @ M
    D_star = la.slash_inversion(D.T, D @ D.T)
    mu_hat = f_T + M @ D_star @ (g_T - Z @ f_T)
    temp = np.eye(n*h) - D_star @ D
    Omega_hat = M @ (D_star * xi @ D_star.T + temp * np.tile(gamma,h) @ temp) @ M.T    
    return mu_hat, Omega_hat


def make_restriction_matrices(restriction_table, p):   
    
    """
    make_restriction_matrices(restriction_table, p)
    creates restriction and coefficient matrices to check restriction validity in later algorithms
    
    parameters:
    restriction_table : ndarray of shape (n_restrictions, 3+n_endogenous)
        matrix of restrictions
    p : int
        number of lags 
          
    returns:
    restriction_matrices : list of length 7
        each list entry stores matrices of restriction and coefficient values
    max_irf_period : int
        maximum number of periods for which IRFs will have to be computed in later algorithms     
    """       

    restriction_matrices = [[]] + [[[],[]] for i in range(6)]
    max_irf_period = 0
    # zero restrictions
    zero_restrictions = restriction_table[restriction_table[:,0]==1]
    restriction_number = len(zero_restrictions)
    if restriction_number != 0:
        indices = np.zeros((restriction_number,3)).astype(int)
        for i in range(restriction_number):
            shock = np.nonzero(zero_restrictions[i,3:])[0][0]
            period = zero_restrictions[i,2]
            indices[i,0] = zero_restrictions[i,1] - 1
            indices[i,1] = shock
            indices[i,2] = period - 1
            max_irf_period = max(max_irf_period, period)
        restriction_matrices[0] = indices
    # restrictions on IRFs: sign restrictions
    restrictions = restriction_table[restriction_table[:,0]==2]    
    occurrences = np.unique(np.nonzero(restrictions[:,3:])[0], return_counts=True)
    sign_occurrences = occurrences[0][occurrences[1]==1]
    sign_restrictions = restrictions[sign_occurrences,:]
    restriction_number = len(sign_occurrences)
    if restriction_number != 0:
        indices = np.zeros((restriction_number,3)).astype(int)
        coefficients = np.zeros((restriction_number,1))
        for i in range(restriction_number):
            shock = np.nonzero(sign_restrictions[i,3:])[0][0]
            period = sign_restrictions[i,2]
            indices[i,0] = sign_restrictions[i,1] - 1
            indices[i,1] = shock
            indices[i,2] = period - 1
            coefficients[i,0] = sign_restrictions[i,3+shock]
            max_irf_period = max(max_irf_period, period)
        restriction_matrices[1][0] = indices
        restriction_matrices[1][1] = coefficients        
    # restrictions on IRFs: magnitude restrictions
    magnitude_occurrences = occurrences[0][occurrences[1]==2]
    magnitude_restrictions = restrictions[magnitude_occurrences,:]
    restriction_number = len(magnitude_occurrences)
    if restriction_number != 0:
        indices = np.zeros((restriction_number,4)).astype(int)
        coefficients = np.zeros((restriction_number,2))
        for i in range(restriction_number):
            shocks = np.nonzero(magnitude_restrictions[i,3:])[0]
            period = magnitude_restrictions[i,2]
            indices[i,0] = magnitude_restrictions[i,1] - 1
            indices[i,1:3] = shocks
            indices[i,3] = period - 1
            coefficients[i,:] = magnitude_restrictions[i,3+shocks]
            max_irf_period = max(max_irf_period, period) 
        restriction_matrices[2][0] = indices
        restriction_matrices[2][1] = coefficients          
    # restrictions on shocks: sign restrictions
    restrictions = restriction_table[restriction_table[:,0]==3]    
    occurrences = np.unique(np.nonzero(restrictions[:,3:])[0], return_counts=True)
    sign_occurrences = occurrences[0][occurrences[1]==1]
    sign_restrictions = restrictions[sign_occurrences,:]
    restriction_number = len(sign_occurrences)
    if restriction_number != 0:
        indices = np.zeros((restriction_number,2)).astype(int)
        coefficients = np.zeros((restriction_number,1))
        for i in range(restriction_number):
            shock = np.nonzero(sign_restrictions[i,3:])[0][0]
            period = sign_restrictions[i,2] - p
            indices[i,0] = period - 1
            indices[i,1] = shock
            coefficients[i,0] = sign_restrictions[i,3+shock]
            max_irf_period = max(max_irf_period, period)
        restriction_matrices[3][0] = indices
        restriction_matrices[3][1] = coefficients          
    # restrictions on shocks: magnitude restrictions
    magnitude_occurrences = occurrences[0][occurrences[1]==2]
    magnitude_restrictions = restrictions[magnitude_occurrences,:]
    restriction_number = len(magnitude_occurrences)
    if restriction_number != 0:
        indices = np.zeros((restriction_number,3)).astype(int)
        coefficients = np.zeros((restriction_number,2))
        for i in range(restriction_number):
            shocks = np.nonzero(magnitude_restrictions[i,3:])[0]
            period = magnitude_restrictions[i,2] - p
            indices[i,0] = period - 1
            indices[i,1:] = shocks
            coefficients[i,:] = magnitude_restrictions[i,3+shocks]
            max_irf_period = max(max_irf_period, period) 
        restriction_matrices[4][0] = indices
        restriction_matrices[4][1] = coefficients          
    # restrictions on historical decomposition: sign restrictions
    restrictions = restriction_table[restriction_table[:,0]==4]    
    occurrences = np.unique(np.nonzero(restrictions[:,3:])[0], return_counts=True)
    sign_occurrences = occurrences[0][occurrences[1]==1]
    sign_restrictions = restrictions[sign_occurrences,:]
    restriction_number = len(sign_occurrences)
    if restriction_number != 0:
        indices = np.zeros((restriction_number,3)).astype(int)
        coefficients = np.zeros((restriction_number,1))
        for i in range(restriction_number):
            shock = np.nonzero(sign_restrictions[i,3:])[0][0]
            period = sign_restrictions[i,2] - p
            indices[i,0] = sign_restrictions[i,1] - 1
            indices[i,1] = shock
            indices[i,2] = period - 1
            coefficients[i,0] = sign_restrictions[i,3+shock]
            max_irf_period = max(max_irf_period, period) 
        restriction_matrices[5][0] = indices
        restriction_matrices[5][1] = coefficients          
    # restrictions on historical decomposition: magnitude restrictions
    magnitude_occurrences = occurrences[0][occurrences[1]==2]
    magnitude_restrictions = restrictions[magnitude_occurrences,:]
    restriction_number = len(magnitude_occurrences)
    if restriction_number != 0:
        indices = np.zeros((restriction_number,4)).astype(int)
        coefficients = np.zeros((restriction_number,2))
        for i in range(restriction_number):
            shocks = np.nonzero(magnitude_restrictions[i,3:])[0]
            period = magnitude_restrictions[i,2] - p
            indices[i,0] = magnitude_restrictions[i,1] - 1
            indices[i,1:3] = shocks
            indices[i,3] = period - 1
            coefficients[i,:] = magnitude_restrictions[i,3+shocks]
            max_irf_period = max(max_irf_period, period)
        restriction_matrices[6][0] = indices
        restriction_matrices[6][1] = coefficients          
    max_irf_period = int(max_irf_period)
    return restriction_matrices, max_irf_period


def make_covariance_restriction_matrices(restriction_table):   
    
    """
    make_covariance_restriction_matrices(restriction_table)
    creates restriction and coefficient matrices to check restriction validity in later algorithms
    
    parameters:
    restriction_table : ndarray of shape (n_restrictions, 3+n_endogenous)
        matrix of restrictions
          
    returns:
    restriction_matrices : list of length 2
        each list entry stores matrices of restriction and coefficient values   
    """       

    restriction_matrices = [[[],[]] for i in range(2)]
    # restrictions on covariance: sign restrictions
    restrictions = restriction_table[restriction_table[:,0]==5]    
    occurrences = np.unique(np.nonzero(restrictions[:,3:])[0], return_counts=True)
    sign_occurrences = occurrences[0][occurrences[1]==1]
    sign_restrictions = restrictions[sign_occurrences,:]
    restriction_number = len(sign_occurrences)
    if restriction_number != 0:
        indices = np.zeros((restriction_number,2)).astype(int)
        coefficients = np.zeros((restriction_number,1))
        for i in range(restriction_number):
            shock = np.nonzero(sign_restrictions[i,3:])[0][0]
            indices[i,0] = sign_restrictions[i,1] - 1
            indices[i,1] = shock
            coefficients[i,0] = sign_restrictions[i,3+shock]
        restriction_matrices[0][0] = indices
        restriction_matrices[0][1] = coefficients  
    # restrictions on covariance: magnitude restrictions
    magnitude_occurrences = occurrences[0][occurrences[1]==2]
    magnitude_restrictions = restrictions[magnitude_occurrences,:]
    restriction_number = len(magnitude_occurrences)
    if restriction_number != 0:
        indices = np.zeros((restriction_number,3)).astype(int)
        coefficients = np.zeros((restriction_number,2))
        for i in range(restriction_number):
            shocks = np.nonzero(magnitude_restrictions[i,3:])[0]
            indices[i,0] = magnitude_restrictions[i,1] - 1
            indices[i,1:] = shocks
            coefficients[i,:] = magnitude_restrictions[i,3+shocks]
        restriction_matrices[1][0] = indices
        restriction_matrices[1][1] = coefficients          
    return restriction_matrices


def make_restriction_irf(mcmc_beta, mcmc_chol_Sigma, iterations, n, p, max_irf_period):

    """
    make_restriction_irf(mcmc_beta, mcmc_chol_Sigma, iterations, n, p, max_irf_period)
    creates preliminary orthogonalized IRFs for restriction algorithm
    
    parameters:
    mcmc_beta : ndarray of shape (k, n, iterations)
        matrix of mcmc values for beta
    mcmc_chol_Sigma : ndarray of shape (n, n, iterations)
        matrix of mcmc values for h(Sigma)
    iterations: int
        number of MCMC iterations
    n : int
        number of endogenous variables       
    p : int
        number of lags
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
            irf = impulse_response_function(mcmc_beta[:,:,i], n, p, max_irf_period)
            structural_irf = structural_impulse_response_function(irf, mcmc_chol_Sigma[:,:,i], n)            
            mcmc_irf[:,:,:,i] = structural_irf
    return mcmc_irf
        

def make_restriction_shocks(mcmc_beta, mcmc_chol_Sigma, Y, X, T, n, iterations, restriction_matrices):

    """
    make_restriction_shocks(mcmc_beta, mcmc_chol_Sigma, Y, X, iterations, restriction_matrices)
    creates preliminary structural shocks for restriction algorithm
    
    parameters:
    mcmc_beta : ndarray of shape (k, n, iterations)
        matrix of mcmc values for beta
    mcmc_chol_Sigma : ndarray of shape (n, n, iterations)
        matrix of mcmc values for h(Sigma)
    Y : ndarray of shape (T,n)
        matrix of endogenous variables
    X : ndarray of shape (T,k)
        matrix of VAR regressors    
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
            E, _ = fit_and_residuals(Y, X, mcmc_beta[:,:,i])
            Xi = la.slash_inversion(E, mcmc_chol_Sigma[:,:,i].T)
            mcmc_shocks[:,:,i] = Xi
    return mcmc_shocks
    
    
def check_irf_sign(indices, coefficients, irf, Q):
    
    """
    check_irf_sign(indices, coefficients, irf, Q)
    checks whether IRF sign restrictions are satisfied
    
    parameters:
    indices : ndarray of shape (n_restrictions,3)
        matrix of IRF indices (variable, shock, period) to check
    coefficients : ndarray of shape (n_restrictions,1)
        matrix of IRF coefficients to apply for positivity/negativity restrictions
    irf : ndarray of shape (n,n,n_periods)
        matrix of preliminary orthogonalized IRFs 
    Q : ndarray of shape (n,n)
        uniform orthogonal matrix 
          
    returns:
    restriction_satisfied : bool
        True if all restrictions are satisfied, False otherwise
    """    
    
    for i in range(len(indices)):
        irf_value = irf[indices[i,0],:,indices[i,2]] @ Q[:,indices[i,1]]
        restriction = (irf_value * coefficients[i])[0]
        if restriction < 0:
            restriction_satisfied = False
            return restriction_satisfied
    restriction_satisfied = True
    return restriction_satisfied
    
    
def check_irf_magnitude(indices, coefficients, irf, Q):
    
    """
    check_irf_magnitude(indices, coefficients, irf, Q)
    checks whether IRF magnitude restrictions are satisfied
    
    parameters:
    indices : ndarray of shape (n_restrictions,4)
        matrix of IRF indices (variable, shock1, shock2, period) to check
    coefficients : ndarray of shape (n_restrictions,2)
        matrix of IRF coefficients to apply for magnitude restrictions
    irf : ndarray of shape (n,n,n_periods)
        matrix of preliminary orthogonalized IRFs 
    Q : ndarray of shape (n,n)
        uniform orthogonal matrix 
          
    returns:
    restriction_satisfied : bool
        True if all restrictions are satisfied, False otherwise
    """    

    for i in range(len(indices)):
        irf_values = np.abs(irf[indices[i,0],:,indices[i,2]] @ Q[:,indices[i,[1,2]]])
        restriction = np.sum(irf_values * coefficients[i,:])
        if restriction < 0:
            restriction_satisfied = False
            return restriction_satisfied
    restriction_satisfied = True
    return restriction_satisfied    
    
    
def check_shock_sign(indices, coefficients, shocks, Q):
    
    """
    check_shock_sign(indices, coefficients, shocks, Q)
    checks whether shock sign restrictions are satisfied
    
    parameters:
    indices : ndarray of shape (n_restrictions,2)
        matrix of shock indices (period, shock) to check
    coefficients : ndarray of shape (n_restrictions,1)
        matrix of shock coefficients to apply for positivity/negativity restrictions
    shocks : ndarray of shape (T,n)
        matrix of preliminary structural shocks
    Q : ndarray of shape (n,n)
        uniform orthogonal matrix 
          
    returns:
    restriction_satisfied : bool
        True if all restrictions are satisfied, False otherwise
    """    

    for i in range(len(indices)):
        shock_value = shocks[indices[i,0],:] @ Q[:,indices[i,1]]
        restriction = (shock_value * coefficients[i])[0]
        if restriction < 0:
            restriction_satisfied = False
            return restriction_satisfied
    restriction_satisfied = True
    return restriction_satisfied    
    
    
def check_shock_magnitude(indices, coefficients, shocks, Q):
    
    """
    check_shock_magnitude(indices, coefficients, shocks, Q)
    checks whether shock magnitude restrictions are satisfied
    
    parameters:
    indices : ndarray of shape (n_restrictions,3)
        matrix of shock indices (period,shock1, shock2) to check
    coefficients : ndarray of shape (n_restrictions,2)
        matrix of shock coefficients to apply for magnitude restrictions
    irf : ndarray of shape (n,n,n_periods)
        matrix of preliminary orthogonalized IRFs 
    Q : ndarray of shape (n,n)
        uniform orthogonal matrix 
          
    returns:
    restriction_satisfied : bool
        True if all restrictions are satisfied, False otherwise
    """    

    for i in range(len(indices)):
        shock_values = np.abs(shocks[indices[i,0],:] @ Q[:,indices[i,[1,2]]])
        restriction = np.sum(shock_values * coefficients[i,:])
        if restriction < 0:
            restriction_satisfied = False
            return restriction_satisfied
    restriction_satisfied = True
    return restriction_satisfied     
    
    
def make_restriction_irf_and_shocks(irf, shocks, Q, n):

    """
    make_restriction_irf_and_shocks(irf, shocks, Q, n)
    generates structural IRFs and shocks for a given Q matrix
    
    parameters:
    irf : ndarray of shape (n,n,n_periods)
        matrix of preliminary orthogonalized IRFs 
    shocks : ndarray of shape (T,n)
        matrix of preliminary structural shocks      
    Q : ndarray of shape (n,n)
        uniform orthogonal matrix 
    n : int
        number of endogenous variables 
          
    returns:
    structural_irf : ndarray of shape (n,n,n_periods)
        matrix of final orthogonalized IRFs 
    structural_shocks : ndarray of shape (T,n)
        matrix of final orthogonalized IRFs         
    """  
    
    structural_irf = structural_impulse_response_function(irf, Q, n)    
    structural_shocks = shocks @ Q
    return structural_irf, structural_shocks
    
    
def check_history_sign(indices, coefficients, irf, shocks):

    """
    check_history_sign(indices, coefficients, irf, shocks)
    checks whether historical sign restrictions are satisfied
    
    parameters:
    indices : ndarray of shape (n_restrictions,3)
        matrix of IRF indices (variable, shock, period) to check
    coefficients : ndarray of shape (n_restrictions,1)
        matrix of historical decomposition coefficients to apply for positivity/negativity restrictions
    irf : ndarray of shape (n,n,n_periods)
        matrix of orthogonalized IRFs 
    shocks : ndarray of shape (T,n)
        matrix of structural shocks     
          
    returns:
    restriction_satisfied : bool
        True if all restrictions are satisfied, False otherwise
    """      

    for i in range(len(indices)):
        restriction_irf = irf[indices[i,0],indices[i,1],:indices[i,2]+1]
        restriction_shocks = shocks[:indices[i,2]+1,indices[i,1]][::-1]
        hd_value = restriction_irf @ restriction_shocks
        restriction = (hd_value * coefficients[i])[0]
        if restriction < 0:
            restriction_satisfied = False
            return restriction_satisfied
    restriction_satisfied = True
    return restriction_satisfied  


def check_history_magnitude(indices, coefficients, irf, shocks):

    """
    check_history_magnitude(indices, coefficients, irf, shocks)
    checks whether historical magnitude restrictions are satisfied
    
    parameters:
    indices : ndarray of shape (n_restrictions,4)
        matrix of IRF indices (variable, shock1, shock2, period) to check
    coefficients : ndarray of shape (n_restrictions,2)
        matrix of historical decomposition coefficients to apply for magnitude restrictions
    irf : ndarray of shape (n,n,n_periods)
        matrix of orthogonalized IRFs 
    shocks : ndarray of shape (T,n)
        matrix of structural shocks     
          
    returns:
    restriction_satisfied : bool
        True if all restrictions are satisfied, False otherwise
    """   

    for i in range(len(indices)):
        restriction_irf_1 = irf[indices[i,0],indices[i,1],:indices[i,3]+1]
        restriction_shocks_1 = shocks[:indices[i,3]+1,indices[i,1]][::-1]
        hd_value_1 = restriction_irf_1 @ restriction_shocks_1
        restriction_1 = np.abs(hd_value_1) * coefficients[i,0]
        restriction_irf_2 = irf[indices[i,0],indices[i,2],:indices[i,3]+1]
        restriction_shocks_2 = shocks[:indices[i,3]+1,indices[i,2]][::-1]
        hd_value_2 = restriction_irf_2 @ restriction_shocks_2
        restriction_2 = np.abs(hd_value_2) * coefficients[i,1]
        restriction = restriction_1 + restriction_2
        if restriction < 0:
            restriction_satisfied = False
            return restriction_satisfied
    restriction_satisfied = True
    return restriction_satisfied  


def check_covariance_sign(indices, coefficients, V, n, h):
    
    """
    check_covariance_sign(indices, coefficients, V, n, h)
    checks whether covariance sign restrictions are satisfied
    
    parameters:
    indices : ndarray of shape (n_restrictions,2)
        matrix of covariance indices (period, shock) to check
    coefficients : ndarray of shape (n_restrictions,1)
        matrix of covariance coefficients to apply for positivity/negativity restrictions
    V : ndarray of shape (h,h)
        matrix of covariance between proxys and structural shocks
    n : int
        number of endogenous variables         
    h : int
        number of proxy variables         
          
    returns:
    restriction_satisfied : bool
        True if all restrictions are satisfied, False otherwise
    """    

    E_r_xi = np.hstack([np.zeros((h,n-h)),V])
    for i in range(len(indices)):
        covariance_value = E_r_xi[indices[i,0],indices[i,1]]
        restriction = (covariance_value * coefficients[i])[0]
        if restriction < 0:
            restriction_satisfied = False
            return restriction_satisfied
    restriction_satisfied = True
    return restriction_satisfied 


def check_covariance_magnitude(indices, coefficients, V, n, h):
    
    """
    check_covariance_magnitude(indices, coefficients, V, n, h)
    checks whether covariance magnitude restrictions are satisfied
    
    parameters:
    indices : ndarray of shape (n_restrictions,3)
        matrix of covariance indices (variable, shock1, shock2) to check
    coefficients : ndarray of shape (n_restrictions,2)
        matrix of covariance coefficients to apply for magnitude restrictions
    V : ndarray of shape (h,h)
        matrix of covariance between proxys and structural shocks
    n : int
        number of endogenous variables         
    h : int
        number of proxy variables  
          
    returns:
    restriction_satisfied : bool
        True if all restrictions are satisfied, False otherwise
    """    

    E_r_xi = np.hstack([np.zeros((h,n-h)),V])
    for i in range(len(indices)):
        covariance_values = np.abs(E_r_xi[indices[i,0],indices[i,[1,2]]])
        restriction = np.sum(covariance_values * coefficients[i,:])
        if restriction < 0:
            restriction_satisfied = False
            return restriction_satisfied
    restriction_satisfied = True
    return restriction_satisfied 

    
def ols_var_mcmc_beta(B, Sigma, XX, k, n, q):
    
    """
    ols_var_mcmc_beta(B, Sigma, XX, k, n)
    generates pseudo MCMC draws for beta for an OLS VAR model
    
    parameters:
    B : matrix of size (k,n)
        matrix of VAR coefficients
    Sigma : matrix of size (n,n)
        variance-covariance matrix of VAR residuals
    XX : matrix of size (k,k)
        covariance matrix of regressors X
    k : int
        number of VAR coefficients by equation
    n : int
        number of endogenous variables   
    q : int
        total number of VAR coefficients
    
    returns:
    mcmc_B : matrix of size (k,n,500)
        matrix of pseudo MCMC draws
    """

    Q = np.kron(Sigma, la.invert_spd_matrix(XX))
    mcmc_beta = la.vec(B).reshape(-1,1) + la.cholesky_nspd(Q) @ nrd.randn(q, 500)
    mcmc_B = np.reshape(mcmc_beta, [k,n,500], order='F') 
    return mcmc_B


def rework_constraint_table(constrained_coefficients_table, lags):
    
    """
    rework_constraint_table(constrained_coefficients_table, lags)
    update constraint table by switching '-1' lags to 'all lags' entries
    
    parameters:
    constrained_coefficients_table : ndarray of size (n_constraint,5)
        matrix of constraints
    lags : int
        number of lags, either -1 or positive
    
    returns:
    secondary_constrained_coefficients_table : ndarray of size (n_constraint,5)
        matrix of updated constraints
    """
    
    secondary_constrained_coefficients_table =  np.empty(shape=[0, 5])
    for i in range(constrained_coefficients_table.shape[0]):
        row = constrained_coefficients_table[i,:]
        lag = constrained_coefficients_table[i,2]
        if lag == -1:
            temp = np.tile(row,[lags,1])
            temp[:,2] = np.arange(1,lags+1)
            secondary_constrained_coefficients_table = np.vstack([\
            secondary_constrained_coefficients_table, temp])
        else:
            secondary_constrained_coefficients_table = np.vstack([\
            secondary_constrained_coefficients_table, row])
    return secondary_constrained_coefficients_table



