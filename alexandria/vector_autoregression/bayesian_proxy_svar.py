# imports
import math as mt
import random as rd
import numpy as np
import numpy.linalg as nla
import numpy.random as nrd
import scipy.linalg as sla
import alexandria.vector_autoregression.var_utilities as vu
import alexandria.math.random_number_generators as rng
import alexandria.console.console_utilities as cu
import alexandria.math.linear_algebra as la
from alexandria.vector_autoregression.vector_autoregression import VectorAutoRegression
from alexandria.vector_autoregression.bayesian_var import BayesianVar
from alexandria.vector_autoregression.maximum_likelihood_var import MaximumLikelihoodVar


class BayesianProxySvar(VectorAutoRegression,BayesianVar):
    
    
    """
    Bayesian proxy-SVAR, developed in section 14.5
    
    Parameters:
    -----------
    endogenous : ndarray of size (n_obs,n_endogenous)
        endogenous variables, defined in (4.11.1)
    
    proxys : ndarray of size (n_obs,n_proxys)
        proxy variables, defined in (4.14.42)
    
    exogenous : ndarray of size (n_obs,n_exogenous), default = []
        exogenous variables, defined in (4.11.1)
    
    structural_identification : int, default = 1
        structural identification scheme, additional to proxy-SVAR
        1 = none, 4 = restrictions
    
    restriction_table : ndarray
        numerical matrix of restrictions for structural identification
        
    lamda : float, default = 0.2
        relevance parameter, defined in (4.14.54)
     
    proxy_prior : int, default = 1
        prior scheme for normal-generalized-normal prior
        1 = uninformative, 2 = Minnesota
    
    lags : int, default = 4
        number of lags, defined in (4.11.1)
    
    constant : bool, default = True
        if True, an intercept is included in the VAR model exogenous
    
    trend : bool, default = False
        if True, a linear trend is included in the VAR model exogenous
    
    quadratic_trend : bool, default = False
        if True, a quadratic trend is included in the VAR model exogenous
    
    ar_coefficients : float or ndarray of size (n_endo,1), default = 0.95
        prior mean delta for AR coefficients, defined in (4.11.16)
    
    pi1 : float, default = 0.1
        overall tightness hyperparameter, defined in (4.11.17)
    
    pi3 : float, default = 1
        lag decay hyperparameter, defined in (4.11.17)    
    
    pi4 : float, default = 100
        exogenous slackness hyperparameter, defined in (4.11.19)       
    
    credibility_level : float, default = 0.95
        VAR model credibility level (between 0 and 1)
    
    iterations : int, default = 2000
        number of Gibbs sampler replications   

    burnin : int, default = 1000
        number of Gibbs sampler burn-in replications 
    
    verbose : bool, default = False
        if True, displays a progress bar 
    
    
    Attributes
    ----------
    endogenous : ndarray of size (n_obs,n_endogenous)
        endogenous variables, defined in (4.11.1)
    
    proxys : ndarray of size (n_obs,n_proxys)
        proxy variables, defined in (4.14.42)
    
    exogenous : ndarray of size (n_obs,n_exogenous), default = []
        exogenous variables, defined in (4.11.1)
    
    structural_identification : int
        structural identification scheme, as defined in section 13.2
        1 = none, 2 = Cholesky, 3 = triangular, 4 = restrictions
    
    restriction_table : ndarray
        numerical matrix of dictural identification restrictions
    
    endogenous : ndarray of size (n_obs,n_endogenous)
        endogenous variables, defined in (4.11.1)
    
    proxys : ndarray of size (n_obs,n_proxys)
        proxy variables, defined in (4.14.42)
    
    exogenous : ndarray of size (n_obs,n_exogenous)
        exogenous variables, defined in (4.11.1)
        
    lags : int
        number of lags, defined in (4.11.1)
    
    constant : bool
        if True, an intercept is included in the VAR model exogenous
    
    trend : bool
        if True, a linear trend is included in the VAR model exogenous
    
    quadratic_trend : bool
        if True, a quadratic trend is included in the VAR model exogenous
    
    ar_coefficients : float or ndarray of size (n_endo,1)
        prior mean delta for AR coefficients, defined in (4.11.16)
    
    pi1 : float
        overall tightness hyperparameter, defined in (4.11.17)
    
    pi2 : float
        cross-variable shrinkage hyperparameter, defined in (4.11.18)
    
    pi4 : float
        exogenous slackness hyperparameter, defined in (4.11.19)    
    
    credibility_level : float
        VAR model credibility level (between 0 and 1)
    
    iterations : int
        number of Gibbs sampler replications  
        
    burnin : int
        number of Gibbs sampler burn-in replications  
    
    verbose : bool
        if True, displays a progress bar      

    R : ndarray of size (T,h)
        in-sample matrix of proxy variables, defined in (4.14.42) 
    
    Y_bar : ndarray of size (T,n+h)
        in-sample matrix of endogenous and proxy regressors, defined in (4.14.49) 
    
    X_bar : ndarray of size (T,m+(n+h)*p)
        in-sample matrix of full regressors, defined in (4.14.49) 
        
    h : int
        number of proxy variables
    
    n_bar : int
        number of endogenous and proxy variables (n+h)
    
    k_bar : int
        total number of proxy-SVAR regressors (m+(n+h)*p)
       
    alpha : float
        prior degrees of freedom, defined in (4.14.50)
    
    inv_W : ndarray of size (k_bar,k_bar)
        prior variance of VAR coefficients, defined in (4.14.50)           
    
    B : ndarray of size (k_bar,n_bar)
        prior mean of VAR coefficients, defined in (4.14.50)           
    
    S : ndarray of size (n_bar,n_bar)
        prior scale matrix, defined in (4.14.50) 
    
    alpha_bar : float
        posterior degrees of freedom, defined in (4.11.33)
    
    W_bar : ndarray of size (k_bar,k_bar)
        posterior variance of VAR coefficients, defined in (4.14.50)           
    
    B_bar : ndarray of size (k_bar,n_bar)
        posterior mean of VAR coefficients, defined in (4.14.50)           
    
    S_bar : ndarray of size (n_bar,n_bar)
        posterior scale matrix, defined in (4.14.50) 
    
    mcmc_beta : ndarray of size (k,n,iterations)
        MCMC values of VAR coefficients   
    
    mcmc_Sigma : ndarray of size (n,n,iterations)
        MCMC values of residual variance-covariance matrix       
    
    mcmc_V : ndarray of size (h,h,iterations)
        MCMC values of covariance matrix V, defined in (4.14.46)
    
    mcmc_min_eigenvalue : ndarray of size (iterations,)
        MCMC values of minimum eigenvalue, defined in (4.14.54)
    
    beta_estimates : ndarray of size (k,n,3)
        estimates of VAR coefficients
        page 1: median, page 2: st dev,  page 3: lower bound, page 4: upper bound
    
    Sigma_estimates : ndarray of size (n,n)
        estimates of variance-covariance matrix of VAR residuals
    
    V_estimates : ndarray of size (h,h)
        posterior estimates of covariance matrix V, defined in (4.14.46)
    
    min_eigenvalue_estimates : float
        posterior estimate of minimum eigenvalue, defined in (4.14.54)          
    
    mcmc_H :  ndarray of size (n,n,iterations)
        MCMC values of structural identification matrix, defined in (4.13.5)
    
    mcmc_Gamma : ndarray of size (iterations,n)
        MCMC values of structural shock variance matrix, defined in definition 13.1
    
    Y : ndarray of size (T,n)
        matrix of in-sample endogenous variables, defined in (4.11.3)
    
    Z : ndarray of size (T,m)
        matrix of in-sample endogenous variables, defined in (4.11.3)
    
    X : ndarray of size (T,k)
        matrix of exogenous and lagged regressors, defined in (4.11.3)
    
    n : int
        number of endogenous variables, defined in (4.11.1)
    
    m : int
        number of exogenous variables, defined in (4.11.1)
    
    p : int
        number of lags, defined in (4.11.1)
    
    T : int
        number of sample periods, defined in (4.11.1)
    
    k : int
        number of VAR coefficients in each equation, defined in (4.11.1)
    
    q : int
        total number of VAR coefficients, defined in (4.11.1)
    
    delta : ndarray of size (n,1)
        prior mean delta for AR coefficients, defined in (4.11.16)
    
    s : ndarray of size (n,1)
        individual AR models residual variance, defined in (4.11.18)
    
    steady_state_estimates : ndarray of size (T,n,3)
        estimates of steady-state, defined in (4.12.30)
    
    fitted_estimates : ndarray of size (T,n,3)
        estimates of in-sample fit, defined in (4.11.2)
    
    residual_estimates : ndarray of size (T,n,3)
        estimates of in-sample residuals, defined in (4.11.2)
    
    structural_shocks_estimates : ndarray of size (T,n,3)
        estimates of in-sample structural shocks, defined in definition 13.1
    
    insample_evaluation : dict
        in-sample evaluation criteria, defined in (4.13.15)-(4.13.17)
    
    mcmc_structural_shocks : ndarray of size (T,n,iterations)
        MCMC values of structural shocks
    
    mcmc_forecasts : ndarray of size (f_periods,n,iterations)
        MCMC values of forecasts
    
    forecast_estimates : ndarray of size (f_periods,n,3)
        forecast estimates, defined in (4.13.12) and (4.13.13)
        page 1: median, page 2: lower bound, page 3: upper bound
    
    forecast_evaluation_criteria : dict
        forecast evaluation criteria, defined in (4.13.18)-(4.13.21)
    
    mcmc_irf : ndarray of size (n,n,irf_periods,iterations)
        MCMC values of impulse response function, defined in section 13.1
    
    mcmc_irf_exo : ndarray of size (n,m,irf_periods,iterations)
        MCMC values of exogenous impulse response function
    
    mcmc_structural_irf : ndarray of size (n,n,irf_periods,iterations)
        MCMC values of structural impulse response function, defined in section 13.2
    
    irf_estimates : ndarray of size (n,n,irf_periods,3)
        posterior estimates of impulse response function, defined in section 13.1 - 13.2
        page 1: median, page 2: lower bound, page 3: upper bound    
    
    exo_irf_estimates : ndarray of size (n,m,irf_periods,3)
        posterior estimates of exogenous impulse response function, if any exogenous variable
        page 1: median, page 2: lower bound, page 3: upper bound
    
    mcmc_fevd : ndarray of size (n,n,fevd_periods,iterations)
        MCMC values of forecast error variance decompositions, defined in section 13.4
    
    fevd_estimates : ndarray of size (n,n,fevd_periods,3)
        posterior estimates of forecast error variance decomposition, defined in section 13.4
        page 1: median, page 2: lower bound, page 3: upper bound 
    
    mcmc_hd : ndarray of size (n,n,T,iterations)
        MCMC values of historical decompositions, defined in section 13.5
    
    hd_estimates : ndarray of size (n,n,T,3)
        posterior estimates of historical decomposition, defined in section 13.5
        page 1: median, page 2: lower bound, page 3: upper bound 
    
    mcmc_conditional_forecasts : ndarray of size (f_periods,n,iterations)
        MCMC values of conditional forecasts, defined in section 14.1
    
    conditional_forecast_estimates : ndarray of size (f_periods,n,3)
        posterior estimates of conditional forecast, defined in section 14.1
        page 1: median, page 2: lower bound, page 3: upper bound
    
    mcmc_structural_conditional_forecasts : ndarray of size (f_periods,n,iterations)
        MCMC values of structural conditional forecasts, defined in section 14.2
    
    structural_conditional_forecast_estimates : ndarray of size (f_periods,n,3)
        structural conditional forecast estimates, defined in section 14.2
        page 1: median, page 2: lower bound, page 3: upper bound
    
    H_estimates : ndarray of size (n,n)
        posterior estimates of structural matrix, defined in section 13.2
    
    Gamma_estimates : ndarray of size (1,n)
        estimates of structural shock variance matrix, defined in section 13.2
    
    
    Methods
    ----------
    estimate
    insample_fit
    forecast
    forecast_evaluation
    impulse_response_function
    forecast_error_variance_decomposition
    historical_decomposition
    conditional_forecast
    structural_conditional_forecast        
    """        
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------    
    

    def __init__(self, endogenous, proxys, exogenous = [], structural_identification = 1,
                 restriction_table = [], lamda = 0.2, proxy_prior = 1, lags = 4, 
                 constant = True, trend = False, quadratic_trend = False, 
                 ar_coefficients = 0.95,  pi1 = 0.1, pi3 = 1, pi4 = 100, 
                 credibility_level = 0.95, iterations = 2000, burnin = 1000, verbose = False):
        
        """
        constructor for the BayesianProxySvar class
        """
        
        self.endogenous = endogenous
        self.proxys = proxys
        self.exogenous = exogenous
        self.structural_identification = structural_identification
        self.restriction_table = restriction_table
        self.lamda = lamda
        self.proxy_prior = proxy_prior
        self.lags = lags
        self.constant = constant
        self.trend = trend
        self.quadratic_trend = quadratic_trend        
        self.ar_coefficients = ar_coefficients        
        self.pi1 = pi1
        self.pi3 = pi3
        self.pi4 = pi4        
        self.credibility_level = credibility_level        
        self.iterations = iterations
        self.burnin = burnin
        self.verbose = verbose     
        # make regular regressors
        self._make_regressors()
        # make proxy SVAR regressors
        self.__make_proxy_svar_regressors()
        # make delta
        self._make_delta()        
        # make individual residual variance
        self._individual_ar_variances()
        # complete with proxy elements
        self.__proxy_delta_and_ar_variances()
        

    def estimate(self):
    
        """
        estimate()
        generates posterior estimates for Bayesian proxy SVAR model parameters
        
        parameters:
        none
        
        returns:
        none    
        """    
        
        # define prior values
        self.__prior()
        # define posterior parameters
        self.__posterior()
        # orthogonal triangular block parameters
        self.__orthogonal_triangular_block_parameters()
        # run MCMC algorithm (Gibbs sampling) for proxy SVAR parameters
        self.__parameter_mcmc()   
        # obtain posterior estimates for regression parameters
        self.__parameter_estimates()
    
        
    #---------------------------------------------------
    # Methods (Access = private)
    #--------------------------------------------------- 


    def __make_proxy_svar_regressors(self):
        
        """ generates proxy regressors Y_bar and X_bar as defined in (4.14.49) """
        
        R = self.proxys[self.lags:]
        Y_bar = np.hstack([self.Y,R])
        X_1 = vu.generate_intercept_and_trends(self.constant, self.trend, self.quadratic_trend, self.T, 0)
        X_2 = vu.generate_exogenous_regressors(self.exogenous, self.lags, self.T)
        X_3 = vu.generate_lagged_endogenous(np.hstack([self.endogenous,self.proxys]), self.lags)
        X_bar = np.hstack([X_1,X_2,X_3])
        h = self.proxys.shape[1]
        n_bar = self.n + h
        k_bar = X_bar.shape[1]
        self.R = R
        self.Y_bar = Y_bar
        self.X_bar = X_bar
        self.h = h
        self.n_bar = n_bar
        self.k_bar = k_bar
       
        
    def __proxy_delta_and_ar_variances(self):        
       
        """ generates proxy elements for delta and AR variances """
       
        delta_bar = np.hstack([self.delta,0.9 * np.ones(self.h)])
        s = np.zeros(self.h)
        for i in range(self.h):
            ar = MaximumLikelihoodVar(self.proxys[:,[i]], lags=self.lags)
            ar.estimate()
            s[i] = ar.Sigma[0,0]
        s_bar = np.hstack([self.s,s])
        self.__delta_bar = delta_bar
        self.__s_bar = s_bar
        
        
    def __prior(self):
        
        """ creates prior elements alpha, inv_W, B and S defined in (4.14.54) """
        
        # if prior is naive, set all elements to uninformative
        if self.proxy_prior == 1:
            alpha = self.n_bar
            inv_W = np.zeros(self.k_bar)
            B = np.zeros((self.k_bar,self.n_bar))
            S = np.zeros((self.n_bar,self.n_bar))
        # if prior is Minnesota, implement parameters that replicate the normal Wishart prior
        elif self.proxy_prior == 2:
            alpha = 0
            inv_W = 1 / vu.make_W(self.__s_bar, self.pi1, self.pi3, self.pi4, self.n_bar, self.m, self.p)
            B = vu.make_B(self.__delta_bar, self.n_bar, self.m, self.p)
            S = vu.make_S(self.__s_bar)
        self.alpha = alpha
        self.inv_W = inv_W
        self.B = B
        self.S = S


    def __posterior(self):
        
        """ creates posterior elements alpha_bar, W_bar, B_bar and S_bar defined in (4.14.54) """
        
        alpha_bar = self.alpha + self.T
        inv_W_bar = np.diag(self.inv_W) + self.X_bar.T @ self.X_bar
        W_bar = la.invert_spd_matrix(inv_W_bar)
        B_bar = W_bar @ (np.diag(self.inv_W) @ self.B + self.X_bar.T @ self.Y_bar)
        S_bar = np.diag(self.S) + self.Y_bar.T @ self.Y_bar + self.B.T @ np.diag(self.inv_W) @ self.B \
        - B_bar.T @ inv_W_bar @ B_bar
        self.alpha_bar = alpha_bar
        self.W_bar = W_bar
        self.__inv_W_bar = inv_W_bar
        self.B_bar = B_bar
        self.S_bar = S_bar
        

    def __orthogonal_triangular_block_parameters(self):
        
        """ creates U, V, H, P, Q and F defined in algorithm 14.9, along with associated elements """

        U = [None] * self.n_bar
        V = [None] * self.n_bar
        K = [None] * self.n_bar
        P = [None] * self.n_bar
        F = [None] * self.n_bar
        FU = [None] * self.n_bar
        C = [None] * self.n_bar
        for j in range(self.n_bar):
            U_j = np.eye(self.n_bar,j+1)
            if j < self.n:
                V_j = sla.block_diag(np.eye(self.m), np.kron(np.eye(self.p), np.eye(self.n_bar,self.n)))
            else:
                V_j = np.eye(self.k_bar)
            inv_H_j = V_j.T @ self.__inv_W_bar @ V_j
            H_j = la.invert_spd_matrix(inv_H_j)
            K_j = la.cholesky_nspd(H_j)
            P_j = H_j @ V_j.T @ self.__inv_W_bar @ self.B_bar @ U_j 
            Q_j = self.alpha_bar * la.invert_spd_matrix(U_j.T @ self.S_bar @ U_j + \
                  U_j.T @ self.B_bar.T @ self.__inv_W_bar @ self.B_bar @ U_j - P_j.T @ inv_H_j @ P_j) 
            F_j = la.cholesky_nspd(Q_j)
            FU_j = F_j.T @ U_j.T
            C_j = [i for i in range(self.n_bar) if i != j]
            U[j] = U_j
            V[j] = V_j
            K[j] = K_j
            P[j] = P_j
            F[j] = F_j
            FU[j] = FU_j
            C[j] = C_j
        z = np.zeros(self.n)
        z[:self.n-self.h] = self.h
        x_dim = (np.ones(self.n) + self.n - np.arange(1,self.n+1) - z).astype(int)
        self.__U = U
        self.__V = V
        self.__K = K
        self.__P = P
        self.__F = F
        self.__FU = FU
        self.__C = C
        self.__x_dim = x_dim
            
                
    def __parameter_mcmc(self):  

        """ Gibbs sampler for proxy SVAR parameters H_bar_0 and H_bar_+, following algorithm 14.12 """

        # step 1: posterior parameters
        U_j = self.__U
        V_j = self.__V
        K_j = self.__K
        P_j = self.__P
        F_j = self.__F
        FU_j = self.__FU
        C_j = self.__C
        x_dim = self.__x_dim
        
        # unpack other parameters
        Y = self.Y
        X = self.X
        n = self.n
        h = self.h
        p = self.p
        k = self.k
        n_bar = self.n_bar
        k_bar = self.k_bar
        alpha_bar = self.alpha_bar
        lamda = self.lamda
        structural_identification = self.structural_identification
        restriction_table = self.restriction_table        
        iterations = self.iterations
        burnin = self.burnin
        verbose = self.verbose
        
        # preallocate storage space
        mcmc_beta = np.zeros((k,n,iterations))
        mcmc_Sigma = np.zeros((n,n,iterations))        
        mcmc_chol_Sigma = np.zeros((n,n,iterations))
        mcmc_H = np.zeros((n,n,iterations))
        mcmc_inv_H = np.zeros((n,n,iterations))
        mcmc_V = np.zeros((h,h,iterations))
        mcmc_min_eigenvalue = np.zeros(iterations)

        # create matrices of restriction and checks, if applicable
        restriction_matrices, covariance_restriction_matrices, max_irf_period, max_zero_irf_period, no_zero_restrictions, \
        shock_history_restrictions = self.__make_restriction_matrices_and_checks(restriction_table, p)
        
        # step 2: set initial value for Lambda_0
        Lambda_bar_0 = np.eye(n_bar)
        
        # iterate over iterations
        iteration = 0
        while iteration < (burnin + iterations): 

            # step 3: sample Lambda_bar_0 and Lambda_bar_plus
            Lambda_bar_0, inv_Lambda_bar_0, Lambda_bar_plus = self.__draw_Lambda_bar_0_and_Lambda_bar_plus(U_j, V_j, \
                                            K_j, P_j, F_j, FU_j, C_j, Lambda_bar_0, alpha_bar, n_bar, k_bar)
                
            # for next steps: recover reduced-form parameters
            B, Sigma, chol_Sigma = self.__recover_reduced_form_parameters(Lambda_bar_0,\
                                   inv_Lambda_bar_0, Lambda_bar_plus, n, V_j[0])            

            # step 4: sample Q
            Q = self.__draw_Q(inv_Lambda_bar_0, chol_Sigma, B, n, h, p, x_dim, \
                no_zero_restrictions, max_zero_irf_period, restriction_matrices)
            
            # step 5: obtain SVAR parameters
            H_bar_0, H_bar_plus, H, inv_H = self.__get_svar_parameters(Lambda_bar_0, inv_Lambda_bar_0, Lambda_bar_plus, Q, n)

            # step 6: check relevance
            V, min_eigenvalue, relevance_satisfied = self.__check_relevance(H_bar_0, inv_Lambda_bar_0, Q, lamda, n, h)
            if not relevance_satisfied:
                continue
            
            # step 7: check restrictions
            if structural_identification == 4:
                restriction_satisfied = self.__check_restrictions(restriction_matrices, covariance_restriction_matrices, \
                                        shock_history_restrictions, max_irf_period, Y, X, B, H, inv_H, V, n, h, p)
                if not restriction_satisfied:
                    continue            

            # save if burn is exceeded
            if iteration >= burnin:
                mcmc_beta[:,:,iteration-burnin] = B
                mcmc_Sigma[:,:,iteration-burnin] = Sigma      
                mcmc_chol_Sigma[:,:,iteration-burnin] = chol_Sigma
                mcmc_H[:,:,iteration-burnin] = H
                mcmc_inv_H[:,:,iteration-burnin] = inv_H
                mcmc_V[:,:,iteration-burnin] = V
                mcmc_min_eigenvalue[iteration-burnin] = min_eigenvalue   

            # display progress bar
            if verbose:
                cu.progress_bar(iteration, iterations+burnin, 'Model parameters:') 
            iteration += 1            
            
        # save as attributes
        self.mcmc_beta = mcmc_beta
        self.mcmc_Sigma = mcmc_Sigma
        self._mcmc_chol_Sigma = mcmc_chol_Sigma
        self.mcmc_H = mcmc_H
        self._mcmc_inv_H = mcmc_inv_H
        self.mcmc_Gamma = np.ones((iterations,n))
        self.mcmc_V = mcmc_V
        self.mcmc_min_eigenvalue = mcmc_min_eigenvalue
        self._svar_index = np.arange(iterations)   

    
    def __make_restriction_matrices_and_checks(self, restriction_table, p):
        
        """ elements for later check of restrictions """
        
        if len(restriction_table) == 0:
            restriction_matrices = []
            covariance_restriction_matrices = []
            max_irf_period = 0
            max_zero_irf_period = 0
            no_zero_restrictions = True
            shock_history_restrictions = False
        else:
            restriction_matrices, max_irf_period = vu.make_restriction_matrices(restriction_table, p)
            covariance_restriction_matrices = vu.make_covariance_restriction_matrices(restriction_table)
            no_zero_restrictions, max_zero_irf_period, shock_history_restrictions = self.__make_restriction_checks(restriction_table)            
        return restriction_matrices, covariance_restriction_matrices, max_irf_period, max_zero_irf_period, no_zero_restrictions, shock_history_restrictions
    

    def __make_restriction_checks(self, restriction_table):
        
        """ boolean for zero/shock resrictions and max IRF period for zero restrictions """

        zero_restriction_table = restriction_table[restriction_table[:,0]==1]
        zero_restriction_number = len(zero_restriction_table)        
        no_zero_restrictions = (zero_restriction_number == 0)
        if no_zero_restrictions:
            max_zero_irf_period = 0
        else:
            max_zero_irf_period = int(np.max(zero_restriction_table[:,2]))
        shock_restriction_table = restriction_table[restriction_table[:,0]==3]
        history_restriction_table = restriction_table[restriction_table[:,0]==4]
        shock_history_restriction_table = np.vstack([shock_restriction_table,history_restriction_table])
        shock_history_restriction_number = len(shock_history_restriction_table)       
        shock_history_restrictions = (shock_history_restriction_number != 0)
        return no_zero_restrictions, max_zero_irf_period, shock_history_restrictions


    def __draw_Lambda_bar_0_and_Lambda_bar_plus(self, U, V, K, P, F, FU, C, Lambda_bar_0, alpha_bar, n_bar, k_bar):
        
        """ generation of proxy SVAR parameters Lambda_bar_0 and Lambda_bar_plus, following algorithm 14.9 """
        
        Lambda_bar_plus = np.zeros((n_bar,k_bar))
        for j in range(n_bar):
            # step 3
            temp = Lambda_bar_0[C[j],:]
            kernel = la.nullspace(temp)
            z = kernel[:,0]
            # step 4
            w_1 = FU[j] @ z
            w_1 = w_1 / nla.norm(w_1)
            # step 5
            w = np.zeros((j+1,j+1))
            w[:,0] = w_1
            for i in range(1,j+1):
                w_i = np.zeros(j+1)
                c_i = w_1[:i+1] @ w_1[:i+1]
                c_i_1 = w_1[:i] @ w_1[:i]
                w_i[i] = - c_i_1
                w_i[:i] = w_1[:i] * w_1[i]
                w_i = w_i / mt.sqrt(c_i * c_i_1)
                w[:,i] = w_i
            # step 6
            beta = np.zeros(j+1)
            s = nrd.randn(alpha_bar+1) / mt.sqrt(alpha_bar) 
            r = s @ s
            beta[0] = rd.choice([-1, 1]) * mt.sqrt(r)
            # step 7
            beta[1:] = nrd.randn(j) / mt.sqrt(alpha_bar)
            # step 8
            gamma_0 = F[j] @ (w @ beta)
            gamma_0 = np.sign(gamma_0[j]) * gamma_0
            # step 9
            gamma_plus = K[j] @ nrd.randn(P[j].shape[0]) + P[j] @ gamma_0
            # step 10
            lambda_0 = U[j] @ gamma_0
            lambda_plus = V[j] @ gamma_plus
            Lambda_bar_0[j,:] = lambda_0
            Lambda_bar_plus[j,:] = lambda_plus
        inv_Lambda_bar_0 = la.invert_lower_triangular_matrix(Lambda_bar_0)
        return Lambda_bar_0, inv_Lambda_bar_0, Lambda_bar_plus
            
    
    def __recover_reduced_form_parameters(self, Lambda_bar_0, inv_Lambda_bar_0, Lambda_bar_plus, n, V_j):
        
        """ recover reduced-form parameters """

        inv_Lambda_0 = inv_Lambda_bar_0[:n,:n]
        chol_Sigma = inv_Lambda_0
        Sigma = chol_Sigma @ chol_Sigma.T
        Lambda_plus = Lambda_bar_plus[:n,:] @ V_j
        B = (inv_Lambda_0 @ Lambda_plus).T
        return B, Sigma, chol_Sigma


    def __draw_Q(self, inv_Lambda_bar_0, chol_Sigma, B, n, h, p, x_dim, no_zero_restrictions, max_zero_irf_period, restriction_matrices):
        
        """ Q rotation matrix, with or without additional zero restrictions """

        if no_zero_restrictions:
            Q = self.__draw_regular_Q(inv_Lambda_bar_0, x_dim, n, h)
        else:
            irf = vu.impulse_response_function(B, n, p, max_zero_irf_period)
            structural_irf = vu.structural_impulse_response_function(irf, chol_Sigma, n)
            Q = self.__draw_zero_restriction_Q(inv_Lambda_bar_0, x_dim, n, h, restriction_matrices[0], structural_irf)  
        return Q


    def __draw_regular_Q(self, inv_Lambda_bar_0, x_dim, n, h):
        
        """ generation of proxy SVAR parameter Q, following algorithm 14.10 """        

        G = inv_Lambda_bar_0[-h:,:n]
        # initiate Q_1 and iterate
        Q_1 = np.zeros((n,n))
        for j in range(n):
            # step 1
            x1_j = nrd.randn(x_dim[j])
            w1_j = x1_j / nla.norm(x1_j)
            # step 2
            if j < n-h:
                M1_j = np.hstack([Q_1[:,:j],G.T]).T
            else:
                M1_j = Q_1[:,:j].T
            K1_j = la.nullspace(M1_j)
            Q_1[:,j] = K1_j @ w1_j
        # step 3
        Q_2 = rng.uniform_orthogonal(h)
        # step 4
        Q = sla.block_diag(Q_1, Q_2).T
        return Q


    def __draw_zero_restriction_Q(self, inv_Lambda_bar_0, x_dim, n, h, restriction_matrix, irf):
        
        """ generation of proxy SVAR parameter Q, following algorithm 14.12 """        

        zero_restriction_shocks = np.unique(restriction_matrix[:,1])
        G = inv_Lambda_bar_0[-h:,:n]
        # initiate Q_1 and iterate
        Q_1 = np.zeros((n,n))
        for j in range(n):
            # G_j for zero restrictions
            if j in zero_restriction_shocks:
                shock_restrictions = restriction_matrix[restriction_matrix[:,1]==j]
                G_j = irf[shock_restrictions[:,0],:,shock_restrictions[:,2]]
                h_j = G_j.shape[0]
            else:
                G_j = np.empty([0,n])
                h_j = 0
            # step 1
            x1_j = nrd.randn(x_dim[j]-h_j)
            w1_j = x1_j / nla.norm(x1_j)
            # step 2
            if j < n-h:
                M1_j = np.hstack([Q_1[:,:j],G.T,G_j.T]).T
            else:
                M1_j = np.hstack([Q_1[:,:j],G_j.T]).T
            K1_j = la.nullspace(M1_j)
            Q_1[:,j] = K1_j @ w1_j
        # step 3
        Q_2 = rng.uniform_orthogonal(h)
        # step 4
        Q = sla.block_diag(Q_1, Q_2).T
        return Q


    def __get_svar_parameters(self, Lambda_bar_0, inv_Lambda_bar_0, Lambda_bar_plus, Q, n):
        
        """ SVAR parameters from orthogonal triangular-block parameterization """  
        
        H_bar_0 = Q @ Lambda_bar_0
        H_bar_plus = Q @ Lambda_bar_plus
        inv_Lambda_0 = inv_Lambda_bar_0[:n,:n]
        Q_1 = Q[:n,:n]
        H = inv_Lambda_0 @ Q_1.T  
        inv_H = H_bar_0[:n,:n]
        return H_bar_0, H_bar_plus, H, inv_H
        

    def __check_relevance(self, H_bar_0, inv_Lambda_bar_0, Q, lamda, n, h):
        
        """ check relevance conditions """

        # create Gamma_01, inv_Gamma_02 and inv_H_0
        Gamma_01 = H_bar_0[n:,:n]
        Q_1 = Q[:n,:n]
        Q_2 = Q[n:,n:] 
        inv_Lambda_0 = inv_Lambda_bar_0[:n,:n]
        inv_Lambda_02 = inv_Lambda_bar_0[n:,n:]
        inv_H_0 = inv_Lambda_0 @ Q_1.T
        inv_Gamma_02 = inv_Lambda_02 @ Q_2.T
        # create V from (4.14.46)
        E_r_xi = - inv_Gamma_02 @ Gamma_01 @ inv_H_0
        V = E_r_xi[:,n-h:]
        # relevance matrix P, from (4.14.54)
        VV = V @ V.T
        P = la.backslash_inversion(inv_Gamma_02 @ inv_Gamma_02.T + VV, VV)
        # minimum eigenvalue
        eigenvalues, _ = nla.eig(P)
        min_eigenvalue = min(eigenvalues)
        if min_eigenvalue > lamda:
            relevance_satisfied = True
        else:
            relevance_satisfied = False
        return V, min_eigenvalue, relevance_satisfied
        
    
    def __check_restrictions(self, restriction_matrices, covariance_restriction_matrices, shock_history_restrictions, max_irf_period, Y, X, B, H, inv_H, V, n, h, p):

        """ check of all structural identification restrictions """
                
        # if no restriction, stop restriction check
        if len(restriction_matrices) == 0:
            restriction_satisfied = True
            return restriction_satisfied
        # preliminary IRFs and shocks
        if max_irf_period != 0:
            irf = vu.impulse_response_function(B, n, p, max_irf_period)
        if shock_history_restrictions:            
            E, _ = vu.fit_and_residuals(Y, X, B)
        # check restrictions: IRF, sign
        irf_sign_index = restriction_matrices[1][0]
        if len(irf_sign_index) != 0:
            irf_sign_coefficients = restriction_matrices[1][1]
            restriction_satisfied = vu.check_irf_sign(irf_sign_index, irf_sign_coefficients, irf, H)
            if not restriction_satisfied:
                return restriction_satisfied
        # check restrictions: IRF, magnitude
        irf_magnitude_index = restriction_matrices[2][0]
        if len(irf_magnitude_index) != 0:
            irf_magnitude_coefficients = restriction_matrices[2][1]
            restriction_satisfied = vu.check_irf_magnitude(irf_magnitude_index, irf_magnitude_coefficients, irf, H)
            if not restriction_satisfied:
                return restriction_satisfied  
        # check restrictions: structural shocks, sign
        shock_sign_index = restriction_matrices[3][0]
        if len(shock_sign_index) != 0:
            shock_sign_coefficients = restriction_matrices[3][1]
            restriction_satisfied = vu.check_shock_sign(shock_sign_index, shock_sign_coefficients, E, inv_H.T)
            if not restriction_satisfied:
                return restriction_satisfied
        # check restrictions: structural shocks, magnitude
        shock_magnitude_index = restriction_matrices[4][0]
        if len(shock_magnitude_index) != 0:
            shock_magnitude_coefficients = restriction_matrices[4][1]
            restriction_satisfied = vu.check_shock_magnitude(shock_magnitude_index, shock_magnitude_coefficients, E, inv_H.T)
            if not restriction_satisfied:
                return restriction_satisfied
        # historical decomposition values if any of sign or magnitude restrictions apply
        history_sign_index = restriction_matrices[5][0]
        history_magnitude_index = restriction_matrices[6][0]
        if len(history_sign_index) != 0 or len(history_magnitude_index) != 0:
            structural_irf = vu.structural_impulse_response_function(irf, H, n)
            structural_shocks = E @ inv_H.T
        # check restrictions: historical decomposition, sign
        if len(history_sign_index) != 0:
            history_sign_coefficients = restriction_matrices[5][1]
            restriction_satisfied = vu.check_history_sign(history_sign_index, \
                                    history_sign_coefficients, structural_irf, structural_shocks)
            if not restriction_satisfied:    
                return restriction_satisfied        
        # check restrictions: historical decomposition, magnitude
        if len(history_magnitude_index) != 0:
            history_magnitude_coefficients = restriction_matrices[6][1]
            restriction_satisfied = vu.check_history_magnitude(history_magnitude_index, \
                                    history_magnitude_coefficients, structural_irf, structural_shocks)
            if not restriction_satisfied:
                return restriction_satisfied
        # check restrictions: covariance, sign
        covariance_sign_index = covariance_restriction_matrices[0][0]
        if len(covariance_sign_index) != 0:
            covariance_sign_coefficients = covariance_restriction_matrices[0][1]
            restriction_satisfied = vu.check_covariance_sign(covariance_sign_index, covariance_sign_coefficients, V, n, h)
            if not restriction_satisfied:
                return restriction_satisfied        
        # check restrictions: covariance, magnitude
        covariance_magnitude_index = covariance_restriction_matrices[1][0]
        if len(covariance_magnitude_index) != 0:
            covariance_magnitude_coefficients = covariance_restriction_matrices[1][1]
            restriction_satisfied = vu.check_covariance_magnitude(covariance_magnitude_index, covariance_magnitude_coefficients, V, n, h)
            if not restriction_satisfied:
                return restriction_satisfied   
        restriction_satisfied = True
        return restriction_satisfied


    def __parameter_estimates(self):
        
        """
        point estimates and credibility intervals for model parameters
        use empirical quantiles from MCMC algorithm
        """
        
        # unpack
        mcmc_beta = self.mcmc_beta
        mcmc_Sigma = self.mcmc_Sigma
        mcmc_H = self.mcmc_H
        mcmc_Gamma = self.mcmc_Gamma
        mcmc_V = self.mcmc_V
        mcmc_min_eigenvalue = self.mcmc_min_eigenvalue
        k, n = self.k, self.n
        credibility_level = self.credibility_level
        # initiate and fill storage for beta, 4 columns: lower bound, median, upper bound, standard deviation
        beta_estimates = np.zeros((k,n,4))
        beta_estimates[:,:,:3] = vu.posterior_estimates(mcmc_beta, credibility_level)
        beta_estimates[:,:,3] = np.std(mcmc_beta,axis=2)
        # other estimates for Sigma, H, Gamma
        Sigma_estimates = np.quantile(mcmc_Sigma,0.5,axis=2)
        H_estimates = np.quantile(mcmc_H,0.5,axis=2)
        Gamma_estimates = np.quantile(mcmc_Gamma,0.5,axis=0)        
        # proxy SVAR specific estimates
        V_estimates = np.quantile(mcmc_V,0.5,axis=2)
        min_eigenvalue_estimates = np.quantile(mcmc_min_eigenvalue,0.5)
        # save as attributes
        self.beta_estimates = beta_estimates
        self.Sigma_estimates = Sigma_estimates
        self.H_estimates = H_estimates
        self.Gamma_estimates = Gamma_estimates
        self.V_estimates = V_estimates
        self.min_eigenvalue_estimates = min_eigenvalue_estimates
        
        
        