# imports
import numpy as np
from alexandria.vector_autoregression.vector_autoregression import VectorAutoRegression
from alexandria.vector_autoregression.bayesian_var import BayesianVar
import alexandria.vector_autoregression.var_utilities as vu
import alexandria.math.linear_algebra as la
import alexandria.console.console_utilities as cu
import alexandria.math.stat_utilities as su
import alexandria.math.random_number_generators as rng


class DummyObservationBayesianVar(VectorAutoRegression,BayesianVar):
    
    
    """
    Dummy observation vector autoregression, developed in section 11.5
    
    Parameters:
    -----------
    endogenous : ndarray of size (n_obs,n)
        endogenous variables, defined in (4.11.1)
    
    exogenous : ndarray of size (n_obs,m), default = []
        exogenous variables, defined in (4.11.1)
    
    structural_identification : int, default = 2
        structural identification scheme, as defined in section 13.2
        1 = none, 2 = Cholesky, 3 = triangular, 4 = restrictions
    
    restriction_table : ndarray
        numerical matrix of restrictions for structural identification
    
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
    
    pi5 : float, default = 1
        sums-of-coefficients hyperparameter, defined in (4.12.6)     
    
    pi6 : float, default = 0.1
        initial observation hyperparameter, defined in (4.12.10)   
    
    pi7 : float, default = 0.1
        long-run hyperparameter, defined in (4.12.16)      
    
    sums_of_coefficients : bool, default = False
        if True, applies sums-of-coefficients, as defined in section 12.2
    
    dummy_initial_observation : bool, default = False
        if True, applies dummy initial observation, as defined in section 12.2    
    
    long_run_prior : bool, default = False
        if True, applies long-run prior, as defined in section 12.2     
    
    long_run_table : ndarray, default = []
        numerical matrix of long-run prior
    
    stationary_prior : bool, default = False
        if True, applies stationary prior, as defined in section 12.4       
    
    credibility_level : float, default = 0.95
        VAR model credibility level (between 0 and 1)
    
    iterations : int, default = 2000
        number of Gibbs sampler replications   
    
    verbose : bool, default = False
        if True, displays a progress bar 
    
    
    Attributes
    ----------
    endogenous : ndarray of size (n_obs,n)
        endogenous variables, defined in (4.11.1)
    
    exogenous : ndarray of size (n_obs,m)
        exogenous variables, defined in (4.11.1)
    
    structural_identification : int
        structural identification scheme, as defined in section 13.2
        1 = none, 2 = Cholesky, 3 = triangular, 4 = restrictions
    
    restriction_table : ndarray
        numerical matrix of dictural identification restrictions
    
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
    
    pi3 : float
        lag decay hyperparameter, defined in (4.11.17)    
    
    pi4 : float
        exogenous slackness hyperparameter, defined in (4.11.19)    
    
    pi5 : float
        sums-of-coefficients hyperparameter, defined in (4.12.6)     
    
    pi6 : float
        initial observation hyperparameter, defined in (4.12.10)   
    
    pi7 : float
        long-run hyperparameter, defined in (4.12.16)  
    
    sums_of_coefficients : bool
        if True, applies sums-of-coefficients, as defined in section 12.2
    
    dummy_initial_observation : bool
        if True, applies dummy initial observation, as defined in section 12.2    
    
    long_run_prior : bool
        if True, applies long-run prior, as defined in section 12.2    
    
    J : ndarray of size (n,n)
        matrix of long-run prior coefficients, defined in (4.12.15)
    
    stationary_prior : bool
        if True, applies stationary prior, as defined in section 12.4       
    
    credibility_level : float
        VAR model credibility level (between 0 and 1)
    
    iterations : int
        number of Gibbs sampler replications   
    
    verbose : bool
        if True, displays a progress bar      
    
    Y_dum : ndarray of size (n,n)
        dummy observation prior Y matrix, defined in (4.12.6)
    
    X_dum : ndarray of size (n,k)
        dummy observation prior X matrix, defined in (4.12.6)
    
    B_hat : ndarray of size (k,n)
        posterior mean of VAR coefficients, defined in (4.11.50)
    
    W_hat : ndarray of size (k,k)
        posterior mean of VAR coefficients, defined in (4.11.50) 
    
    alpha_hat : float
        posterior degrees of freedom, defined in (4.11.50)
    
    S_hat : ndarray of size (n,n)
        posterior scale matrix, defined in (4.11.50) 
        
    alpha_tilde : float
        posterior degrees of freedom, defined in (4.11.53)
    
    S_tilde : ndarray of size (n,n)
        posterior scale matrix, defined in (4.11.53)     
              
    beta_estimates : ndarray of size (k,n,3)
        estimates of VAR coefficients
        page 1: median, page 2: st dev,  page 3: lower bound, page 4: upper bound
    
    Sigma_estimates : ndarray of size (n,n)
        estimates of variance-covariance matrix of VAR residuals
    
    mcmc_beta : ndarray of size (k,n,iterations)
        MCMC values of VAR coefficients   
    
    mcmc_Sigma : ndarray of size (n,n,iterations)
        MCMC values of residual variance-covariance matrix   
    
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
    
    Y_sum : ndarray of size (n,n)
        sums-of-coefficients Y matrix, defined in (4.12.6)
    
    X_sum : ndarray of size (n,k)
        sums-of-coefficients X matrix, defined in (4.12.6)
    
    Y_obs : ndarray of size (1,n)
        dummy initial observation Y matrix, defined in (4.12.10)
    
    X_obs : ndarray of size (1,k)
        dummy initial observation X matrix, defined in (4.12.10)
    
    Y_lrp : ndarray of size (1,n)
        long run prior Y matrix, defined in (4.12.16)
    
    X_lrp : ndarray of size (1,k)
        long run prior X matrix, defined in (4.12.16)
    
    Y_d : ndarray of size (T_d,n)
        full Y matrix combining sample data and dummy observations, defined in (4.11.62)
    
    X_d : ndarray of size (T_d,k)
        full X matrix combining sample data and dummy observations, defined in (4.11.62)
    
    T_d : int
        total number of observations combining sample data and dummy observations, defined in (4.11.62)
    
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
    

    def __init__(self, endogenous, exogenous = [], structural_identification = 2, 
                 restriction_table = [], lags = 4, constant = True, trend = False, 
                 quadratic_trend = False, ar_coefficients = 0.95, pi1 = 0.1, 
                 pi3 = 1, pi4 = 100, pi5 = 1, pi6 = 0.1, pi7 = 0.1, 
                 sums_of_coefficients = False, dummy_initial_observation = False, 
                 long_run_prior = False, long_run_table = [], stationary_prior = False, 
                 credibility_level = 0.95, iterations = 2000, verbose = False):

        """
        constructor for the DummyObservationBayesianVar class
        """
        
        self.endogenous = endogenous
        self.exogenous = exogenous
        self.structural_identification = structural_identification
        self.restriction_table = restriction_table
        self.lags = lags
        self.constant = constant
        self.trend = trend
        self.quadratic_trend = quadratic_trend
        self.ar_coefficients = ar_coefficients
        self.pi1 = pi1
        self.pi3 = pi3
        self.pi4 = pi4
        self.pi5 = pi5
        self.pi6 = pi6
        self.pi7 = pi7
        self.sums_of_coefficients = sums_of_coefficients
        self.dummy_initial_observation = dummy_initial_observation
        self.long_run_prior = long_run_prior
        self.J = long_run_table
        self.stationary_prior = stationary_prior
        self.credibility_level = credibility_level
        self.iterations = iterations
        self.verbose = verbose  
        # make regressors
        self._make_regressors()
        # make delta
        self._make_delta()        
        # make individual residual variance
        self._individual_ar_variances()


    def estimate(self):
    
        """
        estimate()
        generates posterior estimates for Bayesian VAR model parameters beta and Sigma
        
        parameters:
        none
        
        returns:
        none    
        """    
        
        # generate dummy extensions, if applicable
        self._dummy_extensions()
        # define prior values
        self.__prior()
        # define posterior values
        self.__posterior()
        # obtain posterior estimates for regression parameters
        self.__parameter_estimates()   
        # run MCMC algorithm (Gibbs sampling) for VAR parameters
        self.__parameter_mcmc()
        # estimate structural identification
        self._make_structural_identification()


    #---------------------------------------------------
    # Methods (Access = private)
    #--------------------------------------------------- 


    def __prior(self):
        
        """ creates prior elements Y_dum and X_dum defined in (4.11.54) """
        
        # matrices H, J and K
        H = np.diag(self.delta * self.s ** 0.5)
        J = np.diag((np.arange(self.lags) + 1) ** self.pi3)
        K = np.diag(self.s ** 0.5)
        # create the different blocks and concatenate
        block_1 = np.zeros((self.m,self.n))
        block_2 = H / self.pi1
        block_3 = np.zeros((self.n*(self.lags-1),self.n))
        block_4 = K
        Y_dum = np.vstack([block_1,block_2,block_3,block_4])
        block_5 = np.eye(self.m) / (self.pi1 * self.pi4)
        block_6 = np.zeros((self.m,self.n*self.lags))
        block_7 = np.zeros((self.n*self.p,self.m))
        block_8 = np.kron(J,K) / self.pi1
        block_9 = np.zeros((self.n,self.m))
        block_10 = np.zeros((self.n,self.n*self.lags))
        X_dum = np.vstack([np.hstack([block_5,block_6]),np.hstack([block_7,block_8]),\
                           np.hstack([block_9,block_10])])
        self.Y_dum = Y_dum
        self.X_dum = X_dum
        # update data with dummy observations, following (4.11.62)
        self.Y_d = np.vstack([Y_dum,self.Y_d])
        self.X_d = np.vstack([X_dum,self.X_d])
        self.T_d = self.T_d + Y_dum.shape[0]
        self._XX_d = self.X_d.T @ self.X_d
        self._XY_d = self.X_d.T @ self.Y_d
        self._YY_d = self.Y_d.T @ self.Y_d        
        

    def __posterior(self):
        
        """ creates posterior elements B_hat, W_hat, alpha_hat, S_hat, alpha_tilde and S_tilde
            defined in (4.11.50) and (4.11.53) """
        
        # unpack
        Y, X, XX, XY, T = self.Y_d, self.X_d, self._XX_d, self._XY_d, self.T_d
        n, k = self.n, self.k
        # define posterior elements
        inv_XX = la.invert_spd_matrix(XX)
        W_hat = inv_XX
        B_hat = inv_XX @ XY
        alpha_hat = T - k + 2
        E = Y - X @ B_hat
        S_hat = E.T @ E
        alpha_tilde = T - n - k + 3
        S_tilde = S_hat / alpha_tilde
        # save as attributes
        self.B_hat = B_hat
        self.W_hat = W_hat
        self.alpha_hat = alpha_hat
        self.S_hat = S_hat
        self.alpha_tilde = alpha_tilde
        self.S_tilde = S_tilde
        

    def __parameter_estimates(self):
        
        """
        point estimates and credibility intervals for model parameters
        use the inverse Wishart defined in (4.11.51) and matrix Student defined in (4.11.53)
        """
        
        # mean and standard deviation of posterior distribution, using matrix Student properties
        mean = self.B_hat
        scale = np.reshape(np.diag(np.kron(self.S_tilde, self.W_hat)), [self.k,self.n], order='F')
        standard_deviation = (self.alpha_tilde / (self.alpha_tilde - 2) * scale) ** 0.5
        # critical value of Student distribution for credibility level
        credibility_level = self.credibility_level     
        Z = su.student_icdf((1 + credibility_level) / 2, self.alpha_tilde)
        # initiate storage: 4 columns: lower bound, median, upper bound, standard deviation
        beta_estimates = np.zeros((self.k,self.n,4))        
        # fill estimates
        beta_estimates[:,:,0] = mean
        beta_estimates[:,:,1] = mean - Z * scale ** 0.5
        beta_estimates[:,:,2] = mean + Z * scale ** 0.5
        beta_estimates[:,:,3] = standard_deviation        
        # posterior estimates for Sigma, using mean of inverse Wishart
        Sigma_estimates = self.S_hat / (self.alpha_hat - self.n - 1)
        # save as attributes
        self.beta_estimates = beta_estimates
        self.Sigma_estimates = Sigma_estimates


    def __parameter_mcmc(self):
        
        """ Gibbs sampler for VAR parameters beta and Sigma """
        
        # unpack
        B_hat, W_hat, alpha_hat, S_hat = self.B_hat, self.W_hat, self.alpha_hat, self.S_hat
        iterations = self.iterations
        n, m, p, k = self.n, self.m, self.p, self.k
        # initiate storage
        mcmc_beta = np.zeros((k,n,iterations))
        mcmc_Sigma = np.zeros((n,n,iterations)) 
        mcmc_chol_Sigma = np.zeros((n,n,iterations)) 
        # compute constant Cholesky factors
        chol_W_hat = la.cholesky_nspd(W_hat)
        chol_S_hat = la.cholesky_nspd(S_hat)
        # loop over iterations, checking for stationarity if stationary prior is activated
        iteration = 0
        while iteration < iterations:
            # sample Sigma from inverse Wishart posterior (4.11.34)
            Sigma = rng.inverse_wishart(alpha_hat, chol_S_hat, True)
            chol_Sigma = la.cholesky_nspd(Sigma)
            # sample B from matrix normal distribution defined in (4.12.34)
            B = rng.matrix_normal(B_hat, chol_W_hat, Sigma, True, False)
            if self.stationary_prior:
                stationary = vu.check_stationarity(B, n, m, p)
            else:
                stationary = True
            if stationary:
                mcmc_beta[:,:,iteration] = B
                mcmc_Sigma[:,:,iteration] = Sigma
                mcmc_chol_Sigma[:,:,iteration] = chol_Sigma
                # if verbose, display progress bar
                if self.verbose:
                    cu.progress_bar(iteration, iterations, 'Model parameters:')
                iteration += 1
        self.mcmc_beta = mcmc_beta
        self.mcmc_Sigma = mcmc_Sigma
        self._mcmc_chol_Sigma = mcmc_chol_Sigma
                    
            
