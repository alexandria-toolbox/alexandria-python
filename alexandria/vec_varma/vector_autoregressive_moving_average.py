# imports
import numpy as np
import alexandria.vector_autoregression.var_utilities as vu
import alexandria.processor.input_utilities as iu
from alexandria.vector_autoregression.maximum_likelihood_var import MaximumLikelihoodVar
import alexandria.math.linear_algebra as la
import alexandria.math.random_number_generators as rng
import alexandria.state_space.state_space_utilities as ss
import alexandria.console.console_utilities as cu
import alexandria.vec_varma.vec_varma_utilities as vvu
from alexandria.state_space.bayesian_state_space_sampler import BayesianStateSpaceSampler


class VectorAutoregressiveMovingAverage():
    
    
    """
    Vector Autoregressive Moving Average, developed in chapter 16
    
    Parameters:
    -----------
    endogenous : ndarray of size (n_obs,n)
        endogenous variables, defined in (5.16.1)
    
    exogenous : ndarray of size (n_obs,m), default = []
        exogenous variables, defined in (5.16.1)
    
    structural_identification : int, default = 2
        structural identification scheme, as defined in section 16.5
        1 = none, 2 = Cholesky, 3 = triangular, 4 = restrictions
    
    restriction_table : ndarray
        numerical matrix of restrictions for structural identification
    
    lags : int, default = 4
        number of lags, defined in (5.16.1)
    
    residual_lags : int, default = 1
        number of lags in past residuals, defined in (5.16.1)
    
    constant : bool, default = True
        if True, an intercept is included in the VAR model exogenous
    
    trend : bool, default = False
        if True, a linear trend is included in the VAR model exogenous
    
    quadratic_trend : bool, default = False
        if True, a quadratic trend is included in the VAR model exogenous
    
    ar_coefficients : float or ndarray of size (n_endo,), default = 0.95
        prior mean delta for AR coefficients, defined in (5.16.9)
    
    pi1 : float, default = 0.1
        overall tightness hyperparameter, defined in (5.16.9)
    
    pi2 : float, default = 0.5
        cross-variable shrinkage hyperparameter, defined in (5.16.9)
    
    pi3 : float, default = 1
        lag decay hyperparameter, defined in (5.16.9)    
    
    pi4 : float, default = 100
        exogenous slackness hyperparameter, defined in (5.16.9)      
    
    lambda1 : float, default = 0.1
        overall tightness hyperparameter, defined in (5.16.10)
    
    lambda2 : float, default = 0.5
        cross-variable shrinkage hyperparameter, defined in (5.16.10)
    
    lambda3 : float, default = 1
        lag decay hyperparameter, defined in (5.16.10) 
    
    credibility_level : float, default = 0.95
        VAR model credibility level (between 0 and 1)
    
    iterations : int, default = 2000
        number of Gibbs sampler replications   
    
    burnin : int, default = 1000
        number of Gibbs sampler burn-in replications  
    
    verbose : bool, default = False
        if True, displays a progress bar 
    
    
    Properties
    ----------
    endogenous : ndarray of size (n_obs,n)
        endogenous variables, defined in (5.16.1)
    
    exogenous : ndarray of size (n_obs,m), default = []
        exogenous variables, defined in (5.16.1)
    
    structural_identification : int, default = 2
        structural identification scheme, as defined in section 16.5
        1 = none, 2 = Cholesky, 3 = triangular, 4 = restrictions
    
    restriction_table : ndarray
        numerical ndarray of restrictions for structural identification
    
    lags : int, default = 4
        number of lags, defined in (5.16.1)
    
    residual_lags : int, default = 1
        number of residual lags, defined in (5.16.1)
    
    constant : bool, default = True
        if True, an intercept is included in the VAR model exogenous
    
    trend : bool, default = False
        if True, a linear trend is included in the VAR model exogenous
    
    quadratic_trend : bool, default = False
        if True, a quadratic trend is included in the VAR model exogenous
    
    ar_coefficients : float or ndarray of size (n_endo,), default = 0.95
        prior mean delta for AR coefficients, defined in (5.16.9)
    
    pi1 : float, default = 0.1
        overall tightness hyperparameter, defined in (5.16.9)
    
    pi2 : float, default = 0.5
        cross-variable shrinkage hyperparameter, defined in (5.16.9)
    
    pi3 : float, default = 1
        lag decay hyperparameter, defined in (5.16.9)    
    
    pi4 : float, default = 100
        exogenous slackness hyperparameter, defined in (5.16.9)    
    
    lambda1 : float, default = 0.1
        overall tightness hyperparameter, defined in (5.16.10)
    
    lambda2 : float, default = 0.5
        cross-variable shrinkage hyperparameter, defined in (5.16.10)
    
    lambda3 : float, default = 1
        lag decay hyperparameter, defined in (5.16.10) 
    
    credibility_level : float, default = 0.95
        VAR model credibility level (between 0 and 1)
    
    iterations : int, default = 2000
        number of Gibbs sampler replications   
    
    burnin : int, default = 1000
        number of Gibbs sampler burn-in replications  
    
    verbose : bool, default = False
        if True, displays a progress bar 
    
    Y : ndarray of size (T,n)
        ndarray of in-sample endogenous variables, defined in (5.16.3)
    
    X : ndarray of size (T,k1)
        ndarray of exogenous and lagged regressors, defined in (5.16.3)
    
    n : int
        number of endogenous variables, defined in (5.16.1)
    
    m : int
        number of exogenous variables, defined in (5.16.1)
    
    p : int
        number of lags, defined in (5.16.1)
    
    q : int
        number of residual lags, defined in (5.16.1)
        
    T : int
        number of sample periods, defined in (5.16.1)
    
    k1 : int
        number of autoregressive coefficients in each equation, defined in (5.16.1)
    
    k2 : int
        number of lagged residual coefficients in each equation, defined in (5.16.1)
    
    k : int
        total number of coefficients in each equation, defined in (5.16.1)
    
    r1 : int
        total number of autoregressive coefficients in the model, defined in (5.16.1)
    
    r2 : int
        totla number of lagged residual coefficients in the model, defined in (5.16.1)
    
    r : int
        total number of coefficients in the model, defined in (5.16.1)
    
    delta : ndarray of size (n,)
        prior mean delta for AR coefficients, defined in (5.16.9)
    
    s : ndarray of size (n,)
        prior scale ndarray, defined in (5.16.11) 
    
    b : ndarray of size (r1,)
        prior mean of VAR coefficients, defined in (5.16.9)
    
    V : ndarray of size (q,q)
        prior variance of autoregressive coefficients, defined in (5.16.9)           
    
    W : ndarray of size (q,q)
        prior variance of lagged residual coefficients, defined in (5.16.10)           
    
    alpha : float
        prior degrees of freedom, defined in (5.16.11)
    
    S : ndarray of size (n,)
        prior scale ndarray, defined in (5.16.11) 
    
    alpha_bar : float
        posterior degrees of freedom, defined in (5.16.23)
    
    mcmc_beta : ndarray of size (k1,n,iterations)
        MCMC values of autoregressive coefficients   
    
    mcmc_kappa : ndarray of size (k2,n,iterations)
        MCMC values of autoregressive coefficients   
       
    mcmc_Z : ndarray of size (T,k2,iterations)
        MCMC values of lagged residuals  
          
    mcmc_E : ndarray of size (T,n,iterations)
        MCMC values of current residuals  
             
    mcmc_Sigma : ndarray of size (n,n,iterations)
        MCMC values of residual variance-covariance ndarray
     
    mcmc_chol_Sigma : ndarray of size (n,n,iterations)
        MCMC values of residual variance-covariance ndarray (Cholesky factor)
    
    mcmc_inv_Sigma : ndarray of size (n,n,iterations)
        MCMC values of residual variance-covariance ndarray (inverse)
    
    beta_estimates : ndarray of size (k1,n,3)
        estimates of VAR coefficients
        page 1: median, page 2: st dev,  page 3: lower bound, page 4: upper bound
    
    kappa_estimates : ndarray of size (k2,n,3)
        estimates of lagged residual coefficients
        page 1: median, page 2: st dev,  page 3: lower bound, page 4: upper bound
     
    E_estimates : ndarray of size (T,n)
        estimates of current residuals
    
    Z_estimates : ndarray of size (T,k2)
        estimates of lagged residuals
       
    Sigma_estimates : ndarray of size (n,n)
        estimates of variance-covariance ndarray of VAR residuals
    
    mcmc_H :  ndarray of size (n,n,iterations)
        MCMC values of structural identification ndarray, defined in (4.13.5)
    
    mcmc_Gamma : ndarray of size (iterations,n)
        MCMC values of structural shock variance ndarray, defined in definition 13.1
    
    H_estimates : ndarray of size (n,n)
        posterior estimates of structural ndarray, defined in section 13.2
    
    Gamma_estimates : ndarray of size (1,n)
        estimates of structural shock variance ndarray, defined in section 13.2
    
    steady_state_estimates : ndarray of size (T,n,3)
        estimates of steady-state, defined in (4.12.30)
    
    fitted_estimates : ndarray of size (T,n,3)
        estimates of in-sample fit, defined in (4.11.2)
    
    residual_estimates : ndarray of size (T,n,3)
        estimates of in-sample residuals, defined in (4.11.2)
    
    mcmc_structural_shocks : ndarray of size (T,n,iterations)
        MCMC values of structural shocks
    
    structural_shocks_estimates : ndarray of size (T,n,3)
        estimates of in-sample structural shocks, defined in definition 13.1
    
    insample_evaluation : struct
        in-sample evaluation criteria, defined in (4.13.15)-(4.13.17)
    
    mcmc_forecasts : ndarray of size (f_periods,n,iterations)
        MCMC values of forecasts
    
    forecast_estimates : ndarray of size (f_periods,n,3)
        forecast estimates, defined in (4.13.12) and (4.13.13)
        page 1: median, page 2: lower bound, page 3: upper bound
    
    forecast_evaluation_criteria : struct
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
    """    
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------    
    

    def __init__(self, endogenous, exogenous = [], structural_identification = 2, 
                 restriction_table = [], lags = 4, residual_lags = 1,
                 constant = True, trend = False, quadratic_trend = False, 
                 ar_coefficients = 0.95, pi1 = 0.1, pi2 = 0.5, pi3 = 1, pi4 = 100, 
                 lambda1 = 0.1, lambda2 = 0.5, lambda3 = 1, credibility_level = 0.95,
                 iterations = 3000, burnin = 1000, verbose = False):

        """
        constructor for the VectorAutoregressiveMovingAverage class
        """
        
        self.endogenous = endogenous
        self.exogenous = exogenous
        self.structural_identification = structural_identification
        self.restriction_table = restriction_table
        self.lags = lags
        self.residual_lags = residual_lags
        self.constant = constant
        self.trend = trend
        self.quadratic_trend = quadratic_trend
        self.ar_coefficients = ar_coefficients
        self.pi1 = pi1
        self.pi2 = pi2
        self.pi3 = pi3
        self.pi4 = pi4
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.credibility_level = credibility_level
        self.iterations = iterations
        self.burnin = burnin
        self.verbose = verbose
        # make regressors
        self.__make_regressors()   


    def estimate(self):
    
        """
        estimate()
        generates posterior estimates for Bayesian VARMA model parameters
        
        parameters:
        none
        
        returns:
        none    
        """    

        # define prior values
        self.__prior()
        # define posterior values
        self.__posterior()
        # run MCMC algorithm for VARMA parameters
        self.__parameter_mcmc()
        # obtain posterior estimates
        self.__parameter_estimates()
        # estimate structural identification
        self.__make_structural_identification()


    def insample_fit(self):
        
        """
        insample_fit()
        generates in-sample fit and residuals along with evaluation criteria
        
        parameters:
        none
        
        returns:
        none    
        """           
        
        # compute steady-state
        self.__steady_state()
        # compute fitted and residuals
        self.__fitted_and_residual()
        # compute in-sample criteria
        self.__insample_criteria()


    def forecast(self, h, credibility_level, Z_p=[]):
        
        """
        forecast(h, credibility_level, Z_p=[])
        estimates forecasts for the Bayesian VAR model, using algorithm 13.4
        
        parameters:
        h : int
            number of forecast periods
        credibility_level: float between 0 and 1
            credibility level for forecast credibility bands
        Z_p : empty list or numpy array of dimension (h, n_exo)
            empty list unless the model includes exogenous other than constant, trend and quadratic trend
            if not empty, n_exo is the number of additional exogenous variables
        
        returns:
        forecast_estimates : ndarray of shape (h,n,3)
            page 1: median; page 2: interval lower bound; page 3: interval upper bound
        """ 
        
        # get forecast
        self.__make_forecast(h, Z_p)
        # obtain posterior estimates
        self.__forecast_posterior_estimates(credibility_level)
        forecast_estimates = self.forecast_estimates
        return forecast_estimates


    def forecast_evaluation(self, Y):
        
        """
        forecast_evaluation(Y)
        forecast evaluation criteria for the Bayesian VARMA model, as defined in (4.13.18)-(4.13.22)
        
        parameters:
        Y : ndarray of shape (h,n)
            array of realised values for forecast evaluation, h being the number of forecast periods
            
        returns:
        forecast_evaluation_criteria : dictionary
            dictionary with criteria name as keys and corresponding number as value
        """
        
        # unpack
        Y_hat, mcmc_forecast = self.forecast_estimates[:,:,0], self.mcmc_forecast
        # obtain regular forecast evaluation criteria from equations (4.13.18) and (4.13.19)
        standard_evaluation_criteria = vu.forecast_evaluation_criteria(Y_hat, Y)
        # obtain Bayesian forecast evaluation criteria from equations (4.13.21) and (4.13.22)
        bayesian_evaluation_criteria = vu.bayesian_forecast_evaluation_criteria(mcmc_forecast, Y)
        # merge dictionaries
        forecast_evaluation_criteria = iu.concatenate_dictionaries(standard_evaluation_criteria, bayesian_evaluation_criteria)
        # save as attributes
        self.forecast_evaluation_criteria = forecast_evaluation_criteria
        return forecast_evaluation_criteria


    def impulse_response_function(self, h, credibility_level):
        
        """
        impulse_response_function(h, credibility_level)
        impulse response functions, as defined in (4.13.1)-(4.13.9)
        
        parameters:
        h : int
            number of IRF periods
        credibility_level: float between 0 and 1
            credibility level for forecast credibility bands
            
        returns:
        irf_estimates : ndarray of shape (n,n,h,3)
            first 3 dimensions are variable, shock, period; 4th dimension is median, lower bound, upper bound
        exo_irf_estimates : ndarray of shape (n,n,h,3)
            first 3 dimensions are variable, shock, period; 4th dimension is median, lower bound, upper bound
        """
        
        # get regular impulse response funtion
        self.__make_impulse_response_function(h)
        # get exogenous impuse response function
        self.__make_exogenous_impulse_response_function(h)
        # get structural impulse response function
        self.__make_structural_impulse_response_function(h)
        # obtain posterior estimates
        self.__irf_posterior_estimates(credibility_level)
        irf_estimates, exo_irf_estimates = self.irf_estimates, self.exo_irf_estimates
        return irf_estimates, exo_irf_estimates


    def forecast_error_variance_decomposition(self, h, credibility_level):
        
        """
        forecast_error_variance_decomposition(self, h, credibility_level)
        forecast error variance decomposition, as defined in (4.13.31)
        
        parameters:
        h : int
            number of FEVD periods
        credibility_level: float between 0 and 1
            credibility level for forecast credibility bands
            
        returns:
        fevd_estimates : ndarray of shape (n,n,h,3)
            first 3 dimensions are variable, shock, period; 4th dimension is median, lower bound, upper bound
        """
        
        # get forecast error variance decomposition
        self.__make_forecast_error_variance_decomposition(h)
        # obtain posterior estimates
        self.__fevd_posterior_estimates(credibility_level)
        fevd_estimates = self.fevd_estimates
        return fevd_estimates


    def historical_decomposition(self, credibility_level):
        
        """
        historical_decomposition(self, credibility_level)
        historical decomposition, as defined in (4.13.34)-(4.13-36)
        
        parameters:
        credibility_level: float between 0 and 1
            credibility level for forecast credibility bands
            
        returns:
        hd_estimates : ndarray of shape (n,n,T,3)
            first 3 dimensions are variable, shock, period; 4th dimension is median, lower bound, upper bound
        """
        
        # get historical decomposition
        self.__make_historical_decomposition()
        # obtain posterior estimates
        self.__hd_posterior_estimates(credibility_level)
        hd_estimates = self.hd_estimates
        return hd_estimates  
    
    
    def conditional_forecast(self, h, credibility_level, conditions, shocks, conditional_forecast_type, Z_p=[]):

        """
        conditional_forecast(self, h, credibility_level, conditions, shocks, conditional_forecast_type, Z_p=[])
        estimates conditional forecasts for the Bayesian VAR model, using algorithms 14.1 and 14.2
        
        parameters:
        h : int
            number of forecast periods
        credibility_level : float between 0 and 1
            credibility level for forecast credibility bands
        conditions : ndarray of shape (n_conditions,4)
            table defining conditions (column 1: variable, column 2: period, column 3: mean, column 4: variance) 
        shocks: empty list or ndarray of shape (n,)
            vector defining shocks generating the conditions; should be empty if conditional_forecast_type = 1          
        conditional_forecast_type : int
            conditional forecast type (1 = agnostic, 2 = structural)
        Z_p : empty list or ndarray of dimension (h, n_exo)
            empty list unless the model includes exogenous other than constant, trend and quadratic trend
            if not empty, n_exo is the number of additional exogenous variables
        
        returns:
        conditional_forecast_estimates : ndarray of shape (h,n,3)
            page 1: median; page 2: interval lower bound; page 3: interval upper bound
        """         
        
        # if conditional forecast type is agnostic
        if conditional_forecast_type == 1:
            # get conditional forecasts
            self.__make_conditional_forecast(h, conditions, Z_p)
        # if instead conditional forecast type is structural
        elif conditional_forecast_type == 2:
            # establish type of shocks
            shock_type = self.__check_shock_type(h, conditions, shocks)
            # get structural conditional forecasts
            self.__make_structural_conditional_forecast(h, conditions, shocks, Z_p, shock_type)
        # obtain posterior estimates
        self.__conditional_forecast_posterior_estimates(credibility_level)
        conditional_forecast_estimates = self.conditional_forecast_estimates
        return conditional_forecast_estimates       
    

    #---------------------------------------------------
    # Methods (Access = private)
    #--------------------------------------------------- 


    def __make_regressors(self):
    
        """ generates regressors defined in (5.16.3), along with other dimension elements """
              
        # define regressor matrices
        Y, X = self.__make_regressor_matrices()
        # define dimensions
        n, m, p, q, T, k1, k2, k, r1, r2, r = self.__generate_dimensions()
        # define estimation terms
        XX = X.T @ X
        # save as attributes      
        self.Y = Y
        self.X = X
        self.n = n
        self.m = m
        self.p = p
        self.q = q        
        self.T = T
        self.k1 = k1
        self.k2 = k2
        self.k = k
        self.r1 = r1
        self.r2 = r2
        self.r = r        
        self.__XX = XX


    def __make_regressor_matrices(self):
        
        Y = self.endogenous[self.lags:]        
        periods = self.endogenous.shape[0] - self.lags
        X_1 = vu.generate_intercept_and_trends(self.constant, self.trend, self.quadratic_trend, periods, 0)
        X_2 = vu.generate_exogenous_regressors(self.exogenous, self.lags, periods)
        X_3 = vu.generate_lagged_endogenous(self.endogenous, self.lags)
        X = np.hstack([X_1,X_2,X_3])
        return Y, X


    def __generate_dimensions(self):
        
        T = self.endogenous.shape[0] - self.lags
        n = self.endogenous.shape[1]
        m = int(self.constant) + int(self.trend) + int(self.quadratic_trend) 
        if len(self.exogenous) != 0:
            m += self.exogenous.shape[1]   
        p = self.lags
        q = self.residual_lags
        k1 = m + n * p
        k2 = n * q
        k = k1 + k2
        r1 = n * k1
        r2 = n * k2
        r = r1 + r2
        return n, m, p, q, T, k1, k2, k, r1, r2, r


    def __prior(self):
        
        """ creates prior elements b, V, W, alpha, S and F defined in (5.16.9), (5.16.10), (5.16.11) and (5.16.13) """
        
        delta = self.__make_delta()
        s = self.__individual_ar_variances()
        b = vu.make_b(delta, self.n, self.m, self.p)
        V = vu.make_V(s, self.pi1, self.pi2, self.pi3, self.pi4, self.n, self.m, self.p)
        W = vu.make_V(s, self.lambda1, self.lambda2, self.lambda3, 1, self.n, 0, self.q)
        alpha = vu.make_alpha(self.n)
        S = vu.make_S(s)
        F = np.eye(self.n*(self.q+1),k=-self.n)
        M = np.diag(np.hstack((np.ones(1), np.zeros(self.q))))
        self.delta = delta
        self.s = s
        self.b = b
        self.V = V
        self.W = W
        self.alpha = alpha
        self.S = S
        self.__F = F
        self.__M = M


    def __make_delta(self):    
        
        if iu.is_numeric(self.ar_coefficients):
            ar_coefficients = np.array(self.n * [self.ar_coefficients])
        else:
            ar_coefficients = self.ar_coefficients
        delta = ar_coefficients
        return delta
        
    
    def __individual_ar_variances(self):
        
        s = np.zeros(self.n)
        for i in range(self.n):
            ar = MaximumLikelihoodVar(self.endogenous[:,[i]], lags=self.lags)
            ar.estimate()
            s[i] = ar.Sigma[0,0]
        return s


    def __posterior(self):
        
        """ creates posterior elements defined in (5.16.17), (5.16.20) and (5.16.23) """
        
        inv_V, inv_V_b = vu.make_V_b_inverse(self.b, self.V)
        inv_W = np.diag(1 / self.W)
        alpha_bar = self.alpha + self.T
        self.__inv_V = inv_V
        self.__inv_V_b = inv_V_b
        self.__inv_W = inv_W
        self.alpha_bar = alpha_bar


    def __parameter_mcmc(self):
        
        """ Gibbs sampler for VAR parameters beta, kappa, Sigma and Z, following algorithm 16.1 """
        
        # unpack
        Y = self.Y
        X = self.X
        inv_V = self.__inv_V
        inv_V_b = self.__inv_V_b
        inv_W = self.__inv_W
        F = self.__F
        M = self.__M
        XX = self.__XX
        alpha_bar = self.alpha_bar
        S = self.S
        n = self.n
        q = self.q
        k1 = self.k1
        k2 = self.k2
        T = self.T
        iterations = self.iterations
        burnin = self.burnin
        verbose = self.verbose
        
        # preallocate storage space
        mcmc_beta = np.zeros((k1,n,iterations))
        mcmc_kappa = np.zeros((k2,n,iterations))
        mcmc_Z = np.zeros((T,k2,iterations))  
        mcmc_E = np.zeros((T,n,iterations))  
        mcmc_Sigma = np.zeros((n,n,iterations))
        mcmc_chol_Sigma = np.zeros((n,n,iterations))
        mcmc_inv_Sigma = np.zeros((n,n,iterations))

        # set initial values for parameters
        inv_Sigma = np.diag(1 / S)
        Sigma = np.diag(S)
        S = np.diag(S)
        Z = np.zeros((T,k2))
        K = np.zeros((k2,n))
        
        # iterate over iterations
        iteration = 0
        while iteration < (burnin + iterations):
        
            # step 2: sample beta
            beta = self.__draw_beta(Y, X, inv_V, inv_V_b, inv_Sigma, XX, Z, K)
            B = np.reshape(beta,[k1,n],order='F')
            
            # step 3: sample kappa
            kappa = self.__draw_kappa(Y, X, Z, B, inv_W, inv_Sigma)
            K = np.reshape(kappa,[k2,n],order='F')            
            
            # step 4: sample Z
            E, Z = self.__draw_Z(Y, X, B, K, F, M, Sigma, n, T, q, k2)

            # step 5: sample Sigma
            Sigma, inv_Sigma, chol_Sigma = self.__draw_Sigma(E, S, alpha_bar)           
            
            # save if burn is exceeded
            if iteration >= burnin:
                mcmc_beta[:,:,iteration-burnin] = B
                mcmc_kappa[:,:,iteration-burnin] = K
                mcmc_Z[:,:,iteration-burnin] = Z
                mcmc_E[:,:,iteration-burnin] = E
                mcmc_Sigma[:,:,iteration-burnin] = Sigma
                mcmc_chol_Sigma[:,:,iteration-burnin] = chol_Sigma
                mcmc_inv_Sigma[:,:,iteration-burnin] = inv_Sigma
            if verbose:
                cu.progress_bar(iteration, iterations+burnin, 'Model parameters:')
            iteration += 1
                    
        # save as attributes
        self.mcmc_beta = mcmc_beta
        self.mcmc_kappa = mcmc_kappa
        self.mcmc_Z = mcmc_Z
        self.mcmc_E = mcmc_E
        self.mcmc_Sigma = mcmc_Sigma
        self.__mcmc_chol_Sigma = mcmc_chol_Sigma
        self.__mcmc_inv_Sigma = mcmc_inv_Sigma
            

    def __draw_beta(self, Y, X, inv_V, inv_V_b, inv_Sigma, XX, Z, K):
        
        """ draw beta from its conditional posterior defined in (5.16.17) """
        
        # posterior V_bar
        inv_V_bar = inv_V + np.kron(inv_Sigma, XX)
        # posterior b_bar
        b_bar_temp = inv_V_b + la.vec(X.T @ (Y - Z @ K) @ inv_Sigma)  
        # efficient sampling of beta (algorithm 9.4)
        beta = rng.efficient_multivariate_normal(b_bar_temp, inv_V_bar)
        return beta 


    def __draw_kappa(self, Y, X, Z, B, inv_W, inv_Sigma):
        
        """ draw kappa from its conditional posterior defined in (5.16.20) """
        
        # posterior W_bar
        inv_W_bar = inv_W + np.kron(inv_Sigma, Z.T @ Z)
        # posterior g_bar
        g_bar_temp = la.vec(Z.T @ (Y - X @ B) @ inv_Sigma)  
        # efficient sampling of beta (algorithm 9.4)
        kappa = rng.efficient_multivariate_normal(g_bar_temp, inv_W_bar)
        return kappa 
    

    def __draw_Z(self, Y, X, B, K, F, M, Sigma, n, T, q, k2):
        
        """ draw Z from its conditional posterior, using (5.16.12)-(5.16.13) """
        
        # parameters for state-space representation
        G, Y_GX, J, Omega = self.__state_space_representation(Y, X, B, K, M, Sigma, n)
        # initial values for algorithm
        z_00, Upsilon_00 = self.__kalman_filter_initial_values(Sigma, k2, n, q)
        # Carter-Kohn algorithm: forward pass
        Z_tt, Z_tt1, Ups_tt, Ups_tt1 = self.__forward_pass(Y_GX, J, F, Omega, z_00, Upsilon_00, T, n, k2)
        # Carter-Kohn algorithm: backward pass
        Z = self.__backward_pass(Z_tt, Z_tt1, Ups_tt, Ups_tt1, F, T, n, k2)
        E = Z[:,:n]
        Z = Z[:,n:]
        return E, Z
    
    
    def __state_space_representation(self, Y, X, B, K, M, Sigma, n):
        
        G = B.T
        Y_GX = Y - X @ B
        J = np.hstack((np.eye(n),K.T))
        Omega = np.kron(M, Sigma)
        return G, Y_GX, J, Omega

    
    def __kalman_filter_initial_values(self, Sigma, k2, n, q):

        z_00 = np.zeros(k2+n)
        Upsilon_00 = np.kron(np.eye(q+1), Sigma)
        return z_00, Upsilon_00    


    def __forward_pass(self, Y_GX, J, F, Omega, z_00, Upsilon_00, T, n, k2):
        
        Z_tt, Z_tt1, Ups_tt, Ups_tt1 = ss.varma_forward_pass(Y_GX, J, F, Omega, z_00, Upsilon_00, T, n, k2+n)
        return Z_tt, Z_tt1, Ups_tt, Ups_tt1


    def __backward_pass(self, Z_tt, Z_tt1, Ups_tt, Ups_tt1, F, T, n, k2):
        
        Z = ss.varma_backward_pass(Z_tt, Z_tt1, Ups_tt, Ups_tt1, F, T, k2+n)
        return Z


    def __draw_Sigma(self, E, S, alpha_bar):
        
        """ draw Sigma from its conditional posterior defined in (5.16.23) """

        # posterior S_bar
        S_bar = S + E.T @ E
        # sample sigma
        Sigma = rng.inverse_wishart(alpha_bar, S_bar)
        # obtain related elements
        inv_Sigma = la.invert_spd_matrix(Sigma)
        chol_Sigma = la.cholesky_nspd(Sigma)
        return Sigma, inv_Sigma, chol_Sigma  


    def __parameter_estimates(self):
        
        """
        point estimates and credibility intervals for model parameters
        use empirical quantiles from MCMC algorithm
        """

        # posterior estimates for beta
        beta_estimates = np.zeros((self.k1,self.n,4))
        beta_estimates[:,:,:3] = vu.posterior_estimates(self.mcmc_beta, self.credibility_level)
        beta_estimates[:,:,3] = np.std(self.mcmc_beta,axis=2)
        # posterior estimates for K
        kappa_estimates = np.zeros((self.k2,self.n,4))
        kappa_estimates[:,:,:3] = vu.posterior_estimates(self.mcmc_kappa, self.credibility_level)
        kappa_estimates[:,:,3] = np.std(self.mcmc_kappa,axis=2)
        # posterior estimates for E, Z and Sigma
        E_estimates = np.quantile(self.mcmc_E,0.5,axis=2)
        Z_estimates = np.quantile(self.mcmc_Z,0.5,axis=2)        
        Sigma_estimates = np.quantile(self.mcmc_Sigma,0.5,axis=2)
        self.beta_estimates = beta_estimates
        self.kappa_estimates = kappa_estimates
        self.E_estimates = E_estimates
        self.Z_estimates = Z_estimates
        self.Sigma_estimates = Sigma_estimates


    def __make_structural_identification(self):
        
        """ structural identification estimates """
        
        if self.structural_identification == 2:
            self.__svar_by_choleski_factorization()
        elif self.structural_identification == 3:
            self.__svar_by_triangular_factorization()
        elif self.structural_identification == 4:
            self.__svar_by_restrictions()
        if self.structural_identification != 1:
            self.__svar_estimates()


    def __svar_by_choleski_factorization(self):
        
        self.mcmc_H = self.__mcmc_chol_Sigma.copy()
        self.mcmc_Gamma = np.ones((self.iterations,self.n))
        mcmc_inv_H = np.zeros((self.n,self.n,self.iterations))
        for i in range(self.iterations):
            mcmc_inv_H[:,:,i] = la.invert_lower_triangular_matrix(self.mcmc_H[:,:,i])
            if self.verbose:
                cu.progress_bar(i, self.iterations, 'Structural identification:')
        self.__mcmc_inv_H =  mcmc_inv_H
        self.__svar_index = np.arange(self.iterations)


    def __svar_by_triangular_factorization(self):
        
        mcmc_H = np.zeros((self.n,self.n,self.iterations))
        mcmc_inv_H = np.zeros((self.n,self.n,self.iterations))
        mcmc_Gamma = np.zeros((self.iterations,self.n))
        for i in range(self.iterations):
            H, Gamma = la.triangular_factorization(self.__mcmc_chol_Sigma[:,:,i], is_cholesky = True)
            mcmc_H[:,:,i] = H
            mcmc_inv_H[:,:,i] = la.invert_lower_triangular_matrix(H)
            mcmc_Gamma[i,:] = Gamma
            if self.verbose:
                cu.progress_bar(i, self.iterations, 'Structural identification:')
        self.mcmc_H = mcmc_H
        self.mcmc_Gamma = mcmc_Gamma
        self.__mcmc_inv_H = mcmc_inv_H
        self.__svar_index = np.arange(self.iterations)


    def __svar_by_restrictions(self):

        # initiate MCMC elements
        svar_index = np.zeros(self.iterations).astype(int)
        mcmc_H = np.zeros((self.n,self.n,self.iterations))
        mcmc_inv_H = np.zeros((self.n,self.n,self.iterations))
        # create matrices of restriction checks
        restriction_matrices, max_irf_period = vu.make_restriction_matrices(self.restriction_table, self.p)
        # make preliminary orthogonalised impulse response functions, if relevant
        mcmc_irf = vvu.make_varma_restriction_irf(self.mcmc_beta, self.mcmc_kappa, self.__mcmc_chol_Sigma, \
                                            self.iterations, self.n, self.p, self.q, max_irf_period)  
        mcmc_shocks = vvu.make_varma_restriction_shocks(self.mcmc_E, self.__mcmc_chol_Sigma, self.T, self.n, self.iterations, restriction_matrices)            
        # loop over iterations, until desired number of total iterations is obtained
        i = 0
        while i < self.iterations:
            # select a random index in number of iterations
            j = rng.discrete_uniform(0, self.iterations-1)
            # make a random rotation matrix Q: if no zero restrictions, draw from uniform orthogonal distribution
            if len(restriction_matrices[0]) == 0:
                Q = rng.uniform_orthogonal(self.n)
            # if there are zero restrictions, use the zero uniform orthogonal distribution
            else:
                Q = rng.zero_uniform_orthogonal(self.n, restriction_matrices[0], mcmc_irf[:,:,:,j])
            # check restrictions: IRF, sign
            irf_sign_index = restriction_matrices[1][0]
            if len(irf_sign_index) != 0:
                irf_sign_coefficients = restriction_matrices[1][1]
                restriction_satisfied = vu.check_irf_sign(irf_sign_index, irf_sign_coefficients, mcmc_irf[:,:,:,j], Q)
                if not restriction_satisfied:
                    continue
            # check restrictions: IRF, magnitude
            irf_magnitude_index = restriction_matrices[2][0]
            if len(irf_magnitude_index) != 0:
                irf_magnitude_coefficients = restriction_matrices[2][1]
                restriction_satisfied = vu.check_irf_magnitude(irf_magnitude_index, irf_magnitude_coefficients, mcmc_irf[:,:,:,j], Q)
                if not restriction_satisfied:
                    continue                      
            # check restrictions: structural shocks, sign
            shock_sign_index = restriction_matrices[3][0]
            if len(shock_sign_index) != 0:
                shock_sign_coefficients = restriction_matrices[3][1]
                restriction_satisfied = vu.check_shock_sign(shock_sign_index, shock_sign_coefficients, mcmc_shocks[:,:,j], Q)
                if not restriction_satisfied:
                    continue
            # check restrictions: structural shocks, magnitude
            shock_magnitude_index = restriction_matrices[4][0]
            if len(shock_magnitude_index) != 0:
                shock_magnitude_coefficients = restriction_matrices[4][1]
                restriction_satisfied = vu.check_shock_magnitude(shock_magnitude_index, shock_magnitude_coefficients, mcmc_shocks[:,:,j], Q)
                if not restriction_satisfied:
                    continue
            # historical decomposition values if any of sign or magnitude restrictions apply
            history_sign_index = restriction_matrices[5][0]
            history_magnitude_index = restriction_matrices[6][0]
            if len(history_sign_index) != 0 or len(history_magnitude_index) != 0:
                irf, shocks = vu.make_restriction_irf_and_shocks(mcmc_irf[:,:,:,j], mcmc_shocks[:,:,j], Q, self.n)
            # check restrictions: historical decomposition, sign
            if len(history_sign_index) != 0:
                history_sign_coefficients = restriction_matrices[5][1]
                restriction_satisfied = vu.check_history_sign(history_sign_index, history_sign_coefficients, irf, shocks)
                if not restriction_satisfied:    
                    continue
            # check restrictions: historical decomposition, magnitude
            if len(history_magnitude_index) != 0:
                history_magnitude_coefficients = restriction_matrices[6][1]
                restriction_satisfied = vu.check_history_magnitude(history_magnitude_index, history_magnitude_coefficients, irf, shocks)
                if not restriction_satisfied:
                    continue  
            # if all restriction passed, keep the draw and record
            H = self.__mcmc_chol_Sigma[:,:,j] @ Q
            inv_H = Q.T @ la.invert_lower_triangular_matrix(self.__mcmc_chol_Sigma[:,:,j])
            mcmc_H[:,:,i] = H
            mcmc_inv_H[:,:,i] = inv_H
            svar_index[i] = j
            if self.verbose:
                cu.progress_bar(i, self.iterations, 'Structural identification:')  
            i += 1 
        self.mcmc_H = mcmc_H
        self.__mcmc_inv_H = mcmc_inv_H
        self.mcmc_Gamma = np.ones((self.iterations,self.n))
        self.__svar_index = svar_index


    def __svar_estimates(self):

        H_estimates = np.quantile(self.mcmc_H,0.5,axis=2)
        Gamma_estimates = np.quantile(self.mcmc_Gamma,0.5,axis=0)
        self.H_estimates = H_estimates
        self.Gamma_estimates = Gamma_estimates


    def __steady_state(self):

        ss = np.zeros((self.T,self.n,self.iterations))
        for i in range(self.iterations):
            ss[:,:,i] = vvu.varma_steady_state(self.X, self.mcmc_beta[:,:,i], self.n, self.m, self.p, self.T)
            if self.verbose:
                cu.progress_bar(i, self.iterations, 'Steady-state:')            
        ss_estimates = vu.posterior_estimates(ss, self.credibility_level)
        self.steady_state_estimates = ss_estimates


    def __fitted_and_residual(self):
        
        fitted = np.zeros((self.T,self.n,self.iterations))
        residual = np.zeros((self.T,self.n,self.iterations))
        for i in range(self.iterations):
            residual[:,:,i], fitted[:,:,i] = vvu.varma_fit_and_residuals(self.X, self.mcmc_beta[:,:,i],\
                                             self.mcmc_Z[:,:,i], self.mcmc_kappa[:,:,i], self.mcmc_E[:,:,i])
            if self.verbose:
                cu.progress_bar(i, self.iterations, 'Fitted and residual:')
        fitted_estimates = vu.posterior_estimates(fitted, self.credibility_level)
        residual_estimates = vu.posterior_estimates(residual, self.credibility_level)                
        self.fitted_estimates = fitted_estimates
        self.residual_estimates = residual_estimates                
        if self.structural_identification != 1:
            structural_shocks = np.zeros((self.T,self.n,self.iterations))
            for i in range(self.iterations):
                index = self.__svar_index[i]
                structural_shocks[:,:,i] = vu.structural_shocks(residual[:,:,index], self.__mcmc_inv_H[:,:,i])
                if self.verbose:
                    cu.progress_bar(i, self.iterations, 'Structural shocks:')
            structural_shock_estimates = vu.posterior_estimates(structural_shocks, self.credibility_level)
            self.mcmc_structural_shocks = structural_shocks
            self.structural_shock_estimates = structural_shock_estimates


    def __insample_criteria(self):
        
        insample_evaluation = vu.insample_evaluation_criteria(self.Y, \
                              self.residual_estimates[:,:,0], self.T, self.k)
        if self.verbose:
            cu.progress_bar_complete('In-sample evaluation criteria:')
        self.insample_evaluation = insample_evaluation


    def __make_forecast(self, h, Z_p):     
        
        Z_p, Y = vu.make_forecast_regressors(Z_p, self.Y, h, self.p, self.T,
                 self.exogenous, self.constant, self.trend, self.quadratic_trend)
        mcmc_forecast = np.zeros((h,self.n,self.iterations))
        for i in range(self.iterations):
            mcmc_forecast[:,:,i] = vvu.varma_forecast(self.mcmc_beta[:,:,i], self.mcmc_kappa[:,:,i], \
                                   self.__mcmc_chol_Sigma[:,:,i], h, Z_p, Y, self.mcmc_E[-self.q:,:,i], self.n)
            if self.verbose:
                cu.progress_bar(i, self.iterations, 'Forecasts:')
        self.mcmc_forecast = mcmc_forecast


    def __forecast_posterior_estimates(self, credibility_level):
        
        # obtain posterior estimates
        mcmc_forecast = self.mcmc_forecast
        forecast_estimates = vu.posterior_estimates(mcmc_forecast, credibility_level)
        self.forecast_estimates = forecast_estimates  
        
        
    def __make_impulse_response_function(self, h):
        
        mcmc_irf = np.zeros((self.n, self.n, h, self.iterations))
        for i in range(self.iterations):
            # get regular impulse response function for VARMA
            mcmc_irf[:,:,:,i] = vvu.varma_impulse_response_function(self.mcmc_beta[:,:,i], \
                                self.mcmc_kappa[:,:,i], self.n, self.p, self.q, h)
            if self.verbose:    
                cu.progress_bar(i, self.iterations, 'Impulse response function:')
        self.mcmc_irf = mcmc_irf      
        
        
    def __make_exogenous_impulse_response_function(self, h):
        
        if len(self.exogenous) != 0:
            r = self.exogenous.shape[1]
            mcmc_irf_exo = np.zeros((self.n, r, h, self.iterations))
            for i in range(self.iterations):
                # get exogenous IRFs: same as VAR since no shocks are involved in exogenous IRF
                mcmc_irf_exo[:,:,:,i] = vu.exogenous_impulse_response_function(self.mcmc_beta[:,:,i], self.n, self.m, r, self.p, h)
                if self.verbose:    
                    cu.progress_bar(i, self.iterations, 'Exogenous impulse response function:')
        else:
            mcmc_irf_exo = []
        self.mcmc_irf_exo = mcmc_irf_exo   
        
        
    def __make_structural_impulse_response_function(self, h):
        
        if self.structural_identification == 1:
            if self.verbose:    
                cu.progress_bar_complete('Structural impulse response function:')  
            self.mcmc_structural_irf = []
        else:
            mcmc_structural_irf = self.mcmc_irf.copy()
            for i in range(self.iterations):
                index = self.__svar_index[i]
                mcmc_structural_irf[:,:,:,i] = vu.structural_impulse_response_function(self.mcmc_irf[:,:,:,index],\
                                                                                       self.mcmc_H[:,:,i], self.n)
                if self.verbose:
                    cu.progress_bar(i, self.iterations, 'Structural impulse response function:')                
            self.mcmc_structural_irf = mcmc_structural_irf


    def __irf_posterior_estimates(self, credibility_level):

        if self.structural_identification == 1:
            mcmc_irf = self.mcmc_irf
        else:
            mcmc_irf = self.mcmc_structural_irf
        irf_estimates = vu.posterior_estimates_3d(mcmc_irf, credibility_level)
        if len(self.exogenous) != 0:
            exo_irf_estimates = vu.posterior_estimates_3d(self.mcmc_irf_exo, credibility_level)
        else:
            exo_irf_estimates = []
        self.irf_estimates = irf_estimates
        self.exo_irf_estimates = exo_irf_estimates        
        
        
    def __make_forecast_error_variance_decomposition(self, h):
        
        if self.structural_identification == 1:
            if self.verbose:
                cu.progress_bar_complete('Forecast error variance decomposition:')  
            self.mcmc_fevd = []
        else:
            if self.structural_identification == 3:
                mcmc_Gamma = self.mcmc_Gamma.copy()
            else:
                mcmc_Gamma = [[]] * self.iterations
            mcmc_fevd = np.zeros((self.n, self.n, h, self.iterations))
            has_irf = hasattr(self, 'mcmc_structural_irf') and self.mcmc_structural_irf.shape[2] >= h
            for i in range(self.iterations):                
                if has_irf:
                    structural_irf = self.mcmc_structural_irf[:,:,:h,i]
                else:
                    index = self.__svar_index[i]
                    irf = vvu.varma_impulse_response_function(self.mcmc_beta[:,:,index], \
                                        self.mcmc_kappa[:,:,i], self.n, self.p, self.q, h)
                    structural_irf = vu.structural_impulse_response_function(irf, self.mcmc_H[:,:,i], self.n) 
                mcmc_fevd[:,:,:,i] = vu.forecast_error_variance_decomposition(structural_irf, mcmc_Gamma[i], self.n, h)            
                if self.verbose:
                    cu.progress_bar(i, self.iterations, 'Forecast error variance decomposition:')         
            self.mcmc_fevd = mcmc_fevd

        
    def __fevd_posterior_estimates(self, credibility_level):

        if self.structural_identification == 1:
            self.fevd_estimates = []
        else:
            mcmc_fevd = self.mcmc_fevd
            fevd_estimates = vu.posterior_estimates_3d(mcmc_fevd, credibility_level)
            normalized_fevd_estimates = vu.normalize_fevd_estimates(fevd_estimates)
            self.fevd_estimates = normalized_fevd_estimates        
        
        
    def __make_historical_decomposition(self):
        
        if self.structural_identification == 1:
            if self.verbose:
                cu.progress_bar_complete('Historical decomposition:')
            self.mcmc_hd = []        
        else:
            mcmc_hd = np.zeros((self.n, self.n, self.T, self.iterations))   
            has_irf = hasattr(self, 'mcmc_structural_irf') and self.mcmc_structural_irf.shape[2] >= self.T
            has_structural_shocks = hasattr(self, 'mcmc_structural_shocks')
            for i in range(self.iterations):
                index = self.__svar_index[i]
                if has_irf:
                    structural_irf = self.mcmc_structural_irf[:,:,:self.T,i]
                else:       
                    irf = vvu.varma_impulse_response_function(self.mcmc_beta[:,:,index], \
                                        self.mcmc_kappa[:,:,index], self.n, self.p, self.q, self.T)
                    structural_irf = vu.structural_impulse_response_function(irf, self.mcmc_H[:,:,i], self.n)    
                if has_structural_shocks:
                    structural_shocks = self.mcmc_structural_shocks[:,:,i]  
                else:
                    E, _ = vvu.varma_fit_and_residuals(self.X, self.mcmc_beta[:,:,i],\
                                                     self.mcmc_Z[:,:,i], self.mcmc_kappa[:,:,i], self.mcmc_E[:,:,index])
                    structural_shocks = vu.structural_shocks(E, self._mcmc_inv_H[:,:,i])
                mcmc_hd[:,:,:,i] = vu.historical_decomposition(structural_irf, structural_shocks, self.n, self.T)             
                if self.verbose:    
                    cu.progress_bar(i, self.iterations, 'Historical decomposition:')   
            self.mcmc_hd = mcmc_hd
            
               
    def __hd_posterior_estimates(self, credibility_level):

        if self.structural_identification == 1:
            self.hd_estimates = []
        else:
            mcmc_hd = self.mcmc_hd
            hd_estimates = vu.posterior_estimates_3d(mcmc_hd, credibility_level)
            self.hd_estimates = hd_estimates        
        
        
    def __make_conditional_forecast(self, h, conditions, Z_p):
        
        # make regressors Z_p and Y
        Z_p, Y = vu.make_forecast_regressors(Z_p, self.Y, h, self.p, self.T,
                 self.exogenous, self.constant, self.trend, self.quadratic_trend)
        # make conditional forecast regressors y_bar, Z and omega
        y_bar, Q, omega = vvu.varma_conditional_forecast_regressors_1(conditions, h, self.n, self.p, self.q)
        # initiate storage and loop over iterations
        mcmc_conditional_forecast = np.zeros((h,self.n,self.iterations))
        for i in range(self.iterations):
            # recover iteration-specific regressors
            mu, F, K, gamma_00, Upsilon_00 = vvu.varma_conditional_forecast_regressors_2(Y, self.mcmc_E[-self.q:,:,i], 
                                             self.mcmc_beta[:,:,i], self.mcmc_kappa[:,:,i], self.mcmc_Sigma[:,:,i], \
                                             conditions, Z_p, self.n, self.m, self.p, self.q, h)          
            # run Carter Kohn algorithm to obtain conditional forecasts
            bss = BayesianStateSpaceSampler(y_bar, Q, omega, mu, F, K, gamma_00, \
                  Upsilon_00, kalman_type = 'conditional_forecast')
            bss.carter_kohn_algorithm()
            mcmc_conditional_forecast[:,:,i] = bss.Z[:,:self.n]
            if self.verbose:
                cu.progress_bar(i, self.iterations, 'Conditional forecasts:')  
        self.mcmc_conditional_forecast = mcmc_conditional_forecast
        

    def __conditional_forecast_posterior_estimates(self, credibility_level):

        if len(self.mcmc_conditional_forecast) == 0:
            self.conditional_forecast_estimates = []
        else:
            mcmc_conditional_forecast = self.mcmc_conditional_forecast
            conditional_forecast_estimates = vu.posterior_estimates(mcmc_conditional_forecast, credibility_level)
            self.conditional_forecast_estimates = conditional_forecast_estimates          


    def __check_shock_type(self, h, conditions, shocks): 
        
        # check for structural identification
        if self.structural_identification == 1:
            if self.verbose:
                cu.progress_bar_complete('Conditional forecasts:')
            shock_type = 'none'          
        else:
            # identify shocks
            if np.sum(shocks) == self.n:
                shock_type = 'all_shocks'
            else:
                shock_type = 'shock-specific'
        return shock_type


    def __make_structural_conditional_forecast(self, h, conditions, shocks, Z_p, shock_type):

        # if there is an issue, return empty mcmc matrix
        if shock_type == 'none':
            self.mcmc_conditional_forecast = []
            if self.verbose:
                cu.progress_bar_complete('Conditional forecasts:')
        # if condition type is well defined, proceed
        else:
            # make regressors Z_p and Y
            Z_p, Y = vu.make_forecast_regressors(Z_p, self.Y, h, self.p, self.T,
                      self.exogenous, self.constant, self.trend, self.quadratic_trend)
            has_irf = hasattr(self, 'mcmc_structural_irf') and self.mcmc_structural_irf.shape[2] >= h
            # make conditional forecast regressors R, y_bar and omega
            R, y_bar, omega = vu.conditional_forecast_regressors_3(conditions, h, self.n)
            if shock_type == 'shock-specific':
                P, non_generating = vu.conditional_forecast_regressors_5(shocks, h, self.n)
            # initiate storage and loop over iterations
            mcmc_conditional_forecast = np.zeros((h,self.n,self.iterations))
            for i in range(self.iterations): 
                index = self.__svar_index[i]
                # make predictions, absent shocks
                f = vvu.varma_linear_forecast(self.mcmc_beta[:,:,index], self.mcmc_kappa[:,:,index], h, Z_p, Y, \
                                              self.mcmc_E[-self.q:,:,index], self.n)
                # recover structural IRF or estimate them
                if has_irf:
                    structural_irf = self.mcmc_structural_irf[:,:,:h,i]
                else:
                    irf = vvu.varma_impulse_response_function(self.mcmc_beta[:,:,index], \
                          self.mcmc_kappa[:,:,index], self.n, self.p, self.q, h)
                    structural_irf = vu.structural_impulse_response_function(irf, self.mcmc_H[:,:,i], self.n)  
                # recover iteration-specific regressors
                M = vu.conditional_forecast_regressors_4(structural_irf, self.n, h)
                # get posterior mean and variance, depending on condition type                
                if shock_type == 'all_shocks':
                    mu_hat, Omega_hat = vu.conditional_forecast_posterior(y_bar, f, M, R, self.mcmc_Gamma[i,:], omega, self.n, h)
                elif shock_type == 'shock-specific':
                    Gamma_nd = vu.conditional_forecast_regressors_6(self.mcmc_Gamma[i,:], non_generating, h)
                    mu_hat, Omega_hat = vu.shock_specific_conditional_forecast_posterior(\
                                        y_bar, f, M, R, P, self.mcmc_Gamma[i,:], Gamma_nd, omega, self.n, h)                
                # sample values
                mcmc_conditional_forecast[:,:,i] = rng.multivariate_normal(mu_hat, Omega_hat).reshape(h,self.n)
                if self.verbose:
                    cu.progress_bar(i, self.iterations, 'Conditional forecasts:')
            self.mcmc_conditional_forecast = mcmc_conditional_forecast

