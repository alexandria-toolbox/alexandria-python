# imports
import numpy as np
import numpy.linalg as nla
from alexandria.vector_autoregression.vector_autoregression import VectorAutoRegression
from alexandria.vector_autoregression.bayesian_var import BayesianVar
import alexandria.vector_autoregression.var_utilities as vu
import alexandria.math.linear_algebra as la
import alexandria.math.math_utilities as mu
import scipy.optimize as sco
import alexandria.console.console_utilities as cu
import alexandria.math.stat_utilities as su
import alexandria.math.random_number_generators as rng


class NormalWishartBayesianVar(VectorAutoRegression,BayesianVar):
    
    
    """
    Normal-Wishart vector autoregression, developed in section 11.3
    
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
    
    hyperparameter_optimization : bool, default = False
        if True, applies hyperparameter optimization by marginal likelihood 
    
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
    
    hyperparameter_optimization : bool
        if True, applies hyperparameter optimization by marginal likelihood 
    
    stationary_prior : bool
        if True, applies stationary prior, as defined in section 12.4       
    
    credibility_level : float
        VAR model credibility level (between 0 and 1)
    
    iterations : int
        number of Gibbs sampler replications   
    
    verbose : bool
        if True, displays a progress bar      
    
    B : ndarray of size (k,n)
        prior mean of VAR coefficients, defined in (4.11.24)
    
    W : ndarray of size (k,k)
        prior mean of VAR coefficients, defined in (4.11.20) 
    
    alpha : float
        prior degrees of freedom, defined in (4.11.29)
    
    S : ndarray of size (n,n)
        prior scale matrix, defined in (4.11.29) 
    
    B_bar : ndarray of size (k,n)
        posterior mean of VAR coefficients, defined in (4.11.33)
    
    W_bar : ndarray of size (k,k)
        posterior mean of VAR coefficients, defined in (4.11.33) 
    
    alpha_bar : float
        posterior degrees of freedom, defined in (4.11.33)
    
    S_bar : ndarray of size (n,n)
        posterior scale matrix, defined in (4.11.33) 
    
    alpha_hat : float
        posterior degrees of freedom, defined in (4.11.38)
    
    S_hat : ndarray of size (n,n)
        posterior scale matrix, defined in (4.11.38)        
    
    beta_estimates : ndarray of size (k,n,3)
        estimates of VAR coefficients
        page 1: median, page 2: st dev,  page 3: lower bound, page 4: upper bound
    
    Sigma_estimates : ndarray of size (n,n)
        estimates of variance-covariance matrix of VAR residuals
    
    mcmc_beta : ndarray of size (k,n,iterations)
        MCMC values of VAR coefficients   
    
    mcmc_Sigma : ndarray of size (n,n,iterations)
        MCMC values of residual variance-covariance matrix
    
    m_y : float
        log 10 marginal likelihood, defined in (4.12.21)     
    
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
    
    H_estimates : ndarray of size (n,n)
        posterior estimates of structural matrix, defined in section 13.2
    
    Gamma_estimates : ndarray of size (1,n)
        estimates of structural shock variance matrix, defined in section 13.2
    
    
    Methods
    ----------
    estimate
    marginal_likelihood
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
                 restriction_table = [], lags = 4, constant = True, trend = False, 
                 quadratic_trend = False, ar_coefficients = 0.95, pi1 = 0.1, 
                 pi3 = 1, pi4 = 100, pi5 = 1,  pi6 = 0.1, pi7 = 0.1, 
                 sums_of_coefficients = False, dummy_initial_observation = False, 
                 long_run_prior = False, long_run_table = [], 
                 hyperparameter_optimization = False, stationary_prior = False, 
                 credibility_level = 0.95, iterations = 2000, verbose = False):

        """
        constructor for the NormalWishartBayesianVar class
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
        self.hyperparameter_optimization = hyperparameter_optimization
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
        # optimize hyperparameters, if applicable
        self.__optimize_hyperparameters()
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


    def marginal_likelihood(self):
        
        """ 
        log10 marginal likelihood, defined in (4.12.24)
        
        parameters:
        none
        
        returns:
        m_y: float
            log10 marginal likelihood
        """

        # unpack
        n = self.n
        T = self.T
        XX = self._XX
        W = self.W
        S = self.S
        S_bar = self.S_bar
        alpha = self.alpha
        alpha_bar = self.alpha_bar
        # evaluate the log marginal likelihood from equation (4.12.24)
        term_1 = - (n*T/2) * np.log(np.pi)
        term_2 = - n / 2 * la.stable_determinant(W.reshape(-1,1) * XX)
        term_3 = - T / 2 * np.sum(np.log(S))
        term_4 = - alpha_bar / 2 * la.stable_determinant((S_bar - np.diag(S)) / S.reshape(-1,1))
        term_5 = mu.log_multivariate_gamma(alpha_bar/2,n) - mu.log_multivariate_gamma(alpha/2,n)
        log_f_y = term_1 + term_2 + term_3 + term_4 + term_5
        # convert to log10
        m_y = log_f_y / np.log(10)
        # save as attributes
        self.m_y = m_y
        return m_y 


    #---------------------------------------------------
    # Methods (Access = private)
    #--------------------------------------------------- 


    def __optimize_hyperparameters(self):
        
        """ optimize_hyperparameters delta, pi1, pi3 and pi4 """
        
        if self.hyperparameter_optimization:
            # initial value of optimizer
            x_0 = np.array([0.1,1,100] + self.n * [0.9])
            # bounds for parameter values
            bound = [(0.01,1), (1,5), (10,1000)] + [(0, 1)] * self.n
            # optimize
            result = sco.minimize(self.__negative_likelihood, x_0, bounds = bound, method='L-BFGS-B')
            # update hyperparameters with optimized values
            self.pi1 = result.x[0]
            self.pi3 = result.x[1]
            self.pi4 = result.x[2]
            self.delta = result.x[3:]
            # if verbose, display progress bar and success/failure of optimization
            if self.verbose:
                cu.progress_bar_complete('hyperparameter optimization:')
                cu.optimization_completion(result.success)
                

    def __negative_likelihood(self, x):
        
        """ negative log marginal likelihood for normal-Wishart prior """
        
        # unpack
        n, m, p, T, s = self.n, self.m, self.p, self.T, self.s
        XX, XY, YY = self._XX, self._XY, self._YY
        pi1, pi3, pi4 = x[0], x[1], x[2] 
        delta = x[3:]
        # recover prior and posterior elements
        B = vu.make_B(delta, n, m, p)
        W = vu.make_W(s, pi1, pi3, pi4, n, m, p)
        alpha = vu.make_alpha(n)
        S = vu.make_S(s)
        _, _, alpha_bar, S_bar, _, _ = vu.normal_wishart_posterior(B, W, alpha, S, n, T, XX, XY, YY)
        # compute log of marginal likelihood, omitting irrelevant terms
        term_1 = n * la.stable_determinant(W.reshape(-1,1) * XX)
        term_2 = alpha_bar * la.stable_determinant((S_bar - np.diag(S)) / S.reshape(-1,1))
        negative_log_f_y = term_1 + term_2
        return negative_log_f_y 


    def __prior(self):
        
        """ creates prior elements B, W, alpha and S defined in (4.11.27), (4.11.30) and (4.11.33) """
        
        B = vu.make_B(self.delta, self.n, self.m, self.p)
        W = vu.make_W(self.s, self.pi1, self.pi3, self.pi4, self.n, self.m, self.p)
        alpha = vu.make_alpha(self.n)
        S = vu.make_S(self.s)
        self.B = B
        self.W = W
        self.alpha = alpha
        self.S = S


    def __posterior(self):
        
        """ creates posterior elements B_bar, W_bar, alpha_bar, S_bar, alpha_hat and S_hat
            defined in (4.11.33) and (4.11.38) """
        
        # unpack
        B, W, alpha, S = self.B, self.W, self.alpha, self.S
        XX, XY, YY = self._XX_d, self._XY_d, self._YY_d
        n, T = self.n, self.T
        # define posterior elements
        B_bar, W_bar, alpha_bar, S_bar, alpha_hat, S_hat = \
            vu.normal_wishart_posterior(B, W, alpha, S, n, T, XX, XY, YY)
        # store as attributes
        self.B_bar = B_bar
        self.W_bar = W_bar
        self.alpha_bar = alpha_bar
        self.S_bar = S_bar
        self.alpha_hat = alpha_hat
        self.S_hat = S_hat


    def __parameter_estimates(self):
        
        """
        point estimates and credibility intervals for model parameters
        use the inverse Wishart defined in (4.11.34) and matrix Student defined in (4.11.37)
        """
        
        # mean and standard deviation of posterior distribution, using matrix Student properties
        mean = self.B_bar
        scale = np.reshape(np.diag(np.kron(self.S_hat, self.W_bar)), [self.k,self.n], order='F')
        standard_deviation = (self.alpha_hat / (self.alpha_hat - 2) * scale) ** 0.5
        # critical value of Student distribution for credibility level
        credibility_level = self.credibility_level        
        Z = su.student_icdf((1 + credibility_level) / 2, self.alpha_hat)
        # initiate storage: 4 columns: lower bound, median, upper bound, standard deviation
        beta_estimates = np.zeros((self.k,self.n,4))        
        # fill estimates
        beta_estimates[:,:,0] = mean
        beta_estimates[:,:,1] = mean - Z * scale ** 0.5
        beta_estimates[:,:,2] = mean + Z * scale ** 0.5
        beta_estimates[:,:,3] = standard_deviation        
        # posterior estimates for Sigma, using mean of inverse Wishart
        Sigma_estimates = self.S_bar / (self.alpha_bar - self.n - 1)
        # save as attributes
        self.beta_estimates = beta_estimates
        self.Sigma_estimates = Sigma_estimates


    def __parameter_mcmc(self):
        
        """ Gibbs sampler for VAR parameters beta and Sigma """
        
        # unpack
        B_bar, W_bar, alpha_bar, S_bar = self.B_bar, self.W_bar, self.alpha_bar, self.S_bar
        iterations = self.iterations
        n, m, p, k = self.n, self.m, self.p, self.k
        # initiate storage
        mcmc_beta = np.zeros((k,n,iterations))
        mcmc_Sigma = np.zeros((n,n,iterations))
        mcmc_chol_Sigma = np.zeros((n,n,iterations)) 
        # compute constant Cholesky factors
        chol_W_bar = la.cholesky_nspd(W_bar)
        chol_S_bar = la.cholesky_nspd(S_bar)
        # loop over iterations, checking for stationarity if stationary prior is activated
        iteration = 0
        while iteration < iterations:
            # sample Sigma from inverse Wishart posterior (4.11.34)
            Sigma = rng.inverse_wishart(alpha_bar, chol_S_bar, True)
            chol_Sigma = la.cholesky_nspd(Sigma)
            # sample B from matrix normal distribution defined in (4.12.34)
            B = rng.matrix_normal(B_bar, chol_W_bar, Sigma, True, False)
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
                    cu.progress_bar(iteration, self.iterations, 'Model parameters:')
                iteration += 1
        self.mcmc_beta = mcmc_beta
        self.mcmc_Sigma = mcmc_Sigma
        self._mcmc_chol_Sigma = mcmc_chol_Sigma
                    
            
