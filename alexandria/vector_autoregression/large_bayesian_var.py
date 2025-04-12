# imports
import numpy as np
from alexandria.vector_autoregression.vector_autoregression import VectorAutoRegression
from alexandria.vector_autoregression.bayesian_var import BayesianVar
import alexandria.vector_autoregression.var_utilities as vu
import alexandria.math.random_number_generators as rng
import alexandria.math.linear_algebra as la
import alexandria.console.console_utilities as cu


class LargeBayesianVar(VectorAutoRegression,BayesianVar):
    
    
    """
    Large Bayesian vector autoregression, developed in section 11.6
    
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
    
    pi2 : float, default = 0.5
        cross-variable shrinkage hyperparameter, defined in (4.11.18)
    
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
    
    constrained_coefficients : bool, default = False
        if True, applies constrained coefficients, as defined in section 12.1
    
    constrained_coefficients_table : ndarray, default = []
        numerical matrix of constrained coefficients
    
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
        
    burnin : int, default = 1000
        number of Gibbs sampler burn-in replications         
    
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
    
    pi2 : float
        cross-variable shrinkage hyperparameter, defined in (4.11.18)
    
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
    
    constrained_coefficients : bool
        if True, applies constrained coefficients, as defined in section 12.1
    
    constrained_coefficients_table : ndarray
        numerical matrix of constrained coefficients
    
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
        
    burnin : int
        number of Gibbs sampler burn-in replications  
    
    verbose : bool
        if True, displays a progress bar      
    
    alpha_bar : float
        posterior degrees of freedom, defined in (4.11.79)    
    
    mcmc_beta : ndarray of size (k,n,iterations)
        MCMC values of VAR coefficients   
    
    mcmc_Sigma : ndarray of size (n,n,iterations)
        MCMC values of residual variance-covariance matrix       
    
    beta_estimates : ndarray of size (k,n,3)
        estimates of VAR coefficients
        page 1: median, page 2: st dev,  page 3: lower bound, page 4: upper bound
    
    Sigma_estimates : ndarray of size (n,n)
        estimates of variance-covariance matrix of VAR residuals
    
    b : ndarray of size (q,1)
        prior mean of VAR coefficients, defined in (4.11.16)
    
    V : ndarray of size (q,q)
        prior mean of VAR coefficients, defined in (4.11.20)         
    
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
                 pi2 = 0.5, pi3 = 1, pi4 = 100, pi5 = 1, pi6 = 0.1, pi7 = 0.1, 
                 constrained_coefficients = False, constrained_coefficients_table = [], 
                 sums_of_coefficients = False, dummy_initial_observation = False, 
                 long_run_prior = False, long_run_table = [], stationary_prior = False, 
                 credibility_level = 0.95, iterations = 2000, burnin = 1000, verbose = False):

        """
        constructor for the LargeBayesianVar class
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
        self.pi2 = pi2
        self.pi3 = pi3
        self.pi4 = pi4
        self.pi5 = pi5
        self.pi6 = pi6
        self.pi7 = pi7
        self.constrained_coefficients = constrained_coefficients
        self.constrained_coefficients_table = constrained_coefficients_table
        self.sums_of_coefficients = sums_of_coefficients
        self.dummy_initial_observation = dummy_initial_observation
        self.long_run_prior = long_run_prior
        self.J = long_run_table
        self.stationary_prior = stationary_prior
        self.credibility_level = credibility_level
        self.iterations = iterations
        self.burnin = burnin
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
        # define raw prior values
        self.__prior()
        # apply constrained coefficients, if applicable
        self._make_constrained_coefficients()
        # define posterior values
        self.__posterior()
        # run MCMC algorithm (Gibbs sampling) for VAR parameters
        self.__parameter_mcmc()  
        # obtain posterior estimates for regression parameters
        self.__parameter_estimates()   
        # estimate structural identification
        self._make_structural_identification()


    #---------------------------------------------------
    # Methods (Access = private)
    #--------------------------------------------------- 


    def __prior(self):
        
        """ creates prior elements b and V defined in (4.11.16)-(4.11.20) """
        
        # define b and V
        b = vu.make_b(self.delta, self.n, self.m, self.p)
        V = vu.make_V(self.s, self.pi1, self.pi2, self.pi3, self.pi4, self.n, self.m, self.p)
        self.b = b
        self.V = V


    def __posterior(self):
        
        """ creates posterior element alpha_bar defined in (4.11.79) """
        
        # generate preliminary posterior elements
        b = np.reshape(self.b, [self.k, self.n], order='F')
        V = np.reshape(self.V, [self.k, self.n], order='F')
        inv_V = np.zeros((self.k,self.k,self.n))
        for i in range(self.n):
            inv_V[:,:,i] = np.diag(1/V[:,i])
        inv_V_b = b / V
        # alpha is omitted in alpha_bar to simplify as it is arbitrarily close to zero
        alpha_bar = self.T
        self.alpha_bar = alpha_bar
        self.__inv_V = inv_V
        self.__inv_V_b = inv_V_b
        
        
    def __parameter_mcmc(self):
        
        """ Gibbs sampler for VAR parameters beta, lambda and phi, following algorithm 11.2 """
        
        # unpack
        Y = self.Y_d
        X = self.X_d
        XX = self._XX_d
        inv_V = self.__inv_V
        inv_V_b = self.__inv_V_b
        alpha_bar = self.alpha_bar
        n = self.n
        m = self.m
        p = self.p
        k = self.k
        T = self.T_d
        iterations = self.iterations
        burnin = self.burnin
        stationary_prior = self.stationary_prior
        verbose = self.verbose        

        # preallocate storage space
        lamda = np.ones(n)
        phi = [np.ones(i) for i in range(n)]
        mcmc_beta = np.zeros((k,n,iterations))
        mcmc_Sigma = np.zeros((n,n,iterations))
        mcmc_chol_Sigma = np.zeros((n,n,iterations))
        
        # iterate over iterations
        iteration = 0
        while iteration < (burnin + iterations):
        
            # step 2: sample beta
            B, E = self.__draw_beta(inv_V, inv_V_b, XX, X, Y, lamda, phi, k, n, T)
            
            # step 3: sample lamda
            lamda = self.__draw_lamda(alpha_bar, E, phi, n)
            
            # step 4: sample phi
            phi = self.__draw_phi(E, n)
            
            # step 5: sample Sigma
            Sigma, chol_Sigma = self.__draw_Sigma(lamda, phi, n)
            
            # save if burn is exceeded
            if iteration >= burnin:
                
                # check stability if applicable, save and display progress bar
                if stationary_prior:
                    stationary = vu.check_stationarity(B, n, m, p)
                else:
                    stationary = True
                if stationary:
                    if verbose:
                        cu.progress_bar(iteration, iterations+burnin, 'Model parameters:')
                    mcmc_beta[:,:,iteration-burnin] = B
                    mcmc_Sigma[:,:,iteration-burnin] = Sigma
                    mcmc_chol_Sigma[:,:,iteration-burnin] = chol_Sigma
                    iteration += 1
                    
            else:
                if verbose:
                    cu.progress_bar(iteration, iterations+burnin, 'Model parameters:') 
                iteration += 1
                
        # save as attributes
        self.mcmc_beta = mcmc_beta
        self.mcmc_Sigma = mcmc_Sigma
        self._mcmc_chol_Sigma = mcmc_chol_Sigma
       

    def __draw_beta(self, inv_V, inv_V_b, XX, X, Y, lamda, phi, k, n, T):
        
        """ draw beta from its conditional posterior defined in (4.11.75) """
        
        B = np.zeros((k,n))
        E = np.zeros((T,n))
        for i in range(n):
            # posterior Vi_bar
            inv_Vi_bar = inv_V[:,:,i] + XX / lamda[i]
            # posterior b_bar
            if i > 0:
                bi_bar_temp = inv_V_b[:,i] + X.T @ (Y[:,i] + E[:,:i] @ phi[i]) / lamda[i]
            else:
                bi_bar_temp = inv_V_b[:,i] + X.T @ Y[:,i] / lamda[i]
            # efficient sampling of beta (algorithm 9.4)
            B[:,i] = rng.efficient_multivariate_normal(bi_bar_temp, inv_Vi_bar)
            E[:,i] = Y[:,i] - X @ B[:,i]
        return B, E      
    
    
    def __draw_lamda(self, alpha_bar, E, phi, n):
        
        """ draw lamda from its conditional posterior defined in (4.11.78) """
        
        lamda = np.zeros(n)
        for i in range(n):
            # get psi_bar, omitting psi as it is arbitrarily close to zero
            if i > 0:
                temp = E[:,i] + E[:,:i] @ phi[i]
                psi_bar = temp @ temp
            else:
                psi_bar = E[:,i] @ E[:,i]
            lamda[i] = rng.inverse_gamma(alpha_bar / 2, psi_bar / 2)
        return lamda
            
        
    def __draw_phi(self, E, n):
        
        """ draw phi from its conditional posterior defined in (4.11.81) """
        
        phi = [np.ones(i) for i in range(n)]
        for i in range(1,n):
            # compute posterior estimates, ignoring tau as it is arbitrarily close to zero, and simplifying lamda_i
            inv_Zi_bar = E[:,:i].T @ E[:,:i]
            fi_bar_temp = - E[:,:i].T @ E[:,i]
            phi[i] = rng.efficient_multivariate_normal(fi_bar_temp, inv_Zi_bar)
        return phi


    def __draw_Sigma(self, lamda, phi, n):
        
        """ draw Sigma from lamda and phi, using definition (4.11.65) """
        
        Phi = np.eye(n)
        for i in range(1,n):
            Phi[i,:i] = phi[i]
        inv_Phi = la.invert_lower_triangular_matrix(Phi)
        Sigma = (inv_Phi * lamda) @ inv_Phi.T
        chol_Sigma = inv_Phi * lamda ** 0.5
        return Sigma, chol_Sigma

        
    def __parameter_estimates(self):
        
        """
        point estimates and credibility intervals for model parameters
        use empirical quantiles from MCMC algorithm
        """
        
        # unpack
        mcmc_beta = self.mcmc_beta
        mcmc_Sigma = self.mcmc_Sigma
        credibility_level = self.credibility_level
        k, n = self.k, self.n
        # initiate storage: 4 columns: lower bound, median, upper bound, standard deviation
        beta_estimates = np.zeros((k,n,4))
        # fill estimates
        beta_estimates[:,:,:3] = vu.posterior_estimates(mcmc_beta, credibility_level)
        beta_estimates[:,:,3] = np.std(mcmc_beta,axis=2)
        Sigma_estimates = np.quantile(mcmc_Sigma,0.5,axis=2)
        self.beta_estimates = beta_estimates
        self.Sigma_estimates = Sigma_estimates
        

