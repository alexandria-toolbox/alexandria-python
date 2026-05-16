# imports
import numpy as np
import alexandria.nowcasting.nowcasting_utilities as nwu
import alexandria.state_space.state_space_utilities as ss
import alexandria.vector_autoregression.var_utilities as vu
import alexandria.math.linear_algebra as la
import alexandria.math.random_number_generators as rng
from alexandria.vector_autoregression.bayesian_var import BayesianVar
from alexandria.vector_autoregression.maximum_likelihood_var import MaximumLikelihoodVar
import alexandria.processor.input_utilities as iu
import numpy.random as nrd
import alexandria.console.console_utilities as cu


class MixedFrequencyBayesianVar(BayesianVar):
    
    
    """
    Mixed frequency Bayesian VAR, developed in chapter 17
    
    Parameters:
    -----------
    
    endogenous : ndarray of shape (n_obs,n)
        endogenous variables, defined in (6.17.1)
    
    exogenous : ndarray of shape (n_obs,m), default = []
        exogenous variables, defined in (6.17.1)
        
    decomposition : bool
        if True, applies frequency decomposition as developed in section 17.3
        
    decomposition_table : ndarray
        numerical matrix of frequency decomposition       
    
    structural_identification : int, default = 2
        structural identification scheme, as defined in section 13.2
        1 = none, 2 = Cholesky, 3 = triangular, 4 = restrictions
    
    restriction_table : ndarray
        numerical matrix of restrictions for structural identification
    
    lags : int, default = 4
        number of lags, defined in (6.17.1)
    
    constant : bool, default = True
        if True, an intercept is included in the VAR model exogenous
    
    trend : bool, default = False
        if True, a linear trend is included in the VAR model exogenous
    
    quadratic_trend : bool, default = False
        if True, a quadratic trend is included in the VAR model exogenous
    
    ar_coefficients : float or ndarray of shape (n_endo,1), default = 0.95
        prior mean delta for AR coefficients, defined in (6.17.8)
    
    pi1 : float, default = 0.1
        overall tightness hyperparameter, defined in (6.17.8)
    
    pi2 : float, default = 0.5
        cross-variable shrinkage hyperparameter, defined in (6.17.8)
    
    pi3 : float, default = 1
        lag decay hyperparameter, defined in (6.17.8)    
    
    pi4 : float, default = 100
        exogenous slackness hyperparameter, defined in (6.17.8)             
    
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
    endogenous : ndarray of shape (n_obs,n)
        endogenous variables, defined in (6.17.1)
    
    exogenous : ndarray of shape (n_obs,m), default = []
        exogenous variables, defined in (6.17.1)
        
    decomposition : bool
        if True, applies frequency decomposition as developed in section 17.3
        
    decomposition_table : ndarray
        numerical matrix of frequency decomposition       
    
    structural_identification : int, default = 2
        structural identification scheme, as defined in section 13.2
        1 = none, 2 = Cholesky, 3 = triangular, 4 = restrictions
    
    restriction_table : ndarray
        numerical matrix of restrictions for structural identification
    
    lags : int, default = 4
        number of lags, defined in (6.17.1)
    
    constant : bool, default = True
        if True, an intercept is included in the VAR model exogenous
    
    trend : bool, default = False
        if True, a linear trend is included in the VAR model exogenous
    
    quadratic_trend : bool, default = False
        if True, a quadratic trend is included in the VAR model exogenous
    
    ar_coefficients : float or ndarray of shape (n_endo,1), default = 0.95
        prior mean delta for AR coefficients, defined in (6.17.8)
    
    pi1 : float, default = 0.1
        overall tightness hyperparameter, defined in (6.17.8)
    
    pi2 : float, default = 0.5
        cross-variable shrinkage hyperparameter, defined in (6.17.8)
    
    pi3 : float, default = 1
        lag decay hyperparameter, defined in (6.17.8)    
    
    pi4 : float, default = 100
        exogenous slackness hyperparameter, defined in (6.17.8)             
    
    credibility_level : float, default = 0.95
        VAR model credibility level (between 0 and 1)
    
    iterations : int, default = 2000
        number of Gibbs sampler replications   

    burnin : int, default = 1000
        number of Gibbs sampler burn-in replications 
    
    verbose : bool, default = False
        if True, displays a progress bar 
    
    Z : ndarray of shape (T,m)
        exogenous variables, defined in (6.17.3)    
    
    Z_ : ndarray of shape (T+p,m)
        exogenous variables with initial conditions  
    
    n : int
        number of endogenous variables, defined in (6.17.1)
    
    m : int
        number of exogenous variables, defined in (6.17.1)
    
    p : int
        number of lags, defined in (6.17.1)
    
    T : int
        number of sample periods, defined in (6.17.1)
    
    k : int
        number of VAR coefficients in each equation, defined in (6.17.1)
    
    q : int
        total number of VAR coefficients, defined in (6.17.1)    
    
    yo_ : list of len (T+p)
        list of observed endogenous variables, defined in (6.17.16)    
    
    L_ : list of len (T+p)
        list of selection matrices, defined in (6.17.21)      
    
    b : ndarray of shape (q,1)
        prior mean of VAR coefficients, defined in (6.17.8)
    
    V : ndarray of shape (q,q)
        prior mean of VAR coefficients, defined in (6.17.8)       
    
    alpha : float
        prior degrees of freedom, defined in (6.17.9)
    
    S : ndarray of shape (n,n)
        prior scale matrix, defined in (6.17.9) 
    
    alpha_bar : float
        posterior degrees of freedom, defined in (6.17.15)      
    
    mcmc_beta : ndarray of shape (k,n,iterations)
        MCMC values of VAR coefficients   
    
    mcmc_Sigma : ndarray of shape (n,n,iterations)
        MCMC values of residual variance-covariance matrix     
    
    mcmc_W : ndarray of shape (T,n,iterations)
        MCMC values of latent endogenous variables    
    
    beta_estimates : ndarray of shape (k,n,4)
        estimates of VAR coefficients
        page 1: median, page 2: st dev,  page 3: lower bound, page 4: upper bound
    
    Sigma_estimates : ndarray of shape (n,n)
        estimates of variance-covariance matrix of VAR residuals    
    
    W_estimates : ndarray of shape (T,n,3)
        estimates of latent endogenous variables 
        page 1: median, page 2: lower bound, page 3: upper bound
    
    Y : ndarray of shape (T,n)
        matrix of in-sample endogenous variables, obtained from W
    
    X : ndarray of shape (T,k)
        matrix of exogenous and lagged regressors, defined in (6.17.3)   
    
    mcmc_H :  ndarray of shape (n,n,iterations)
        MCMC values of structural identification matrix, defined in (4.13.5)
    
    mcmc_Gamma : ndarray of shape (iterations,n)
        MCMC values of structural shock variance matrix, defined in definition 13.1    
    
    delta : ndarray of shape (n,1)
        prior mean delta for AR coefficients, defined in (4.11.16)
    
    s : ndarray of shape (n,1)
        individual AR models residual variance, defined in (4.11.18)
    
    Y_sum : ndarray of shape (n,n)
        sums-of-coefficients Y matrix, defined in (4.12.6)
    
    X_sum : ndarray of shape (n,k)
        sums-of-coefficients X matrix, defined in (4.12.6)
    
    Y_obs : ndarray of shape (1,n)
        dummy initial observation Y matrix, defined in (4.12.10)
    
    X_obs : ndarray of shape (1,k)
        dummy initial observation X matrix, defined in (4.12.10)
    
    Y_lrp : ndarray of shape (1,n)
        long run prior Y matrix, defined in (4.12.16)
    
    X_lrp : ndarray of shape (1,k)
        long run prior X matrix, defined in (4.12.16)
    
    Y_d : ndarray of shape (T_d,n)
        full Y matrix combining sample data and dummy observations, defined in (4.11.62)
    
    X_d : ndarray of shape (T_d,k)
        full X matrix combining sample data and dummy observations, defined in (4.11.62)
    
    T_d : int
        total number of observations combining sample data and dummy observations, defined in (4.11.62)    
    
    steady_state_estimates : ndarray of shape (T,n,3)
        estimates of steady-state, defined in (4.12.30)
    
    fitted_estimates : ndarray of shape (T,n,3)
        estimates of in-sample fit, defined in (4.11.2)
    
    residual_estimates : ndarray of shape (T,n,3)
        estimates of in-sample residuals, defined in (4.11.2)
    
    structural_shocks_estimates : ndarray of shape (T,n,3)
        estimates of in-sample structural shocks, defined in definition 13.1
    
    insample_evaluation : dict
        in-sample evaluation criteria, defined in (4.13.15)-(4.13.17)
    
    mcmc_structural_shocks : ndarray of shape (T,n,iterations)
        MCMC values of structural shocks
    
    mcmc_forecasts : ndarray of shape (f_periods,n,iterations)
        MCMC values of forecasts
    
    forecast_estimates : ndarray of shape (f_periods,n,3)
        forecast estimates, defined in (4.13.12) and (4.13.13)
        page 1: median, page 2: lower bound, page 3: upper bound
    
    forecast_evaluation_criteria : dict
        forecast evaluation criteria, defined in (4.13.18)-(4.13.21)
    
    mcmc_irf : ndarray of shape (n,n,irf_periods,iterations)
        MCMC values of impulse response function, defined in section 13.1
    
    mcmc_irf_exo : ndarray of shape (n,m,irf_periods,iterations)
        MCMC values of exogenous impulse response function
    
    mcmc_structural_irf : ndarray of shape (n,n,irf_periods,iterations)
        MCMC values of structural impulse response function, defined in section 13.2
    
    irf_estimates : ndarray of shape (n,n,irf_periods,3)
        posterior estimates of impulse response function, defined in section 13.1 - 13.2
        page 1: median, page 2: lower bound, page 3: upper bound    
    
    exo_irf_estimates : ndarray of shape (n,m,irf_periods,3)
        posterior estimates of exogenous impulse response function, if any exogenous variable
        page 1: median, page 2: lower bound, page 3: upper bound
    
    mcmc_fevd : ndarray of shape (n,n,fevd_periods,iterations)
        MCMC values of forecast error variance decompositions, defined in section 13.4
    
    fevd_estimates : ndarray of shape (n,n,fevd_periods,3)
        posterior estimates of forecast error variance decomposition, defined in section 13.4
        page 1: median, page 2: lower bound, page 3: upper bound 
    
    mcmc_hd : ndarray of shape (n,n,T,iterations)
        MCMC values of historical decompositions, defined in section 13.5
    
    hd_estimates : ndarray of shape (n,n,T,3)
        posterior estimates of historical decomposition, defined in section 13.5
        page 1: median, page 2: lower bound, page 3: upper bound 
    
    mcmc_conditional_forecasts : ndarray of shape (f_periods,n,iterations)
        MCMC values of conditional forecasts, defined in section 14.1
    
    conditional_forecast_estimates : ndarray of shape (f_periods,n,3)
        posterior estimates of conditional forecast, defined in section 14.1
        page 1: median, page 2: lower bound, page 3: upper bound
    
    H_estimates : ndarray of shape (n,n)
        posterior estimates of structural matrix, defined in section 13.2
    
    Gamma_estimates : ndarray of shape (1,n)
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
    

    def __init__(self, endogenous, exogenous = [], decomposition = False, 
                 decomposition_table = [], structural_identification = 2, 
                 restriction_table = [], lags = 4, constant = True, trend = False, 
                 quadratic_trend = False, ar_coefficients = 0.95, pi1 = 0.1, 
                 pi2 = 0.5, pi3 = 1, pi4 = 100, credibility_level = 0.95, 
                 iterations = 2000, burnin = 1000, verbose = False):

        """
        constructor for the MixedFrequencyBayesianVar class
        """
        
        self.endogenous = endogenous
        self.exogenous = exogenous
        self.decomposition = decomposition
        self.decomposition_table = decomposition_table
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
        self.credibility_level = credibility_level
        self.iterations = iterations
        self.burnin = burnin
        self.verbose = verbose    
        # make regressors
        self.__make_regressors()
        
    
    def estimate(self):
    
        """
        estimate()
        generates posterior estimates for Bayesian VAR model parameters beta and Sigma
        
        parameters:
        none
        
        returns:
        none    
        """    

        # define prior values
        self.__prior()
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
    
    
    def __make_regressors(self):
        
        """ creates regressors and dimensions """

        # make exogenous regressors
        self.Z, self.Z_ = self.__make_exogenous_regressors()
        # generate dimensions
        self.n, self.m, self.p, self.T, self.k, self.q = self.__generate_dimensions()
        # generate delta
        self.delta = self.__make_delta()
        # generate s
        self.s = self.__individual_ar_variances()
        # generate state-space regressors
        self.yo_, self.L_, self.__T_, self.__r = self.__make_state_space_regressors()


    def __make_exogenous_regressors(self):

        """ creates exogenous regressors """
        
        periods = self.endogenous.shape[0]
        X_1 = vu.generate_intercept_and_trends(self.constant, self.trend, self.quadratic_trend, periods, 0)        
        X_2 = vu.generate_exogenous_regressors(self.exogenous, 0, periods)
        Z_ = np.hstack([X_1,X_2])
        Z = Z_[self.lags:]
        return Z, Z_

    
    def __generate_dimensions(self):

        """ creates VAR dimension """        

        T = self.endogenous.shape[0] - self.lags
        n = self.endogenous.shape[1]
        p = self.lags
        m = int(self.constant) + int(self.trend) + int(self.quadratic_trend)    
        if len(self.exogenous) != 0:
            m += self.exogenous.shape[1]
        k = m + n * p
        q = n * k
        return n, m, p, T, k, q        
        
        
    def __make_delta(self):    
        
        """ creates delta hyperparameter """
        
        if iu.is_numeric(self.ar_coefficients):
            ar_coefficients = np.array(self.n * [self.ar_coefficients])
        else:
            ar_coefficients = self.ar_coefficients
        delta = ar_coefficients
        return delta
        
    
    def __individual_ar_variances(self):
        
        """ creates individual AR variances """
        
        s = np.zeros(self.n)
        for i in range(self.n):
            endogenous = self.endogenous[:,[i]]
            endogenous = endogenous[~np.isnan(endogenous)].reshape(-1,1)
            ar = MaximumLikelihoodVar(endogenous, lags=self.lags)
            ar.estimate()
            s[i] = ar.Sigma[0,0]
        return s
        
    
    def __make_state_space_regressors(self):
        
        """ creates period-specific parameters for state-space formulation """
        
        # identify maximum lag for state-space sampler
        if self.decomposition:
            r = max(self.p, max(self.decomposition_table))
        else:
            r = self.p
        # create full selection matrix
        L = np.zeros((self.n,self.n*r))
        eye_matrix = np.eye(self.n)
        zero_matrix = np.zeros((self.n,self.n))
        if self.decomposition:
            decomposition_table = self.decomposition_table
        else:
            decomposition_table = np.ones(self.n, dtype=int)
        for i in range(self.n):
            periods = decomposition_table[i]
            temp = np.hstack([np.tile(eye_matrix,periods),np.tile(zero_matrix,r-periods)])
            L[i,:] = temp[i,:]
        # initiate observation and selection matrices
        T_ = self.endogenous.shape[0]
        yo_ = [None] * T_
        L_ = [None] * T_
        # loop over periods to obtain period-specific matrices
        for t in range(T_):
            endogenous_t = self.endogenous[t,:]
            non_nan_entries = np.logical_not(np.isnan(endogenous_t)).nonzero()[0]
            yo_[t] = endogenous_t[non_nan_entries]
            L_[t] = L[non_nan_entries]
        return yo_, L_, T_, r
    
    
    def __prior(self):
        
        """ creates prior elements b and V """
        
        self.b = vu.make_b(self.delta, self.n, self.m, self.p)
        self.V = vu.make_V(self.s, self.pi1, self.pi2, self.pi3, self.pi4, self.n, self.m, self.p)
        self.alpha = vu.make_alpha(self.n)
        self.S = vu.make_S(self.s)


    def __posterior(self):
        
        """ creates posterior elements """
        
        # generate preliminary posterior elements
        inv_V, inv_V_b = vu.make_V_b_inverse(self.b, self.V)
        # generate posterior alpha_bar
        alpha_bar = self.alpha + self.T
        self.alpha_bar = alpha_bar
        self.__inv_V = inv_V
        self.__inv_V_b = inv_V_b        
    
    
    
    def __parameter_mcmc(self):
        
        """ Gibbs sampler for VAR parameters beta, Sigma and W, following algorithm 17.1 """

        # unpack
        endogenous = self.endogenous
        Z = self.Z
        Z_ = self.Z_
        inv_V = self.__inv_V
        inv_V_b = self.__inv_V_b
        alpha_bar = self.alpha_bar
        S = self.S
        n = self.n
        m = self.m
        p = self.p
        k = self.k
        T = self.T
        yo_ = self.yo_
        L_ = self.L_
        T_ = self.__T_
        r = self.__r
        iterations = self.iterations
        burnin = self.burnin
        verbose = self.verbose
        
        # preallocate storage space
        mcmc_beta = np.zeros((k,n,iterations))
        mcmc_Sigma = np.zeros((n,n,iterations))
        mcmc_W = np.zeros((T,n,iterations))
        mcmc_X = np.zeros((T,k,iterations))
        mcmc_chol_Sigma = np.zeros((n,n,iterations))
        mcmc_inv_Sigma = np.zeros((n,n,iterations))
        # set initial values
        inv_Sigma = np.diag(1 / S)
        S = np.diag(S)    
        W = nrd.randn(T,n)
        X = nrd.randn(T,k)
        
        # iterate over iterations
        iteration = 0
        while iteration < (burnin + iterations):
            
            # step 2: sample beta
            beta = self.__draw_beta(inv_V, inv_V_b, inv_Sigma, W, X)
            B = np.reshape(beta,[k,n],order='F')
            
            # step 3: sample Sigma
            Sigma, inv_Sigma, chol_Sigma = self.__draw_Sigma(W, X, B, S, alpha_bar)
            
            # step 4: sample gamma
            Gamma, W, X = self.__draw_gamma(Z, Z_, B, Sigma, yo_, L_, n, m, p, T, T_)

            # save if burn is exceeded
            if iteration >= burnin:
                
                # save parameter values
                mcmc_beta[:,:,iteration-burnin] = B
                mcmc_Sigma[:,:,iteration-burnin] = Sigma
                mcmc_W[:,:,iteration-burnin] = W
                mcmc_X[:,:,iteration-burnin] = X
                mcmc_chol_Sigma[:,:,iteration-burnin] = chol_Sigma
                mcmc_inv_Sigma[:,:,iteration-burnin] = inv_Sigma
                
            # if verbose, display progress bar
            if verbose:
                cu.progress_bar(iteration, iterations+burnin, 'Model parameters:') 
                
            # update iterations    
            iteration += 1
              
        # save as attributes
        self.mcmc_beta = mcmc_beta
        self.mcmc_Sigma = mcmc_Sigma
        self.mcmc_W = mcmc_W
        self.__mcmc_X = mcmc_X
        self._mcmc_chol_Sigma = mcmc_chol_Sigma
        self.__mcmc_inv_Sigma = mcmc_inv_Sigma
    
    
    def __draw_beta(self, inv_V, inv_V_b, inv_Sigma, W, X):
        
        """ draw beta from its conditional posterior defined in (6.17.11) """
        
        # posterior V_bar
        inv_V_bar = inv_V + np.kron(inv_Sigma, X.T @ X)
        # posterior b_bar
        b_bar_temp = inv_V_b + la.vec(X.T @ W @ inv_Sigma)
        # efficient sampling of beta (algorithm 9.4)
        beta = rng.efficient_multivariate_normal(b_bar_temp, inv_V_bar)
        return beta     
    
    
    def __draw_Sigma(self, W, X, B, S, alpha_bar):
        
        """ draw Sigma from its conditional posterior defined in (6.17.14) """

        # compute residuals
        residuals = W - X @ B
        # compute S_bar
        S_bar = S + residuals.T @ residuals
        # sample sigma
        Sigma = rng.inverse_wishart(alpha_bar, S_bar)
        inv_Sigma = la.invert_spd_matrix(Sigma)
        chol_Sigma = la.cholesky_nspd(Sigma)
        return Sigma, inv_Sigma, chol_Sigma
    
    
    def __draw_gamma(self, Z, Z_, B, Sigma, yo_, L_, n, m, p, T, T_):
        
        """ draw Gamma from its conditional posterior, as defined in (6.17.20)-(6.17.21) """
        
        # compute state-space parameters
        F, mu_, Upsilon = nwu.make_mfbvar_state_regressors(Z_, B, Sigma, n, m, p, T_)
        # get initial values for algorithm
        gamma_00, Upsilon_00 = nwu.mfbvar_kalman_initial_values(Sigma, n, p)
        # run forward pass
        Gamma_tt, Gamma_tt1, Ups_tt, Ups_tt1 = ss.mfbvar_forward_pass(yo_, L_, \
                               F, mu_, n, p, Upsilon, T_, gamma_00, Upsilon_00)
        # run backward pass
        Gamma = ss.mfbvar_backward_pass(Gamma_tt, Gamma_tt1, Ups_tt, Ups_tt1, F, T_, n, p)
        # update state regressors
        W, X = nwu.update_mfbvar_state_regressors(Z, Gamma, n, p)
        return Gamma, W, X
    
    
    def __parameter_estimates(self):
        
        """
        point estimates and credibility intervals for model parameters
        use empirical quantiles from MCMC algorithm
        """

        # unpack
        mcmc_beta = self.mcmc_beta
        mcmc_Sigma = self.mcmc_Sigma
        mcmc_W = self.mcmc_W
        mcmc_X = self.__mcmc_X
        credibility_level = self.credibility_level
        k, n = self.k, self.n
        # initiate storage: 4 columns: lower bound, median, upper bound, standard deviation
        beta_estimates = np.zeros((k,n,4))
        # fill estimates
        beta_estimates[:,:,:3] = vu.posterior_estimates(mcmc_beta, credibility_level)
        beta_estimates[:,:,3] = np.std(mcmc_beta,axis=2)
        Sigma_estimates = np.quantile(mcmc_Sigma,0.5,axis=2)
        W_estimates = vu.posterior_estimates(mcmc_W, credibility_level)
        X = np.quantile(mcmc_X,0.5,axis=2)
        Y = W_estimates[:,:,0]
        self.beta_estimates = beta_estimates
        self.Sigma_estimates = Sigma_estimates    
        self.W_estimates = W_estimates
        self.Y = Y
        self.X = X
    
    
    
    
    