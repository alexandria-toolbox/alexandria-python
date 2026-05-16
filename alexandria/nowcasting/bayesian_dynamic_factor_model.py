# imports
import numpy as np
import numpy.random as nrd
import alexandria.nowcasting.nowcasting_utilities as nwu
import alexandria.vector_autoregression.var_utilities as vu
import alexandria.math.random_number_generators as rng
import alexandria.state_space.state_space_utilities as ss
import alexandria.console.console_utilities as cu
import alexandria.processor.input_utilities as iu


class BayesianDynamicFactorModel(object):
    
    
    """
    Bayesian dynamic factor model, developed in chapter 18
    
    Parameters:
    -----------
    endogenous : ndarray of shape (n_obs,n)
        endogenous variables, defined in (6.18.1)

    factors : int, default = 3
        number of latent factors, defined in (6.18.1)
        
    loadings_lags : int, default = 2
        number of loadings lags, defined in (6.18.2)

    factor_lags : int, default = 2
        number of factor lags, defined in (6.18.6)

    residual_lags : int, default = 1
        number of residual lags, defined in (6.18.7)
        
    sigma : float, default = 0.1
        variance on residual shock e_it, defined in (6.18.7)        
        
    omega : float, default = 0.1
        variance on factor shock xi_t, defined in (6.18.6) 

    delta1 : float, default = 0.1
        overall tightness hyperparameter on lambda_ij, defined in (6.18.21)
    
    pi1 : float, default = 0.1
        overall tightness hyperparameter, defined in (6.18.25)
    
    pi2 : float, default = 0.5
        cross-variable shrinkage hyperparameter, defined in (6.18.25)
    
    pi3 : float, default = 1
        lag decay hyperparameter, defined in (6.18.25)  
        
    omega_1 : float, default = 0.1
        overall tightness hyperparameter on gamma_i, defined in (6.18.27)     
    
    credibility_level : float, default = 0.95
        VAR model credibility level (between 0 and 1)

    burnin : int, default = 1000
        number of Gibbs sampler burn-in replications  
        
    iterations : int, default = 2000
        number of Gibbs sampler replications   
    
    verbose : bool, default = False
        if True, displays a progress bar 
    
    
    Attributes
    ----------
    endogenous : ndarray of size (n_obs,n)
        endogenous variables, defined in (6.18.1)
        
    factors : int, default = 3
        number of latent factors, defined in (6.18.1)
        
    loadings_lags : int, default = 2
        number of loadings lags, defined in (6.18.2)

    factor_lags : int, default = 2
        number of factor lags, defined in (6.18.6)

    residual_lags : int, default = 1
        number of residual lags, defined in (6.18.7)
        
    sigma : float, default = 0.1
        variance on residual shock e_it, defined in (6.18.7)        
        
    omega : float, default = 0.1
        variance on factor shock xi_t, defined in (6.18.6) 

    delta1 : float, default = 0.1
        overall tightness hyperparameter on lambda_ij, defined in (6.18.21)
    
    pi1 : float, default = 0.1
        overall tightness hyperparameter, defined in (6.18.25)
    
    pi2 : float, default = 0.5
        cross-variable shrinkage hyperparameter, defined in (6.18.25)
    
    pi3 : float, default = 1
        lag decay hyperparameter, defined in (6.18.25)  
        
    omega_1 : float, default = 0.1
        overall tightness hyperparameter on gamma_i, defined in (6.18.27)     
    
    credibility_level : float, default = 0.95
        VAR model credibility level (between 0 and 1)

    burnin : int, default = 1000
        number of Gibbs sampler burn-in replications  
        
    iterations : int, default = 2000
        number of Gibbs sampler replications   
    
    verbose : bool, default = False
        if True, displays a progress bar         
        
    y : ndarray of size (n_obs,n)
        standardized endogenous variables
        
    c : ndarray of size (n_obs,)
        mean of endogenous variables, prior to standardization   
        
    S : ndarray of size (n_obs,)
        standard deviation of endogenous variables, prior to standardization  
      
    T : int
        number of sample periods, defined in (6.18.1)  
        
    n : int
        number of endogenous variables, defined in (6.18.1)
    
    t_ : list of length (n)
        cell of observed sample periods t_i, defined in (6.18.9) 
    
    T_ : list of length (n)
        cell of observed sample length T_i, defined in (6.18.9)   
        
    x : list of length (n)
        cell of sample observations x_i, defined in (6.18.10)         
      
    x_ : list of length (T)
        cell of sample observations, periods by periods

    J : list of length (T)
        selection matrices J_t, defined in (6.18.3)

    d : list of length (T)
        number of observations d_t for each sample period, defined in (6.18.3)
    
    m : int
        number of latent factors, defined in (6.18.1)

    q : int
        number of loadings lags, defined in (6.18.2)
        
    p : int
        number of factor lags, defined in (6.18.6)
    
    r : int
        number of residual lags, defined in (6.18.7)
    
    s : int
        max(q,p), defined in (6.18.39)    

    k : int
        factor VAR coefficients per equation, defined in (6.18.15)

    l : int
        regression coefficients per loading equation, defined in (6.18.11)
        
    mcmc_lambda : ndarray of shape (n,l,iterations)
        MCMC values of loadings coefficients lambda      
        
    mcmc_beta : ndarray of shape (k,m,iterations)
        MCMC values of VAR coefficients beta         
        
    mcmc_gamma : ndarray of shape (n,r,iterations)
        MCMC values of AR coefficients gamma           
        
    mcmc_f : ndarray of shape (T,m*(p+1),iterations)
        MCMC values of latent factors f            
        
    mcmc_eps : ndarray of shape (T,n*(r+1),iterations)
        MCMC values of latent residuals epsilon

    lambda_estimates : ndarray of shape (n,l,4)
        posterior estimates for lambda
        page 1: median, page 2: lower bound, page 3: upper bound, page 4: standard deviation

    beta_estimates : ndarray of shape (k,m,4)
        posterior estimates for beta
        page 1: median, page 2: lower bound, page 3: upper bound, page 4: standard deviation

    gamma_estimates : ndarray of shape (n,r,4)
        posterior estimates for gamma
        page 1: median, page 2: lower bound, page 3: upper bound, page 4: standard deviation

    f_estimates : ndarray of shape (T,m,4)
        posterior estimates for factors
        page 1: median, page 2: lower bound, page 3: upper bound, page 4: standard deviation

    mcmc_forecasts : ndarray of shape (f_periods,n,iterations)
        MCMC values of forecasts

    mcmc_f_forecasts : ndarray of shape (f_periods,m,iterations)
        MCMC values of factor forecasts
        
    forecast_estimates : ndarray of shape (f_periods,n,3)
        forecast estimates
        page 1: median, page 2: lower bound, page 3: upper bound

    f_forecast_estimates : ndarray of shape (f_periods,m,3)
        forecast estimates for latent factors
        page 1: median, page 2: lower bound, page 3: upper bound

    mcmc_irf : ndarray of shape (n,m+1,irf_periods,iterations)
        MCMC values of impulse response function
    
    irf_estimates : ndarray of size (n,m+1,irf_periods,3)
        posterior estimates of impulse response function
        page 1: median, page 2: lower bound, page 3: upper bound    
    
    mcmc_fevd : ndarray of size (n,m+1,fevd_periods,iterations)
        MCMC values of forecast error variance decompositions
    
    fevd_estimates : ndarray of size (n,m+1,fevd_periods,3)
        posterior estimates of forecast error variance decomposition
        page 1: median, page 2: lower bound, page 3: upper bound 
    
    mcmc_hd : ndarray of size (n,m+1,T,iterations)
        MCMC values of historical decompositions
    
    hd_estimates : ndarray of size (n,m+1,T,3)
        posterior estimates of historical decomposition
        page 1: median, page 2: lower bound, page 3: upper bound 
        
    fitted_estimates : ndarray of size (T,n,3)
        estimates of in-sample fit
    
    residual_estimates : ndarray of size (T,n,3)
        estimates of in-sample residuals
        
    factor_residual_estimates : ndarray of size (T,m,3)
        estimates of in-sample residuals     
        
    insample_evaluation : dict
        in-sample evaluation criteria        
        
    forecast_evaluation_criteria : dict
        forecast evaluation criteria
        
    
    Methods
    ----------
    estimate
    insample_fit
    forecast
    forecast_evaluation
    impulse_response_function
    forecast_error_variance_decomposition
    historical_decomposition
    """    
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------    
    

    def __init__(self, endogenous, factors = 3, loadings_lags = 2, factor_lags = 2, 
                 residual_lags = 1, sigma = 0.1, omega = 0.1, delta1 = 0.1, 
                 pi1 = 0.1, pi2 = 0.5, pi3 = 1, omega1 = 0.1, credibility_level = 0.95, 
                 burnin = 1000,iterations = 2000, verbose = False):

        """
        constructor for the BayesianDynamicFactorModel class
        """
        
        self.endogenous = endogenous
        self.factors = factors
        self.loadings_lags = loadings_lags
        self.factor_lags = factor_lags
        self.residual_lags = residual_lags
        self.sigma = sigma
        self.omega = omega
        self.delta1 = delta1
        self.delta2 = 1e-10
        self.pi1 = pi1
        self.pi2 = pi2
        self.pi3 = pi3
        self.omega1 = omega1
        self.credibility_level = credibility_level
        self.burnin = burnin
        self.iterations = iterations
        self.verbose = verbose
        # make regressors
        self.__make_regressors()
    
    
    def estimate(self):
    
        """
        estimate()
        generates posterior estimates for the Bayesian dynamic factor model
        
        parameters:
        none
        
        returns:
        none    
        """    
        
        # define prior values
        self.__prior()
        # run MCMC algorithm (Gibbs sampling) for dfm parameters
        self.__parameter_mcmc()
        # compute bridge equations
        self.__bridge_equations()
        # obtain posterior estimates for dfm parameters
        self.__parameter_estimates()   


    def insample_fit(self):
        
        """
        insample_fit()
        generates in-sample fit and residuals
        
        parameters:
        none
        
        returns:
        none    
        """           
        
        # compute fitted and residuals
        self.__fitted_and_residual()
        # compute in-sample criteria
        self.__insample_criteria()
        

    def forecast(self, h, credibility_level):
        
        """
        forecast(h, credibility_level)
        estimates forecasts for the Bayesian dynamic factor model, using algorithm 18.3
        
        parameters:
        h : int
            number of forecast periods
        credibility_level: float between 0 and 1
            credibility level for forecast credibility bands
        
        returns:
        forecast_estimates : ndarray of shape (h,n,3)
            page 1: median; page 2: interval lower bound; page 3: interval upper bound
        """ 
        
        # get forecast
        self.__make_forecast(h)
        # obtain posterior estimates
        self.__forecast_posterior_estimates(credibility_level)
        forecast_estimates = self.forecast_estimates
        return forecast_estimates


    def forecast_evaluation(self, Y):
        
        """
        forecast_evaluation(Y)
        forecast evaluation criteria for the Bayesian DFM
        
        parameters:
        Y : ndarray of shape (h,n)
            array of realised values for forecast evaluation, h being the number of forecast periods
            
        returns:
        forecast_evaluation_criteria : dictionary
            dictionary with criteria name as keys and corresponding number as value
        """

        # unpack
        Y_hat, mcmc_forecast = self.forecast_estimates[:,:,0], self.mcmc_forecast
        # obtain regular forecast evaluation criteria 
        standard_evaluation_criteria = vu.forecast_evaluation_criteria(Y_hat, Y)
        # obtain Bayesian forecast evaluation criteria 
        bayesian_evaluation_criteria = vu.bayesian_forecast_evaluation_criteria(mcmc_forecast, Y)
        # merge dictionaries
        forecast_evaluation_criteria = iu.concatenate_dictionaries(standard_evaluation_criteria, bayesian_evaluation_criteria)
        # save as attributes
        self.forecast_evaluation_criteria = forecast_evaluation_criteria
        return forecast_evaluation_criteria
    

    def impulse_response_function(self, h, credibility_level):
        
        """
        impulse_response_function(h, credibility_level)
        impulse response functions, as defined in (6.18.46)-(6.18.47)
        
        parameters:
        h : int
            number of IRF periods
        credibility_level: float between 0 and 1
            credibility level for forecast credibility bands
            
        returns:
        irf_estimates : ndarray of shape (n,m+1,h,3)
            first 3 dimensions are variable, shock, period; 4th dimension is median, lower bound, upper bound
        """        
        
        # get regular impulse response funtion
        self.__make_impulse_response_function(h)    
        # obtain posterior estimates
        self.__irf_posterior_estimates(credibility_level)    
        irf_estimates = self.irf_estimates
        return irf_estimates
        
    
    def forecast_error_variance_decomposition(self, h, credibility_level):
        
        """
        forecast_error_variance_decomposition(self, h, credibility_level)
        forecast error variance decomposition, as defined in (6.18.48)-(6.18.50)
        
        parameters:
        h : int
            number of FEVD periods
        credibility_level: float between 0 and 1
            credibility level for forecast credibility bands
            
        returns:
        fevd_estimates : ndarray of shape (n,m+1,h,3)
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
        historical decomposition, as defined in (6.18.51)-(6.18.53)
        
        parameters:
        credibility_level: float between 0 and 1
            credibility level for forecast credibility bands
            
        returns:
        hd_estimates : ndarray of shape (n,m+1,T,3)
            first 3 dimensions are variable, shock, period; 4th dimension is median, lower bound, upper bound
        """
        
        # get historical decomposition
        self.__make_historical_decomposition()
        # obtain posterior estimates
        self.__hd_posterior_estimates(credibility_level)
        hd_estimates = self.hd_estimates
        return hd_estimates      
    
    
    #---------------------------------------------------
    # Methods (Access = private)
    #--------------------------------------------------- 


    def __make_regressors(self):
        
        """ generates regressors, hyperparameters and dimensions """
        
        # verify data integrity
        nwu.check_data_integrity(self.endogenous)
        # standardize data
        y, c, S = nwu.standardize_data(self.endogenous)
        # create regressors
        T, n, t_, T_, x, x_, J, d = nwu.make_dfm_regressors(y)
        # get dimensions
        m, q, p, r, s, k, l = nwu.make_dfm_dimensions(self.factors, \
                              self.loadings_lags, self.factor_lags, self.residual_lags)
        # save as attributes
        self.y = y
        self.c = c
        self.S = S
        self.T = T
        self.n = n
        self.t_ = t_
        self.T_ = T_
        self.x = x
        self.x_ = x_
        self.J = J
        self.d = d
        self.m = m
        self.q = q
        self.p = p
        self.r = r
        self.s = s
        self.k = k
        self.l = l

        
    def __prior(self):

        """ create prior elements for the dynamic factor model """    

        # define prior hyperparameters for lambda
        self.__inv_U, self.__inv_U_h = nwu.make_lambda_prior(self.n, self.m, \
                                       self.q, self.l, self.delta1, self.delta2)
        # define prior hyperparameters for beta
        self.__inv_V = nwu.make_beta_prior(self.m, self.p, self.pi1, self.pi2, self.pi3)            
        # define prior hyperparameters for gamma
        self.__inv_Q = nwu.make_gamma_prior(self.r, self.omega1)
        
        
    def __parameter_mcmc(self):
        
        """ Gibbs sampler for DFM parameters  """
        
        # unpack
        n = self.n
        m = self.m
        p = self.p
        q = self.q
        r = self.r
        s = self.s
        k = self.k
        l = self.l
        T = self.T
        sigma = self.sigma
        omega = self.omega
        inv_U = self.__inv_U
        inv_U_h = self.__inv_U_h
        inv_V = self.__inv_V
        inv_Q = self.__inv_Q
        t_ = self.t_
        x = self.x
        x_ = self.x_
        J = self.J
        c = self.c
        S = self.S
        burnin = self.burnin
        iterations = self.iterations
        verbose = self.verbose
        # preallocate storage space
        mcmc_lambda = np.zeros((n,l,iterations))
        mcmc_beta = np.zeros((k,m,iterations))
        mcmc_gamma = np.zeros((n,r,iterations))
        mcmc_W = np.zeros((T,l,iterations))
        mcmc_f = np.zeros((T,m*(s+1),iterations))
        mcmc_eps = np.zeros((T,n*(r+1),iterations))
        mcmc_E_ = [None] * iterations
        mcmc_xi = np.zeros((T,m,iterations)) 
        mcmc_e = np.zeros((T,n,iterations)) 
        mcmc_Y = np.zeros((T,n,iterations))
        
        # step 1: set initial values
        z = nrd.randn(T,m*(s+1)+n*(r+1))
        gamma = np.zeros((n,r))
        beta = np.zeros((k,m))
        W, F, Z, eps_, E_, E, xi, e = nwu.make_state_regressors(beta, gamma, z, m, q, p, s, n, r, T)
        
        # iterate over iterations
        iteration = 0
        while iteration < (burnin + iterations):   

            # step 2: draw lambda_i
            lamda = self.__draw_lambda(x, sigma, inv_U, inv_U_h, W, E_, gamma, t_, n, l)
  
            # step3: draw beta_j
            beta = self.__draw_beta(F, omega, inv_V, Z, m, k)
            
            # step 4: draw gamma_i
            gamma = self.__draw_gamma(eps_, E_, sigma, inv_Q, t_, n, r)

            # step 5: draw z
            z = self.__draw_z(x_, J, lamda, beta, gamma, sigma, omega, n, m, q, p, s, r, T)
            W, F, Z, eps_, E_, E, xi, e = nwu.make_state_regressors(beta, gamma, z, m, q, p, s, n, r, T)

            # recover nowcasts
            Y = self.__nowcasts(W, lamda, E, c, S)
            
            # save if burn is exceeded
            if iteration >= burnin:    
                
                # save parameter values
                mcmc_lambda[:,:,iteration-burnin] = lamda
                mcmc_beta[:,:,iteration-burnin] = beta
                mcmc_gamma[:,:,iteration-burnin] = gamma
                mcmc_W[:,:,iteration-burnin] = W
                mcmc_f[:,:,iteration-burnin] = z[:,:m*(s+1)]
                mcmc_eps[:,:,iteration-burnin] = z[:,-n*(r+1):]
                mcmc_E_[iteration-burnin] = E_
                mcmc_xi[:,:,iteration-burnin] = xi
                mcmc_e[:,:,iteration-burnin] = e
                mcmc_Y[:,:,iteration-burnin] = Y
                
            # if verbose, display progress bar
            if verbose:
                cu.progress_bar(iteration, iterations+burnin, 'Model parameters:')

            # update iterations
            iteration += 1           

        # save as attributes
        self.mcmc_lambda = mcmc_lambda
        self.mcmc_beta = mcmc_beta
        self.mcmc_gamma = mcmc_gamma
        self.mcmc_f = mcmc_f
        self.mcmc_eps = mcmc_eps
        self.__mcmc_W = mcmc_W
        self.__mcmc_E_ = mcmc_E_
        self.__mcmc_xi = mcmc_xi
        self.__mcmc_e = mcmc_e
        self.__mcmc_Y = mcmc_Y
        
        
    def __draw_lambda(self, x, sigma, inv_U, inv_U_h, W, E_, gamma, t_, n, l):
        
        """ draw lambda from its conditional posterior defined in (6.18.30) """

        lamda = np.zeros((n,l))
        lamda[:l,:l] = np.eye(l)
        for i in range(l,n):
            # get regressors
            x_i = x[i]
            E_i = E_[i][t_[i]]
            gamma_i = gamma[i,:]
            W_i = W[t_[i]]
            inv_U_i = inv_U[i]
            inv_U_h_i = inv_U_h[i]
            # posterior U_i_bar
            inv_U_i_bar = inv_U_i + W_i.T @ W_i / sigma
            # posterior h_i_bar     
            h_i_bar_temp = inv_U_h_i + W_i.T @ (x_i - E_i @ gamma_i) / sigma
            # efficient sampling of lambda_i (algorithm 9.4)
            lambda_i = rng.efficient_multivariate_normal(h_i_bar_temp, inv_U_i_bar)
            lamda[i,:] = lambda_i
        return lamda
        
            
    def __draw_beta(self, F, omega, inv_V, Z, m, k):

        """ draw beta from its conditional posterior defined in (6.18.33) """

        beta = np.zeros((k,m))
        for j in range(m):
            # get regressors
            f_j = F[:,j]
            inv_V_j = inv_V[j]
            # posterior V_j_bar
            inv_V_j_bar = inv_V_j + Z.T @ Z / omega
            # posterior b_j_bar
            b_j_bar_temp = Z.T @ f_j / omega
            # efficient sampling of beta (algorithm 9.4)
            beta_j = rng.efficient_multivariate_normal(b_j_bar_temp, inv_V_j_bar)
            beta[:,j] = beta_j
        return beta


    def __draw_gamma(self, eps_, E_, sigma, inv_Q, t_, n, r):
        
        """ draw gamma from its conditional posterior defined in (6.18.36) """

        gamma = np.zeros((n,r))
        if r > 0:
            for i in range(n):
                # get regressors
                eps_i = eps_[i][t_[i]]
                E_i = E_[i][t_[i]]
                # posterior Q_i_bar
                inv_Q_i_bar = inv_Q + E_i.T @ E_i / sigma
                # posterior g_i_bar
                g_i_bar_temp = E_i.T @ eps_i / sigma
                # efficient sampling of beta (algorithm 9.4)
                gamma_i = rng.efficient_multivariate_normal(g_i_bar_temp, inv_Q_i_bar)
                gamma[i,:] = gamma_i
        return gamma
            

    def __draw_z(self, x_, J, lamda, beta, gamma, sigma, omega, n, m, q, p, s, r, T):
        
        """ draw z from the state-space representation (6.18.39) """
        
        # get parameters for state-space representation
        Lambda, B, Upsilon = nwu.dfm_state_space_representation(lamda, beta, gamma, \
                             sigma, omega, n, m, q, p, s, r)
        # get initial values for algorithm
        z_00, Upsilon_00 = nwu.dfm_kalman_initial_values(sigma, omega, s, r, m, n) 
        # run forward pass
        Z_tt, Z_tt1, Ups_tt, Ups_tt1 = ss.dfm_forward_pass(x_, Lambda, B, Upsilon, \
                                       z_00, Upsilon_00, T, m, n, s, r, J)
        # run backward pass
        Z = ss.dfm_backward_pass(Z_tt, Z_tt1, Ups_tt, Ups_tt1, B, T, m, s, n, r)   
        return Z
            

    def __nowcasts(self, W, lamda, E, c, S):
        
        """ draw Y from the dynamic factor model (6.18.2) """

        # recover all values jointly from compact representation
        Y_hat = W @ lamda.T + E
        # rescale the fitted values
        Y = Y_hat * S + c
        return Y


    def __bridge_equations(self):
        
        """ bridge equations for DFM """
        
        # unpack
        n = self.n
        r = self.r        
        l = self.l
        T = self.T
        sigma = self.sigma  
        inv_U = self.__inv_U
        inv_U_h = self.__inv_U_h
        inv_Q = self.__inv_Q
        t_ = self.t_
        x = self.x
        x_ = self.x_
        J = self.J        
        iterations = self.iterations  
        verbose = self.verbose
        mcmc_W = self.__mcmc_W
        gamma = self.mcmc_gamma[:l,:,0]
        E_ = self.__mcmc_E_[0]
        
        # iterate over iterations
        for iteration in range(iterations):  
            
            # get iteration parameters
            W = mcmc_W[:,:,iteration]
            
            # step 1: update lambda
            lamda = self.__update_lambda(iteration, x, sigma, inv_U, inv_U_h, W, E_, gamma, t_, n, l)

            # step 2: update epsilon
            z, eps_, E_, E, e = self.__update_epsilon(x_, lamda, W, gamma, J, sigma, l, r, T)
            
            # step 3: update gamma
            gamma = self.__update_gamma(l, r, eps_, t_, E_, inv_Q, sigma)
            
            # update MCMC values
            self.__update_mcmc_values(lamda, gamma, z, E_, e, n, l, r, iteration)

            # if verbose, display progress bar
            if verbose:
                cu.progress_bar(iteration, iterations, 'Bridge equations:')

            # update iterations
            iteration += 1                 
        
        
    def __update_lambda(self, iteration, x, sigma, inv_U, inv_U_h, W, E_, gamma, t_, n, l):
        
        """ loadings lambda for the restricted variables """

        lamda = np.zeros((l,l))
        for i in range(l):            
            # get regressors
            x_i = x[i]
            E_i = E_[i][t_[i]]
            gamma_i = gamma[i,:]
            W_i = W[t_[i]]
            inv_U_i = inv_U[i]
            inv_U_h_i = inv_U_h[i]
            # posterior U_i_bar
            inv_U_i_bar = inv_U_i + W_i.T @ W_i / sigma
            # posterior h_i_bar     
            h_i_bar_temp = inv_U_h_i + W_i.T @ (x_i - E_i @ gamma_i) / sigma
            # efficient sampling of lambda_i (algorithm 9.4)
            lambda_i = rng.efficient_multivariate_normal(h_i_bar_temp, inv_U_i_bar)
            # update lambda
            lamda[i,:] = lambda_i                
        return lamda


    def __update_epsilon(self, x_, lamda, W, gamma, J, sigma, l, r, T):
        
        """ residuals epsilon for the restricted variables """
        
        # get parameters for state-space representation
        A, B, Upsilon = nwu.epsilon_state_space_representation(gamma, sigma, l, r)
        # get initial values for algorithm
        z_00, Upsilon_00 = nwu.epsilon_kalman_initial_values(sigma, r, l)
        # run forward pass
        Z_tt, Z_tt1, Ups_tt, Ups_tt1 = ss.epsilon_forward_pass(x_, lamda, W, A, B, Upsilon, z_00, Upsilon_00, T, l, r, J)
        # run backward pass
        z = ss.epsilon_backward_pass(Z_tt, Z_tt1, Ups_tt, Ups_tt1, B, T, l, r) 
        # update regressors
        eps_, E_, E, e = nwu.update_epsilon_regressors(gamma, z, l, r, T)
        return z, eps_, E_, E, e


    def __update_gamma(self, l, r, eps_, t_, E_, inv_Q, sigma):
        
        """ AR coefficients gamma for the restricted variables """

        gamma = np.zeros((l,r))
        if r > 0:
            for i in range(l):
                # get regressors
                eps_i = eps_[i][t_[i]]
                E_i = E_[i][t_[i]]
                # posterior Q_i_bar
                inv_Q_i_bar = inv_Q + E_i.T @ E_i / sigma
                # posterior g_i_bar
                g_i_bar_temp = E_i.T @ eps_i / sigma
                # efficient sampling of beta (algorithm 9.4)
                gamma_i = rng.efficient_multivariate_normal(g_i_bar_temp, inv_Q_i_bar)
                gamma[i,:] = gamma_i
        return gamma


    def __update_mcmc_values(self, lamda, gamma, z, E_, e, n, l, r, iteration):

        """ update mcmc values of restricted variables """

        self.mcmc_lambda[:l,:,iteration] = lamda
        self.mcmc_gamma[:l,:,iteration] = gamma  
        for i in range(r+1):
            self.mcmc_eps[:,i*n:i*n+l,iteration] = z[:,i*l:(i+1)*l]
        for i in range(l):
            self.__mcmc_E_[iteration][i] = E_[i]
        self.__mcmc_e[:,:l,iteration] = e


    def __parameter_estimates(self):
        
        """ point estimates and credibility intervals for model parameters """

        # unpack
        mcmc_lambda = self.mcmc_lambda
        mcmc_beta = self.mcmc_beta
        mcmc_gamma = self.mcmc_gamma
        mcmc_f = self.mcmc_f
        credibility_level = self.credibility_level
        m = self.m
        # recover posterior estimates
        lambda_estimates = nwu.posterior_estimates(mcmc_lambda, credibility_level)
        beta_estimates = nwu.posterior_estimates(mcmc_beta, credibility_level)
        gamma_estimates = nwu.posterior_estimates(mcmc_gamma, credibility_level)
        f_estimates = nwu.posterior_estimates(mcmc_f[:,:m,:], credibility_level)
        # save as attributes
        self.lambda_estimates = lambda_estimates
        self.beta_estimates = beta_estimates
        self.gamma_estimates = gamma_estimates
        self.f_estimates = f_estimates


    def __fitted_and_residual(self):
        
        """ point estimates and credibility intervals for fitted and residuals """
        
        mcmc_eps = self.mcmc_eps
        mcmc_xi = self.__mcmc_xi
        mcmc_Y = self.__mcmc_Y      
        n, m = self.n, self.m
        credibility_level = self.credibility_level
        eps_estimates = nwu.posterior_estimates(mcmc_eps[:,:n,:], credibility_level)
        xi_estimates = nwu.posterior_estimates(mcmc_xi[:,:m,:], credibility_level)
        Y_estimates = nwu.posterior_estimates(mcmc_Y, credibility_level) 
        self.fitted_estimates = Y_estimates
        self.residual_estimates = eps_estimates
        self.factor_residual_estimates = xi_estimates
       
           
    def __insample_criteria(self):
        
        """ in-sample fit evaluation criteria """
        
        insample_evaluation = vu.insample_evaluation_criteria(self.fitted_estimates[:,:,0], \
                              np.quantile(self.__mcmc_e, 0.5, axis=2), self.T, self.l)        
        if self.verbose:
            cu.progress_bar_complete('In-sample evaluation criteria:')
        self.insample_evaluation = insample_evaluation        
    
       
    def __make_forecast(self, h):  
        
        """ forecasts for the dynamic factor model """
        
        # initiate storage and loop over iterations
        mcmc_forecast = np.zeros((h,self.n,self.iterations))
        mcmc_f_forecast = np.zeros((h,self.m,self.iterations))
        for i in range(self.iterations):
            # make MCMC simulation for beta and Sigma
            mcmc_forecast[:,:,i], mcmc_f_forecast[:,:,i] = nwu.dfm_forecast(self.mcmc_lambda[:,:,i], \
                         self.mcmc_beta[:,:,i], self.mcmc_gamma[:,:,i], self.sigma, self.omega,  \
                         self.mcmc_f[-1,:,i], self.mcmc_eps[-1,:,i], h, self.m, self.n, self.p, self.r, self.l)
            if self.verbose:
                cu.progress_bar(i, self.iterations, 'Forecasts:')
        self.mcmc_forecast = mcmc_forecast
        self.mcmc_f_forecast = mcmc_f_forecast


    def __forecast_posterior_estimates(self, credibility_level):
        
        """ posterior estimates for forecasts """
        
        # obtain posterior estimates
        mcmc_forecast = self.mcmc_forecast
        mcmc_f_forecast = self.mcmc_f_forecast
        forecast_estimates = nwu.posterior_estimates(mcmc_forecast, credibility_level)
        f_forecast_estimates = nwu.posterior_estimates(mcmc_f_forecast, credibility_level)
        self.forecast_estimates = forecast_estimates[:,:,:3]
        self.f_forecast_estimates = f_forecast_estimates[:,:,:3]


    def __make_impulse_response_function(self, h):

        """ impulse response function for the dynamic factor model """
        
        mcmc_irf = np.zeros((self.n, self.m+1, h, self.iterations))
        for i in range(self.iterations):
            # get impulse response function
            mcmc_irf[:,:,:,i] = nwu.dfm_impulse_response_function(self.mcmc_lambda[:,:,i], \
                                self.mcmc_beta[:,:,i], self.mcmc_gamma[:,:,i], self.n, \
                                self.m, self.q, self.p, self.r, h)
            if self.verbose:    
                cu.progress_bar(i, self.iterations, 'Impulse response function:')
        self.mcmc_irf = mcmc_irf
        

    def __irf_posterior_estimates(self, credibility_level):
        
        """ posterior estimates for impulse response function """
        
        mcmc_irf = self.mcmc_irf
        irf_estimates = nwu.posterior_estimates_3d(mcmc_irf, credibility_level)
        self.irf_estimates = irf_estimates
        

    def __make_forecast_error_variance_decomposition(self, h):
        
        """ forecast error variance decomposition for the dynamic factor model """        

        mcmc_fevd = np.zeros((self.n, self.m+1, h, self.iterations))
        has_irf = hasattr(self, 'mcmc_irf') and self.mcmc_irf.shape[2] >= h
        for i in range(self.iterations):                
            # recover structural IRF or estimate them
            if has_irf:
                irf = self.mcmc_irf[:,:,:h,i]
            else:
                irf = nwu.dfm_impulse_response_function(self.mcmc_lambda[:,:,i], self.mcmc_beta[:,:,i], \
                      self.mcmc_gamma[:,:,i], self.n, self.m, self.q, self.p, self.r, h)
            # recover fevd
            mcmc_fevd[:,:,:,i] = nwu.dfm_forecast_error_variance_decomposition(irf, \
                                 self.sigma, self.omega, self.n, self.m, h)            
            if self.verbose:
                cu.progress_bar(i, self.iterations, 'Forecast error variance decomposition:')         
        self.mcmc_fevd = mcmc_fevd


    def __fevd_posterior_estimates(self, credibility_level):

        """ posterior estimates for forecast error variance decomposition """

        mcmc_fevd = self.mcmc_fevd
        fevd_estimates = nwu.posterior_estimates_3d(mcmc_fevd, credibility_level)
        normalized_fevd_estimates = nwu.normalize_fevd_estimates(fevd_estimates)
        self.fevd_estimates = normalized_fevd_estimates
        
    
    def __make_historical_decomposition(self):
        
        """ historical decomposition for the dynamic factor model """  
        
        # initiate MCMC storage for HD and loop over iterations
        mcmc_hd = np.zeros((self.n, self.m+1, self.T, self.iterations))   
        has_irf = hasattr(self, 'mcmc_irf') and self.mcmc_irf.shape[2] >= self.T
        for i in range(self.iterations):
            # recover structural IRF or estimate them
            if has_irf:
                irf = self.mcmc_irf[:,:,:self.T,i]
            else:
                irf = nwu.dfm_impulse_response_function(self.mcmc_lambda[:,:,i], \
                      self.mcmc_beta[:,:,i], self.mcmc_gamma[:,:,i], self.n, \
                      self.m, self.q, self.p, self.r, self.T)                
            xi = self.__mcmc_xi[:,:,i]
            e = self.__mcmc_e[:,:,i]
            # get historical decomposition
            mcmc_hd[:,:,:,i] = nwu.dfm_historical_decomposition(irf, xi, e, self.n, self.m, self.T) 
            if self.verbose:    
                cu.progress_bar(i, self.iterations, 'Historical decomposition:')                 
        self.mcmc_hd = mcmc_hd    
    
    
    def __hd_posterior_estimates(self, credibility_level):

        """ posterior estimates for historical decomposition """

        mcmc_hd = self.mcmc_hd
        hd_estimates = nwu.posterior_estimates_3d(mcmc_hd, credibility_level)
        self.hd_estimates = hd_estimates    
    
    
    
    
    
    
    
