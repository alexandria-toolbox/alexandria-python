# imports
import numpy as np
import numpy.random as nrd
import scipy.stats as sst
import alexandria.math.stat_utilities as su
import alexandria.math.linear_algebra as la
import alexandria.processor.input_utilities as iu
import alexandria.console.console_utilities as cu
import alexandria.math.random_number_generators as rng
import alexandria.linear_regression.regression_utilities as ru


class BayesianMidasRegression(object):
    
    
    """  
    Bayesian MIDAS regression, developed in chapter 19
    
    Parameters:
    -----------
    endogenous : ndarray of shape (n_obs,)
        endogenous variables, defined in (6.19.1)
        
    exogenous : ndarray of shape (n_obs,n_exogenous)
        exogenous variables, defined in (6.19.1)        
        
    endogenous_lags : int, default = 1
        number of endogenous lags, defined in (6.19.1)

    exogenous_lags : int or ndarray of shape (n_exogenous,), default = 4
        number of exogenous lags, defined in (6.19.1) 
    
    representation : str, default = 'unrestricted'
        applicable representation, among 'unrestricted', 'almon', 'fourier' 
        
    prior_type : str, default = 'minnesota'
        applicable prior, among 'minnesota', 'horseshoe', 'lasso'     
    
    omega1 : float, default = 0.01
        overall tightness hyperparameter on alpha, defined in (6.19.10)

    omega2 : float, default = 1
        cross-variable shrinkage on alpha, defined in (6.19.10)
        
    upsilon1 : float, default = 0.1
        overall tightness hyperparameter on beta and delta, defined in (6.19.11)

    upsilon2 : float, default = 1
        cross-variable shrinkage on beta and delta, defined in (6.19.11)

    polynomial_order : int, default = 3
        order of Alman or fourier polynomial, defined in (6.19.65) and (6.19.71)  

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
    endogenous : ndarray of shape (n_obs,)
        endogenous variables, defined in (6.19.1)
        
    exogenous : ndarray of shape (n_obs,n_exogenous)
        exogenous variables, defined in (6.19.1)        
        
    endogenous_lags : int, default = 1
        number of endogenous lags, defined in (6.19.1)

    exogenous_lags : int or ndarray of shape (n_exogenous,), default = 4
        number of exogenous lags, defined in (6.19.1) 
    
    representation : str, default = 'unrestricted'
        applicable representation, among 'unrestricted', 'almon', 'fourier' 
        
    prior_type : str, default = 'minnesota'
        applicable prior, among 'minnesota', 'horseshoe', 'lasso'     
    
    omega1 : float, default = 0.01
        overall tightness hyperparameter on alpha, defined in (6.19.10)

    omega2 : float, default = 1
        cross-variable shrinkage on alpha, defined in (6.19.10)
        
    upsilon1 : float, default = 0.1
        overall tightness hyperparameter on beta and delta, defined in (6.19.11)

    upsilon2 : float, default = 1
        cross-variable shrinkage on beta and delta, defined in (6.19.11)

    polynomial_order : int, default = 3
        order of Alman or fourier polynomial, defined in (6.19.65) and (6.19.71)      

    credibility_level : float, default = 0.95
        VAR model credibility level (between 0 and 1)

    burnin : int, default = 1000
        number of Gibbs sampler burn-in replications  
        
    iterations : int, default = 2000
        number of Gibbs sampler replications   
    
    verbose : bool, default = False
        if True, displays a progress bar 
        
    p : int
        number of endogenous lags, defined in (6.19.1)
        
    y : ndarray of shape (T,)
        endogenous variables, defined in (6.19.7)        
        
    Y : ndarray of shape (T,p)
        lagged endogenous variables, defined in (6.19.7)          
        
    T : int
        number of realigned sample periods, defined in (6.19.1)        
        
    n : int
        number of exogenous variables, defined in (6.19.1)    

    p_ : ndarray of shape (n,)
        number of exogenous lags p_i, defined in (6.19.1)     

    X_ : list of len (n)
        list of endogenous regressors X_i, defined in (6.19.7)
    
    X : ndarray of shape (T,l)
        full matrix of regressors X, defined in (6.19.7)
    
    l : int
        number of regression coefficients, defined in (6.19.5)  
        
    Xp : ndarray of shape (l,1)
        matrix of exogenous predictors, defined in (6.19.75)     
        
    q_ : ndarray of shape (n,)
        polynomial orders q_i, defined in (6.19.65) and (6.19.71)        

    Q_ : list of len (n)
        list of parsimonious representation matrices Q_i, defined in (6.19.76) and (6.19.72)         

    Z_ : list of len (n)
        list of parsimonious regressors Z_i, defined in (6.19.70)           

    Z : ndarray of shape (T,n_parsimonious)
        full matrix of parsimonious regressors Z, defined in (6.19.70) 
    
    kappa : float
        prior shape on sigma, defined in (6.19.14)

    lamda : float
        prior scale on sigma, defined in (6.19.14)
 
    f : float
        prior shape on mu, defined in (6.19.50)

    g : float
        prior scale on mu, defined in (6.19.50)

    mcmc_beta : list of len (f_periods)
        list of mcmc values for regression coefficients beta_i, for each forecast period

    mcmc_sigma : list of len (f_periods)
        list of mcmc values for sigma, for each forecast period

    mcmc_forecast : list of len (f_periods)
        list of mcmc values for predictions, for each forecast period

    forecast_estimates : list of len (f_periods)
        list of forecast estimates, for each forecast period
        entries are median, lower bounds, upper bounds
    
    beta_estimates: list of len (n)
        median, lower bound, upper bound and variance of regression coefficeints beta_i
    
    sigma_estimates: ndarray of shape (4,)
        median, lower bound, upper bound and variance of residual variance sigma
        
    fitted_estimates: ndarray of shape (T,3)
        estimates of in-sample fit

    residual_estimates: ndarray of size (T,3)
        estimates of in-sample residuals
        
    insample_evaluation: list of size(3)
        estimates of in-sample evaluation criteria
        
    forecast_evaluation_criteria: dict
        dictionary of forecast evaluation criteria        
        

    Methods
    ----------
    estimate
    insample_fit
    forecast
    forecast_evaluation
    """    
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------    
    

    def __init__(self, endogenous, exogenous, endogenous_lags = 1, 
                 exogenous_lags = 4, representation = 'unrestricted', 
                 prior_type = 'minnesota', omega1 = 0.01, omega2 = 1,
                 upsilon1 = 0.1, upsilon2 = 1, polynomial_order = 2, 
                 credibility_level = 0.95, iterations = 2000, 
                 burnin = 1000, verbose = False):    
    
        """
        constructor for the BayesianMidasRegression class
        """  
        
        self.endogenous = endogenous
        self.exogenous = exogenous
        self.endogenous_lags = endogenous_lags
        self.exogenous_lags = exogenous_lags
        self.representation = representation
        self.prior_type = prior_type
        self.omega1 = omega1
        self.omega2 = omega2
        self.upsilon1 = upsilon1
        self.upsilon2 = upsilon2
        self.polynomial_order = polynomial_order
        self.credibility_level = credibility_level
        self.iterations = iterations
        self.burnin = burnin
        self.verbose = verbose
        # make regressors
        self.__make_regressors()   
        # define prior values
        self.__prior()
        # initialize record elements
        self.__initialize_records()
    
    
    def estimate(self):
        
        """
        estimate()
        generates model estimates and nowcasts for the Bayesian Midas model
        
        parameters:
        none
        
        returns:
        none
        """ 
        
        # update recording parameters
        self.__update_records(1)
        # run MCMC algorithm for regression parameters
        self.__parameter_mcmc(1)
        # obtain posterior estimates for regression parameters
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
        generates model estimates and nowcasts for the Bayesian Midas model
        
        parameters:
        h : int
            number of forecast periods
        credibility_level: float between 0 and 1
            credibility level for forecast credibility bands
        
        returns:
        none
        """         

        # update recording parameters
        self.__update_records(h)
        # run MCMC algorithm for regression parameters if not already done
        if len(self.mcmc_beta[h-1]) == 0:
            self.__parameter_mcmc(h)
        # run mcmc algorithm for predictions if not already done
        if len(self.mcmc_forecast[h-1]) == 0:
            self.__make_forecast(h)
            self.__forecast_posterior_estimates(h, credibility_level)
        
        
    def forecast_evaluation(self, y):
        
        """
        forecast_evaluation(y)
        forecast evaluation criteria for the linear regression model
        
        parameters:
        y : ndarray of shape (m,)
            array of realised values for forecast evaluation
            
        returns:
        forecast_evaluation_criteria: dict
            dictionary of forecast evaluation criteria
        """
        
        # recover forecasts

        iterations = self.iterations
        indices = [index for index, value in enumerate(self.mcmc_forecast) if len(value) != 0]
        m = len(indices)
        y_hat = np.zeros(m)
        y_ = np.zeros(m)
        mcmc_forecast = np.zeros((m,iterations))
        for i in range(m):
            y_hat[i] = self.forecast_estimates[indices[i]][0]
            y_[i] = y[indices[i]]
            mcmc_forecast[i,:] = self.mcmc_forecast[indices[i]]
        # get regular forecast evaluation criteria
        standard_evaluation_criteria = ru.forecast_evaluation_criteria(y_hat, y_)  
        # obtain Bayesian forecast evaluation criteria
        bayesian_evaluation_criteria = self.__bayesian_forecast_evaluation_criteria(y, mcmc_forecast, iterations, m)   
        # merge dictionaries
        forecast_evaluation_criteria = iu.concatenate_dictionaries(standard_evaluation_criteria, bayesian_evaluation_criteria)        
        # save as attributes
        self.forecast_evaluation_criteria = forecast_evaluation_criteria        

  
    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------       


    def __make_regressors(self):
        
        """ generates regressors using frequency alignment """
        
        # determine initial observation
        self.__make_initial_observation()
        # make endogenous regressors
        self.__make_endogenous_regressors()
        # make exogenous regressors
        self.__make_exogenous_regressors()
        # make predictors
        self.__make_predictors()
        # make parsimonious representation
        self.__make_parsimonious_representation()        
        
    
    def __make_initial_observation(self):
        
        """ initial endogenous variable observation so that lags are well defined """
        
        self.n = self.exogenous.shape[1]
        if iu.is_numeric(self.exogenous_lags):
            self.p_ = np.array(self.n * [self.exogenous_lags])
        else:
            self.p_ = self.exogenous_lags        
        indices = np.zeros(self.n+1)
        for i in range(self.n):
            non_nan_exogenous, non_nan_entries = la.dropna(self.exogenous[:,i])
            indices[i] = non_nan_entries[self.p_[i]]  
        self.p = self.endogenous_lags
        non_nan_endogenous, non_nan_entries = la.dropna(self.endogenous)     
        indices[-1] = non_nan_entries[self.p]
        self.__endogenous_indices = non_nan_entries[non_nan_entries >= np.max(indices)]

    
    def __make_endogenous_regressors(self):
        
        """ endogenous regressors y and Y defined in (6.19.3) """

        self.T = self.__endogenous_indices.shape[0]
        self.y = np.zeros(self.T)
        self.Y = np.zeros((self.T,self.p))      
        for t in range(self.T):
            y_t, _ = la.dropna(self.endogenous[:self.__endogenous_indices[t]+1])
            self.y[t] = y_t[-1]
            if self.p > 0:
                self.Y[t,:] = y_t[-(self.p+1):-1][::-1]
             

    def __make_exogenous_regressors(self):
        
        """ exogenous regressors 1_T and X_i defined in (6.19.3) """
        
        self.__one_T = np.ones((self.T,1))
        X = [self.__one_T,self.Y]
        X_ = [None] * self.n
        for i in range(self.n):
            temp = self.exogenous[:,i]
            X_i = np.zeros((self.T,self.p_[i]+1))
            for t in range(self.T):
                X_it, _ = la.dropna(temp[:self.__endogenous_indices[t]+1])
                X_it = X_it[-self.p_[i]-1:][::-1]
                X_i[t,:] = X_it
            X_[i] = X_i
            X.append(X_i)
        self.X_ = X_
        X = np.hstack(X)
        self.X = X            
        self.l = 1 + self.p + np.sum(self.p_) + self.n
          

    def __make_predictors(self):
        
        """ regression predictors  """

        Xp_ = [None] * self.n
        for i in range(self.n):
            temp, _ = la.dropna(self.exogenous[:,i])
            Xp_i = temp[-self.p_[i]-1:]
            Xp_[i] = Xp_i
        if self.p == 0:
            Yp = np.zeros(0)
        else:
            temp, _ = la.dropna(self.endogenous)
            Yp = temp[-self.p:][::-1]
        self.Xp = np.hstack([1, Yp, np.hstack(Xp_)])


    def __make_parsimonious_representation(self):
        
        """ parsimonious representations, defined in (6.19.6)-(6.19.11) """        

        if iu.is_numeric(self.polynomial_order):
            self.q_ = np.array(self.n * [self.polynomial_order])
        else:
            self.q_ = self.polynomial_order
        Q_ = [None] * self.n
        Z_ = [None] * self.n
        Z =  [self.__one_T,self.Y]
        for i in range(self.n):
            if self.representation == 'almon':
                temp_1 = np.arange(0,self.p_[i]+1).reshape(-1,1) * np.ones((self.p_[i]+1,self.q_[i]+1))
                temp_2 = np.arange(0,self.q_[i]+1) * np.ones((self.p_[i]+1,self.q_[i]+1))
                Q_i = temp_1 ** temp_2
            elif self.representation == 'fourier': 
                temp_1 = np.ones((self.p_[i]+1,1))
                temp_2 = 2 * np.pi / (self.p_[i] + 1) * (np.arange(0,self.p_[i]+1).reshape(-1,1) @ np.arange(1,self.q_[i]+1).reshape(1,-1))
                temp_3 = np.cos(temp_2)
                temp_4 = np.sin(temp_2)
                Q_i = np.hstack([temp_1,temp_3,temp_4])
            if self.representation == 'almon' or self.representation == 'fourier': 
                Q_[i] = Q_i
                Z_i = self.X_[i] @ Q_i
                Z_[i] = Z_i
                Z.append(Z_i)
        self.Q_ = Q_
        self.Z_ = Z_ 
        if self.representation == 'unrestricted':
            self.Z = []
            self.__ll = 1 + self.p + self.n + np.sum(self.p_)            
        elif self.representation == 'almon':
            self.Z = np.hstack(Z)
            self.__ll = 1 + self.p + self.n + np.sum(self.q_)            
        elif self.representation == 'fourier':                 
            self.Z = np.hstack(Z)
            self.__ll = 1 + self.p + self.n + 2 * np.sum(self.q_)  
            
            
    def __prior(self):
        
        """ creates prior elements """

        # Minnesota prior hyperparameters, defined in (6.19.12)
        if self.prior_type == 'minnesota':
            u = 10000
            w = (self.omega1 / (np.arange(1,self.p+1) ** self.omega2)) ** 2
            v = [u, w]
            for i in range(self.n):
                if self.representation == 'unrestricted':
                    v_i = (self.upsilon1 / (np.arange(0,self.p_[i]+1) ** self.upsilon2)) ** 2
                elif self.representation == 'almon':
                    v_i = (self.upsilon1 / (np.arange(0,self.q_[i]+1) ** self.upsilon2)) ** 2
                elif self.representation == 'fourier':
                    v_i = (self.upsilon1 / (np.hstack([np.arange(0,self.q_[i]+1),np.arange(1,self.q_[i]+1)]) ** self.upsilon2)) ** 2
                v_i[0] = (2 * self.upsilon1) ** 2
                v.append(v_i)
            v = np.hstack(v)
            inv_V = np.diag(1 / v)
            self.__inv_V = inv_V
        # Bayesian lasso hyperparameters, defined in (6.19.50)
        elif self.prior_type == 'lasso':
            self.__f = 0.001
            self.__g = 0.001
        # generate kappa and lambda, defined in (6.19.14), (6.19.27) and (6.19.51)
        self.__kappa = 0.001
        self.__lamda = 0.001
        
        
    def __initialize_records(self):
        
        """ initialize recording elements """

        self.mcmc_beta = []
        self.mcmc_sigma = []
        self.mcmc_forecast = []        
        self.forecast_estimates = []
        
        
    def __update_records(self, horizon):
            
        """ update recording elements """

        length = len(self.mcmc_beta)
        if length < horizon:
            self.mcmc_beta = self.mcmc_beta + [[]] * (horizon - length)
            self.mcmc_sigma = self.mcmc_sigma + [[]] * (horizon - length)    
            self.mcmc_forecast = self.mcmc_forecast + [[]] * (horizon - length)    
            self.forecast_estimates = self.forecast_estimates + [[]] * (horizon - length)            

          
    def __parameter_mcmc(self, horizon):
        
        """ posterior distributions for parameters from algorithms 19.1-19.4 """
        
        # unpack
        representation = self.representation
        prior_type = self.prior_type
        n = self.n
        l = self.l
        ll = self.__ll
        p = self.p
        p_ = self.p_
        q_ = self.q_
        Q_ = self.Q_
        kappa = self.__kappa
        lamda = self.__lamda
        if prior_type == 'minnesota':
            inv_V = self.__inv_V
        elif prior_type == 'lasso':
            f = self.__f
            g = self.__g
        burnin = self.burnin
        iterations = self.iterations
        verbose = self.verbose  
        if verbose and horizon == 1:
            verbose_string = 'Model parameters, 1 period ahead:'
        elif verbose and horizon > 1:
            verbose_string = 'Model parameters, ' + str(horizon) + ' periods ahead:'        

        # initialize storage
        mcmc_delta = np.zeros((ll,iterations))
        mcmc_sigma = np.zeros((iterations))  

        # realign regressors
        y, X, T = self.__realign_regressors(horizon)

        # prior, posterior and initial values, depending on prior type
        if prior_type == 'minnesota':
            XX, Xy, kappa_bar, sigma = self.__minnesota_parameters(y, X, T, kappa)
            
        elif prior_type == 'horseshoe':
            XX, Xy, gamma_nu, gamma_tau, gamma_eta, gamma_psi, kappa_bar, sigma, tau2, psi2 \
                = self.__horseshoe_parameters(y, X, T, ll, kappa)
            
        elif prior_type == 'lasso':
            inv_XX, b_bar, f_bar, kappa_bar, sigma, xi, mu = \
                self.__lasso_parameters(y, X, T, ll, kappa, f) 

        # iterate over iterations
        iteration = 0
        while iteration < (burnin + iterations): 
            
            # move on with Minnesota prior if selected
            if prior_type == 'minnesota':            
            
                # step 2: draw beta
                beta = self.__draw_minnesota_beta(Xy, XX, sigma, inv_V)
            
                # step 3: draw sigma
                sigma = self.__draw_minnesota_sigma(kappa_bar, y, X, beta, lamda)    

            # else, move on with horseshoe prior if selected
            elif prior_type == 'horseshoe':
                
                # step 4: draw beta
                beta, beta2 = self.__draw_horseshoe_beta(Xy, XX, tau2, psi2, sigma)

                # step 5: draw nu
                nu = self.__draw_nu(gamma_nu, tau2)

                # step 6: draw tau2
                tau2 = self.__draw_tau2(gamma_tau, nu, sigma, beta2, psi2)
                
                # step 7: draw eta
                eta = self.__draw_eta(gamma_eta, psi2, ll)
                
                # step 8: draw psi2
                psi2 = self.__draw_psi2(gamma_psi, eta, sigma, beta2, tau2, ll)
                
                # step 9: draw sigma
                sigma = self.__draw_horseshoe_sigma(kappa_bar, y, X, beta, beta2, tau2, psi2, lamda)
                
            # else, move on with Bayesian lasso if selected
            elif prior_type == 'lasso':

                # step 4: draw beta
                beta, beta2 = self.__draw_lasso_beta(inv_XX, b_bar, sigma, xi)

                # step 5: draw xi
                xi = self.__draw_xi(beta, mu, sigma, ll)

                # step 6: draw mu
                mu = self.__draw_mu(f_bar, g, xi)

                # step 7: draw sigma
                sigma = self.__draw_lasso_sigma(kappa_bar, y, X, beta, beta2, xi, lamda)   

            # save if burn is exceeded
            if iteration >= burnin:  
                
                # save parameter values
                mcmc_delta[:,iteration-burnin] = beta
                mcmc_sigma[iteration-burnin] = sigma
            
            # if verbose, display progress bar
            if verbose:
                cu.progress_bar(iteration, iterations+burnin, verbose_string)

            # update iterations
            iteration += 1   

        # recover beta if parsimonious representation was selected
        mcmc_beta = self.__recover_beta_from_parsimonious(mcmc_delta, \
                    iterations, representation, n, l, p, p_, q_, Q_)

        # save as attributes
        self.mcmc_beta[horizon-1] = mcmc_beta
        self.mcmc_sigma[horizon-1] = mcmc_sigma
        

    def __realign_regressors(self, h):
        
        """ realign regressors to forecast horizon """

        y = self.y[h-1:]
        T = y.shape[0]
        one_T = self.__one_T[:T,:]
        if self.p == 0:
            Y = np.zeros((T,0))
        else:
            Y = self.Y[:T]
        X = [one_T,Y]
        for i in range(self.n):
            if self.representation == 'unrestricted':
                X_i = self.X_[i][:T]
                X.append(X_i)
            elif self.representation == 'almon' or self.representation == 'fourier':
                Z_i = self.Z_[i][:T] 
                X.append(Z_i)
        X = np.hstack(X)
        return y, X, T
            
        
    def __minnesota_parameters(self, y, X, T, kappa):

        """ prior and posterior values for Minnesota """

        XX = X.T @ X
        Xy = X.T @ y
        kappa_bar = (T + kappa) / 2
        sigma = 1
        return XX, Xy, kappa_bar, sigma
        
        
    def __horseshoe_parameters(self, y, X, T, ll, kappa):
        
        """ prior and posterior values for horeseshoe """
        
        XX = X.T @ X
        Xy = X.T @ y
        gamma_nu = 1
        gamma_tau = (ll + 1) / 2
        gamma_eta = 1
        gamma_psi = 1
        kappa_bar = (T + ll + kappa) / 2
        sigma = 1
        tau2 = 5
        psi2_ = 5 * np.ones(ll)        
        return XX, Xy, gamma_nu, gamma_tau, gamma_eta, gamma_psi, kappa_bar, sigma, tau2, psi2_
        
    
    def __lasso_parameters(self, y, X, T, ll, kappa, f):
        
        """ prior and posterior values for lasso """

        inv_XX = la.robust_covariance_matrix(X)
        Xy = X.T @ y
        b_bar = inv_XX @ Xy
        f_bar = f + 2 * ll
        kappa_bar = kappa + T + ll
        sigma = 1 
        xi = 5 * np.ones(ll)
        mu = 1
        return inv_XX, b_bar, f_bar, kappa_bar, sigma, xi, mu


    def __draw_minnesota_beta(self, Xy, XX, sigma, inv_V):
        
        """ draw beta from its conditional posterior defined in (6.19.16) """  
        
        inv_V_bar = inv_V + XX / sigma
        b_bar_temp = Xy / sigma
        beta = rng.efficient_multivariate_normal(b_bar_temp, inv_V_bar) 
        return beta
       
    
    def __draw_minnesota_sigma(self, kappa_bar, y, X, beta, lamda):      
    
        """ draw sigma from its conditional posterior defined in (6.19.19) """ 
        
        residuals = y - X @ beta
        lamda_bar = residuals @ residuals + lamda
        sigma = rng.inverse_gamma(kappa_bar, lamda_bar)
        return sigma
    
    
    def __draw_horseshoe_beta(self, Xy, XX, tau2, psi2, sigma):
        
        """ draw beta from its conditional posterior defined in (6.19.30) """       
        
        inv_V_star = np.diag(1 / (psi2 * tau2)) + XX
        V_star = la.invert_spd_matrix(inv_V_star)
        V_bar = sigma * V_star
        b_bar = V_star @ Xy        
        beta = rng.multivariate_normal(b_bar, V_bar)
        beta2 = beta ** 2
        return beta, beta2
    
    
    def __draw_nu(self, gamma_nu, tau2):
        
        """ draw nu from its conditional posterior defined in (6.19.36) """        
        
        phi_nu = 1 / tau2 + 1
        nu = rng.inverse_gamma(gamma_nu, phi_nu)
        return nu
    

    def __draw_tau2(self, gamma_tau, nu, sigma, beta2, psi2):  
 
        """ draw tau2 from its conditional posterior defined in (6.19.33) """ 

        phi_tau = 1 / nu + (beta2 / psi2).sum() / (2 * sigma)
        tau2 = rng.inverse_gamma(gamma_tau, phi_tau) + 1e-12
        return tau2
    
  
    def __draw_eta(self, gamma_eta, psi2, ll):
        
        """ draw eta from its conditional posterior defined in (6.19.42) """  

        eta = np.zeros(ll)
        phi_eta = 1 / psi2 + 1
        for h in range(ll):
            eta[h] = rng.inverse_gamma(gamma_eta, phi_eta[h])
        return eta
 
    
    def __draw_psi2(self, gamma_psi, eta, sigma, beta2, tau2, ll):

        """ draw psi2 from its conditional posterior defined in (6.19.39) """        

        psi2 = np.zeros(ll)
        phi_psi = 1 / eta + beta2 / (2 * sigma * tau2)
        for h in range(ll):
            psi2[h] = rng.inverse_gamma(gamma_psi, phi_psi[h])
        return psi2
    
  
    def __draw_horseshoe_sigma(self, kappa_bar, y, X, beta, beta2, tau2, psi2, lamda):
        
        """ draw sigma from its conditional posterior defined in (6.19.45) """ 

        residuals = y - X @ beta
        ratios = (beta2 / (tau2 * psi2)).sum()
        lamda_bar = (residuals @ residuals + ratios + lamda) / 2
        sigma = rng.inverse_gamma(kappa_bar, lamda_bar)
        return sigma    
  
    
    def __draw_lasso_beta(self, inv_XX, b_bar, sigma, xi):
        
        """ draw beta from its conditional posterior defined in (6.19.54) """          

        V_bar = sigma * inv_XX
        chi = sigma ** 0.5 * xi
        beta = rng.truncated_multivariate_normal(b_bar, V_bar, -chi, chi)
        beta2 = beta ** 2
        return beta, beta2
        
        
    def __draw_xi(self, beta, mu, sigma, ll):
        
        """ draw xi from its conditional posterior defined in (6.19.57) """      
        
        sqrt_sigma = sigma ** 0.5
        xi = np.zeros(ll)
        for h in range(ll):
            rho_h = np.abs(beta[h]) / sqrt_sigma
            xi[h] = rng.gamma(1,mu) + rho_h
        return xi
            
    
    def __draw_mu(self, f_bar, g, xi):    
        
        """ draw mu from its conditional posterior defined in (6.19.60) """
        
        g_bar = g + xi.sum()
        mu = rng.inverse_gamma(f_bar,g_bar)
        return mu
    
    
    def __draw_lasso_sigma(self, kappa_bar, y, X, beta, beta2, xi, lamda):

        """ draw sigma from its conditional posterior defined in (6.19.63) """ 
    
        residuals = y - X @ beta
        lamda_bar = (lamda + residuals @ residuals) / 2
        iota_bar = max(beta2 / xi ** 2)
        gamma_scale = 1 / lamda_bar
        try:
            p_max = sst.gamma.cdf(1 / iota_bar, kappa_bar, scale=gamma_scale)
        except:
            import pdb; pdb.set_trace()    
        sigma = 1 / sst.gamma.ppf(nrd.uniform(high=p_max), kappa_bar, scale=gamma_scale)
        return sigma      
  
    
    def __recover_beta_from_parsimonious(self, mcmc_delta, iterations, representation, n, l, p, p_, q_, Q_):
        
        """ recover beta from delta using the equivalence defined in (6.19.66) """
        
        if representation == 'unrestricted':
            mcmc_beta = mcmc_delta
        else:
            mcmc_beta = np.zeros((l, iterations))
            mcmc_beta[:1+p] = mcmc_delta[:1+p]
            beta_index = np.cumsum(np.hstack([1+p,p_+1]))
            if representation == 'almon':
                delta_index = np.cumsum(np.hstack([1+p,q_+1]))
            elif representation == 'fourier':
                delta_index = np.cumsum(np.hstack([1+p,2*q_+1]))
            for i in range(n):
                Q_i = Q_[i]
                mcmc_delta_i = mcmc_delta[delta_index[i]:delta_index[i+1],:]
                mcmc_beta_i = Q_i @ mcmc_delta_i
                mcmc_beta[beta_index[i]:beta_index[i+1],:] = mcmc_beta_i
        return mcmc_beta  
    

    def __parameter_estimates(self):

        """ posterior estimates for midas regression parameters """ 
       
        beta_estimates = np.zeros((self.l,4))
        beta_estimates[:,0] = np.quantile(self.mcmc_beta[0],0.5,axis=1)
        beta_estimates[:,1] = np.quantile(self.mcmc_beta[0],(1-self.credibility_level)/2,axis=1)
        beta_estimates[:,2] = np.quantile(self.mcmc_beta[0],(1+self.credibility_level)/2,axis=1)
        beta_estimates[:,3] = np.std(self.mcmc_beta[0],axis=1)
        sigma_estimates = np.zeros(4,)
        sigma_estimates[0] = np.quantile(self.mcmc_sigma[0],0.5)
        sigma_estimates[1] = np.quantile(self.mcmc_sigma[0],(1-self.credibility_level)/2)
        sigma_estimates[2] = np.quantile(self.mcmc_sigma[0],(1+self.credibility_level)/2)    
        sigma_estimates[3] = np.std(self.mcmc_sigma[0])
        self.beta_estimates = beta_estimates
        self.sigma_estimates = sigma_estimates


    def __fitted_and_residual(self):
        
        """ fitted and residual values """

        y = self.y
        X = self.X
        T = self.T
        mcmc_beta = self.mcmc_beta[0]
        mcmc_fitted = np.zeros((T,self.iterations))
        mcmc_residual = np.zeros((T,self.iterations))        
        for i in range(self.iterations):
            y_hat = X @ mcmc_beta[:,i]
            residuals = y - y_hat
            mcmc_fitted[:,i] = y_hat
            mcmc_residual[:,i] = residuals            
        fitted_estimates = np.zeros((T,3))
        fitted_estimates[:,0] = np.quantile(mcmc_fitted,0.5,axis=1)
        fitted_estimates[:,1] = np.quantile(mcmc_fitted,(1-self.credibility_level)/2,axis=1)
        fitted_estimates[:,2] = np.quantile(mcmc_fitted,(1+self.credibility_level)/2,axis=1)         
        residual_estimates = np.zeros((T,3))
        residual_estimates[:,0] = np.quantile(mcmc_residual,0.5,axis=1)
        residual_estimates[:,1] = np.quantile(mcmc_residual,(1-self.credibility_level)/2,axis=1)
        residual_estimates[:,2] = np.quantile(mcmc_residual,(1+self.credibility_level)/2,axis=1)           
        self.fitted_estimates = fitted_estimates
        self.residual_estimates = residual_estimates            
 

    def __insample_criteria(self):
        
        """ in-sample fit evaluation criteria """

        insample_evaluation = ru.insample_evaluation_criteria(self.y, self.residual_estimates[:,0], self.T, self.l)
        self.insample_evaluation = insample_evaluation
        

    def __make_forecast(self, h):   

        """ forecasts for the midas regression model """        

        mcmc_forecast = np.zeros((self.iterations))
        if self.verbose and h == 1:
            verbose_string = 'Forecasts, 1 period ahead:'
        elif self.verbose and h > 1:
            verbose_string = 'Forecasts, ' + str(h) + ' periods ahead:' 
        X = self.Xp
        mcmc_beta = self.mcmc_beta[h-1]
        mcmc_sigma  = self.mcmc_sigma[h-1]
        for i in range(self.iterations):
            try:
                beta = mcmc_beta[:,i]
            except:
                pass
                # import pdb; pdb.set_trace()
            sigma  = mcmc_sigma[i]
            e = sigma ** 0.5 * nrd.randn()
            yp = X @ beta + e
            mcmc_forecast[i] = yp
            if self.verbose:
                cu.progress_bar(i, self.iterations, verbose_string)
        self.mcmc_forecast[h-1] = mcmc_forecast


    def __forecast_posterior_estimates(self, h, credibility_level):
        
        """ posterior estimates for forecasts """

        # obtain posterior estimates
        mcmc_forecast = self.mcmc_forecast[h-1]
        forecast_estimates = np.zeros(3)
        forecast_estimates[0] = np.quantile(mcmc_forecast, 0.5)
        forecast_estimates[1] = np.quantile(mcmc_forecast, (1-credibility_level)/2)
        forecast_estimates[2] = np.quantile(mcmc_forecast, (1+credibility_level)/2)
        self.forecast_estimates[h-1] = forecast_estimates
        
        
    def __bayesian_forecast_evaluation_criteria(self, y, mcmc_forecast, iterations, m):
        
        """ Bayesian forecast evaluation criteria from equations from equations (3.10.13) and (3.10.17) """   

        log_score = np.zeros(m)
        crps = np.zeros(m)
        for i in range(m):
            # get actual, prediction mean, prediction variance    
            y_i = y[i]
            forecasts = mcmc_forecast[i,:]
            mu_i = np.mean(forecasts)
            sigma_i = np.var(forecasts)
            # get log score from equation (3.10.14)
            log_pdf, _ = su.normal_pdf(y_i, mu_i, sigma_i)
            log_score[i] = - log_pdf
            # get CRPS from equation (3.10.17)
            term_1 = np.sum(np.abs(forecasts - y_i))
            term_2 = 0
            for j in range(iterations):
                term_2 += np.sum(np.abs(forecasts[j] - forecasts))
            crps[i] = term_1 / iterations - term_2 / (2 * iterations**2)
        log_score = np.mean(log_score)
        crps = np.mean(crps)
        bayesian_forecast_evaluation_criteria = {}
        bayesian_forecast_evaluation_criteria['log_score'] = log_score
        bayesian_forecast_evaluation_criteria['crps'] = crps
        return bayesian_forecast_evaluation_criteria   
