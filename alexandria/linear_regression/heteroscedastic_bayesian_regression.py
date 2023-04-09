# imports
import numpy as np
import numpy.random as nrd
import numpy.linalg as nla
from scipy.stats import chi2
import alexandria.math.linear_algebra as la
import alexandria.math.math_utilities as mu
import alexandria.math.stat_utilities as su
import alexandria.console.console_utilities as cu
import alexandria.math.random_number_generators as rng
from alexandria.linear_regression.linear_regression import LinearRegression



class HeteroscedasticBayesianRegression(LinearRegression):


    """
    Heteroscedastic Bayesian linear regression, developed in section 9.5
    
    Parameters:
    -----------
    endogenous : ndarray of shape (n_obs,)
        endogenous variable, defined in (3.9.1)
    
    exogenous : ndarray of shape (n_obs,n_regressors)
        exogenous variables, defined in (3.9.1)
   
    heteroscedastic : ndarray of shape (n_obs,h), default = exogenous
        heteroscedasticity variables, defined in (3.9.37)
    
    constant : bool, default = True
        if True, an intercept is included in the regression
   
    trend : bool, default = False
        if True, a linear trend is included in the regression
   
    quadratic_trend : bool, default = False
        if True, a quadratic trend is included in the regression
   
    b_exogenous : float or ndarray of shape (n_regressors,), default = 0
        prior mean for regressors
   
    V_exogenous : float or ndarray of shape (n_regressors,), default = 1
        prior variance for regressors (positive)
   
    b_constant : float, default = 0
        prior mean for constant term
   
    V_constant : float, default = 1
        prior variance for constant (positive)
    
    b_trend : float, default = 0
        prior mean for trend
   
    V_trend : float, default = 1
        prior variance for trend (positive)
    
    b_quadratic_trend : float, default = 0
        prior mean for quadratic_trend
   
    V_quadratic_trend : float, default = 1
        prior variance for quadratic_trend (positive)
   
    alpha : float, default = 1e-4
        prior shape, defined in (3.9.21)
   
    delta : float, default = 1e-4
        prior scale, defined in (3.9.21)
   
    g : float or ndarray of shape (h,), default = 0
        prior mean, defined in (3.9.43)
   
    Q : float or ndarray of shape (h,), default = 100
        prior variance, defined in (3.9.43)
   
    tau : float, default = 0.001
         variance of the random walk shock, defined in (3.9.50)
   
    iterations : int, default = 2000
        post burn-in iterations for MCMC algorithm
   
    burn : int, default = 1000
        burn-in iterations for MCMC algorithm
    
    thinning : bool, default = False
        if True, thinning is applied to posterior draws from MCMC algorithm
    
    thinning_frequency : int, default = 10
        if thinning is True, retains only one out of so many draws from MCMC algorithm
   
    credibility_level : float, default = 0.95
        credibility level (between 0 and 1)
    
    verbose : bool, default = False
        if True, displays a progress bar  
    
    
    Attributes
    ----------
    endogenous : ndarray of shape (n_obs,)
        endogenous variable, defined in (3.9.1)
    
    exogenous : ndarray of shape (n_obs,n_regressors)
        exogenous variables, defined in (3.9.1)
   
    heteroscedastic : ndarray of shape (n_obs,h), default = exogenous
        heteroscedasticity variables, defined in (3.9.37)
    
    constant : bool
        if True, an intercept is included in the regression
   
    trend : bool
        if True, a linear trend is included in the regression
   
    quadratic_trend : bool
        if True, a quadratic trend is included in the regression
   
    b_exogenous : float or ndarray of shape (n_regressors,)
        prior mean for regressors
   
    V_exogenous : float or ndarray of shape (n_regressors,)
        prior variance for regressors (positive)
  
    b_constant : float
        prior mean for constant term
   
    V_constant : float
        prior variance for constant (positive)
    
    b_trend : float
        prior mean for trend
   
    V_trend : float
        prior variance for trend (positive)
    
    b_quadratic_trend : float
        prior mean for quadratic_trend
   
    V_quadratic_trend : float
        prior variance for quadratic_trend (positive)
   
    b : ndarray of shape (k,)
        prior mean, defined in (3.9.10)
   
    V : ndarray of shape (k,k)
        prior variance, defined in (3.9.10)
   
    alpha : float
        prior shape, defined in (3.9.21)
   
    delta : float
        prior scale, defined in (3.9.21)
   
    g : float or ndarray of shape (h,)
        prior mean, defined in (3.9.43)
   
    Q : float or ndarray of shape (h,)
        prior variance, defined in (3.9.43)
   
    tau : float
         variance of the random walk shock, defined in (3.9.50)    
   
    iterations : int
        post burn-in iterations for MCMC algorithm
   
    burn : int
        burn-in iterations for MCMC algorithm
   
    thinning : bool, default = False
        if True, thinning is applied to posterior draws from MCMC algorithm
    
    thinning_frequency : int, default = 10
        if thinning is True, retains only one out of so many draws from MCMC algorithm
    
    credibility_level : float
        credibility level (between 0 and 1)
    
    verbose : bool
        if True, displays a progress bar during MCMC algorithms
   
    y : ndarray of shape (n,)
        explained variables, defined in (3.9.3)
    
    X : ndarray of shape (n,k)
        explanatory variables, defined in (3.9.3)
    
    Z : ndarray of shape (n,h)
        heteroscedasticity variables, defined in (3.9.39)
   
    n : int
        number of observations, defined in (3.9.1)
    
    k : int
        dimension of beta, defined in (3.9.1)
    
    h : int
        dimension of gamma, defined in (3.9.37)    
   
    alpha_bar : float
        posterior scale, defined in (3.9.35)
   
    mcmc_beta : ndarray of shape (k,iterations)
        storage of mcmc values for beta
   
    mcmc_sigma : ndarray of shape (iterations,)
        storage of mcmc values for sigma
   
    mcmc_gamma : ndarray of shape (h,iterations)
        storage of mcmc values for gamma
   
    estimates_beta : ndarray of shape (k,4)
        posterior estimates for beta
        column 1: interval lower bound; column 2: median; 
        column 3: interval upper bound; column 4: standard deviation
   
    estimates_sigma : float
        posterior estimate for sigma
   
    estimates_gamma : ndarray of shape (h,3)
        posterior estimates for gamma
        column 1: interval lower bound; column 2: median; 
        column 3: interval upper bound
   
    X_hat : ndarray of shape (m,k)
        predictors for the model 
   
    Z_hat : ndarray of shape (m,h)
        heteroscedasticity predictors for the model
   
    m : int
        number of predicted observations, defined in (3.10.1)   
   
    mcmc_forecasts : ndarray of shape (m,iterations)
        storage of mcmc values for forecasts
   
    estimates_forecasts : ndarray of shape (m,3)
        posterior estimates for predictions   
        column 1: interval lower bound; column 2: median; 
        column 3: interval upper bound
    
    estimates_fit : ndarray of shape (n,)
        posterior estimates (median) for in sample-fit
   
    estimates_residuals : ndarray of shape (n,)
        posterior estimates (median) for residuals    
            
    insample_evaluation : dict
        in-sample fit evaluation (SSR, R2, adj-R2)
                                
    forecast_evaluation_criteria : dict
        out-of-sample forecast evaluation (RMSE, MAE, ...)
   
    m_y : float
        log10 marginal likelihood
   
   
    Methods
    ----------
    estimate
    forecast
    fit_and_residuals
    forecast_evaluation
    marginal_likelihood
    """


    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self, endogenous, exogenous, heteroscedastic = None,
            constant = True, trend = False, quadratic_trend = False,    
            b_exogenous = 0, V_exogenous = 1, b_constant = 0, V_constant = 1,  
            b_trend = 0, V_trend = 1, b_quadratic_trend = 0, V_quadratic_trend = 1,
            alpha = 1e-4, delta = 1e-4, g = 0, Q = 100, tau = 0.001, 
            iterations = 2000, burn = 1000, thinning = False, 
            thinning_frequency = 10, credibility_level = 0.95, verbose = False):
        
        """
        constructor for the HeteroscedasticBayesianRegression class
        """
        
        self.endogenous = endogenous
        self.exogenous = exogenous
        # by default, set heteroscedastic regressors to exogenous regressors
        if heteroscedastic is None:
            self.heteroscedastic = exogenous
        else:
            self.heteroscedastic = heteroscedastic
        self.constant = constant
        self.trend = trend
        self.quadratic_trend = quadratic_trend
        self.b_exogenous = b_exogenous
        self.V_exogenous = V_exogenous
        self.b_constant = b_constant
        self.V_constant = V_constant
        self.b_trend = b_trend
        self.V_trend = V_trend
        self.b_quadratic_trend = b_quadratic_trend
        self.V_quadratic_trend = V_quadratic_trend 
        self.alpha = alpha
        self.delta = delta
        self.g = g
        self.Q = Q
        self.tau = tau
        self.iterations = iterations
        self.burn = burn
        self.thinning = thinning
        self.thinning_frequency = thinning_frequency
        self.credibility_level = credibility_level
        self.verbose = verbose
        # make regressors
        self._make_regressors()
    
    
    def estimate(self):
        
        """
        estimate()
        generates posterior estimates for linear regression model parameters beta, sigma and gamma
        
        parameters:
        none
        
        returns:
        none
        """
        
        # define prior values
        self.__prior()
        # define posterior values
        self.__posterior()
        # run MCMC algorithm for regression parameters
        self.__parameter_mcmc()
        # obtain posterior estimates for regression parameters
        self.__parameter_estimates()    
    
    
    def forecast(self, X_hat, credibility_level, Z_hat = None):

        """
        forecast(X_hat, credibility_level, Z_hat = None)
        predictions for the linear regression model using algorithm 10.1
        
        parameters:
        X_hat : matrix of shape (m,k)
            array of predictors
        credibility_level : float
            credibility level for predictions (between 0 and 1)
        Z_hat : matrix of shape (m,k), optional
            array of heteroscedasticity predictors
        
        returns:
        estimates_forecasts : matrix of size (m,3)
            posterior estimates for predictions
            column 1: interval lower bound; column 2: median; 
            column 3: interval upper bound
        """
        
        # by default, set heteroscedastic regressors to exogenous regressors
        if Z_hat is None:
            Z_hat = X_hat
        # run mcmc algorithm for predictive density
        mcmc_forecasts, m = self.__forecast_mcmc(X_hat, Z_hat)
        # obtain posterior estimates
        estimates_forecasts = self.__forecast_estimates(mcmc_forecasts, credibility_level)
        # save as attributes
        self.X_hat = X_hat
        self.Z_hat = Z_hat
        self.m = m
        self.mcmc_forecasts = mcmc_forecasts
        self.estimates_forecasts = estimates_forecasts
        return estimates_forecasts    
    
    
    def fit_and_residuals(self):
        
        """
        fit_and_residuals()
        estimates of in-sample fit and regression residuals
        
        parameters:
        none
        
        returns:
        none
        """
        
        # unpack
        y = self.y        
        X = self.X
        beta = self.estimates_beta[:,1]
        k = self.k
        n = self.n      
        # estimate fits and residuals
        estimates_fit = X @ beta
        estimates_residuals = y - X @ beta
        # estimate in-sample prediction criteria from equation (3.10.8)
        res = estimates_residuals
        ssr = res @ res
        tss = (y - np.mean(y)) @ (y - np.mean(y))
        r2 = 1 - ssr / tss
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)
        insample_evaluation = {}
        insample_evaluation['ssr'] = ssr
        insample_evaluation['r2'] = r2
        insample_evaluation['adj_r2'] = adj_r2   
        # save as attributes
        self.estimates_fit = estimates_fit
        self.estimates_residuals = estimates_residuals
        self.insample_evaluation = insample_evaluation
        
        
    def forecast_evaluation(self, y):
        
        """
        forecast_evaluation(y)
        forecast evaluation criteria for the linear regression model
        
        parameters:
        y : ndarray of shape (m,)
            array of realised values for forecast evaluation
            
        returns:
        none
        """
        
        # unpack
        mcmc_forecasts = self.mcmc_forecasts
        estimates_forecasts = self.estimates_forecasts
        m = self.m
        iterations = self.iterations
        # calculate forecast error
        y_hat = estimates_forecasts[:,1]
        err = y - y_hat
        # compute forecast evaluation from equation (3.10.11)
        rmse = np.sqrt(err @ err / m)
        mae = np.sum(np.abs(err)) / m
        mape = 100 * np.sum(np.abs(err / y)) / m
        theil_u = np.sqrt(err @ err) / (np.sqrt(y @ y) + np.sqrt(y_hat @ y_hat))
        bias = np.sum(err) / np.sum(np.abs(err))        
        # loop over the m predictions
        log_score = np.zeros(m)
        crps = np.zeros(m)        
        for i in range(m):
            # get actual, prediction mean, prediction variance    
            y_i = y[i]
            forecasts = mcmc_forecasts[i,:]
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
        forecast_evaluation_criteria = {}
        forecast_evaluation_criteria['rmse'] = rmse
        forecast_evaluation_criteria['mae'] = mae
        forecast_evaluation_criteria['mape'] = mape
        forecast_evaluation_criteria['theil_u'] = theil_u
        forecast_evaluation_criteria['bias'] = bias
        forecast_evaluation_criteria['log_score'] = log_score
        forecast_evaluation_criteria['crps'] = crps
        # save as attributes
        self.forecast_evaluation_criteria = forecast_evaluation_criteria


    def marginal_likelihood(self):
        
        """ 
        log10 marginal likelihood, defined in (3.10.29)
        
        parameters:
        none
        
        returns:
        m_y: float
            log10 marginal likelihood
        """
        
        # unpack
        y = self.y
        X = self.X
        Z = self.Z
        n = self.n
        k = self.k
        h = self.h
        b = self.b
        inv_V = self.__inv_V
        V = self.V
        g = self.g
        Q = self.Q
        inv_Q = self.__inv_Q
        alpha = self.alpha
        delta = self.delta
        mcmc_beta = self.mcmc_beta
        mcmc_sigma = self.mcmc_sigma
        mcmc_gamma = self.mcmc_gamma
        iterations = self.iterations
        # generate theta_hat and Sigma_hat
        mcmc_theta = np.vstack((mcmc_beta, mcmc_sigma, mcmc_gamma))
        theta_hat = np.mean(mcmc_theta, 1)
        Sigma_hat = np.cov(mcmc_theta)
        inv_Sigma_hat = la.invert_spd_matrix(Sigma_hat)
        # generate parameters for truncation of the Chi2
        omega = 0.5
        bound = chi2.ppf(omega, k + h + 1)
        # compute the log of first row of (3.10.29)
        J = iterations
        term_1 = - np.log(omega * J)
        term_2 = (n - 1) / 2 * np.log(2 * np.pi)
        term_3 = -0.5 * np.log(nla.det(Sigma_hat))
        term_4 = 0.5 * np.log(nla.det(V))
        term_5 = 0.5 * np.log(nla.det(Q))
        term_6 = np.log(mu.gamma(alpha / 2))
        term_7 = - alpha / 2 * np.log(delta / 2)
        row_1 = - (term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7)
        # for second row of (3.10.29), compute the log of each term in summation
        summation = np.zeros(iterations)
        for i in range(iterations):
            theta = mcmc_theta[:,i]
            beta = mcmc_beta[:,i]
            sigma = mcmc_sigma[i]
            gamma = mcmc_gamma[:,i]
            quadratic_form = (theta - theta_hat) @ inv_Sigma_hat @ (theta - theta_hat)
            if quadratic_form > bound:
                summation[i] = -1000;
            else:
                W = np.exp(Z @ gamma)
                term_1 = 0.5 * np.sum(np.log(W))
                term_2 = ((alpha + n) / 2 + 1) * np.log(sigma)
                residuals = y - X @ beta
                inv_sigma_W = 1 / (sigma * W)
                term_3 = 0.5 * residuals * inv_sigma_W @ residuals
                term_4 = 0.5 * (beta - b) @ inv_V @ (beta - b)
                term_5 = 0.5 * delta / sigma
                term_6 = 0.5 * (gamma - g) @ inv_Q @ (gamma - g)
                term_7 = - 0.5 * quadratic_form
                summation[i] = term_1 + term_2 + term_3 + term_4 + term_5 \
                               + term_6 + term_7
        # turn sum of the logs into log of the sum
        row_2 = - mu.log_sum_exp(summation)
        # sum the two rows and convert to log10
        log_f_y = row_1 + row_2
        m_y = log_f_y / np.log(10)
        self.m_y = m_y
        return m_y    
    

    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------


    def _make_regressors(self):
        
        """generates regressors Z defined in (3.9.39)"""
        
        # run superclass function
        LinearRegression._make_regressors(self)
        # unpack
        heteroscedastic = self.heteroscedastic
        # define Z        
        Z = heteroscedastic
        # get dimensions
        h = Z.shape[1]
        # save as attributes
        self.Z = Z
        self.h = h
    
    
    def __prior(self):
        
        """creates prior elements b, V, g and Q defined in (3.9.10) and (3.9.43)"""
        
        # generate b
        self.__generate_b()
        # generate V
        self.__generate_V()
        # generate g
        self.__generate_g()
        # generate Q
        self.__generate_Q()
        
    
    def __generate_b(self):
        
        """creates prior element b"""
        
        # unpack
        constant = self.constant
        trend = self.trend
        quadratic_trend = self.quadratic_trend
        b_exogenous = self.b_exogenous
        b_constant = self.b_constant
        b_trend = self.b_trend
        b_quadratic_trend = self.b_quadratic_trend
        n_exogenous = self._n_exogenous
        # if b_exogenous is a scalar, turn it into a vector replicating the value
        if isinstance(b_exogenous, (int,float)):
            b_exogenous = b_exogenous * np.ones(n_exogenous)
        b = b_exogenous
        # if quadratic trend is included, add to prior mean
        if quadratic_trend:
            b = np.hstack((b_quadratic_trend, b))
        # if trend is included, add to prior mean
        if trend:
            b = np.hstack((b_trend, b))
        # if constant is included, add to prior mean
        if constant:
            b = np.hstack((b_constant, b))
        # save as attribute
        self.b = b 
    
    
    def __generate_V(self):
        
        """creates prior element V"""
        
        # unpack
        b = self.b        
        constant = self.constant
        trend = self.trend
        quadratic_trend = self.quadratic_trend
        V_exogenous = self.V_exogenous
        V_constant = self.V_constant
        V_trend = self.V_trend
        V_quadratic_trend = self.V_quadratic_trend
        n_exogenous = self._n_exogenous
        # if V_exogenous is a scalar, turn it into a vector replicating the value
        if isinstance(V_exogenous, (int,float)):
            V_exogenous = V_exogenous * np.ones(n_exogenous)
        V = V_exogenous
        # if quadratic trend is included, add to prior mean
        if quadratic_trend:
            V = np.hstack((V_quadratic_trend, V))
        # if trend is included, add to prior mean
        if trend:
            V = np.hstack((V_trend, V))
        # if constant is included, add to prior mean
        if constant:
            V = np.hstack((V_constant, V))        
        # convert the vector V into an array
        inv_V_b = b / V
        inv_V = np.diag(1/V)
        V = np.diag(V)
        # save as attributes
        self.V = V
        self.__inv_V = inv_V
        self.__inv_V_b = inv_V_b     
    
    
    def __generate_g(self):
        
        """creates prior element g"""
        
        # unpack
        g = self.g
        # if g is a scalar, turn it into a vector replicating the value
        if isinstance(g, (int,float)):
            h = self.h
            g = g * np.ones(h)
        # save as attribute
        self.g = g   
    
    
    def __generate_Q(self):
        
        """creates prior element Q"""
        
        # unpack
        Q = self.Q
        # if Q is a scalar, turn it into a vector replicating the value
        if isinstance(Q, (int,float)):
            h = self.h
            inv_Q = np.identity(h) / Q
            Q = np.identity(h) * Q
        # if Q is a vector, turn it into a diagonal array
        else:
            inv_Q = np.diag(1/Q)
            Q = np.diag(Q)
        # save as attributes
        self.Q = Q
        self.__inv_Q = inv_Q    
    

    def __posterior(self):
        
        """creates constant posterior element alpha_bar defined in (3.9.48)"""
        
        # unpack
        alpha = self.alpha
        n = self.n
        # set value
        alpha_bar = alpha + n
        # save as attribute
        self.alpha_bar = alpha_bar


    def __parameter_mcmc(self):
        
        """posterior distribution for parameters from algorithm 9.2"""
        
        # unpack
        y = self.y
        X = self.X
        Z = self.Z
        k = self.k
        h = self.h
        inv_V = self.__inv_V
        inv_V_b = self.__inv_V_b
        alpha_bar = self.alpha_bar
        delta = self.delta
        iterations = self.iterations
        burn = self.burn
        thinning = self.thinning
        thinning_frequency = self.thinning_frequency
        verbose = self.verbose
        # if thinning, multiply iterations accordingly
        if thinning:
            iterations *= thinning_frequency
            burn *= thinning_frequency
        # preallocate storage space
        mcmc_beta = np.zeros((k,iterations))
        mcmc_sigma = np.zeros(iterations)
        mcmc_gamma = np.zeros((h, iterations))
        total_iterations = iterations + burn
        # set initial values
        beta = np.zeros(k)
        inv_sigma = 1
        gamma = np.zeros(h)
        inv_W = 1 / np.exp(Z @ gamma)
        acceptance_rate = 0
        # run algorithm 9.2 (Gibbs sampling for the model parameters)
        for i in range(total_iterations):
            # draw beta from its conditional posterior
            beta, residuals = self.__draw_beta(inv_V, inv_V_b, inv_W, inv_sigma, X, y)
            # draw sigma from its conditional posterior
            sigma, inv_sigma = self.__draw_sigma(inv_W, delta, alpha_bar, residuals)
            # draw gamma, using a Metropolis_Hastings step
            gamma, inv_W, accept = self.__draw_gamma(gamma, residuals, inv_sigma, Z)
            # if burn-in sample is over, record value
            if i >= burn:
                acceptance_rate += accept
                mcmc_beta[:,i-burn] = beta
                mcmc_sigma[i-burn] = sigma
                mcmc_gamma[:,i-burn] = gamma
            # if verbose, display progress bar
            if verbose:
                cu.progress_bar(i, total_iterations, 'Model parameters:')
        # calculate aceptance rate
        acceptance_rate /= iterations
        if verbose:
            cu.print_message('Acceptance rate on Metropolis-Hastings algorithm: ' \
                             + str(round(100 * acceptance_rate, 2)) + '%.')            
        # trim if trimming is applied
        if thinning:
            mcmc_beta = mcmc_beta[:,::thinning_frequency]
            mcmc_sigma = mcmc_sigma[::thinning_frequency]
            mcmc_gamma = mcmc_gamma[:,::thinning_frequency]            
        # save as attributes
        self.mcmc_beta = mcmc_beta
        self.mcmc_sigma = mcmc_sigma
        self.mcmc_gamma = mcmc_gamma            


    def __draw_beta(self, inv_V, inv_V_b, inv_W, inv_sigma, X, y):
        
        """draw beta from its conditional posterior defined in (3.9.45)"""

        # posterior parameters for beta, defined in (3.9.46)
        XW = X.T * inv_W
        inv_V_bar = inv_V + inv_sigma * XW @ X
        # posterior b_bar            
        b_bar_temp = inv_V_b + inv_sigma * XW @ y
        # efficient sampling of beta (algorithm 9.4)
        beta = rng.efficient_multivariate_normal(b_bar_temp, inv_V_bar)
        # compute residuals
        residuals = y - X @ beta
        return beta, residuals


    def __draw_sigma(self, inv_W, delta, alpha_bar, residuals):
        
        """draw sigma from its conditional posterior defined in (3.9.35)"""

        # compute delta_bar
        delta_bar = delta + residuals * inv_W @ residuals
        # sample sigma
        sigma = rng.inverse_gamma(alpha_bar / 2, delta_bar / 2)
        inv_sigma = 1 / sigma
        return sigma, inv_sigma


    def __draw_gamma(self, gamma, residuals, inv_sigma, Z):
        
        """sample gamma, using a Metropolis_Hastings step"""

        # run the Metropolis-Hastings algorithm
        gamma, accept = self.__metropolis_hastings(gamma, residuals, inv_sigma)
        inv_W = 1 / np.exp(Z @ gamma)
        return gamma, inv_W, accept


    def __metropolis_hastings(self, gamma, residuals, inv_sigma):

        """Metropolis-hastings step, using (3.9.50) and (3.9.51)"""
        
        # unpack
        n = self.n
        h = self.h
        Z = self.Z
        tau = self.tau
        g = self.g
        inv_Q = self.__inv_Q
        # get candidate value from (3.9.50)
        gamma_tilde = gamma + np.sqrt(tau) * nrd.randn(h)
        # compute acceptance probability from (3.9.51)
        term_1 = np.ones(n) @ Z @ (gamma_tilde - gamma)
        term_2 = residuals * (np.exp(-Z @ gamma_tilde) - np.exp(-Z @ gamma)) \
                 @ residuals * inv_sigma
        term_3 = (gamma_tilde - g) @ inv_Q @ (gamma_tilde - g)
        term_4 = - (gamma - g) @ inv_Q @ (gamma - g)
        prob = min(1, np.exp(-0.5 * (term_1 + term_2 + term_3 + term_4)))
        # get uniform random number to decide on acceptance or rejection
        u = nrd.rand()
        if u <= prob:
            gamma = gamma_tilde
            accept = 1
        else:
            accept = 0
        return gamma, accept


    def __parameter_estimates(self):
        
        """
        point estimates and credibility intervals for model parameters
        uses quantiles of the empirical posterior distribution
        """
        
        # unpack
        mcmc_beta = self.mcmc_beta
        mcmc_sigma = self.mcmc_sigma
        mcmc_gamma = self.mcmc_gamma
        credibility_level = self.credibility_level
        k = self.k
        h = self.h
        # initiate storage: 4 columns: lower bound, median, upper bound, standard deviation
        estimates_beta = np.zeros((k,4))
        # fill estimates
        estimates_beta[:,0] = np.quantile(mcmc_beta, (1-credibility_level)/2, 1)
        estimates_beta[:,1] = np.quantile(mcmc_beta, 0.5, 1)
        estimates_beta[:,2] = np.quantile(mcmc_beta, (1+credibility_level)/2, 1)
        estimates_beta[:,3] = np.std(mcmc_beta, 1)
        # get median for sigma
        estimates_sigma = np.quantile(mcmc_sigma, 0.5)
        # get estimates for gamma
        estimates_gamma = np.zeros((h,4))
        estimates_gamma[:,0] = np.quantile(mcmc_gamma, (1-credibility_level)/2, 1)
        estimates_gamma[:,1] = np.quantile(mcmc_gamma, 0.5, 1)
        estimates_gamma[:,2] = np.quantile(mcmc_gamma, (1+credibility_level)/2, 1)
        estimates_gamma[:,3] = np.std(mcmc_gamma, 1)
        # save as attributes
        self.estimates_beta = estimates_beta
        self.estimates_sigma = estimates_sigma
        self.estimates_gamma = estimates_gamma


    def __forecast_mcmc(self, X_hat, Z_hat = None):
        
        """posterior predictive distribution from algorithm 10.2""" 
        
        # unpack
        mcmc_beta = self.mcmc_beta
        mcmc_sigma = self.mcmc_sigma
        mcmc_gamma = self.mcmc_gamma
        iterations = self.iterations
        verbose = self.verbose
        # add constant and trends if included
        m = X_hat.shape[0]
        X_hat = self._add_intercept_and_trends(X_hat, False)
        # initiate storage, loop over simulations and simulate predictions
        mcmc_forecasts = np.zeros((m, iterations))        
        for i in range(iterations):
            beta = mcmc_beta[:,i]
            sigma = mcmc_sigma[i]
            gamma = mcmc_gamma[:,i]
            W = np.exp(Z_hat @ gamma)
            y_hat = X_hat @ beta + np.sqrt(sigma * W) * nrd.randn(m)
            mcmc_forecasts[:,i] = y_hat
            if verbose:
                cu.progress_bar(i, iterations, 'Predictions:')
        return mcmc_forecasts, m


    def __forecast_estimates(self, mcmc_forecasts, credibility_level):
        
        """point estimates and credibility intervals for predictions""" 
        
        m = mcmc_forecasts.shape[0]
        # initiate estimate storage; 3 columns: lower bound, median, upper bound
        estimates_forecasts = np.zeros((m,3))
        # fill estimates
        estimates_forecasts[:,0] = np.quantile(mcmc_forecasts, (1-credibility_level)/2, 1)
        estimates_forecasts[:,1] = np.quantile(mcmc_forecasts, 0.5, 1)
        estimates_forecasts[:,2] = np.quantile(mcmc_forecasts, (1+credibility_level)/2, 1)
        return estimates_forecasts





