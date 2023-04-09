# imports
import numpy as np
import numpy.random as nrd
import alexandria.math.linear_algebra as la
import alexandria.math.math_utilities as mu
import alexandria.math.stat_utilities as su
import alexandria.console.console_utilities as cu
import alexandria.math.random_number_generators as rng
from alexandria.linear_regression.linear_regression import LinearRegression



class IndependentBayesianRegression(LinearRegression):


    """
    Independent Bayesian linear regression, developed in section 9.4
    
    Parameters:
    -----------
    endogenous : ndarray of shape (n_obs,)
        endogenous variable, defined in (3.9.1)
    
    exogenous : ndarray of shape (n_obs,n_regressors)
        exogenous variables, defined in (3.9.1)
    
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
   
    iterations : int, default = 2000
        post burn-in iterations for MCMC algorithm
   
    burn : int, default = 1000
        burn-in iterations for MCMC algorithm
    
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
   
    iterations : int
        post burn-in iterations for MCMC algorithm
   
    burn : int
        burn-in iterations for MCMC algorithm    
    
    credibility_level : float
        credibility level (between 0 and 1)
    
    verbose : bool
        if True, displays a progress bar during MCMC algorithms
   
    y : ndarray of shape (n,)
        explained variables, defined in (3.9.3)
    
    X : ndarray of shape (n,k)
        explanatory variables, defined in (3.9.3)
   
    n : int
        number of observations, defined in (3.9.1)
    
    k : int
        dimension of beta, defined in (3.9.1)
   
    alpha_bar : float
        posterior scale, defined in (3.9.35)
   
    mcmc_beta : ndarray of shape (k, iterations)
        storage of mcmc values for beta
   
    mcmc_sigma : ndarray of shape (iterations,)
        storage of mcmc values for sigma
   
    estimates_beta : ndarray of shape (k,4)
        posterior estimates for beta
        column 1: interval lower bound; column 2: median; 
        column 3: interval upper bound; column 4: standard deviation
   
    estimates_sigma : float
        posterior estimate for sigma
   
    X_hat : ndarray of shape (m,k)
        predictors for the model 
   
    m : int
        number of predicted observations, defined in (3.10.1)   
   
    mcmc_forecasts : matrix of size ndarray of shape (m, iterations)
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


    def __init__(self, endogenous, exogenous, constant = True, trend = False, 
                 quadratic_trend = False, b_exogenous = 0, V_exogenous = 1,
                 b_constant = 0, V_constant = 1, b_trend = 0, V_trend = 1,
                 b_quadratic_trend = 0, V_quadratic_trend = 1,
                 alpha = 1e-4, delta = 1e-4, iterations = 2000, burn = 1000,
                 credibility_level = 0.95, verbose = False):
        
        """
        constructor for the IndependentBayesianRegression class
        """        
        
        self.endogenous = endogenous
        self.exogenous = exogenous
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
        self.iterations = iterations
        self.burn = burn
        self.credibility_level = credibility_level
        self.verbose = verbose  
        # make regressors
        self._make_regressors()


    def estimate(self):
        
        """
        estimate()
        generates posterior estimates for linear regression model parameters beta and sigma 
        
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


    def forecast(self, X_hat, credibility_level):
        
        """
        forecast(X_hat, credibility_level)
        predictions for the linear regression model using algorithm 10.1

        parameters:
        X_hat : ndarray of shape (m,k)
            array of predictors
        credibility_level : float
            credibility level for predictions (between 0 and 1)

        returns:
        estimates_forecasts : ndarray of shape (m,3)
            posterior estimates for predictions
            column 1: interval lower bound; column 2: median; 
            column 3: interval upper bound
        """
        
        # run mcmc algorithm for predictive density
        mcmc_forecasts, m = self.forecast_mcmc(X_hat)
        # obtain posterior estimates
        estimates_forecasts = self.forecast_estimates(mcmc_forecasts, credibility_level)
        # save as attributes
        self.X_hat = X_hat
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
        marginal_likelihood()
        log10 marginal likelihood, defined in (3.10.27)
        
        parameters:
        none
        
        returns:
        m_y: float
            log10 marginal likelihood
        """

        # unpack
        y = self.y
        X = self.X
        XX = self._XX
        Xy = self._Xy
        b = self.b
        V = self.V
        inv_V = self.__inv_V
        inv_V_b = self.__inv_V_b
        alpha = self.alpha
        delta = self.delta
        alpha_bar = self.alpha_bar
        estimates_beta = self.estimates_beta
        mcmc_sigma = self.mcmc_sigma
        n = self.n
        iterations = self.iterations
        # generate high density values
        beta = estimates_beta[:,1]
        res = y - X @ beta
        delta_bar = delta + res @ res
        # generate the vector of summation terms
        summation = np.zeros(iterations)
        VXX = V @ XX
        for i in range(iterations):
            inv_sigma = 1 / mcmc_sigma[i]
            inv_V_bar = inv_V + inv_sigma * XX
            V_bar = la.invert_spd_matrix(inv_V_bar)
            b_bar = V_bar @ (inv_V_b + inv_sigma * Xy)
            summation[i] = 0.5 * (la.stable_determinant(inv_sigma * VXX) \
                           - (beta - b_bar) @ inv_V_bar @ (beta - b_bar))
        # compute the marginal likelihood, part by part
        term_1 = -0.5 * n * np.log(np.pi) \
            - 0.5 * (beta - b) @ inv_V @ (beta - b) + np.log(iterations)
        term_2 = - mu.log_sum_exp(summation)
        term_3 = 0.5 * (alpha * np.log(delta) - alpha_bar * np.log(delta_bar))
        term_4 = np.log(mu.gamma(alpha_bar / 2)) - np.log(mu.gamma(alpha / 2))
        log_f_y = term_1 + term_2 + term_3 + term_4
        m_y = log_f_y / np.log(10)
        self.m_y = m_y
        return m_y


    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------


    def __prior(self):
        
        """creates prior elements b and V defined in (3.9.10)"""
        
        # generate b
        self.__generate_b()
        # generate V
        self.__generate_V()


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


    def __posterior(self):
        
        """creates constant posterior element alpha_bar defined in (3.9.35)"""
        
        # unpack
        alpha = self.alpha
        n = self.n
        # set value
        alpha_bar = alpha + n
        # save as attribute
        self.alpha_bar = alpha_bar


    def __parameter_mcmc(self):
        
        """posterior distribution for parameters from algorithm 9.1"""
        
        # unpack
        X = self.X
        y = self.y
        XX = self._XX
        Xy = self._Xy
        k = self.k
        inv_V = self.__inv_V
        inv_V_b = self.__inv_V_b
        alpha_bar = self.alpha_bar
        delta = self.delta
        iterations = self.iterations
        burn = self.burn
        verbose = self.verbose
        # preallocate storage space
        mcmc_beta = np.zeros((k,iterations))
        mcmc_sigma = np.zeros(iterations)
        total_iterations = iterations + burn
        # set initial values
        beta = np.zeros(k)
        inv_sigma = 1
        # run algorithm 9.1 (Gibbs sampling for the model parameters)
        for i in range(total_iterations):
            # draw beta from its conditional posterior
            beta = self.__draw_beta(inv_V, inv_V_b, inv_sigma, XX, Xy)
            # draw sigma from its conditional posterior
            sigma, inv_sigma = self.__draw_sigma(y, X, beta, delta, alpha_bar)
            # if burn-in sample is over, record value
            if i >= burn:
                mcmc_beta[:,i-burn] = beta
                mcmc_sigma[i-burn] = sigma
            # if verbose, display progress bar
            if verbose:
                cu.progress_bar(i, total_iterations, 'Model parameters:')
        # save as attributes
        self.mcmc_beta = mcmc_beta
        self.mcmc_sigma = mcmc_sigma


    def __draw_beta(self, inv_V, inv_V_b, inv_sigma, XX, Xy):
        
        """draw beta from its conditional posterior defined in (3.9.33)"""
        
        # posterior V_bar
        inv_V_bar = inv_V + inv_sigma * XX
        # posterior b_bar
        b_bar_temp = inv_V_b + inv_sigma * Xy
        # efficient sampling of beta (algorithm 9.4)
        beta = rng.efficient_multivariate_normal(b_bar_temp, inv_V_bar)
        return beta


    def __draw_sigma(self, y, X, beta, delta, alpha_bar):
        
        """draw sigma from its conditional posterior defined in (3.9.35)"""

        # compute residuals
        residuals = y - X @ beta
        # compute delta_bar
        delta_bar = delta + residuals @ residuals
        # sample sigma
        sigma = rng.inverse_gamma(alpha_bar / 2, delta_bar / 2)
        inv_sigma = 1 / sigma
        return sigma, inv_sigma


    def __parameter_estimates(self):
        
        """
        point estimates and credibility intervals for model parameters
        uses quantiles of the empirical posterior distribution
        """
        
        # unpack
        mcmc_beta = self.mcmc_beta
        mcmc_sigma = self.mcmc_sigma
        credibility_level = self.credibility_level
        k = self.k
        # initiate storage: 4 columns: lower bound, median, upper bound, standard deviation
        estimates_beta = np.zeros((k,4))
        # fill estimates
        estimates_beta[:,0] = np.quantile(mcmc_beta, (1-credibility_level)/2, 1)
        estimates_beta[:,1] = np.quantile(mcmc_beta, 0.5, 1)
        estimates_beta[:,2] = np.quantile(mcmc_beta, (1+credibility_level)/2, 1)
        estimates_beta[:,3] = np.std(mcmc_beta, 1)
        # get point estimate for sigma
        estimates_sigma = np.quantile(mcmc_sigma, 0.5)
        # save as attributes
        self.estimates_beta = estimates_beta
        self.estimates_sigma = estimates_sigma


    def forecast_mcmc(self, X_hat):
        
        """posterior predictive distribution from algorithm 10.1""" 
        
        # unpack
        mcmc_beta = self.mcmc_beta
        mcmc_sigma = self.mcmc_sigma
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
            y_hat = X_hat @ beta + np.sqrt(sigma) * nrd.randn(m)
            mcmc_forecasts[:,i] = y_hat
            if verbose:
                cu.progress_bar(i, iterations, 'Predictions:')
        return mcmc_forecasts, m


    def forecast_estimates(self, mcmc_forecasts, credibility_level):
        
        """point estimates and credibility intervals for predictions""" 
        
        m = mcmc_forecasts.shape[0]
        # initiate estimate storage; 3 columns: lower bound, median, upper bound
        estimates_forecasts = np.zeros((m,3))
        # fill estimates
        estimates_forecasts[:,0] = np.quantile(mcmc_forecasts, (1-credibility_level)/2, 1)
        estimates_forecasts[:,1] = np.quantile(mcmc_forecasts, 0.5, 1)
        estimates_forecasts[:,2] = np.quantile(mcmc_forecasts, (1+credibility_level)/2, 1)
        return estimates_forecasts




