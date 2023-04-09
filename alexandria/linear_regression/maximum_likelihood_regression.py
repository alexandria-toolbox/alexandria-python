# imports
import numpy as np
import alexandria.math.linear_algebra as la
import alexandria.math.stat_utilities as su
import alexandria.console.console_utilities as cu
from alexandria.linear_regression.linear_regression import LinearRegression



class MaximumLikelihoodRegression(LinearRegression):
    
    
    """
    Maximum likelihood linear regression, developed in section 9.1
    
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
    
    credibility_level : float
        credibility level (between 0 and 1)
    
    verbose : bool
        if True, displays a progress bar during MCMC algorithms
       
    y : ndarray of shape (n,)
        endogenous variable, defined in (3.9.3)
    
    X : ndarray of shape (n,k)
        exogenous variables, defined in (3.9.3)
       
    n : int
        number of observations, defined in (3.9.1)
    
    k : int
        dimension of beta, defined in (3.9.1)
    
    beta : ndarray of shape (k,)
        regression coefficients
       
    sigma : float
        residual variance, defined in (3.9.1)
    
    estimates_beta : ndarray of shape (k,4)
        posterior estimates for beta
        column 1: interval lower bound; column 2: point estimate; 
        column 3: interval upper bound; column 4: standard deviation
       
    X_hat : ndarray of shape (m,k)
        predictors for the model 
       
    m : int
        number of predicted observations, defined in (3.10.1)
       
    estimates_forecasts : ndarray of shape (m,3)
        estimates for predictions   
        column 1: interval lower bound; column 2: point estimate; 
        column 3: interval upper bound
    
    estimates_fit : ndarray of shape (n,)
        posterior estimates (median) for in sample-fit
       
    estimates_residuals : ndarray of shape (n,)
        posterior estimates (median) for residuals
        
    insample_evaluation : dict
        in-sample fit evaluation (SSR, R2, adj-R2)
        
    forecast_evaluation_criteria : dict
        out-of-sample forecast evaluation (RMSE, MAE, ...)
    
    
     Methods
     ----------
     estimate
     forecast
     fit_and_residuals
     forecast_evaluation
    """


    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------

    
    def __init__(self, endogenous, exogenous, constant = True, trend = False, 
                 quadratic_trend = False, credibility_level = 0.95, verbose = False):
        
        """
        constructor for the MaximumLikelihoodRegression class
        """
        
        self.endogenous = endogenous
        self.exogenous = exogenous
        self.constant = constant
        self.trend = trend
        self.quadratic_trend = quadratic_trend      
        self.credibility_level = credibility_level
        self.verbose = verbose
        # make regressors
        self._make_regressors()    
        
        
    def estimate(self):
        
        """
        estimate()
        estimates parameters beta and sigma of linear regression model
        
        parameters:
        none
        
        returns:
        none
        """
        
        # fit to obtain maximum likelihood estimates
        self.__fit()
        # obtain posterior estimates for regression parameters
        self.__parameter_estimates()         
        
        
    def forecast(self, X_hat, credibility_level):
        
        """
        forecast(X_hat, credibility_level)
        predictions for the linear regression model using (3.10.2)

        parameters:
        X_hat : ndarray of shape (m,k)
            array of predictors
        credibility_level : float
            credibility level for predictions (between 0 and 1)

        returns:
        estimates_forecasts : ndarray of shape (m, 3)
            posterior estimates for predictions
            column 1: interval lower bound; column 2: median;
            column 3: interval upper bound
        """
        
        # unpack
        XX = self._XX
        verbose = self.verbose
        beta = self.beta
        sigma = self.sigma
        n = self.n
        k = self.k
        # add constant and trends if included
        X_hat = self._add_intercept_and_trends(X_hat, False)
        # obtain prediction location, scale and degrees of freedom from (3.10.2)
        m = X_hat.shape[0]
        location = X_hat @ beta
        scale = sigma * np.identity(m) \
            + sigma * X_hat @ la.invert_spd_matrix(XX) @ X_hat.T
        s = np.sqrt(np.diag(scale))        
        # if verbose, display progress bar
        if verbose:
            cu.progress_bar_complete('Predictions:')
        # critical value of Student distribution for credibility level
        df = n - k
        Z = su.student_icdf((1 + credibility_level) / 2, df)
        # initiate estimate storage; 3 columns: lower bound, median, upper bound
        estimates_forecasts = np.zeros((m,3))        
        # fill estimates
        estimates_forecasts[:,0] = location - Z * s
        estimates_forecasts[:,1] = location
        estimates_forecasts[:,2] = location + Z * s
        # save as attributes
        self.X_hat = X_hat
        self.m = m
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
        X = self.X
        y = self.y
        beta = self.beta
        sigma = self.sigma
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
        # additionally estimate AIC and BIC from equation (3.10.9)
        aic = 2 * k / n + np.log(sigma)
        bic = k * np.log(n) / n + np.log(sigma)
        insample_evaluation = {}
        insample_evaluation['ssr'] = ssr
        insample_evaluation['r2'] = r2
        insample_evaluation['adj_r2'] = adj_r2
        insample_evaluation['aic'] = aic
        insample_evaluation['bic'] = bic
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
        estimates_forecasts = self.estimates_forecasts
        m = self.m
        # calculate forecast error
        y_hat = estimates_forecasts[:,1]
        err = y - y_hat
        # compute forecast evaluation from equation (3.10.11)
        rmse = np.sqrt(err @ err / m)
        mae = np.sum(np.abs(err)) / m
        mape = 100 * np.sum(np.abs(err / y)) / m
        theil_u = np.sqrt(err @ err) / (np.sqrt(y @ y) + np.sqrt(y_hat @ y_hat))
        bias = np.sum(err) / np.sum(np.abs(err))
        forecast_evaluation_criteria = {}
        forecast_evaluation_criteria['rmse'] = rmse
        forecast_evaluation_criteria['mae'] = mae
        forecast_evaluation_criteria['mape'] = mape
        forecast_evaluation_criteria['theil_u'] = theil_u
        forecast_evaluation_criteria['bias'] = bias
        # save as attributes
        self.forecast_evaluation_criteria = forecast_evaluation_criteria
        
        
    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------       
        
    
    def __fit(self):
        
        """estimates beta_hat and sigma_hat from (3.9.7)"""

        # unpack        
        y = self.y
        X = self.X
        XX = self._XX
        Xy = self._Xy
        n = self.n
        verbose = self.verbose
        # estimate beta_hat and sigma_hat
        beta, sigma = self._ols_regression(y, X, XX, Xy, n)
        # if verbose, display progress bar
        if verbose:
            cu.progress_bar_complete('Model parameters:')
        self.beta = beta
        self.sigma = sigma
        
        
    def __parameter_estimates(self):
        
        """estimates and intervals from Student distribution in (3.9.8)"""
        
        # unpack
        beta = self.beta
        sigma = self.sigma
        credibility_level = self.credibility_level
        n = self.n
        k = self.k
        XX = self._XX
        # obtain scale for Student distribution of beta from equation (3.9.8)
        S = la.invert_spd_matrix(XX)
        s = np.sqrt(sigma * np.diag(S))
        # critical value of Student distribution for credibility level
        df = n - k
        Z = su.student_icdf((1 + credibility_level) / 2, df)
        # initiate storage: 4 columns: lower bound, median, upper bound, standard deviation
        estimates_beta = np.zeros((k,4))
        # fill estimates
        estimates_beta[:,0] = beta - Z * s
        estimates_beta[:,1] = beta
        estimates_beta[:,2] = beta + Z * s
        estimates_beta[:,3] = np.sqrt(df / (df - 2)) * s
        # save as attributes
        self.estimates_beta = estimates_beta 
    
    
          
        
