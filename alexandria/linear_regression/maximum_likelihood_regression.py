# imports
import numpy as np
import alexandria.math.linear_algebra as la
import alexandria.math.stat_utilities as su
import alexandria.console.console_utilities as cu
from alexandria.linear_regression.linear_regression import LinearRegression
import alexandria.linear_regression.regression_utilities as ru


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
    
    beta_estimates : ndarray of shape (k,4)
        posterior estimates for beta
        column 1: point estimate; column 2: interval lower bound; 
        column 3: interval upper bound; column 4: standard deviation
       
    X_hat : ndarray of shape (m,k)
        predictors for the model 
       
    m : int
        number of predicted observations, defined in (3.10.1)
       
    forecast_estimates : ndarray of shape (m,3)
        estimates for predictions   
        column 1: interval lower bound; column 2: point estimate; 
        column 3: interval upper bound
    
    fitted_estimates : ndarray of shape (n,)
        posterior estimates (median) for in sample-fit
       
    residual_estimates : ndarray of shape (n,)
        posterior estimates (median) for residuals
        
    insample_evaluation : dict
        in-sample fit evaluation (SSR, R2, adj-R2)
        
    forecast_evaluation_criteria : dict
        out-of-sample forecast evaluation (RMSE, MAE, ...)
    
    
     Methods
     ----------
     estimate
     forecast
     insample_fit
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


    def insample_fit(self):
        
        """
        insample_fit()
        generates in-sample fit and residuals along with evaluation criteria
        
        parameters:
        none
        
        returns:
        none    
        """           
        

        # compute fitted and residuals
        self.__fitted_and_residual()
        # compute in-sample criteria
        self.__insample_criteria()


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
        constant = self.constant
        trend = self.trend
        quadratic_trend = self.quadratic_trend
        # add constant and trends if included
        X_hat = ru.add_intercept_and_trends(X_hat, constant, trend, quadratic_trend, n)        
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
        forecast_estimates = np.zeros((m,3))        
        # fill estimates
        forecast_estimates[:,0] = location
        forecast_estimates[:,1] = location - Z * s
        forecast_estimates[:,2] = location + Z * s
        # save as attributes
        self.X_hat = X_hat
        self.m = m
        self.forecast_estimates = forecast_estimates
        return forecast_estimates 

        
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
        forecast_estimates = self.forecast_estimates
        # obtain regular forecast evaluation criteria
        y_hat = forecast_estimates[:,0]
        forecast_evaluation_criteria = ru.forecast_evaluation_criteria(y_hat, y) 
        # save as attributes
        self.forecast_evaluation_criteria = forecast_evaluation_criteria        
        
    
    def __fit(self):
        
        """ estimates beta_hat and sigma_hat from (3.9.7) """

        # estimate beta_hat and sigma_hat
        beta, sigma = self._ols_regression()
        # if verbose, display progress bar
        if self.verbose:
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
        beta_estimates = np.zeros((k,4))
        # fill estimates
        beta_estimates[:,0] = beta
        beta_estimates[:,1] = beta - Z * s
        beta_estimates[:,2] = beta + Z * s
        beta_estimates[:,3] = np.sqrt(df / (df - 2)) * s
        # save as attributes
        self.beta_estimates = beta_estimates
    
    
    def __fitted_and_residual(self):
        
        """ in-sample fitted and residuals """
    
        fitted, residual = ru.fitted_and_residual(self.y, self.X, self.beta_estimates[:,0])
        self.fitted_estimates = fitted
        self.residual_estimates = residual
    
    
    def __insample_criteria(self):
        
        """ in-sample fit evaluation criteria """
    
        insample_evaluation = ru.ml_insample_evaluation_criteria(self.y, self.residual_estimates, self.n, self.k, self.sigma)
        self.insample_evaluation = insample_evaluation          
        
