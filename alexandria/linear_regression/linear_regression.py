# imports
import alexandria.linear_regression.regression_utilities as ru


class LinearRegression(object):
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------
    
    
    def __init__(self):
        pass
    

    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------   


    def _make_regressors(self):
        
        """ generates regressors X and y defined in (3.9.2) """
        
        # define y
        y = self.endogenous
        # define X, adding constant and trends if included
        X = ru.add_intercept_and_trends(self.exogenous, self.constant, self.trend, self.quadratic_trend, 0)
        # get dimensions
        n_exogenous = self.exogenous.shape[1]
        n = X.shape[0]
        k = X.shape[1]
        # define terms for posterior distribution
        XX = X.T @ X
        Xy = X.T @ y
        yy = y @ y
        # save as attributes      
        self.y = y
        self.X = X
        self._n_exogenous = n_exogenous
        self.n = n
        self.k = k
        self._XX = XX
        self._Xy = Xy 
        self._yy = yy


    def _ols_regression(self):
        
        """ maximum likelihood estimates for beta and sigma, from (3.9.7) """
        
        beta_hat, sigma_hat  = ru.ols_regression(self.y, self.X, self._XX, self._Xy, self.n)
        return beta_hat, sigma_hat 

