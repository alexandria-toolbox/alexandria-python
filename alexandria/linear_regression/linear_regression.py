# imports
import numpy as np
import numpy.linalg as nla



class LinearRegression(object):
    
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------
    
    
    def __init__(self):
        pass        
    
    
    def _make_regressors(self):
        
        """generates regressors X and y defined in (3.9.2)"""
        
        # unpack
        endogenous = self.endogenous
        exogenous = self.exogenous
        # define y
        y = endogenous
        # define X, adding constant and trends if included
        X = self._add_intercept_and_trends(exogenous, True)
        # get dimensions
        n_exogenous = exogenous.shape[1]
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
    
    
    def _add_intercept_and_trends(self, X, in_sample):
        
        """add constant, trend and quadratic trend to regressors if selected"""
        
        # unpack
        constant = self.constant
        trend = self.trend
        quadratic_trend = self.quadratic_trend
        n_rows = X.shape[0]
        # consider quadratic trend
        if quadratic_trend:
            # if in_sample, quadratic trend starts at 1
            if in_sample:
                quadratic_trend_column = np.arange(1,n_rows+1).reshape(-1,1) ** 2
            # if out of sample, quadratic trend starts at (n+1) ** 2
            else:
                n = self.n
                quadratic_trend_column = (n + np.arange(1,n_rows+1).reshape(-1,1)) ** 2
            X = np.hstack((quadratic_trend_column, X))
        # consider trend
        if trend:
            # if in_sample, trend starts at 1
            if in_sample:
                trend_column = np.arange(1,n_rows+1).reshape(-1,1)
            # if out of sample, trend starts at n+1
            else:
                n = self.n
                trend_column = n + np.arange(1,n_rows+1).reshape(-1,1)
            X = np.hstack((trend_column, X))
        # consider intercept
        if constant:
            constant_column = np.ones((n_rows,1))
            X = np.hstack((constant_column, X))
        return X    
    
    
    def _ols_regression(self, y, X, XX, Xy, n):
        
        """maximum likelihood estimates for beta and sigma, from (3.9.7)"""
        
        beta_hat = nla.solve(XX, Xy)
        res = y - X @ beta_hat
        sigma_hat = res @ res / n
        return beta_hat, sigma_hat   
    
    