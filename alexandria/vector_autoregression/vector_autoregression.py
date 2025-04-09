# imports
import numpy as np
import numpy.linalg as nla
import alexandria.vector_autoregression.var_utilities as vu


class VectorAutoRegression(object):
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------
    
    
    def __init__(self):
        pass
    
    
    def _make_regressors(self):
    
        """generates regressors Y, X as defined in (4.11.3), along with other dimension elements"""
              
        # define Y
        Y = self.__make_endogenous_matrix()
        # define X
        Z, X = self.__make_regressor_matrix()
        # define dimensions
        n, m, p, T, k, q = self.__generate_dimensions()
        # define estimation terms
        XX = X.T @ X
        XY = X.T @ Y
        YY = Y.T @ Y
        # save as attributes      
        self.Y = Y
        self.Z = Z
        self.X = X
        self.n = n
        self.m = m
        self.p = p
        self.T = T
        self.k = k
        self.q = q
        self._XX = XX
        self._XY = XY 
        self._YY = YY


    def __make_endogenous_matrix(self):
        
        # unpack, recover endogenous after trimming inital conditions
        endogenous = self.endogenous
        lags = self.lags
        Y = endogenous[lags:]
        return Y
        
    
    def __make_regressor_matrix(self):
        
        # unpack
        endogenous = self.endogenous
        exogenous = self.exogenous
        lags = self.lags
        constant = self.constant
        trend = self.trend
        quadratic_trend = self.quadratic_trend  
        periods = endogenous.shape[0] - lags
        # get automated exogenous: constant, trend, quadratic trend
        X_1 = vu.generate_intercept_and_trends(constant, trend, quadratic_trend, periods, 0)
        # recover other exogenous
        X_2 = vu.generate_exogenous_regressors(exogenous, lags, periods)
        # get lagged endogenous
        X_3 = vu.generate_lagged_endogenous(endogenous, lags)
        # concat to obtain final regressor matrix
        Z = np.hstack([X_1,X_2])
        X = np.hstack([X_1,X_2,X_3])
        return Z, X


    def __generate_dimensions(self):

        endogenous = self.endogenous
        exogenous = self.exogenous
        lags = self.lags
        constant = self.constant
        trend = self.trend
        quadratic_trend = self.quadratic_trend        
        T = endogenous.shape[0] - lags
        n = endogenous.shape[1]
        p = lags
        m = int(constant) + int(trend) + int(quadratic_trend)    
        if len(exogenous) != 0:
            m += exogenous.shape[1]
        k = m + n * p
        q = n * k
        return n, m, p, T, k, q

    
    def _ols_var(self, Y, X, XX, XY, T):
        
        """maximum likelihood estimates for B and Sigma, from (4.11.9)"""
        
        B_hat = nla.solve(XX, XY)
        E_hat = Y - X @ B_hat
        Sigma_hat = E_hat.T @ E_hat / T
        return B_hat, Sigma_hat      
        
        
        
        