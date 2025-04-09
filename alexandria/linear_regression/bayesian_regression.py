# imports
import numpy as np
import alexandria.linear_regression.regression_utilities as ru


class BayesianRegression(object):
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------
    
    
    def __init__(self):
        pass
    
        
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
    

    #---------------------------------------------------
    # Methods (Access = private)
    #--------------------------------------------------- 


    def _prior(self):
        
        # generate b
        b = ru.generate_b(self.b_exogenous, self.b_constant, self.b_trend, self.b_quadratic_trend, \
                          self.constant, self.trend, self.quadratic_trend, self._n_exogenous)
        # generate V
        V = ru.generate_V(self.V_exogenous, self.V_constant, self.V_trend, self.V_quadratic_trend, \
                          self.constant, self.trend, self.quadratic_trend, self._n_exogenous)
        V, inv_V, inv_V_b = ru.generate_b_and_V_arrays(b, V)
        # save as attributes
        self.b = b
        self.V = V
        self._inv_V = inv_V
        self._inv_V_b = inv_V_b    

    
    def __fitted_and_residual(self):
        
        """ in-sample fitted and residuals """
    
        fitted, residual = ru.fitted_and_residual(self.y, self.X, self.beta_estimates[:,0])
        self.fitted_estimates = fitted
        self.residual_estimates = residual
    
    
    def __insample_criteria(self):
        
        """ in-sample fit evaluation criteria """
    
        insample_evaluation = ru.insample_evaluation_criteria(self.y, self.residual_estimates, self.n, self.k)
        self.insample_evaluation = insample_evaluation
    

    
    
    
    
    
    
