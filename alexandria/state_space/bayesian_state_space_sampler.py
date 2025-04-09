# imports
import numpy as np
import alexandria.state_space.state_space_utilities as ss


class BayesianStateSpaceSampler(object):
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------
    
    
    def __init__(self, X, A, Omega, C, B, Upsilon, z_00 = [], Upsilon_00 = [], kalman_type = 'standard'):

        """
        constructor for the BayesianStateSpaceSampler class
        """
        
        self.X = X
        self.A = A
        self.Omega = Omega
        self.C = C
        self.B = B
        self.Upsilon = Upsilon
        self.z_00 = z_00
        self.Upsilon_00 = Upsilon_00
        self.kalman_type = kalman_type
        self.T = X.shape[0]
        self.n = X.shape[1]
        self.k = A.shape[1]
    
    
    def carter_kohn_algorithm(self):
        
        """
        carter_kohn_algorithm()
        Bayesian sampler for a given state-space system, using algorithm k.2
        
        parameters:
        none
        
        returns:
        none    
        """   
        
        self.__forward_pass()
        self.__backward_pass()
        
    
    #---------------------------------------------------
    # Methods (Access = private)
    #--------------------------------------------------- 


    def __forward_pass(self):
        
        """ forward pass of the algorithm, based on Kalman filter """
        
        if self.kalman_type == 'standard':
            Z_tt, Z_tt1, Ups_tt, Ups_tt1 = ss.kalman_filter(self.X, self.A, self.Omega,\
                                           self.C, self.B, self.Upsilon, self.T, self.n, self.k)
        elif self.kalman_type == 'conditional_forecast':
            Z_tt, Z_tt1, Ups_tt, Ups_tt1 = ss.conditional_forecast_kalman_filter(self.X, self.A, \
                                           self.Omega, self.C, self.B, self.Upsilon, self.z_00, \
                                           self.Upsilon_00, self.T, self.n, self.k)            
        self.Z_tt = Z_tt
        self.Z_tt1 = Z_tt1
        self.Ups_tt = Ups_tt
        self.Ups_tt1 = Ups_tt1


    def __backward_pass(self):
        
        """ backward pass of the algorithm, based on Kalman filter """
        
        if self.kalman_type == 'standard':
            Z = ss.backward_pass(self.Z_tt, self.Z_tt1, self.Ups_tt, self.Ups_tt1, self.B, self.T, self.k)
        elif self.kalman_type == 'conditional_forecast':
            Z = ss.static_backward_pass(self.Z_tt, self.Z_tt1, self.Ups_tt, self.Ups_tt1, self.B, self.T, self.k)
        self.Z = Z




