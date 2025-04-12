# imports
import numpy as np
import alexandria.processor.input_utilities as iu
import alexandria.math.linear_algebra as la
import alexandria.math.random_number_generators as rng
import alexandria.vector_autoregression.var_utilities as vu
import alexandria.console.console_utilities as cu
from alexandria.vector_autoregression.maximum_likelihood_var import MaximumLikelihoodVar
from alexandria.state_space.bayesian_state_space_sampler import BayesianStateSpaceSampler


class BayesianVar(object):
    
    
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
        
        # compute steady-state
        self.__steady_state()
        # compute fitted and residuals
        self.__fitted_and_residual()
        # compute in-sample criteria
        self.__insample_criteria()


    def forecast(self, h, credibility_level, Z_p=[]):
        
        """
        forecast(h, credibility_level, Z_p=[])
        estimates forecasts for the Bayesian VAR model, using algorithm 13.4
        
        parameters:
        h : int
            number of forecast periods
        credibility_level: float between 0 and 1
            credibility level for forecast credibility bands
        Z_p : empty list or numpy array of dimension (h, n_exo)
            empty list unless the model includes exogenous other than constant, trend and quadratic trend
            if not empty, n_exo is the number of additional exogenous variables
        
        returns:
        forecast_estimates : ndarray of shape (h,n,3)
            page 1: median; page 2: interval lower bound; page 3: interval upper bound
        """ 
        
        # get forecast
        self.__make_forecast(h, Z_p)
        # obtain posterior estimates
        self.__forecast_posterior_estimates(credibility_level)
        forecast_estimates = self.forecast_estimates
        return forecast_estimates


    def forecast_evaluation(self, Y):
        
        """
        forecast_evaluation(Y)
        forecast evaluation criteria for the Bayesian VAR model, as defined in (4.13.18)-(4.13.22)
        
        parameters:
        Y : ndarray of shape (h,n)
            array of realised values for forecast evaluation, h being the number of forecast periods
            
        returns:
        forecast_evaluation_criteria : dictionary
            dictionary with criteria name as keys and corresponding number as value
        """
        
        # unpack
        Y_hat, mcmc_forecast = self.forecast_estimates[:,:,0], self.mcmc_forecast
        # obtain regular forecast evaluation criteria from equations (4.13.18) and (4.13.19)
        standard_evaluation_criteria = vu.forecast_evaluation_criteria(Y_hat, Y)
        # obtain Bayesian forecast evaluation criteria from equations (4.13.21) and (4.13.22)
        bayesian_evaluation_criteria = vu.bayesian_forecast_evaluation_criteria(mcmc_forecast, Y)
        # merge dictionaries
        forecast_evaluation_criteria = iu.concatenate_dictionaries(standard_evaluation_criteria, bayesian_evaluation_criteria)
        # save as attributes
        self.forecast_evaluation_criteria = forecast_evaluation_criteria
        return forecast_evaluation_criteria


    def impulse_response_function(self, h, credibility_level):
        
        """
        impulse_response_function(h, credibility_level)
        impulse response functions, as defined in (4.13.1)-(4.13.9)
        
        parameters:
        h : int
            number of IRF periods
        credibility_level: float between 0 and 1
            credibility level for forecast credibility bands
            
        returns:
        irf_estimates : ndarray of shape (n,n,h,3)
            first 3 dimensions are variable, shock, period; 4th dimension is median, lower bound, upper bound
        exo_irf_estimates : ndarray of shape (n,n,h,3)
            first 3 dimensions are variable, shock, period; 4th dimension is median, lower bound, upper bound
        """
        
        # get regular impulse response funtion
        self.__make_impulse_response_function(h)
        # get exogenous impuse response function
        self.__make_exogenous_impulse_response_function(h)
        # get structural impulse response function
        self.__make_structural_impulse_response_function(h)
        # obtain posterior estimates
        self.__irf_posterior_estimates(credibility_level)
        irf_estimates, exo_irf_estimates = self.irf_estimates, self.exo_irf_estimates
        return irf_estimates, exo_irf_estimates
    
    
    def forecast_error_variance_decomposition(self, h, credibility_level):
        
        """
        forecast_error_variance_decomposition(self, h, credibility_level)
        forecast error variance decomposition, as defined in (4.13.31)
        
        parameters:
        h : int
            number of FEVD periods
        credibility_level: float between 0 and 1
            credibility level for forecast credibility bands
            
        returns:
        fevd_estimates : ndarray of shape (n,n,h,3)
            first 3 dimensions are variable, shock, period; 4th dimension is median, lower bound, upper bound
        """
        
        # get forecast error variance decomposition
        self.__make_forecast_error_variance_decomposition(h)
        # obtain posterior estimates
        self.__fevd_posterior_estimates(credibility_level)
        fevd_estimates = self.fevd_estimates
        return fevd_estimates
    
    
    def historical_decomposition(self, credibility_level):
        
        """
        historical_decomposition(self, credibility_level)
        historical decomposition, as defined in (4.13.34)-(4.13-36)
        
        parameters:
        credibility_level: float between 0 and 1
            credibility level for forecast credibility bands
            
        returns:
        hd_estimates : ndarray of shape (n,n,T,3)
            first 3 dimensions are variable, shock, period; 4th dimension is median, lower bound, upper bound
        """
        
        # get historical decomposition
        self.__make_historical_decomposition()
        # obtain posterior estimates
        self.__hd_posterior_estimates(credibility_level)
        hd_estimates = self.hd_estimates
        return hd_estimates    
    
    
    def conditional_forecast(self, h, credibility_level, conditions, shocks, conditional_forecast_type, Z_p=[]):

        """
        conditional_forecast(self, h, credibility_level, conditions, shocks, conditional_forecast_type, Z_p=[])
        estimates conditional forecasts for the Bayesian VAR model, using algorithms 14.1 and 14.2
        
        parameters:
        h : int
            number of forecast periods
        credibility_level : float between 0 and 1
            credibility level for forecast credibility bands
        conditions : ndarray of shape (n_conditions,4)
            table defining conditions (column 1: variable, column 2: period, column 3: mean, column 4: variance) 
        shocks: empty list or ndarray of shape (n,)
            vector defining shocks generating the conditions; should be empty if conditional_forecast_type = 1          
        conditional_forecast_type : int
            conditional forecast type (1 = agnostic, 2 = structural)
        Z_p : empty list or ndarray of dimension (h, n_exo)
            empty list unless the model includes exogenous other than constant, trend and quadratic trend
            if not empty, n_exo is the number of additional exogenous variables
        
        returns:
        conditional_forecast_estimates : ndarray of shape (h,n,3)
            page 1: median; page 2: interval lower bound; page 3: interval upper bound
        """         
        
        # if conditional forecast type is agnostic
        if conditional_forecast_type == 1:
            # get conditional forecasts
            self.__make_conditional_forecast(h, conditions, Z_p)
        # if instead conditional forecast type is structural
        elif conditional_forecast_type == 2:
            # establish type of shocks
            shock_type = self.__check_shock_type(h, conditions, shocks)
            # get structural conditional forecasts
            self.__make_structural_conditional_forecast(h, conditions, shocks, Z_p, shock_type)
        # obtain posterior estimates
        self.__conditional_forecast_posterior_estimates(credibility_level)
        conditional_forecast_estimates = self.conditional_forecast_estimates
        return conditional_forecast_estimates       
    

    #---------------------------------------------------
    # Methods (Access = private)
    #--------------------------------------------------- 


    def _make_delta(self):    
        
        if iu.is_numeric(self.ar_coefficients):
            ar_coefficients = np.array(self.n * [self.ar_coefficients])
        else:
            ar_coefficients = self.ar_coefficients
        self.delta = ar_coefficients
        
    
    def _individual_ar_variances(self):
        
        s = np.zeros(self.n)
        for i in range(self.n):
            ar = MaximumLikelihoodVar(self.endogenous[:,[i]], lags=self.lags)
            ar.estimate()
            s[i] = ar.Sigma[0,0]
        self.s = s


    def _dummy_extensions(self):
        
       # sums-of-coefficients
       Y_sum, X_sum = vu.sums_of_coefficients_extension(self.sums_of_coefficients,\
                                                        self.pi5, self.Y, self.n, self.m, self.p)
       # dummy initial observation
       Y_obs, X_obs = vu.dummy_initial_observation_extension(self.dummy_initial_observation,\
                                                             self.pi6, self.Y, self.X, self.n, self.m, self.p)
       # long run prior
       Y_lrp, X_lrp = vu.long_run_prior_extension(self.long_run_prior, self.pi7, self.J,\
                                                  self.Y, self.n, self.m, self.p)
       # concatenate to Y, X, update T as in 4.11.62
       Y_d = np.vstack([Y_sum,Y_obs,Y_lrp,self.Y])
       X_d = np.vstack([X_sum,X_obs,X_lrp,self.X])
       T_d = np.vstack([Y_sum,Y_obs,Y_lrp]).shape[0] + self.T
       self.Y_sum = Y_sum
       self.X_sum = X_sum
       self.Y_obs = Y_obs
       self.X_obs = X_obs      
       self.Y_lrp = Y_lrp
       self.X_lrp = X_lrp
       self._XX_d = X_d.T @ X_d
       self._XY_d = X_d.T @ Y_d
       self._YY_d = Y_d.T @ Y_d
       self.Y_d = Y_d
       self.X_d = X_d
       self.T_d = T_d
       

    def _make_constrained_coefficients(self):
        
        # apply only if constrained coefficients is activated
        if self.constrained_coefficients:
            # unpack
            n, m, k, lags = self.n, self.m, self.k, self.lags
            constant, trend, quadratic_trend = self.constant, self.trend, self.quadratic_trend
            constrained_coefficients_table = self.constrained_coefficients_table
            secondary_constrained_coefficients_table = vu.rework_constraint_table(constrained_coefficients_table, lags)
            self._secondary_constrained_coefficients_table = secondary_constrained_coefficients_table
            # reshape b and V as k * n arrays
            B = np.reshape(self.b.copy(),newshape=(k,n),order='F')
            V = np.reshape(self.V.copy(),newshape=(k,n),order='F')
            # get updated parameters with constrained coefficients applied
            new_b, new_V = vu.make_constrained_coefficients(B, V, n, m, k, lags, constant,\
                           trend, quadratic_trend, secondary_constrained_coefficients_table)
            self.b = new_b
            self.V = new_V
           

    def _make_structural_identification(self):
        
        # SVAR by Choleski factorization, if selected
        if self.structural_identification == 2:
            self.__svar_by_choleski_factorization()
        # SVAR by triangular factorization, if selected
        elif self.structural_identification == 3:
            self.__svar_by_triangular_factorization()
        # SVAR by restrictions, if selected
        elif self.structural_identification == 4:
            self.__svar_by_restrictions()
        # get posterior estimates
        if self.structural_identification != 1:
            self.__svar_estimates()
               

    def __svar_by_choleski_factorization(self):
        
        # compute H, Gamma, and inverse of H
        self.mcmc_H = self._mcmc_chol_Sigma.copy()
        self.mcmc_Gamma = np.ones((self.iterations,self.n))
        # check if Sigma is attribute, which means prior is Minnesota
        if hasattr(self, 'Sigma'):
            self._mcmc_inv_H = np.dstack([la.invert_lower_triangular_matrix(self.mcmc_H[:,:,0])] * self.iterations)
            if self.verbose:
                cu.progress_bar_complete('Structural identification:')
        # for any other prior, use mcmc draws
        else:
            mcmc_inv_H = np.zeros((self.n,self.n,self.iterations))
            for i in range(self.iterations):
                mcmc_inv_H[:,:,i] = la.invert_lower_triangular_matrix(self.mcmc_H[:,:,i])
                if self.verbose:
                    cu.progress_bar(i, self.iterations, 'Structural identification:')
            self._mcmc_inv_H =  mcmc_inv_H
        # create index correspondance for later applications
        self._svar_index = np.arange(self.iterations)

        
    def __svar_by_triangular_factorization(self):
        
        # check if Sigma is attribute, which means prior is Minnesota
        if hasattr(self, 'Sigma'):
            H, Gamma = la.triangular_factorization(self.Sigma)
            mcmc_H = np.dstack([H] * self.iterations)
            mcmc_inv_H = np.dstack([la.invert_lower_triangular_matrix(H)] * self.iterations)
            mcmc_Gamma = np.tile(Gamma.reshape(1,-1),(self.iterations,1))
            if self.verbose:
                cu.progress_bar_complete('Structural identification:')
        # for any other prior, use mcmc draws
        else:
            mcmc_H = np.zeros((self.n,self.n,self.iterations))
            mcmc_inv_H = np.zeros((self.n,self.n,self.iterations))
            mcmc_Gamma = np.zeros((self.iterations,self.n))
            for i in range(self.iterations):
                H, Gamma = la.triangular_factorization(self._mcmc_chol_Sigma[:,:,i], is_cholesky = True)
                mcmc_H[:,:,i] = H
                mcmc_inv_H[:,:,i] = la.invert_lower_triangular_matrix(H)
                mcmc_Gamma[i,:] = Gamma
                if self.verbose:
                    cu.progress_bar(i, self.iterations, 'Structural identification:')
        self.mcmc_H = mcmc_H
        self._mcmc_inv_H = mcmc_inv_H
        self.mcmc_Gamma = mcmc_Gamma
        # create index correspondance for later applications
        self._svar_index = np.arange(self.iterations)
        
        
    def __svar_by_restrictions(self):
        
        # initiate MCMC elements
        svar_index = np.zeros(self.iterations).astype(int)
        mcmc_H = np.zeros((self.n,self.n,self.iterations))
        mcmc_inv_H = np.zeros((self.n,self.n,self.iterations))
        # create matrices of restriction checks
        restriction_matrices, max_irf_period = vu.make_restriction_matrices(self.restriction_table, self.p)
        # make preliminary orthogonalised impulse response functions, if relevant
        mcmc_irf = vu.make_restriction_irf(self.mcmc_beta, self._mcmc_chol_Sigma, \
                                            self.iterations, self.n, self.p, max_irf_period)
        # make preliminary structural shocks, if relevant
        mcmc_shocks = vu.make_restriction_shocks(self.mcmc_beta, self._mcmc_chol_Sigma, self.Y, self.X, \
                                             self.T, self.n, self.iterations, restriction_matrices)
        # loop over iterations, until desired number of total iterations is obtained
        i = 0
        while i < self.iterations:
            # select a random index in number of iterations
            j = rng.discrete_uniform(0, self.iterations-1)
            # make a random rotation matrix Q: if no zero restrictions, draw from uniform orthogonal distribution
            if len(restriction_matrices[0]) == 0:
                Q = rng.uniform_orthogonal(self.n)
            # if there are zero restrictions, use the zero uniform orthogonal distribution
            else:
                Q = rng.zero_uniform_orthogonal(self.n, restriction_matrices[0], mcmc_irf[:,:,:,j])
            # check restrictions: IRF, sign
            irf_sign_index = restriction_matrices[1][0]
            if len(irf_sign_index) != 0:
                irf_sign_coefficients = restriction_matrices[1][1]
                restriction_satisfied = vu.check_irf_sign(irf_sign_index, irf_sign_coefficients, mcmc_irf[:,:,:,j], Q)
                if not restriction_satisfied:
                    continue
            # check restrictions: IRF, magnitude
            irf_magnitude_index = restriction_matrices[2][0]
            if len(irf_magnitude_index) != 0:
                irf_magnitude_coefficients = restriction_matrices[2][1]
                restriction_satisfied = vu.check_irf_magnitude(irf_magnitude_index, irf_magnitude_coefficients, mcmc_irf[:,:,:,j], Q)
                if not restriction_satisfied:
                    continue                      
            # check restrictions: structural shocks, sign
            shock_sign_index = restriction_matrices[3][0]
            if len(shock_sign_index) != 0:
                shock_sign_coefficients = restriction_matrices[3][1]
                restriction_satisfied = vu.check_shock_sign(shock_sign_index, shock_sign_coefficients, mcmc_shocks[:,:,j], Q)
                if not restriction_satisfied:
                    continue
            # check restrictions: structural shocks, magnitude
            shock_magnitude_index = restriction_matrices[4][0]
            if len(shock_magnitude_index) != 0:
                shock_magnitude_coefficients = restriction_matrices[4][1]
                restriction_satisfied = vu.check_shock_magnitude(shock_magnitude_index, shock_magnitude_coefficients, mcmc_shocks[:,:,j], Q)
                if not restriction_satisfied:
                    continue         
            # historical decomposition values if any of sign or magnitude restrictions apply
            history_sign_index = restriction_matrices[5][0]
            history_magnitude_index = restriction_matrices[6][0]
            if len(history_sign_index) != 0 or len(history_magnitude_index) != 0:
                irf, shocks = vu.make_restriction_irf_and_shocks(mcmc_irf[:,:,:,j], mcmc_shocks[:,:,j], Q, self.n)
            # check restrictions: historical decomposition, sign
            if len(history_sign_index) != 0:
                history_sign_coefficients = restriction_matrices[5][1]
                restriction_satisfied = vu.check_history_sign(history_sign_index, history_sign_coefficients, irf, shocks)
                if not restriction_satisfied:    
                    continue
            # check restrictions: historical decomposition, magnitude
            if len(history_magnitude_index) != 0:
                history_magnitude_coefficients = restriction_matrices[6][1]
                restriction_satisfied = vu.check_history_magnitude(history_magnitude_index, history_magnitude_coefficients, irf, shocks)
                if not restriction_satisfied:
                    continue  
            # if all restriction passed, keep the draw and record
            H = self._mcmc_chol_Sigma[:,:,j] @ Q
            inv_H = Q.T @ la.invert_lower_triangular_matrix(self._mcmc_chol_Sigma[:,:,j])
            mcmc_H[:,:,i] = H
            mcmc_inv_H[:,:,i] = inv_H
            svar_index[i] = j
            if self.verbose:
                cu.progress_bar(i, self.iterations, 'Structural identification:')  
            i += 1 
        self.mcmc_H = mcmc_H
        self._mcmc_inv_H = mcmc_inv_H
        self.mcmc_Gamma = np.ones((self.iterations,self.n))
        self._svar_index = svar_index


    def __svar_estimates(self):

        H_estimates = np.quantile(self.mcmc_H,0.5,axis=2)
        Gamma_estimates = np.quantile(self.mcmc_Gamma,0.5,axis=0)
        self.H_estimates = H_estimates
        self.Gamma_estimates = Gamma_estimates


    def __steady_state(self):
        
        ss = np.zeros((self.T,self.n,self.iterations))
        for i in range(self.iterations):
            ss[:,:,i] = vu.steady_state(self.Z, self.mcmc_beta[:,:,i], self.n, self.m, self.p, self.T)
            if self.verbose:
                cu.progress_bar(i, self.iterations, 'Steady-state:')            
        ss_estimates = vu.posterior_estimates(ss, self.credibility_level)
        self.steady_state_estimates = ss_estimates
        
        
    def __fitted_and_residual(self):
        
        fitted = np.zeros((self.T,self.n,self.iterations))
        residual = np.zeros((self.T,self.n,self.iterations))
        for i in range(self.iterations):
            residual[:,:,i], fitted[:,:,i] = vu.fit_and_residuals(self.Y, self.X, self.mcmc_beta[:,:,i])
            if self.verbose:
                cu.progress_bar(i, self.iterations, 'Fitted and residual:')
        fitted_estimates = vu.posterior_estimates(fitted, self.credibility_level)
        residual_estimates = vu.posterior_estimates(residual, self.credibility_level)                
        self.fitted_estimates = fitted_estimates
        self.residual_estimates = residual_estimates                
        if self.structural_identification != 1:
            structural_shocks = np.zeros((self.T,self.n,self.iterations))
            for i in range(self.iterations):
                index = self._svar_index[i]
                structural_shocks[:,:,i] = vu.structural_shocks(residual[:,:,index], self._mcmc_inv_H[:,:,i])
                if self.verbose:
                    cu.progress_bar(i, self.iterations, 'Structural shocks:')
            structural_shock_estimates = vu.posterior_estimates(structural_shocks, self.credibility_level)
            self.mcmc_structural_shocks = structural_shocks
            self.structural_shock_estimates = structural_shock_estimates
        

    def __insample_criteria(self):
        
        insample_evaluation = vu.insample_evaluation_criteria(self.Y, \
                              self.residual_estimates[:,:,0], self.T, self.k)
        if self.verbose:
            cu.progress_bar_complete('In-sample evaluation criteria:')
        self.insample_evaluation = insample_evaluation
       
        
    def __make_forecast(self, h, Z_p):     
        
        # make regressors
        Z_p, Y = vu.make_forecast_regressors(Z_p, self.Y, h, self.p, self.T,
                 self.exogenous, self.constant, self.trend, self.quadratic_trend)
        # initiate storage and loop over iterations
        mcmc_forecast = np.zeros((h,self.n,self.iterations))
        for i in range(self.iterations):
            # make MCMC simulation for beta and Sigma
            mcmc_forecast[:,:,i] = vu.forecast(self.mcmc_beta[:,:,i], \
                         self._mcmc_chol_Sigma[:,:,i], h, Z_p, Y, self.n)
            if self.verbose:
                cu.progress_bar(i, self.iterations, 'Forecasts:')
        self.mcmc_forecast = mcmc_forecast
        
       
    def __forecast_posterior_estimates(self, credibility_level):
        
        # obtain posterior estimates
        mcmc_forecast = self.mcmc_forecast
        forecast_estimates = vu.posterior_estimates(mcmc_forecast, credibility_level)
        self.forecast_estimates = forecast_estimates     
        

    def __make_impulse_response_function(self, h):
        
        mcmc_irf = np.zeros((self.n, self.n, h, self.iterations))
        for i in range(self.iterations):
            # get regular impulse response function
            mcmc_irf[:,:,:,i] = vu.impulse_response_function(self.mcmc_beta[:,:,i], self.n, self.p, h)
            if self.verbose:    
                cu.progress_bar(i, self.iterations, 'Impulse response function:')
        self.mcmc_irf = mcmc_irf


    def __make_exogenous_impulse_response_function(self, h):
        
        # create exogenous IRFs only if there are exogenous (other than constant, trend, quadratic trend)
        if len(self.exogenous) != 0:
            # identify the number of exogenous (other than constant, trend, quadratic trend)
            r = self.exogenous.shape[1]
            mcmc_irf_exo = np.zeros((self.n, r, h, self.iterations))
            for i in range(self.iterations):
                # get exogenous IRFs
                mcmc_irf_exo[:,:,:,i] = vu.exogenous_impulse_response_function(self.mcmc_beta[:,:,i], self.n, self.m, r, self.p, h)
                if self.verbose:    
                    cu.progress_bar(i, self.iterations, 'Exogenous impulse response function:')
        else:
            mcmc_irf_exo = []
        self.mcmc_irf_exo = mcmc_irf_exo        
        
        
    def __make_structural_impulse_response_function(self, h):
        
        if self.structural_identification == 1:
            if self.verbose:    
                cu.progress_bar_complete('Structural impulse response function:')  
            self.mcmc_structural_irf = []
        else:
            mcmc_structural_irf = self.mcmc_irf.copy()
            for i in range(self.iterations):
                # get index in case svar comes from restrictions
                index = self._svar_index[i]
                # recover structural impulse response function
                mcmc_structural_irf[:,:,:,i] = vu.structural_impulse_response_function(self.mcmc_irf[:,:,:,index],\
                                                                                       self.mcmc_H[:,:,i], self.n)
                if self.verbose:
                    cu.progress_bar(i, self.iterations, 'Structural impulse response function:')                
            self.mcmc_structural_irf = mcmc_structural_irf


    def __irf_posterior_estimates(self, credibility_level):

        if self.structural_identification == 1:
            mcmc_irf = self.mcmc_irf
        else:
            mcmc_irf = self.mcmc_structural_irf
        irf_estimates = vu.posterior_estimates_3d(mcmc_irf, credibility_level)
        if len(self.exogenous) != 0:
            exo_irf_estimates = vu.posterior_estimates_3d(self.mcmc_irf_exo, credibility_level)
        else:
            exo_irf_estimates = []
        self.irf_estimates = irf_estimates
        self.exo_irf_estimates = exo_irf_estimates
        
        
    def __make_forecast_error_variance_decomposition(self, h):
        
        # if no structural identification, FEVD is not computed
        if self.structural_identification == 1:
            if self.verbose:
                cu.progress_bar_complete('Forecast error variance decomposition:')  
            self.mcmc_fevd = []
        # if there is some structural identification, proceed
        else:
            # recover H from MCMC record, and Gamma if not identity
            if self.structural_identification == 3:
                mcmc_Gamma = self.mcmc_Gamma.copy()
            else:
                mcmc_Gamma = [[]] * self.iterations
            # initiate MCMC storage for FEVD and loop over iterations
            mcmc_fevd = np.zeros((self.n, self.n, h, self.iterations))
            has_irf = hasattr(self, 'mcmc_structural_irf') and self.mcmc_structural_irf.shape[2] >= h
            for i in range(self.iterations):                
                # recover structural IRF or estimate them
                if has_irf:
                    structural_irf = self.mcmc_structural_irf[:,:,:h,i]
                else:
                    index = self._svar_index[i]
                    irf = vu.impulse_response_function(self.mcmc_beta[:,:,index], self.n, self.p, h)
                    structural_irf = vu.structural_impulse_response_function(irf, self.mcmc_H[:,:,i], self.n) 
                # recover fevd
                mcmc_fevd[:,:,:,i] = vu.forecast_error_variance_decomposition(structural_irf, mcmc_Gamma[i], self.n, h)            
                if self.verbose:
                    cu.progress_bar(i, self.iterations, 'Forecast error variance decomposition:')         
            self.mcmc_fevd = mcmc_fevd

        
    def __fevd_posterior_estimates(self, credibility_level):

        if self.structural_identification == 1:
            self.fevd_estimates = []
        else:
            mcmc_fevd = self.mcmc_fevd
            fevd_estimates = vu.posterior_estimates_3d(mcmc_fevd, credibility_level)
            normalized_fevd_estimates = vu.normalize_fevd_estimates(fevd_estimates)
            self.fevd_estimates = normalized_fevd_estimates
     
        
    def __make_historical_decomposition(self):
        
        # if no structural identification, historical decomposition is not computed
        if self.structural_identification == 1:
            if self.verbose:
                cu.progress_bar_complete('Historical decomposition:')
            self.mcmc_hd = []        
        # if there is some structural identification, proceed
        else:
            # initiate MCMC storage for HD and loop over iterations
            mcmc_hd = np.zeros((self.n, self.n, self.T, self.iterations))   
            has_irf = hasattr(self, 'mcmc_structural_irf') and self.mcmc_structural_irf.shape[2] >= self.T
            has_structural_shocks = hasattr(self, 'mcmc_structural_shocks')
            for i in range(self.iterations):
                index = self._svar_index[i]
                # recover structural IRF or estimate them
                if has_irf:
                    structural_irf = self.mcmc_structural_irf[:,:,:self.T,i]
                else:
                    irf = vu.impulse_response_function(self.mcmc_beta[:,:,index], self.n, self.p, self.T)
                    structural_irf = vu.structural_impulse_response_function(irf, self.mcmc_H[:,:,i], self.n)    
                # recover structural shocks or estimate them
                if has_structural_shocks:
                    structural_shocks = self.mcmc_structural_shocks[:,:,i]  
                else:
                    E, _ = vu.fit_and_residuals(self.Y, self.X, self.mcmc_beta[:,:,index])
                    structural_shocks = vu.structural_shocks(E, self._mcmc_inv_H[:,:,i])
                # get historical decomposition
                mcmc_hd[:,:,:,i] = vu.historical_decomposition(structural_irf, structural_shocks, self.n, self.T)             
                if self.verbose:    
                    cu.progress_bar(i, self.iterations, 'Historical decomposition:')   
            self.mcmc_hd = mcmc_hd
            
               
    def __hd_posterior_estimates(self, credibility_level):

        if self.structural_identification == 1:
            self.hd_estimates = []
        else:
            mcmc_hd = self.mcmc_hd
            hd_estimates = vu.posterior_estimates_3d(mcmc_hd, credibility_level)
            self.hd_estimates = hd_estimates


    def __make_conditional_forecast(self, h, conditions, Z_p):
        
        # make regressors Z_p and Y
        Z_p, Y = vu.make_forecast_regressors(Z_p, self.Y, h, self.p, self.T,
                 self.exogenous, self.constant, self.trend, self.quadratic_trend)
        # make conditional forecast regressors y_bar, Z, omega and gamma_00
        y_bar, Q, omega, gamma_00 = vu.conditional_forecast_regressors_1(conditions, h, Y, self.n, self.p)
        # initiate storage and loop over iterations
        mcmc_conditional_forecast = np.zeros((h,self.n,self.iterations))
        for i in range(self.iterations):
            # recover iteration-specific regressors
            mu, F, K, Upsilon_00 = vu.conditional_forecast_regressors_2(self.mcmc_beta[:,:,i], \
                                   self.mcmc_Sigma[:,:,i], conditions, Z_p, self.n, self.m, self.p, h)
            # run Carter Kohn algorithm to obtain conditional forecasts
            bss = BayesianStateSpaceSampler(y_bar, Q, omega, mu, F, K, gamma_00, \
                  Upsilon_00, kalman_type = 'conditional_forecast')
            bss.carter_kohn_algorithm()
            mcmc_conditional_forecast[:,:,i] = bss.Z[:,:self.n]
            if self.verbose:
                cu.progress_bar(i, self.iterations, 'Conditional forecasts:')  
        self.mcmc_conditional_forecast = mcmc_conditional_forecast
        

    def __check_shock_type(self, h, conditions, shocks): 
        
        # check for structural identification
        if self.structural_identification == 1:
            if self.verbose:
                cu.progress_bar_complete('Conditional forecasts:')
            shock_type = 'none'          
        else:
            # identify shocks
            if np.sum(shocks) == self.n:
                shock_type = 'all_shocks'
            else:
                shock_type = 'shock-specific'
        return shock_type
        

    def __make_structural_conditional_forecast(self, h, conditions, shocks, Z_p, shock_type):

        # if there is an issue, return empty mcmc matrix
        if shock_type == 'none':
            self.mcmc_conditional_forecast = []
            if self.verbose:
                cu.progress_bar_complete('Conditional forecasts:')
        # if condition type is well defined, proceed
        else:
            # make regressors Z_p and Y
            Z_p, Y = vu.make_forecast_regressors(Z_p, self.Y, h, self.p, self.T,
                      self.exogenous, self.constant, self.trend, self.quadratic_trend)
            has_irf = hasattr(self, 'mcmc_structural_irf') and self.mcmc_structural_irf.shape[2] >= h
            # make conditional forecast regressors R, y_bar and omega
            R, y_bar, omega = vu.conditional_forecast_regressors_3(conditions, h, self.n)
            if shock_type == 'shock-specific':
                P, non_generating = vu.conditional_forecast_regressors_5(shocks, h, self.n)
            # initiate storage and loop over iterations
            mcmc_conditional_forecast = np.zeros((h,self.n,self.iterations))
            for i in range(self.iterations): 
                index = self._svar_index[i]
                # make predictions, absent shocks
                f = vu.linear_forecast(self.mcmc_beta[:,:,index], h, Z_p, Y, self.n)
                # recover structural IRF or estimate them
                if has_irf:
                    structural_irf = self.mcmc_structural_irf[:,:,:h,i]
                else:
                    irf = vu.impulse_response_function(self.mcmc_beta[:,:,index], self.n, self.p, h)
                    structural_irf = vu.structural_impulse_response_function(irf, self.mcmc_H[:,:,i], self.n)  
                # recover iteration-specific regressors
                M = vu.conditional_forecast_regressors_4(structural_irf, self.n, h)
                # get posterior mean and variance, depending on condition type                
                if shock_type == 'all_shocks':
                    mu_hat, Omega_hat = vu.conditional_forecast_posterior(y_bar, f, M, R, self.mcmc_Gamma[i,:], omega, self.n, h)
                elif shock_type == 'shock-specific':
                    Gamma_nd = vu.conditional_forecast_regressors_6(self.mcmc_Gamma[i,:], non_generating, h)
                    mu_hat, Omega_hat = vu.shock_specific_conditional_forecast_posterior(\
                                        y_bar, f, M, R, P, self.mcmc_Gamma[i,:], Gamma_nd, omega, self.n, h)                
                # sample values
                mcmc_conditional_forecast[:,:,i] = rng.multivariate_normal(mu_hat, Omega_hat).reshape(h,self.n)
                if self.verbose:
                    cu.progress_bar(i, self.iterations, 'Conditional forecasts:')
            self.mcmc_conditional_forecast = mcmc_conditional_forecast
        
    
    def __conditional_forecast_posterior_estimates(self, credibility_level):

        if len(self.mcmc_conditional_forecast) == 0:
            self.conditional_forecast_estimates = []
        else:
            mcmc_conditional_forecast = self.mcmc_conditional_forecast
            conditional_forecast_estimates = vu.posterior_estimates(mcmc_conditional_forecast, credibility_level)
            self.conditional_forecast_estimates = conditional_forecast_estimates      


        

        
   
    
   
    
   
    