# imports
import numpy as np
import numpy.linalg as nla
import numpy.random as nrd
import alexandria.math.linear_algebra as la
import alexandria.math.stat_utilities as su
import alexandria.vector_autoregression.var_utilities as vu
import alexandria.processor.input_utilities as iu
import alexandria.console.console_utilities as cu
from alexandria.vector_autoregression.vector_autoregression import VectorAutoRegression


class MaximumLikelihoodVar(VectorAutoRegression):


    """
    Maximum likelihood vector autoregression, developed in section 11.1
    
    Parameters:
    -----------
    endogenous : ndarray of size (n_obs,n)
        endogenous variables, defined in (4.11.1)
    
    exogenous : ndarray of size (n_obs,m), default = []
        exogenous variables, defined in (4.11.1)
    
    structural_identification : int, default = 2
        structural identification scheme, as defined in section 13.2
        1 = none, 2 = Cholesky, 3 = triangular, 4 = restrictions
    
    lags : int, default = 4
        number of lags, defined in (4.11.1)
    
    constant : bool, default = True
        if True, an intercept is included in the VAR model exogenous
    
    trend : bool, default = False
        if True, a linear trend is included in the VAR model exogenous
    
    quadratic_trend : bool, default = False
        if True, a quadratic trend is included in the VAR model exogenous
    
    credibility_level : float, default = 0.95
        VAR model credibility level (between 0 and 1)
    
    verbose : bool, default = False
        if True, displays a progress bar  
    
    
    Attributes
    ----------
    endogenous : ndarray of size (n_obs,n)
        endogenous variables, defined in (4.11.1)
    
    exogenous : ndarray of size (n_obs,m)
        exogenous variables, defined in (4.11.1)
    
    structural_identification : int
        structural identification scheme, as defined in section 13.2
        1 = none, 2 = Cholesky, 3 = triangular, 4 = restrictions
    
    lags : int
        number of lags, defined in (4.11.1)
    
    constant : bool
        if True, an intercept is included in the VAR model exogenous
    
    trend : bool
        if True, a linear trend is included in the VAR model exogenous
    
    quadratic_trend : bool
        if True, a quadratic trend is included in the VAR model exogenous
    
    credibility_level : float
        VAR model credibility level (between 0 and 1)
    
    verbose : bool, default = False
        if True, displays a progress bar  
    
    B : ndarray of size (k,n)
        matrix of VAR coefficients, defined in (4.11.2)
    
    Sigma : ndarray of size (n,n)
        variance-covariance ndarray of VAR residuals, defined in (4.11.1)
    
    beta_estimates : ndarray of size (k,n,3)
        estimates of VAR coefficients
        page 1: median, page 2: st dev,  page 3: lower bound, page 4: upper bound
    
    Sigma_estimates : ndarray of size (n,n)
        estimates of variance-covariance ndarray of VAR residuals
    
    H : ndarray of size (n,n)
        structural identification ndarray, defined in (4.13.5)
    
    Gamma : ndarray of size (n,)
        diagonal of structural shock variance ndarray, defined in section 13.2
    
    Gamma_estimates : ndarray of size (n,)
        estimates of structural shock variance ndarray, defined in section 13.2
    
    steady_state_estimates : ndarray of size (T,n,3)
        estimates of steady-state, defined in (4.12.30)
    
    fitted_estimates : ndarray of size (T,n,3)
        estimates of in-sample fit, defined in (4.11.2)
    
    residual_estimates : ndarray of size (T,n,3)
        estimates of in-sample residuals, defined in (4.11.2)
    
    structural_shocks_estimates : ndarray of size (T,n,3)
        estimates of in-sample structural shocks, defined in definition 13.1
    
    insample_evaluation : dict
        in-sample evaluation criteria, defined in (4.13.15)-(4.13.17)
    
    forecast_estimates : ndarray of size (f_periods,n,3)
        forecast estimates, defined in (4.13.12) and (4.13.13)
        page 1: median, page 2: lower bound, page 3: upper bound
    
    forecast_evaluation_criteria : dict
        forecast evaluation criteria, defined in (4.13.18)-(4.13.19)
    
    irf_estimates : ndarray of size (n,n,irf_periods,3)
        estimates of impulse response function, defined in (4.13.1) or (4.13.9)
        page 1: median, page 2: lower bound, page 3: upper bound    
    
    exo_irf_estimates : ndarray of size (n,m,irf_periods,3)
        estimates of exogenous impulse response function, if any exogenous variable
        page 1: median, page 2: lower bound, page 3: upper bound
    
    fevd_estimates : ndarray of size (n,n,fevd_periods,3)
        estimates of forecast error variance decomposition, defined in (4.13.31)
        page 1: median, page 2: lower bound, page 3: upper bound 
    
    hd_estimates : ndarray of size (n,n,T,3)
        estimates of historical decomposition, defined in (4.13.35)
        page 1: median, page 2: lower bound, page 3: upper bound 
    
    Y : ndarray of size (T,n)
        ndarray of in-sample endogenous variables, defined in (4.11.3)
    
    Z : ndarray of size (T,m)
        ndarray of in-sample endogenous variables, defined in (4.11.3)
    
    X : ndarray of size (T,k)
        ndarray of exogenous and lagged regressors, defined in (4.11.3)
    
    n : int
        number of endogenous variables, defined in (4.11.1)
    
    m : int
        number of exogenous variables, defined in (4.11.1)
    
    p : int
        number of lags, defined in (4.11.1)
    
    T : int
        number of sample periods, defined in (4.11.1)
    
    k : int
        number of VAR coefficients in each equation, defined in (4.11.1)
    
    q : int
        total number of VAR coefficients, defined in (4.11.1)
    
    
    Methods
    ----------
    estimate
    insample_fit
    forecast
    forecast_evaluation
    impulse_response_function
    forecast_error_variance_decomposition
    historical_decomposition
    """    


    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------

    
    def __init__(self, endogenous, exogenous = [], structural_identification = 2, 
                 lags = 4, constant = True, trend = False, quadratic_trend = False, 
                 credibility_level = 0.95, verbose = False):
        
        """
        constructor for the MaximumLikelihoodRegression class
        """
        
        self.endogenous = endogenous
        self.exogenous = exogenous
        self.structural_identification = structural_identification
        self.lags = lags
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
        estimates parameters B and Sigma of VAR model
        
        parameters:
        none
        
        returns:
        none
        """
        
        # fit to obtain maximum likelihood estimates
        self.__fit()
        # obtain posterior estimates for regression parameters
        self.__parameter_estimates()
        # estimate structural VAR
        self.__make_structural_identification()
          
        
    def insample_fit(self):
        
        """
        insample_fit()
        estimates of in-sample fit elements (fit, residuals, in-sample criteria, steady-state, structural shocks)
        
        parameters:
        none
        
        returns:
        none
        """ 

        # compute steady-state
        self.__steady_state()        
        # obtain fit and residuals
        self.__fitted_and_residuals()
        # obtain in-sample evaluation criteria
        self.__insample_criteria()

        
    def forecast(self, h, credibility_level, Z_p=[]):
        
        """
        forecast(periods, credibility_level, , Z_p=[])
        predictions for the maximum likelihood VAR model using (4.13.12)
        
        parameters:
        h : int
            number of forecast periods
        credibility_level : float
            credibility level for predictions (between 0 and 1)
        Z_p : empty list or numpy array of dimension (h, n_exo)
            empty list unless the model includes exogenous other than constant, trend and quadratic trend
            if not empty, n_exo is the number of additional exogenous variables
            
        returns:
        forecast_estimates : ndarray of shape (periods, n, 3)
            posterior estimates for predictions
            page 1: median; page 2: interval lower bound;
            page 3: interval upper bound
        """
        
        # unpack
        Y = self.Y
        B = self.B
        Sigma = self.Sigma
        constant = self.constant
        trend = self.trend
        quadratic_trend = self.quadratic_trend
        n = self.n
        p = self.p
        T = self.T
        exogenous = self.exogenous
        # make regressors
        Z_p, Y = vu.make_forecast_regressors(Z_p, Y, h, p, T, exogenous, constant, trend, quadratic_trend)
        # obtain point estimates
        forecasts = vu.linear_forecast(B, h, Z_p, Y, n)
        # obtain confidence intervals
        lower_bound, upper_bound = self.__maximum_likelihood_forecast_credibility( \
        forecasts, credibility_level, B, Sigma, h, n, p)
        # concatenate
        forecast_estimates = np.dstack([forecasts,lower_bound,upper_bound])
        self.forecast_estimates = forecast_estimates
        return forecast_estimates    

        
    def forecast_evaluation(self, Y):
        
        """
        forecast_evaluation(Y)
        forecast evaluation criteria for the maximum likelihood VAR model
        
        parameters:
        Y : ndarray of shape (perods,n)
            array of realised values for forecast evaluation
            
        returns:
        none
        """
        
        # recover forecasts
        Y_hat = self.forecast_estimates[:,:,0]
        # obtain forecast evaluation criteria from equations (4.13.18) and (4.13.19)
        forecast_evaluation_criteria = vu.forecast_evaluation_criteria(Y_hat, Y)
        # save as attributes
        self.forecast_evaluation_criteria = forecast_evaluation_criteria
        
        
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
        exo_irf_estimates : ndarray of shape (n,m,h,3)
            first 3 dimensions are variable, shock, period; 4th dimension is median, lower bound, upper bound
        """
        
        # get regular impulse response funtion
        irf, mcmc_irf, irf_exo, mcmc_irf_exo = self.__make_impulse_response_function(h)      
        # get structural impulse response function
        structural_irf, mcmc_structural_irf = self.__make_structural_impulse_response_function(irf, mcmc_irf, h)
        # obtain posterior estimates
        irf_estimates, exo_irf_estimates = self.__irf_posterior_estimates(credibility_level, irf, mcmc_irf, \
                                           irf_exo, mcmc_irf_exo, structural_irf, mcmc_structural_irf)
        self.irf_estimates = irf_estimates
        self.exo_irf_estimates = exo_irf_estimates            
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
        fevd_estimates : ndarray of shape (n,n,h)
            dimensions are variable, shock, period
        """

        # get forecast error variance decomposition
        fevd, mcmc_fevd = self.__make_forecast_error_variance_decomposition(h)
        # obtain posterior estimates
        fevd_estimates = self.__fevd_posterior_estimates(credibility_level, fevd, mcmc_fevd)
        self.fevd_estimates = fevd_estimates
        if self.verbose:
            cu.progress_bar_complete('Forecast error variance decomposition:')
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
        hd, mcmc_hd = self.__make_historical_decomposition()
        # obtain posterior estimates
        hd_estimates = self.__hd_posterior_estimates(credibility_level, hd, mcmc_hd)
        self.hd_estimates = hd_estimates
        if self.verbose:
            cu.progress_bar_complete('Historical decomposition:')
        return hd_estimates  

    
    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------       
    
    
    def __fit(self):
        
        """ estimates B_hat and Sigma_hat from (4.11.9) """

        # unpack        
        Y = self.Y
        X = self.X
        XX = self._XX
        XY = self._XY
        T = self.T
        verbose = self.verbose
        # estimate B_hat and Sigma_hat
        B, Sigma = self._ols_var(Y, X, XX, XY, T)
        # if verbose, display progress bar
        if verbose:
            cu.progress_bar_complete('VAR parameters:')
        self.B = B
        self.Sigma = Sigma
        
        
    def __parameter_estimates(self):
        
        """ estimates and intervals from Normal distribution in (4.11.10) """
        
        # unpack
        B = self.B
        Sigma = self.Sigma
        credibility_level = self.credibility_level
        XX = self._XX
        k = self.k
        n = self.n
        # get coefficients variance Q
        Q = np.kron(Sigma, la.invert_spd_matrix(XX))
        Qii = np.reshape(np.sqrt(np.diag(Q)),[k,n],'F')
        # critical value of Normal distribution for credibility level
        Z = su.normal_icdf((1 + credibility_level) / 2)
        # initiate storage: 4 dimensions: median, standard deviation, lower bound,  upper bound
        beta_estimates = np.zeros((k,n,4))  
        beta_estimates[:,:,0] = B
        beta_estimates[:,:,1] = B - Z * Qii
        beta_estimates[:,:,2] = B + Z * Qii
        beta_estimates[:,:,3] = Qii
        # save as attributes
        self.beta_estimates = beta_estimates 
        self.Sigma_estimates = self.Sigma
        
        
    def __make_structural_identification(self):
        
        """ computes structural VAR estimates """
        
        # unpack
        n = self.n
        structural_identification = self.structural_identification
        Sigma = self.Sigma
        verbose = self.verbose
        # estimate Cholesky SVAR, if selected
        if structural_identification == 2:
            H = la.cholesky_nspd(Sigma)
            inv_H = la.invert_lower_triangular_matrix(H)
            self.H = H
            self.__inv_H = inv_H
            self.Gamma = np.ones(n)
            self.Gamma_estimates = np.ones(n)
            if verbose:
                cu.progress_bar_complete('SVAR parameters:')
        # estimate triangular SVAR, if selected
        elif structural_identification == 3:
            H, Gamma = la.triangular_factorization(Sigma)
            inv_H = la.invert_lower_triangular_matrix(H)
            self.H = H
            self.__inv_H = inv_H
            self.Gamma = Gamma    
            self.Gamma_estimates = Gamma
            if verbose:
                cu.progress_bar_complete('SVAR parameters:')   
        
        
    def __steady_state(self): 
    
        """ computes steady-state for the VAR model """
        
        # point estimates
        Z = self.Z
        B = self.B
        Sigma = self.Sigma
        n = self.n
        m = self.m
        p = self.p
        T = self.T
        k = self.k
        q = self.q
        XX = self._XX
        ss = vu.steady_state(Z, B, n, m, p, T)
        # simulated coefficients for credibility interval
        mcmc_B = vu.ols_var_mcmc_beta(B, Sigma, XX, k, n, q)
        mcmc_ss = np.zeros((T,n,500))
        for j in range(500):
            mcmc_ss[:,:,j] = vu.steady_state(Z, mcmc_B[:,:,j], n, m, p, T)
        if self.verbose: 
            cu.progress_bar_complete('Steady-state:')
        ss_estimates = vu.posterior_estimates(mcmc_ss, self.credibility_level)
        ss_estimates[:,:,0] = ss
        self.steady_state_estimates = ss_estimates     
        

    def __fitted_and_residuals(self):
        
        """ computes fitted, residuals, and structural shocks """

        # point estimates
        Y = self.Y      
        X = self.X           
        B = self.B
        Sigma = self.Sigma
        n = self.n
        T = self.T
        k = self.k
        q = self.q
        XX = self._XX
        structural_identification = self.structural_identification
        # simulated coefficients for credibility interval
        mcmc_B = vu.ols_var_mcmc_beta(B, Sigma, XX, k, n, q)
        mcmc_fitted = np.zeros((T,n,500))
        mcmc_residuals = np.zeros((T,n,500))
        for j in range(500):
            E, Y_hat = vu.fit_and_residuals(Y, X, mcmc_B[:,:,j]) 
            mcmc_fitted[:,:,j] = Y_hat
            mcmc_residuals[:,:,j] = E
        if self.verbose:
            cu.progress_bar_complete('Fitted and residual:')
        fitted_estimates = vu.posterior_estimates(mcmc_fitted, self.credibility_level)
        residual_estimates = vu.posterior_estimates(mcmc_residuals, self.credibility_level)
        E, Y_hat = vu.fit_and_residuals(Y, X, B)
        fitted_estimates[:,:,0] = Y_hat
        residual_estimates[:,:,0] = E
        if structural_identification in [2,3]:
            inv_H = self.__inv_H
            mcmc_structural_shocks = np.zeros((T,n,500))
            for j in range(500):
                Xi = vu.structural_shocks(mcmc_residuals[:,:,j], inv_H)
                mcmc_structural_shocks[:,:,j] = Xi
            if self.verbose:
                cu.progress_bar_complete('Structural shocks:')
            structural_shock_estimates = vu.posterior_estimates(mcmc_structural_shocks, self.credibility_level)
            [E, Y_hat] = vu.fit_and_residuals(Y, X, B)
            Xi = vu.structural_shocks(E, inv_H)
            structural_shock_estimates[:,:,0] = Xi
        else:
            structural_shock_estimates = []
        self.fitted_estimates = fitted_estimates
        self.residual_estimates = residual_estimates
        self.structural_shock_estimates = structural_shock_estimates


    def __insample_criteria(self):
        
        """ computes in-sample evluation criteria """
        
        Y = self.Y    
        E = self.residual_estimates[:,:,0]
        T = self.T        
        k = self.k
        Sigma = self.Sigma
        q = self.q
        # estimate general criteria (SSR, R2, adj-R2)
        insample_evaluation_1 = vu.insample_evaluation_criteria(Y, E, T, k)
        # estimate criteria specific to maximum likelihood VAR (aic, bic, hq)
        insample_evaluation_2 = self.__maximum_likelihood_evaluation_criteria(Sigma, q, T)
        # merge in-sample criteria
        insample_evaluation = iu.concatenate_dictionaries(insample_evaluation_1, insample_evaluation_2)
        self.insample_evaluation = insample_evaluation 
    
    
    def __maximum_likelihood_evaluation_criteria(self, Sigma, q, T):    
    
        """ computes aic, bic and hq for maximum likelihood VAR """
        
        log_det_Sigma = np.log(nla.det(Sigma))
        aic = 2 * q / T + log_det_Sigma
        bic = q * np.log(T) / T + log_det_Sigma
        hq = 2 * q * np.log(np.log(T)) / T + log_det_Sigma
        insample_evaluation = {}
        insample_evaluation['aic'] = aic
        insample_evaluation['bic'] = bic
        insample_evaluation['hq'] = hq
        return insample_evaluation   
    

    def __maximum_likelihood_forecast_credibility(self, forecasts, credibility_level, B, Sigma, periods, n, p):
    
        """ create forecast credibilty intervals for the maximum likelihood VAR """

        lower_bound = np.zeros((periods,n))
        upper_bound = np.zeros((periods,n))
        Q_h = np.zeros((n,n))
        Z = su.normal_icdf((1 + credibility_level) / 2)
        irf = vu.impulse_response_function(B, n, p, periods)
        for h in range(periods):
            y_h = forecasts[h,:]
            Phi_h = irf[:,:,h]
            Q_h += Phi_h @ Sigma @ Phi_h.T
            s_h = np.sqrt(np.diag(Q_h))
            lower_bound[h,:] = y_h - Z * s_h
            upper_bound[h,:] = y_h + Z * s_h
        return lower_bound, upper_bound


    def __make_impulse_response_function(self, h):
        
        """ point estimates and simulations for impulse response functions """
        
        # simulated coefficients for credibility interval
        mcmc_B = vu.ols_var_mcmc_beta(self.B, self.Sigma, self._XX, self.k, self.n, self.q)     
        # impulse response functions, point estimates
        irf = vu.impulse_response_function(self.B, self.n, self.p, h)
        # impulse response functions, simulated values
        mcmc_irf = np.zeros((self.n, self.n, h, 500))
        for i in range(500):
            mcmc_irf[:,:,:,i] = vu.impulse_response_function(mcmc_B[:,:,i], self.n, self.p, h)
        if self.verbose:    
            cu.progress_bar_complete('Impulse response function:')
        # exogenous impulse response functions, point estimates
        if len(self.exogenous) != 0:
            r = self.exogenous.shape[1]
            irf_exo = vu.exogenous_impulse_response_function(self.B, self.n, self.m, r, self.p, h)        
            # impulse response functions, simulated values
            mcmc_irf_exo = np.zeros((self.n, r, h, 500))
            for i in range(500):
                mcmc_irf_exo[:,:,:,i] = vu.exogenous_impulse_response_function(mcmc_B[:,:,i], self.n, self.m, r, self.p, h)    
            if self.verbose:    
                cu.progress_bar_complete('Exogenous impulse response function:')
        else:
            irf_exo = []
            mcmc_irf_exo = []            
        return irf, mcmc_irf, irf_exo, mcmc_irf_exo     
        
  
    def __make_structural_impulse_response_function(self, irf, mcmc_irf, h):
        
        """ structural impulse response function """

        # get structural impulse response function
        if self.structural_identification == 1:
            structural_irf = []
            mcmc_structural_irf = []
        else:
            structural_irf = vu.structural_impulse_response_function(irf, self.H, self.n)
            mcmc_structural_irf = np.zeros((self.n, self.n, h, 500))
            for i in range(500):
                mcmc_structural_irf[:,:,:,i] = vu.structural_impulse_response_function(mcmc_irf[:,:,:,i], self.H, self.n)
            if self.verbose:    
                cu.progress_bar_complete('Structural impulse response function:')         
        return structural_irf, mcmc_structural_irf
    

    def __irf_posterior_estimates(self, credibility_level, irf, mcmc_irf, irf_exo, \
                                  mcmc_irf_exo, structural_irf, mcmc_structural_irf):
        
        """ posterior estimates of impulse response function """
        
        # posterior estimates for endogenous
        if self.structural_identification == 1:
            irf_estimates = vu.posterior_estimates_3d(mcmc_irf, credibility_level)
            irf_estimates[:,:,:,0] = irf
        else:
            irf_estimates = vu.posterior_estimates_3d(mcmc_structural_irf, credibility_level)
            irf_estimates[:,:,:,0] = structural_irf
        # posterior estimates for exogenous
        if len(self.exogenous) != 0:
            exo_irf_estimates = vu.posterior_estimates_3d(mcmc_irf_exo, credibility_level)
        else:
            exo_irf_estimates = []  
        return irf_estimates, exo_irf_estimates


    def __make_forecast_error_variance_decomposition(self, h):
        
        """ forecast error variance decomposition """
        
        # if no structural identification, fevd is not computed
        if self.structural_identification == 1:
            fevd = []
            mcmc_fevd = []
        else:
            # fevd, point estimates
            irf = vu.impulse_response_function(self.B, self.n, self.p, h)
            structural_irf = vu.structural_impulse_response_function(irf, self.H, self.n) 
            fevd = vu.forecast_error_variance_decomposition(structural_irf, self.Gamma, self.n, h)
            # fevd, simulated values
            mcmc_B = vu.ols_var_mcmc_beta(self.B, self.Sigma, self._XX, self.k, self.n, self.q)
            mcmc_fevd = np.zeros((self.n, self.n, h, 500))
            for i in range(500):
                irf = vu.impulse_response_function(mcmc_B[:,:,i], self.n, self.p, h)
                structural_irf = vu.structural_impulse_response_function(irf, self.H, self.n) 
                mcmc_fevd[:,:,:,i] = vu.forecast_error_variance_decomposition(structural_irf, self.Gamma, self.n, h)
        return fevd, mcmc_fevd


    def __fevd_posterior_estimates(self, credibility_level, fevd, mcmc_fevd):

        """ posterior estimates of forecast error variance decomposition """
        
        # posterior estimates
        if self.structural_identification == 1:
            fevd_estimates = []
        else:
            fevd_estimates = vu.posterior_estimates_3d(mcmc_fevd, credibility_level)
            fevd_estimates[:,:,:,0] = fevd
        return fevd_estimates


    def __make_historical_decomposition(self):
        
        """ historical decomposition """
        
        # if no structural identification, historical decomposition is not computed
        if self.structural_identification == 1:
            hd = []
            mcmc_hd = []
        else:
            # hd, point estimates
            irf = vu.impulse_response_function(self.B, self.n, self.p, self.T)
            structural_irf = vu.structural_impulse_response_function(irf, self.H, self.n) 
            E, _ = vu.fit_and_residuals(self.Y, self.X, self.B)
            structural_shocks = vu.structural_shocks(E, self.__inv_H)            
            hd = vu.historical_decomposition(structural_irf, structural_shocks, self.n, self.T)   
            # hd, simulated values
            mcmc_B = vu.ols_var_mcmc_beta(self.B, self.Sigma, self._XX, self.k, self.n, self.q)
            mcmc_hd = np.zeros((self.n, self.n, self.T, 500))
            for i in range(500):
                irf = vu.impulse_response_function(mcmc_B[:,:,i], self.n, self.p, self.T)
                structural_irf = vu.structural_impulse_response_function(irf, self.H, self.n)
                E, _ = vu.fit_and_residuals(self.Y, self.X, mcmc_B[:,:,i])
                structural_shocks = vu.structural_shocks(E, self.__inv_H)                  
                mcmc_hd[:,:,:,i] = vu.historical_decomposition(structural_irf, structural_shocks, self.n, self.T)  
        return hd, mcmc_hd


    def __hd_posterior_estimates(self, credibility_level, hd, mcmc_hd):

        """ posterior estimates of historical decomposition """
        
        # posterior estimates
        if self.structural_identification == 1:
            hd_estimates = []
        else:
            hd_estimates = vu.posterior_estimates_3d(mcmc_hd, credibility_level)
            hd_estimates[:,:,:,0] = hd
        return hd_estimates


