# imports
import numpy as np
from alexandria.vector_autoregression.bayesian_var import BayesianVar
import alexandria.vector_autoregression.var_utilities as vu
import alexandria.vec_varma.vec_varma_utilities as vvu
import alexandria.math.linear_algebra as la
import alexandria.math.random_number_generators as rng
import alexandria.console.console_utilities as cu


class VectorErrorCorrection(BayesianVar):
      
    
    """
    Vector Error Correction, developed in chapter 15
    
    Parameters:
    -----------
    endogenous : ndarray of size (n_obs,n)
        endogenous variables, defined in (5.16.1)
    
    exogenous : ndarray of size (n_obs,m), default = []
        exogenous variables, defined in (5.16.1)
    
    structural_identification : int, default = 2
        structural identification scheme, as defined in section 16.5
        1 = none, 2 = Cholesky, 3 = triangular, 4 = restrictions
    
    restriction_table : ndarray
        numerical matrix of restrictions for structural identification
    
    lags : int, default = 4
        number of lags, defined in (5.16.1)
    
    max_cointegration_rank : int, default = 1
        maximum number of cointegration relations
    
    prior_type : int, default = 1
        prior for VEC model
        1 = uninformative, 2 = horseshoe, 3 = selection
    
    error_correction_type : int, default = 1
        error correction type for VEC model
        1 = general, 2 = reduced-rank
    
    constant : bool, default = True
        if True, an intercept is included in the VAR model exogenous
    
    trend : bool, default = False
        if True, a linear trend is included in the VAR model exogenous
    
    quadratic_trend : bool, default = False
        if True, a quadratic trend is included in the VAR model exogenous
    
    pi1 : float, default = 0.1
        overall tightness hyperparameter, defined in (5.15.18)
    
    pi2 : float, default = 0.5
        cross-variable shrinkage hyperparameter, defined in (5.15.18)
    
    pi3 : float, default = 1
        lag decay hyperparameter, defined in (5.15.18)    
    
    pi4 : float, default = 100
        exogenous slackness hyperparameter, defined in (5.15.18)  
    
    credibility_level : float, default = 0.95
        VAR model credibility level (between 0 and 1)
    
    iterations : int, default = 2000
        number of Gibbs sampler replications   
    
    burnin : int, default = 1000
        number of Gibbs sampler burn-in replications  
    
    verbose : bool, default = False
        if True, displays a progress bar 
    
    
    Properties
    ----------
    endogenous : ndarray of size (n_obs,n)
        endogenous variables, defined in (5.16.1)
    
    exogenous : ndarray of size (n_obs,m), default = []
        exogenous variables, defined in (5.16.1)
    
    structural_identification : int, default = 2
        structural identification scheme, as defined in section 16.5
        1 = none, 2 = Cholesky, 3 = triangular, 4 = restrictions
    
    restriction_table : ndarray
        numerical ndarray of restrictions for structural identification
    
    lags : int, default = 4
        number of lags, defined in (5.16.1)
    
    max_cointegration_rank : int, default = 1
        maximum number of cointegration relations
    
    prior_type : int, default = 1
        prior for VEC model
        1 = uninformative, 2 = horseshoe, 3 = selection
    
    error_correction_type : int, default = 1
        error correction type for VEC model
        1 = general, 2 = reduced-rank
    
    constant : bool, default = True
        if True, an intercept is included in the VAR model exogenous
    
    trend : bool, default = False
        if True, a linear trend is included in the VAR model exogenous
    
    quadratic_trend : bool, default = False
        if True, a quadratic trend is included in the VAR model exogenous
    
    pi1 : float, default = 0.1
        overall tightness hyperparameter, defined in (5.15.18)
    
    pi2 : float, default = 0.5
        cross-variable shrinkage hyperparameter, defined in (5.15.18)
    
    pi3 : float, default = 1
        lag decay hyperparameter, defined in (5.15.18)    
    
    pi4 : float, default = 100
        exogenous slackness hyperparameter, defined in (5.15.18)  
    
    credibility_level : float, default = 0.95
        VAR model credibility level (between 0 and 1)
    
    iterations : int, default = 2000
        number of Gibbs sampler replications   
    
    burnin : int, default = 1000
        number of Gibbs sampler burn-in replications  
    
    verbose : bool, default = False
        if True, displays a progress bar 
    
    Y : ndarray of size (T,n)
        ndarray of in-sample endogenous variables, defined in (4.11.3)
    
    X : ndarray of size (T,k)
        ndarray of exogenous and lagged regressors, defined in (4.11.3)
    
    DY : ndarray of size (T,n)
        ndarray of differenced endogenous variables, defined in (5.15.8)
    
    Y_1 : ndarray of size (T,n)
        ndarray of lagged endogenous variables, defined in (5.15.8)
    
    Z : ndarray of size (T,k)
        ndarray of exogenous and differenced lagged regressors, defined in (5.15.8)
    
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
    
    r : int
        maximum number of cointegration relations
    
    Q : ndarray of size (q,1)
        prior variance of autoregressive coefficients, defined in (5.15.18)           
    
    alpha : float
        prior degrees of freedom, defined in (5.15.19)
    
    S : ndarray of size (n,n)
        prior scale ndarray, defined in (5.15.19)
    
    mcmc_Xi : ndarray of size (n,n,iterations)
        MCMC values of error correction coefficients   
    
    mcmc_Phi : ndarray of size (k,n,iterations)
        MCMC values of autoregressive coefficients   
        
    mcmc_Sigma : ndarray of size (n,n,iterations)
        MCMC values of residual variance-covariance ndarray
       
    mcmc_chol_Sigma : ndarray of size (n,n,iterations)
        MCMC values of residual variance-covariance ndarray (Cholesky factor)
    
    mcmc_K : ndarray of size (n,r,iterations)
        MCMC values of cointegration relations  
    
    mcmc_Lamda : ndarray of size (n,r,iterations)
        MCMC values of cointegration loadings  
    
    mcmc_beta : ndarray of size (k,n,iterations)
        MCMC values of VAR coefficients   
    
    mcmc_nu : ndarray of size (iterations,1)
        MCMC values of nu coefficients, defined in (5.15.40)
    
    mcmc_tau_2 : ndarray of size (iterations,1)
        MCMC values of tau_2 coefficients, defined in (5.15.39)
    
    mcmc_eta : ndarray of size (iterations,1)
        MCMC values of eta coefficients, defined in (5.15.42)
    
    mcmc_psi_2 : ndarray of size (iterations,1)
        MCMC values of psi_2 coefficients, defined in (5.15.39)
    
    mcmc_omega : ndarray of size (iterations,1)
        MCMC values of nu coefficients, defined in (5.15.63)
    
    mcmc_zeta_2 : ndarray of size (iterations,1)
        MCMC values of zeta_2 coefficients, defined in (5.15.58)
    
    mcmc_delta : ndarray of size (iterations,1)
        MCMC values of delta coefficients, defined in (5.15.84)
    
    mcmc_chi : ndarray of size (iterations,1)
        MCMC values of chi coefficients, defined in (5.15.90)
    
    mcmc_gamma : ndarray of size (iterations,1)
        MCMC values of gamma coefficients, defined in (5.15.92)
        
    Xi_estimates : ndarray of size (n,n,4)
        estimates of error correction coefficients
        page 1: median, page 2: st dev,  page 3: lower bound, page 4: upper bound
    
    Phi_estimates : ndarray of size (k,n,3)
        estimates of autoregressive coefficients
        page 1: median, page 2: st dev,  page 3: lower bound, page 4: upper bound
    
    beta_estimates : ndarray of size (k,n,3)
        estimates of VAR coefficients
        page 1: median, page 2: st dev,  page 3: lower bound, page 4: upper bound
    
    Sigma_estimates : ndarray of size (n,n)
        estimates of variance-covariance ndarray of VAR residuals
    
    tau_2_estimates : float or ndarray of size (r,1)
        estimates of tightness coefficients tau_2
    
    psi_2_estimates : float or ndarray of size (n,1)
        estimates of tightness coefficients psi_2
    
    zeta_2_estimates : float or ndarray of size (n,1)
        estimates of tightness coefficients zeta_2
    
    K_estimates : ndarray of size (n,r,4)
        estimates of cointegration relations K
        page 1: median, page 2: st dev,  page 3: lower bound, page 4: upper bound
    
    Lamda_estimates : ndarray of size (n,r,4)
        estimates of loadings ndarray Lamda
        page 1: median, page 2: st dev,  page 3: lower bound, page 4: upper bound    
    
    delta_estimates : float or ndarray of size (n,n)
        estimates of binary coefficients delta (mean of all iterations)
    
    chi_estimates : float or ndarray of size (n,n)
        estimates of binary coefficients chi (mean of all iterations)
    
    gamma_estimates : float or ndarray of size (n,n)
        estimates of binary coefficients gamma (mean of all iterations)
    
    mcmc_H :  ndarray of size (n,n,iterations)
        MCMC values of structural identification ndarray, defined in (4.13.5)
    
    mcmc_Gamma : ndarray of size (iterations,n)
        MCMC values of structural shock variance ndarray, defined in definition 13.1
    
    s : ndarray of size (n,1)
        prior scale ndarray, defined in (5.15.19) 
    
    steady_state_estimates : ndarray of size (T,n,3)
        estimates of steady-state, defined in (4.12.30)
    
    fitted_estimates : ndarray of size (T,n,3)
        estimates of in-sample fit, defined in (4.11.2)
    
    residual_estimates : ndarray of size (T,n,3)
        estimates of in-sample residuals, defined in (4.11.2)
    
    structural_shocks_estimates : ndarray of size (T,n,3)
        estimates of in-sample structural shocks, defined in definition 13.1
    
    insample_evaluation : struct
        in-sample evaluation criteria, defined in (4.13.15)-(4.13.17)
    
    mcmc_structural_shocks : ndarray of size (T,n,iterations)
        MCMC values of structural shocks
    
    mcmc_forecasts : ndarray of size (f_periods,n,iterations)
        MCMC values of forecasts
    
    forecast_estimates : ndarray of size (f_periods,n,3)
        forecast estimates, defined in (4.13.12) and (4.13.13)
        page 1: median, page 2: lower bound, page 3: upper bound
    
    forecast_evaluation_criteria : struct
        forecast evaluation criteria, defined in (4.13.18)-(4.13.21)
    
    mcmc_irf : ndarray of size (n,n,irf_periods,iterations)
        MCMC values of impulse response function, defined in section 13.1
    
    mcmc_irf_exo : ndarray of size (n,m,irf_periods,iterations)
        MCMC values of exogenous impulse response function
    
    mcmc_structural_irf : ndarray of size (n,n,irf_periods,iterations)
        MCMC values of structural impulse response function, defined in section 13.2
    
    irf_estimates : ndarray of size (n,n,irf_periods,3)
        posterior estimates of impulse response function, defined in section 13.1 - 13.2
        page 1: median, page 2: lower bound, page 3: upper bound    
    
    exo_irf_estimates : ndarray of size (n,m,irf_periods,3)
        posterior estimates of exogenous impulse response function, if any exogenous variable
        page 1: median, page 2: lower bound, page 3: upper bound
    
    mcmc_fevd : ndarray of size (n,n,fevd_periods,iterations)
        MCMC values of forecast error variance decompositions, defined in section 13.4
    
    fevd_estimates : ndarray of size (n,n,fevd_periods,3)
        posterior estimates of forecast error variance decomposition, defined in section 13.4
        page 1: median, page 2: lower bound, page 3: upper bound 
    
    mcmc_hd : ndarray of size (n,n,T,iterations)
        MCMC values of historical decompositions, defined in section 13.5
    
    hd_estimates : ndarray of size (n,n,T,3)
        posterior estimates of historical decomposition, defined in section 13.5
        page 1: median, page 2: lower bound, page 3: upper bound 
    
    mcmc_conditional_forecasts : ndarray of size (f_periods,n,iterations)
        MCMC values of conditional forecasts, defined in section 14.1
    
    conditional_forecast_estimates : ndarray of size (f_periods,n,3)
        posterior estimates of conditional forecast, defined in section 14.1
        page 1: median, page 2: lower bound, page 3: upper bound
    H_estimates : ndarray of size (n,n)
        posterior estimates of structural ndarray, defined in section 13.2
    
    Gamma_estimates : ndarray of size (1,n)
        estimates of structural shock variance ndarray, defined in section 13.2
    
    
    Methods
    ----------
    estimate
    insample_fit
    forecast
    forecast_evaluation
    impulse_response_function
    forecast_error_variance_decomposition
    historical_decomposition
    conditional_forecast    
    """    
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------    
    

    def __init__(self, endogenous, exogenous = [], structural_identification = 2, 
                 restriction_table = [], lags = 4, max_cointegration_rank = 1,
                 prior_type = 1, error_correction_type = 1, constant = True, trend = False, 
                 quadratic_trend = False, pi1 = 0.1, pi2 = 0.5, pi3 = 1, pi4 = 100, 
                 credibility_level = 0.95, iterations = 3000, burnin = 1000, verbose = False):

        """
        constructor for the VectorErrorCorrection class
        """
        
        self.endogenous = endogenous
        self.exogenous = exogenous
        self.structural_identification = structural_identification
        self.restriction_table = restriction_table
        self.lags = lags
        self.max_cointegration_rank = max_cointegration_rank
        self.prior_type = prior_type
        self.error_correction_type = error_correction_type
        self.constant = constant
        self.trend = trend
        self.quadratic_trend = quadratic_trend
        self.pi1 = pi1
        self.pi2 = pi2
        self.pi3 = pi3
        self.pi4 = pi4
        self.credibility_level = credibility_level
        self.iterations = iterations
        self.burnin = burnin
        self.verbose = verbose
        # make regressors
        self.__make_regressors()       

    
    def estimate(self):
    
        """
        estimate()
        generates posterior estimates for Bayesian VEC model parameters
        
        parameters:
        none
        
        returns:
        none    
        """    

        # define prior values
        self.__prior()    
        # define posterior values
        self.__posterior()
        # run MCMC algorithm (Gibbs sampling) for VAR parameters
        self.__parameter_mcmc()
        # obtain posterior estimates for vector error correction parameters
        self.__parameter_estimates()
        # estimate structural identification
        self._make_structural_identification()
        

    #---------------------------------------------------
    # Methods (Access = private)
    #--------------------------------------------------- 


    def __make_regressors(self):
    
        """ generates VEC regressors along with other dimension elements """
        
        # first generate VAR regressors, for later use once VEC is converted to VAR
        Y, Z, X = vvu.make_var_regressors(self.endogenous, self.exogenous, self.lags+1, \
                                       self.constant, self.trend, self.quadratic_trend)
        # then generate VEC regressors, following (5.15.8)
        DY, Y_1, Z_vec = vvu.make_vec_regressors(self.endogenous, self.exogenous, self.lags, \
                                       self.constant, self.trend, self.quadratic_trend)
        # define dimensions
        n, m, p_vec, p, T_vec, T, k_vec, k, q_vec, q, r = vvu.generate_dimensions(Y, DY, self.exogenous, self.lags, \
                              self.max_cointegration_rank, self.constant, self.trend, self.quadratic_trend)   
        # get individual ar variances
        s = vvu.individual_ar_variances(n, self.endogenous, self.lags)
        # define estimation terms
        ZZ = Z_vec.T @ Z_vec
        ZDY = Z_vec.T @ DY
        ZY = Z_vec.T @ Y_1
        YY = Y_1.T @ Y_1
        YDY = Y_1.T @ DY
        YZ = Y_1.T @ Z_vec
        # save as attributes      
        self.Y = Y
        self.X = X
        self.DY = DY
        self.Y_1 = Y_1
        self._Z_vec = Z_vec
        self.Z = Z
        self.n = n
        self.m = m
        self._p_vec = p_vec   
        self.p = p
        self._T_vec = T_vec        
        self.T = T
        self._k_vec = k_vec        
        self.k = k
        self._q_vec = q_vec       
        self.q = q
        self.r = r
        self.s = s
        self._ZZ = ZZ
        self._ZDY = ZDY
        self._ZY = ZY
        self._YY = YY
        self._YDY = YDY
        self._YZ = YZ


    def __prior(self):
        
        """ creates prior elements Q, alpha and S defined in (5.15.18) and (5.15.19) """
        
        Q = vu.make_V(self.s, self.pi1, self.pi2, self.pi3, self.pi4, self.n, self.m, self.lags)
        alpha = vu.make_alpha(self.n)
        S = np.diag(vu.make_S(self.s))
        self.Q = Q
        self.alpha = alpha
        self.S = S


    def __posterior(self):
        
        """ creates posterior elements """

        inv_Q = np.diag(1 / self.Q)
        alpha_bar = self.alpha + self.T
        self._inv_Q = inv_Q
        self.alpha_bar = alpha_bar
        

    def __parameter_mcmc(self):
        
        """ Gibbs sampler for VEC parameters, depending on prior and error correction type """
        
        # if prior is uninformative and error correction is general
        if self.prior_type == 1 and self.error_correction_type == 1:
            self.__mcmc_uninformative_general()
        # else, if prior is uninformative and error correction is reduced-rank
        elif self.prior_type == 1 and self.error_correction_type == 2:
            self.__mcmc_uninformative_reduced_rank()
        # else, if prior is horseshoe and error correction is general
        elif self.prior_type == 2 and self.error_correction_type == 1:
            self.__mcmc_horseshoe_general()
        # else, if prior is horseshoe and error correction is reduced-rank
        elif self.prior_type == 2 and self.error_correction_type == 2:
            self.__mcmc_horseshoe_reduced_rank()    
        # else, if prior is selection and error correction is general
        elif self.prior_type == 3 and self.error_correction_type == 1:
            self.__mcmc_selection_general()
        # else, if prior is selection and error correction is reduced-rank
        elif self.prior_type == 3 and self.error_correction_type == 2:
            self.__mcmc_selection_reduced_rank()           


    def __mcmc_uninformative_general(self):
        
        """ Gibbs sampler for VEC with uninformative prior and general approach: algorithm 15.1 """
        
        # unpack
        DY = self.DY
        Y_1 = self.Y_1
        Z = self._Z_vec
        ZZ = self._ZZ
        ZDY = self._ZDY
        ZY = self._ZY
        YY = self._YY
        YDY = self._YDY
        YZ = self._YZ      
        inv_Q = self._inv_Q
        alpha_bar = self.alpha_bar
        S = self.S
        n = self.n
        m = self.m
        p = self._p_vec
        k = self._k_vec
        iterations = self.iterations
        burnin = self.burnin
        verbose = self.verbose

        # other prior values
        inv_U = np.diag(np.ones(n*n) / 10)
        
        # preallocate storage space
        mcmc_Xi = np.zeros((n,n,iterations))
        mcmc_Phi = np.zeros((k,n,iterations))
        mcmc_Sigma = np.zeros((n,n,iterations))
        mcmc_chol_Sigma = np.zeros((n,n,iterations))
        mcmc_inv_Sigma = np.zeros((n,n,iterations))
        mcmc_beta = np.zeros((k+n,n,iterations))

        # step 1: set initial values for MCMC algorithm
        Phi = np.zeros((k,n))
        inv_Sigma = np.diag(1 / np.diag(S))
        
        # iterate over iterations
        iteration = 0
        while iteration < (burnin + iterations):

            # step 2: sample xi
            xi, Xi_T, Xi = self.__draw_xi(inv_U, inv_Sigma, YY, YDY, YZ, Phi, n)

            # step 3: sample phi
            phi, Phi = self.__draw_phi(inv_Q, inv_Sigma, ZZ, ZDY, ZY, Xi_T, k, n)
            
            # step 4: sample Sigma
            Sigma, inv_Sigma, chol_Sigma = self.__draw_Sigma(alpha_bar, S, DY, Y_1, Xi_T, Z, Phi)
            
            # convert into VAR model
            B = vvu.vec_to_var(Xi_T, Phi, n, m, p, k)

            # save if burn is exceeded, and display progress bar
            if iteration >= burnin:
                mcmc_Xi[:,:,iteration-burnin] = Xi
                mcmc_Phi[:,:,iteration-burnin] = Phi           
                mcmc_Sigma[:,:,iteration-burnin] = Sigma
                mcmc_chol_Sigma[:,:,iteration-burnin] = chol_Sigma
                mcmc_inv_Sigma[:,:,iteration-burnin] = inv_Sigma
                mcmc_beta[:,:,iteration-burnin] = B
            if verbose:
                cu.progress_bar(iteration, iterations+burnin, 'Model parameters:')
            iteration += 1
            
        # save as attributes
        self.mcmc_Xi = mcmc_Xi
        self.mcmc_Phi = mcmc_Phi
        self.mcmc_Sigma = mcmc_Sigma
        self._mcmc_chol_Sigma = mcmc_chol_Sigma
        self.__mcmc_inv_Sigma = mcmc_inv_Sigma
        self.mcmc_beta = mcmc_beta


    def __mcmc_uninformative_reduced_rank(self):
        
        """ Gibbs sampler for VEC with uninformative prior and reduced-rank approach: algorithm 15.2 """
        
        # unpack
        DY = self.DY
        Y_1 = self.Y_1
        Z = self._Z_vec
        ZZ = self._ZZ
        ZDY = self._ZDY
        ZY = self._ZY
        YY = self._YY
        YDY = self._YDY
        YZ = self._YZ      
        inv_Q = self._inv_Q
        alpha_bar = self.alpha_bar
        S = self.S
        n = self.n
        m = self.m
        p = self._p_vec
        k = self._k_vec
        r = self.r
        iterations = self.iterations
        burnin = self.burnin
        verbose = self.verbose

        # other prior values
        inv_R = np.diag(np.ones(n*r) / 10)
        inv_P = np.diag(np.ones(n*r) / 10)
        
        # preallocate storage space
        mcmc_K = np.zeros((n,r,iterations))
        mcmc_Lamda = np.zeros((n,r,iterations))
        mcmc_Phi = np.zeros((k,n,iterations))
        mcmc_Xi = np.zeros((n,n,iterations))
        mcmc_Sigma = np.zeros((n,n,iterations))
        mcmc_chol_Sigma = np.zeros((n,n,iterations))
        mcmc_inv_Sigma = np.zeros((n,n,iterations))
        mcmc_beta = np.zeros((k+n,n,iterations))

        # step 1: set initial values for MCMC algorithm
        YYZ = np.eye(n)
        Lamda = np.eye(n,r)
        Phi = np.zeros((k,n))
        inv_Sigma = np.diag(1 / np.diag(S))
        
        # iterate over iterations
        iteration = 0
        while iteration < (burnin + iterations):

            # step 2: sample kappa
            kappa, K = self.__draw_kappa(inv_R, Lamda, inv_Sigma, YY, YYZ, n, r)

            # step 3: sample lambda
            lamda, Lamda_T, Lamda, Xi_T, Xi = self.__draw_lamda(inv_P, inv_Sigma, K, YY, YYZ, n, r)
            
            # step 4: sample phi
            phi, Phi = self.__draw_phi(inv_Q, inv_Sigma, ZZ, ZDY, ZY, Xi_T, k, n)
            
            # step 5: sample Sigma
            Sigma, inv_Sigma, chol_Sigma = self.__draw_Sigma(alpha_bar, S, DY, Y_1, Xi_T, Z, Phi)
            YYZ = (YDY - YZ @ Phi) @ inv_Sigma
            
            # convert into VAR model
            B = vvu.vec_to_var(Xi_T, Phi, n, m, p, k)

            # save if burn is exceeded, and display progress bar
            if iteration >= burnin:
                mcmc_K[:,:,iteration-burnin] = K
                mcmc_Lamda[:,:,iteration-burnin] = Lamda
                mcmc_Xi[:,:,iteration-burnin] = Xi
                mcmc_Phi[:,:,iteration-burnin] = Phi           
                mcmc_Sigma[:,:,iteration-burnin] = Sigma
                mcmc_chol_Sigma[:,:,iteration-burnin] = chol_Sigma
                mcmc_inv_Sigma[:,:,iteration-burnin] = inv_Sigma
                mcmc_beta[:,:,iteration-burnin] = B
            if verbose:
                cu.progress_bar(iteration, iterations+burnin, 'Model parameters:')
            iteration += 1
            
        # save as attributes
        self.mcmc_K = mcmc_K
        self.mcmc_Lamda = mcmc_Lamda
        self.mcmc_Xi = mcmc_Xi
        self.mcmc_Phi = mcmc_Phi
        self.mcmc_Sigma = mcmc_Sigma
        self._mcmc_chol_Sigma = mcmc_chol_Sigma
        self.__mcmc_inv_Sigma = mcmc_inv_Sigma
        self.mcmc_beta = mcmc_beta


    def __mcmc_horseshoe_general(self):
        
        """ Gibbs sampler for VEC with horseshoe prior and general approach: algorithm 15.3 """
        
        # unpack
        DY = self.DY
        Y_1 = self.Y_1
        Z = self._Z_vec
        ZZ = self._ZZ
        ZDY = self._ZDY
        ZY = self._ZY
        YY = self._YY
        YDY = self._YDY
        YZ = self._YZ      
        inv_Q = self._inv_Q
        alpha_bar = self.alpha_bar
        S = self.S
        n = self.n
        m = self.m
        p = self._p_vec
        k = self._k_vec
        iterations = self.iterations
        burnin = self.burnin
        verbose = self.verbose

        # other prior values
        a_nu = 1
        a_tau = (n*n+1) / 2
        a_eta = 1
        a_psi = 1
        
        # preallocate storage space
        mcmc_nu = np.zeros((iterations))
        mcmc_tau_2 = np.zeros((iterations))
        mcmc_eta = np.zeros((n,n,iterations))
        mcmc_psi_2 = np.zeros((n,n,iterations))
        mcmc_Xi = np.zeros((n,n,iterations))
        mcmc_Phi = np.zeros((k,n,iterations))
        mcmc_Sigma = np.zeros((n,n,iterations))
        mcmc_chol_Sigma = np.zeros((n,n,iterations))
        mcmc_inv_Sigma = np.zeros((n,n,iterations))
        mcmc_beta = np.zeros((k+n,n,iterations))

        # step 1: set initial values for MCMC algorithm
        tau_2 = 1
        psi_2 = np.ones((n,n))
        Xi = np.ones((n,n))
        Phi = np.zeros((k,n))
        inv_Sigma = np.diag(1 / np.diag(S))
        
        # iterate over iterations
        iteration = 0
        while iteration < (burnin + iterations):
            
            # step 2: sample nu
            nu = self.__draw_nu(a_nu, tau_2)
            
            # step 3: sample tau_2
            tau_2 = self.__draw_tau_2(a_tau, Xi, psi_2, nu)
            
            # step 4: sample eta
            eta = self.__draw_eta(a_eta, psi_2, n)  
            
            # step 5: sample psi_2
            psi_2 = self.__draw_psi_2(a_psi, Xi, tau_2, eta, n)     
            inv_U = np.diag(1 / la.vec(tau_2 * psi_2.T))
            
            # step 6: sample xi
            xi, Xi_T, Xi = self.__draw_xi(inv_U, inv_Sigma, YY, YDY, YZ, Phi, n)

            # step 7: sample phi
            phi, Phi = self.__draw_phi(inv_Q, inv_Sigma, ZZ, ZDY, ZY, Xi_T, k, n)
            
            # step 8: sample Sigma
            Sigma, inv_Sigma, chol_Sigma = self.__draw_Sigma(alpha_bar, S, DY, Y_1, Xi_T, Z, Phi)
            
            # convert into VAR model
            B = vvu.vec_to_var(Xi_T, Phi, n, m, p, k)
            
            # save if burn is exceeded, and display progress bar
            if iteration >= burnin:
                mcmc_nu[iteration-burnin] = nu
                mcmc_tau_2[iteration-burnin] = tau_2
                mcmc_eta[:,:,iteration-burnin] = eta
                mcmc_psi_2[:,:,iteration-burnin] = psi_2                
                mcmc_Xi[:,:,iteration-burnin] = Xi
                mcmc_Phi[:,:,iteration-burnin] = Phi           
                mcmc_Sigma[:,:,iteration-burnin] = Sigma
                mcmc_chol_Sigma[:,:,iteration-burnin] = chol_Sigma
                mcmc_inv_Sigma[:,:,iteration-burnin] = inv_Sigma
                mcmc_beta[:,:,iteration-burnin] = B
            if verbose:
                cu.progress_bar(iteration, iterations+burnin, 'Model parameters:')
            iteration += 1
            
        # save as attributes
        self.mcmc_nu = mcmc_nu
        self.mcmc_tau_2 = mcmc_tau_2
        self.mcmc_eta = mcmc_eta
        self.mcmc_psi_2 = mcmc_psi_2  
        self.mcmc_Xi = mcmc_Xi
        self.mcmc_Phi = mcmc_Phi
        self.mcmc_Sigma = mcmc_Sigma
        self._mcmc_chol_Sigma = mcmc_chol_Sigma
        self.__mcmc_inv_Sigma = mcmc_inv_Sigma
        self.mcmc_beta = mcmc_beta
        

    def __mcmc_horseshoe_reduced_rank(self):
        
        """ Gibbs sampler for VEC with horseshoe prior and reduced-rank approach: algorithm 15.4 """
        
        # unpack
        DY = self.DY
        Y_1 = self.Y_1
        Z = self._Z_vec
        ZZ = self._ZZ
        ZDY = self._ZDY
        ZY = self._ZY
        YY = self._YY
        YDY = self._YDY
        YZ = self._YZ      
        inv_Q = self._inv_Q
        alpha_bar = self.alpha_bar
        S = self.S
        n = self.n
        m = self.m
        p = self._p_vec
        k = self._k_vec
        r = self.r
        iterations = self.iterations
        burnin = self.burnin
        verbose = self.verbose

        # other prior values
        a_nu = 1
        a_tau = n + 1 / 2
        a_eta = 1
        a_psi = (r + 1) / 2
        a_omega = 1
        a_zeta = (r + 1) / 2
        
        # preallocate storage space
        mcmc_nu = np.zeros((r,iterations))
        mcmc_tau_2 = np.zeros((r,iterations))        
        mcmc_eta = np.zeros((n,iterations))
        mcmc_psi_2 = np.zeros((n,iterations))
        mcmc_omega = np.zeros((n,iterations))
        mcmc_zeta_2 = np.zeros((n,iterations))
        mcmc_K = np.zeros((n,r,iterations))
        mcmc_Lamda = np.zeros((n,r,iterations))
        mcmc_Phi = np.zeros((k,n,iterations))
        mcmc_Xi = np.zeros((n,n,iterations))
        mcmc_Sigma = np.zeros((n,n,iterations))
        mcmc_chol_Sigma = np.zeros((n,n,iterations))
        mcmc_inv_Sigma = np.zeros((n,n,iterations))
        mcmc_beta = np.zeros((k+n,n,iterations))

        # step 1: set initial values for MCMC algorithm
        tau_2 = np.ones((r))
        psi_2 = np.ones((n))
        zeta_2 = np.ones((n))
        K = np.eye(n,r)
        Lamda = np.eye(n,r)
        inv_Sigma = np.diag(1 / np.diag(S))        
        YYZ = np.eye(n)

        # iterate over iterations
        iteration = 0
        while iteration < (burnin + iterations):

            # step 2: sample nu
            nu = self.__draw_nu_j(a_nu, tau_2, r)

            # step 3: sample tau_2
            tau_2 = self.__draw_tau_2_j(a_tau, K, Lamda, psi_2, zeta_2, nu, r)

            # step 4: sample eta
            eta = self.__draw_eta_i(a_eta, psi_2, n)

            # step 5: sample psi_2
            psi_2 = self.__draw_psi_2_i(a_psi, K, tau_2, eta, n)          
            inv_R = np.diag(1 / la.vec(psi_2.reshape(-1,1) @ tau_2.reshape(1,-1)))
            
            # step 6: sample omega
            omega = self.__draw_omega_i(a_omega, zeta_2, n)
            
            # step 7: sample zeta_2
            zeta_2 = self.__draw_zeta_2_i(a_zeta, Lamda, tau_2, omega, n)          
            inv_P = np.diag(1 / la.vec((zeta_2.reshape(-1,1) @ tau_2.reshape(1,-1)).T))            
            
            # step 8: sample kappa
            kappa, K = self.__draw_kappa(inv_R, Lamda, inv_Sigma, YY, YYZ, n, r)

            # step 9: sample lambda
            lamda, Lamda_T, Lamda, Xi_T, Xi = self.__draw_lamda(inv_P, inv_Sigma, K, YY, YYZ, n, r)
            
            # step 10: sample phi
            phi, Phi = self.__draw_phi(inv_Q, inv_Sigma, ZZ, ZDY, ZY, Xi_T, k, n)
            
            # step 11: sample Sigma
            Sigma, inv_Sigma, chol_Sigma = self.__draw_Sigma(alpha_bar, S, DY, Y_1, Xi_T, Z, Phi)
            YYZ = (YDY - YZ @ Phi) @ inv_Sigma
            
            # convert into VAR model
            B = vvu.vec_to_var(Xi_T, Phi, n, m, p, k)

            # save if burn is exceeded, and display progress bar
            if iteration >= burnin:
                mcmc_nu[:,iteration-burnin] = nu
                mcmc_tau_2[:,iteration-burnin] = tau_2    
                mcmc_eta[:,iteration-burnin] = eta
                mcmc_psi_2[:,iteration-burnin] = psi_2
                mcmc_omega[:,iteration-burnin] = omega
                mcmc_zeta_2[:,iteration-burnin] = zeta_2
                mcmc_K[:,:,iteration-burnin] = K
                mcmc_Lamda[:,:,iteration-burnin] = Lamda
                mcmc_Xi[:,:,iteration-burnin] = Xi
                mcmc_Phi[:,:,iteration-burnin] = Phi           
                mcmc_Sigma[:,:,iteration-burnin] = Sigma
                mcmc_chol_Sigma[:,:,iteration-burnin] = chol_Sigma
                mcmc_inv_Sigma[:,:,iteration-burnin] = inv_Sigma
                mcmc_beta[:,:,iteration-burnin] = B
            if verbose:
                cu.progress_bar(iteration, iterations+burnin, 'Model parameters:')
            iteration += 1

        # save as attributes
        self.mcmc_nu = mcmc_nu
        self.mcmc_tau_2 = mcmc_tau_2
        self.mcmc_eta = mcmc_eta
        self.mcmc_psi_2 = mcmc_psi_2
        self.mcmc_omega = mcmc_omega
        self.mcmc_zeta_2 = mcmc_zeta_2
        self.mcmc_K = mcmc_K
        self.mcmc_Lamda = mcmc_Lamda
        self.mcmc_Xi = mcmc_Xi
        self.mcmc_Phi = mcmc_Phi
        self.mcmc_Sigma = mcmc_Sigma
        self._mcmc_chol_Sigma = mcmc_chol_Sigma
        self.__mcmc_inv_Sigma = mcmc_inv_Sigma
        self.mcmc_beta = mcmc_beta
        

    def __mcmc_selection_general(self):
        
        """ Gibbs sampler for VEC with selection prior and general approach: algorithm 15.5 """
        
        # unpack
        DY = self.DY
        Y_1 = self.Y_1
        Z = self._Z_vec
        ZZ = self._ZZ
        ZDY = self._ZDY
        ZY = self._ZY
        YY = self._YY
        YDY = self._YDY
        YZ = self._YZ      
        inv_Q = self._inv_Q
        alpha_bar = self.alpha_bar
        S = self.S
        n = self.n
        m = self.m
        p = self._p_vec
        k = self._k_vec
        iterations = self.iterations
        burnin = self.burnin
        verbose = self.verbose

        # other prior values
        mu = 0.5
        tau_1_2 = 10
        tau_1 = tau_1_2 ** 0.5
        tau_2_2 = 0.01
        tau_2 = tau_2_2 ** 0.5
        temp_1 = mu / tau_1
        temp_2 = - 0.5 / tau_1_2
        temp_3 = (1 - mu) / tau_2
        temp_4 = - 0.5 / tau_2_2
    
        # preallocate storage space
        mcmc_delta = np.zeros((n*n,iterations))
        mcmc_Xi = np.zeros((n,n,iterations))
        mcmc_Phi = np.zeros((k,n,iterations))
        mcmc_Sigma = np.zeros((n,n,iterations))
        mcmc_chol_Sigma = np.zeros((n,n,iterations))
        mcmc_inv_Sigma = np.zeros((n,n,iterations))
        mcmc_beta = np.zeros((k+n,n,iterations))

        # step 1: set initial values for MCMC algorithm
        xi = np.ones((n*n))
        Phi = np.zeros((k,n))
        inv_Sigma = np.diag(1 / np.diag(S))
        
        # iterate over iterations
        iteration = 0
        while iteration < (burnin + iterations):
            
            # step 2: sample delta
            delta = self.__draw_delta(xi, temp_1, temp_2, temp_3, temp_4, n)
            inv_U = np.diag(1 / (delta * tau_1_2 + (1 - delta) * tau_2_2))

            # step 3: sample xi
            xi, Xi_T, Xi = self.__draw_xi(inv_U, inv_Sigma, YY, YDY, YZ, Phi, n)

            # step 4: sample phi
            phi, Phi = self.__draw_phi(inv_Q, inv_Sigma, ZZ, ZDY, ZY, Xi_T, k, n)
            
            # step 5: sample Sigma
            Sigma, inv_Sigma, chol_Sigma = self.__draw_Sigma(alpha_bar, S, DY, Y_1, Xi_T, Z, Phi)
            
            # convert into VAR model
            B = vvu.vec_to_var(Xi_T, Phi, n, m, p, k)

            # save if burn is exceeded, and display progress bar
            if iteration >= burnin:
                mcmc_delta[:,iteration-burnin] = delta
                mcmc_Xi[:,:,iteration-burnin] = Xi
                mcmc_Phi[:,:,iteration-burnin] = Phi           
                mcmc_Sigma[:,:,iteration-burnin] = Sigma
                mcmc_chol_Sigma[:,:,iteration-burnin] = chol_Sigma
                mcmc_inv_Sigma[:,:,iteration-burnin] = inv_Sigma
                mcmc_beta[:,:,iteration-burnin] = B
            if verbose:
                cu.progress_bar(iteration, iterations+burnin, 'Model parameters:')
            iteration += 1     
        # save as attributes
        self.mcmc_delta = mcmc_delta
        self.mcmc_Xi = mcmc_Xi
        self.mcmc_Phi = mcmc_Phi
        self.mcmc_Sigma = mcmc_Sigma
        self._mcmc_chol_Sigma = mcmc_chol_Sigma
        self.__mcmc_inv_Sigma = mcmc_inv_Sigma
        self.mcmc_beta = mcmc_beta


    def __mcmc_selection_reduced_rank(self):
        
        """ Gibbs sampler for VEC with selection prior and reduced-rank approach: algorithm 15.6 """
        
        # unpack
        DY = self.DY
        Y_1 = self.Y_1
        Z = self._Z_vec
        ZZ = self._ZZ
        ZDY = self._ZDY
        ZY = self._ZY
        YY = self._YY
        YDY = self._YDY
        YZ = self._YZ      
        inv_Q = self._inv_Q
        alpha_bar = self.alpha_bar
        S = self.S
        n = self.n
        m = self.m
        p = self._p_vec
        k = self._k_vec
        r = self.r
        iterations = self.iterations
        burnin = self.burnin
        verbose = self.verbose

        # other prior values
        mu = 0.5
        tau_1_2 = 10
        tau_1 = tau_1_2 ** 0.5
        tau_2_2 = 0.01
        tau_2 = tau_2_2 ** 0.5
        temp_1 = mu / tau_1
        temp_2 = - 0.5 / tau_1_2
        temp_3 = (1 - mu) / tau_2
        temp_4 = - 0.5 / tau_2_2
        
        # preallocate storage space
        mcmc_chi = np.zeros((n*r,iterations))
        mcmc_gamma = np.zeros((n*r,iterations))
        mcmc_K = np.zeros((n,r,iterations))
        mcmc_Lamda = np.zeros((n,r,iterations))
        mcmc_Phi = np.zeros((k,n,iterations))
        mcmc_Xi = np.zeros((n,n,iterations))
        mcmc_Sigma = np.zeros((n,n,iterations))
        mcmc_chol_Sigma = np.zeros((n,n,iterations))
        mcmc_inv_Sigma = np.zeros((n,n,iterations))
        mcmc_beta = np.zeros((k+n,n,iterations))

        # step 1: set initial values for MCMC algorithm
        kappa = np.ones((n*r))
        lamda = np.ones((n*r))
        Phi = np.zeros((k,n))
        inv_Sigma = np.diag(1 / np.diag(S))
        YYZ = np.eye(n)
        Lamda = np.eye(n,r)
        Phi = np.zeros((k,n))
        inv_Sigma = np.diag(1 / np.diag(S))
        
        # iterate over iterations
        iteration = 0
        while iteration < (burnin + iterations):
            
            # step 2: sample chi
            chi = self.__draw_chi(kappa, temp_1, temp_2, temp_3, temp_4, n, r)
            inv_R = np.diag(1 / (chi * tau_1_2 + (1 - chi) * tau_2_2))            
            
            # step 3: sample kappa
            kappa, K = self.__draw_kappa(inv_R, Lamda, inv_Sigma, YY, YYZ, n, r)

            # step 4: sample gamma
            gama = self.__draw_gamma(lamda, temp_1, temp_2, temp_3, temp_4, n, r)
            inv_P = np.diag(1 / (gama * tau_1_2 + (1 - gama) * tau_2_2))

            # step 5: sample lambda
            lamda, Lamda_T, Lamda, Xi_T, Xi = self.__draw_lamda(inv_P, inv_Sigma, K, YY, YYZ, n, r)
            
            # step 6: sample phi
            phi, Phi = self.__draw_phi(inv_Q, inv_Sigma, ZZ, ZDY, ZY, Xi_T, k, n)
            
            # step 7: sample Sigma
            Sigma, inv_Sigma, chol_Sigma = self.__draw_Sigma(alpha_bar, S, DY, Y_1, Xi_T, Z, Phi)
            YYZ = (YDY - YZ @ Phi) @ inv_Sigma
            
            # convert into VAR model
            B = vvu.vec_to_var(Xi_T, Phi, n, m, p, k)

            # save if burn is exceeded, and display progress bar
            if iteration >= burnin:
                mcmc_chi[:,iteration-burnin] = chi
                mcmc_gamma[:,iteration-burnin] = gama
                mcmc_K[:,:,iteration-burnin] = K
                mcmc_Lamda[:,:,iteration-burnin] = Lamda
                mcmc_Xi[:,:,iteration-burnin] = Xi
                mcmc_Phi[:,:,iteration-burnin] = Phi           
                mcmc_Sigma[:,:,iteration-burnin] = Sigma
                mcmc_chol_Sigma[:,:,iteration-burnin] = chol_Sigma
                mcmc_inv_Sigma[:,:,iteration-burnin] = inv_Sigma
                mcmc_beta[:,:,iteration-burnin] = B
            if verbose:
                cu.progress_bar(iteration, iterations+burnin, 'Model parameters:')
            iteration += 1
            
        # save as attributes
        self.mcmc_chi = mcmc_chi
        self.mcmc_gamma = mcmc_gamma
        self.mcmc_K = mcmc_K
        self.mcmc_Lamda = mcmc_Lamda
        self.mcmc_Xi = mcmc_Xi
        self.mcmc_Phi = mcmc_Phi
        self.mcmc_Sigma = mcmc_Sigma
        self._mcmc_chol_Sigma = mcmc_chol_Sigma
        self.__mcmc_inv_Sigma = mcmc_inv_Sigma
        self.mcmc_beta = mcmc_beta


    def __draw_xi(self, inv_U, inv_Sigma, YY, YDY, YZ, Phi, n):
        
        """ draw xi from its conditional posterior defined in (5.15.29) """
        
        inv_U_bar = inv_U + np.kron(inv_Sigma, YY)
        d_bar_temp = la.vec((YDY - YZ @ Phi) @ inv_Sigma)
        xi = rng.efficient_multivariate_normal(d_bar_temp, inv_U_bar)
        Xi_T = np.reshape(xi,[n,n],order='F')
        Xi = Xi_T.T
        return xi, Xi_T, Xi


    def __draw_phi(self, inv_Q, inv_Sigma, ZZ, ZDY, ZY, Xi_T, k, n):
        
        """ draw phi from its conditional posterior defined in (5.15.22) """
        
        inv_Q_bar = inv_Q + np.kron(inv_Sigma, ZZ)
        f_bar_temp = la.vec((ZDY - ZY @ Xi_T) @ inv_Sigma)
        phi = rng.efficient_multivariate_normal(f_bar_temp, inv_Q_bar)
        Phi = np.reshape(phi,[k,n],order='F')
        return phi, Phi


    def __draw_Sigma(self, alpha_bar, S, DY, Y_1, Xi_T, Z, Phi):
        
        """ draw Sigma from its conditional posterior defined in (5.15.25) """
        
        residuals = DY - Y_1 @ Xi_T - Z @ Phi
        S_bar = S + residuals.T @ residuals
        Sigma = rng.inverse_wishart(alpha_bar, S_bar)
        inv_Sigma = la.invert_spd_matrix(Sigma)
        chol_Sigma = la.cholesky_nspd(Sigma)
        return Sigma, inv_Sigma, chol_Sigma


    def __draw_kappa(self, inv_R, Lamda, inv_Sigma, YY, YYZ, n, r):
        
        """ draw kappa from its conditional posterior defined in (5.15.35) """        

        inv_R_bar = inv_R + np.kron(Lamda.T @ inv_Sigma @ Lamda, YY)
        g_bar_temp = la.vec(YYZ @ Lamda)
        kappa = rng.efficient_multivariate_normal(g_bar_temp, inv_R_bar)
        K = np.reshape(kappa,[n,r],order='F')
        return kappa, K     
        
        
    def __draw_lamda(self, inv_P, inv_Sigma, K, YY, YYZ, n, r):
        
        """ draw lamda from its conditional posterior defined in (5.15.38) """        

        inv_P_bar = inv_P + np.kron(inv_Sigma, K.T @ YY @ K)
        h_bar_temp = la.vec(K.T @ YYZ)
        lamda = rng.efficient_multivariate_normal(h_bar_temp, inv_P_bar)
        Lamda_T = np.reshape(lamda,[r,n],order='F')
        Lamda = Lamda_T.T
        Xi_T = K @ Lamda_T
        Xi = Xi_T.T
        return lamda, Lamda_T, Lamda, Xi_T, Xi


    def __draw_nu(self, a_nu, tau_2):
        
        """ draw nu from its conditional posterior defined in (5.15.50) """        

        b_nu = 1 / tau_2 + 1
        nu = rng.inverse_gamma(a_nu, b_nu)
        return nu


    def __draw_tau_2(self, a_tau, Xi, psi_2, nu):

        """ draw tau_2 from its conditional posterior defined in (5.15.47) """ 

        b_tau = np.sum((Xi * Xi) / psi_2) / 2 + 1 / nu
        tau_2 = rng.inverse_gamma(a_tau, b_tau)
        return tau_2
        
        
    def __draw_eta(self, a_eta, psi_2, n):
        
        """ draw eta from its conditional posterior defined in (5.15.56) """        

        b_eta = 1 / psi_2 + 1
        eta = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                eta[i,j] = rng.inverse_gamma(a_eta, b_eta[i,j])
        return eta
    

    def __draw_psi_2(self, a_psi, Xi, tau_2, eta, n):
        
        """ draw psi_2 from its conditional posterior defined in (5.15.53) """        

        b_psi = (Xi * Xi) / (2 * tau_2) + 1 / eta
        psi_2 = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                psi_2[i,j] = rng.inverse_gamma(a_psi, b_psi[i,j])
        return psi_2
    

    def __draw_nu_j(self, a_nu, tau_2, r):
        
        """ draw nu_j from its conditional posterior defined in (5.15.71) """        
        
        nu = np.zeros((r))
        for j in range(r):
            b_nu = 1 / tau_2[j] + 1
            nu[j] = rng.inverse_gamma(a_nu, b_nu)
        return nu


    def __draw_tau_2_j(self, a_tau, K, Lamda, psi_2, zeta_2, nu, r):
        
        """ draw tau_2_j from its conditional posterior defined in (5.15.68) """

        tau_2 = np.zeros((r))
        for j in range(r):  
            b_tau = np.sum((K[:,j] * K[:,j]) / psi_2 + (Lamda[:,j] * Lamda[:,j]) / zeta_2) / 2 + 1 / nu[j]
            tau_2[j] = rng.inverse_gamma(a_tau, b_tau)
        return tau_2


    def __draw_eta_i(self, a_eta, psi_2, n):

        """ draw eta_i from its conditional posterior defined in (5.15.77) """        

        eta = np.zeros((n))
        for i in range(n):
            b_eta = 1 / psi_2[i] + 1
            eta[i] = rng.inverse_gamma(a_eta, b_eta)
        return eta


    def __draw_psi_2_i(self, a_psi, K, tau_2, eta, n):
        
        """ draw psi_2_i from its conditional posterior defined in (5.15.74) """

        psi_2 = np.zeros((n))
        for i in range(n):
            b_psi = np.sum((K[i,:] * K[i,:]) / tau_2) / 2 + 1 / eta[i]
            psi_2[i] = rng.inverse_gamma(a_psi, b_psi)
        return psi_2


    def __draw_omega_i(self, a_omega, zeta_2, n):

        """ draw omega_i from its conditional posterior defined in (5.15.83) """        

        omega = np.zeros((n))
        for i in range(n):
            b_omega = 1 / zeta_2[i] + 1
            omega[i] = rng.inverse_gamma(a_omega, b_omega)
        return omega

 
    def __draw_zeta_2_i(self, a_zeta, Lamda, tau_2, omega, n):
        
        """ draw zeta_2_i from its conditional posterior defined in (5.15.80) """

        zeta_2 = np.zeros((n))
        for i in range(n):
            b_zeta = np.sum((Lamda[i,:] * Lamda[i,:]) / tau_2) / 2 + 1 / omega[i]
            zeta_2[i] = rng.inverse_gamma(a_zeta, b_zeta)
        return zeta_2 


    def __draw_delta(self, xi, temp_1, temp_2, temp_3, temp_4, n):
        
        """ draw delta from its conditional posterior defined in (5.15.89) """
        
        xi_2 = xi * xi
        a_1 = temp_1 * np.exp(temp_2 * xi_2)
        a_2 = temp_3 * np.exp(temp_4 * xi_2)
        mu_xi = a_1 / (a_1 + a_2)
        delta = rng.multivariate_bernoulli(mu_xi, n*n)     
        return delta


    def __draw_chi(self, kappa, temp_1, temp_2, temp_3, temp_4, n, r):
        
        """ draw chi from its conditional posterior defined in (5.15.97) """
        
        kappa_2 = kappa * kappa
        a_1 = temp_1 * np.exp(temp_2 * kappa_2)
        a_2 = temp_3 * np.exp(temp_4 * kappa_2)
        mu_kappa = a_1 / (a_1 + a_2)
        chi = rng.multivariate_bernoulli(mu_kappa, n*r)     
        return chi


    def __draw_gamma(self, lamda, temp_1, temp_2, temp_3, temp_4, n, r):
        
        """ draw gamma from its conditional posterior defined in (5.15.100) """
        
        lamda_2 = lamda * lamda
        a_1 = temp_1 * np.exp(temp_2 * lamda_2)
        a_2 = temp_3 * np.exp(temp_4 * lamda_2)
        mu_lamda = a_1 / (a_1 + a_2)
        gama = rng.multivariate_bernoulli(mu_lamda, n*r)     
        return gama


    def __parameter_estimates(self):
        
        """
        point estimates and credibility intervals for model parameters
        use empirical quantiles from MCMC algorithm
        """
        
        # common parameters
        mcmc_Xi = self.mcmc_Xi
        mcmc_Phi = self.mcmc_Phi        
        mcmc_Sigma = self.mcmc_Sigma
        mcmc_beta = self.mcmc_beta
        credibility_level = self.credibility_level
        k_vec, k, n = self._k_vec, self.k, self.n
        Xi_estimates = np.zeros((n,n,4))
        Phi_estimates = np.zeros((k_vec,n,4))
        beta_estimates = np.zeros((k,n,4))
        Xi_estimates[:,:,:3] = vu.posterior_estimates(mcmc_Xi, credibility_level)
        Xi_estimates[:,:,3] = np.std(mcmc_Xi,axis=2)
        Phi_estimates[:,:,:3] = vu.posterior_estimates(mcmc_Phi, credibility_level)
        Phi_estimates[:,:,3] = np.std(mcmc_Phi,axis=2)
        beta_estimates[:,:,:3] = vu.posterior_estimates(mcmc_beta, credibility_level)
        beta_estimates[:,:,3] = np.std(mcmc_beta,axis=2)
        Sigma_estimates = np.quantile(mcmc_Sigma,0.5,axis=2)
        self.Xi_estimates = Xi_estimates
        self.Phi_estimates = Phi_estimates
        self.beta_estimates = beta_estimates
        self.Sigma_estimates = Sigma_estimates
        # model-specific parameters
        if self.prior_type == 1 and self.error_correction_type == 2:
            r = self.r  
            mcmc_K = self.mcmc_K
            K_estimates = np.zeros((n,r,4))
            K_estimates[:,:,:3] = vu.posterior_estimates(mcmc_K, credibility_level)
            K_estimates[:,:,3] = np.std(mcmc_K,axis=2)
            mcmc_Lamda = self.mcmc_Lamda
            Lamda_estimates = np.zeros((n,r,4))
            Lamda_estimates[:,:,:3] = vu.posterior_estimates(mcmc_Lamda, credibility_level)
            Lamda_estimates[:,:,3] = np.std(mcmc_Lamda,axis=2)
            self.K_estimates = K_estimates
            self.Lamda_estimates = Lamda_estimates        
        elif self.prior_type == 2 and self.error_correction_type == 1:
            mcmc_tau_2 = self.mcmc_tau_2
            tau_2_estimates = np.quantile(mcmc_tau_2,0.5)
            mcmc_psi_2 = self.mcmc_psi_2
            psi_2_estimates = np.zeros((n,n,4))
            psi_2_estimates[:,:,:3] = vu.posterior_estimates(mcmc_psi_2, credibility_level)
            psi_2_estimates[:,:,3] = np.std(mcmc_psi_2,axis=2) 
            self.tau_2_estimates = tau_2_estimates
            self.psi_2_estimates = psi_2_estimates
        elif self.prior_type == 2 and self.error_correction_type == 2:
            r = self.r
            mcmc_tau_2 = self.mcmc_tau_2
            tau_2_estimates = np.quantile(mcmc_tau_2,0.5,axis=1)
            mcmc_psi_2 = self.mcmc_psi_2
            psi_2_estimates = np.quantile(mcmc_psi_2,0.5,axis=1)
            mcmc_zeta_2 = self.mcmc_zeta_2
            zeta_2_estimates = np.quantile(mcmc_zeta_2,0.5,axis=1)
            mcmc_K = self.mcmc_K
            K_estimates = np.zeros((n,r,4))
            K_estimates[:,:,:3] = vu.posterior_estimates(mcmc_K, credibility_level)
            K_estimates[:,:,3] = np.std(mcmc_K,axis=2)
            mcmc_Lamda = self.mcmc_Lamda
            Lamda_estimates = np.zeros((n,r,4))
            Lamda_estimates[:,:,:3] = vu.posterior_estimates(mcmc_Lamda, credibility_level)
            Lamda_estimates[:,:,3] = np.std(mcmc_Lamda,axis=2)
            self.tau_2_estimates = tau_2_estimates
            self.psi_2_estimates = psi_2_estimates
            self.zeta_2_estimates = zeta_2_estimates
            self.K_estimates = K_estimates
            self.Lamda_estimates = Lamda_estimates
        elif self.prior_type == 3 and self.error_correction_type == 1: 
            mcmc_delta = self.mcmc_delta
            delta_estimates = np.reshape(np.mean(mcmc_delta,1),(n,n),order='F').T   
            self.delta_estimates = delta_estimates
        elif self.prior_type == 3 and self.error_correction_type == 2:
            r = self.r  
            mcmc_chi = self.mcmc_chi
            chi_estimates = np.reshape(np.mean(mcmc_chi,1),(n,r),order='F')              
            mcmc_K = self.mcmc_K
            K_estimates = np.zeros((n,r,4))
            K_estimates[:,:,:3] = vu.posterior_estimates(mcmc_K, credibility_level)
            K_estimates[:,:,3] = np.std(mcmc_K,axis=2)
            mcmc_gamma = self.mcmc_gamma
            gamma_estimates = np.reshape(np.mean(mcmc_gamma,1),(r,n),order='F').T            
            mcmc_Lamda = self.mcmc_Lamda
            Lamda_estimates = np.zeros((n,r,4))
            Lamda_estimates[:,:,:3] = vu.posterior_estimates(mcmc_Lamda, credibility_level)
            Lamda_estimates[:,:,3] = np.std(mcmc_Lamda,axis=2)
            self.chi_estimates = chi_estimates
            self.K_estimates = K_estimates
            self.gamma_estimates = gamma_estimates
            self.Lamda_estimates = Lamda_estimates            

            