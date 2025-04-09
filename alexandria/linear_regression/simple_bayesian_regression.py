# imports
import numpy as np
import numpy.linalg as nla
import scipy.optimize as sco
import alexandria.math.linear_algebra as la
import alexandria.math.math_utilities as mu
import alexandria.math.stat_utilities as su
import alexandria.processor.input_utilities as iu
import alexandria.console.console_utilities as cu
import alexandria.linear_regression.regression_utilities as ru
from alexandria.linear_regression.linear_regression import LinearRegression
from alexandria.linear_regression.bayesian_regression import BayesianRegression


class SimpleBayesianRegression(LinearRegression, BayesianRegression):
    
    
    """    
    Simplest Bayesian linear regression, developed in section 9.2
    
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
   
    b_exogenous : float or ndarray of shape (n_regressors,), default = 0
        prior mean for regressors
   
    V_exogenous : float or ndarray of shape (n_regressors,), default = 1
        prior variance for regressors (positive)
   
    b_constant : float, default = 0
        prior mean for constant term
   
    V_constant : float, default = 1
        prior variance for constant (positive)
    
    b_trend : float, default = 0
        prior mean for trend
   
    V_trend : float, default = 1
        prior variance for trend (positive)
    
    b_quadratic_trend : float, default = 0
        prior mean for quadratic_trend
   
    V_quadratic_trend : float, default = 1
        prior variance for quadratic_trend (positive)
        
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
   
    b_exogenous : float or ndarray of shape (n_regressors,)
        prior mean for regressors
   
    V_exogenous : float or ndarray of shape (n_regressors,)
        prior variance for regressors (positive)
   
    b_constant : float
        prior mean for constant term
   
    V_constant : float
        prior variance for constant (positive)
    
    b_trend : float
        prior mean for trend
   
    V_trend : float
        prior variance for trend (positive)
    
    b_quadratic_trend : float
        prior mean for quadratic trend
   
    V_quadratic_trend : float
        prior variance for quadratic trend (positive)
        
    hyperparameter_optimization : bool
        if True, performs hyperparameter optimization by marginal likelihood
        
    optimization_type : int, 1 or 2
        if 1, simple optimization (scalar v); if 2, full optimization (vector V)  
        
    b : ndarray of shape (k,)
        prior mean, defined in (3.9.10)
   
    V : ndarray of shape (k,k)
        prior variance, defined in (3.9.10)        
    
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
   
    sigma : float
        residual variance, defined in (3.9.1)
   
    b_bar : ndarray of shape (k,)
        posterior mean, defined in (3.9.14)
   
    V_bar : ndarray of shape (k,k)
        posterior variance, defined in (3.9.14)    
    
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
   
    m_y : float
        log10 marginal likelihood
   
   
    Methods
    ----------
    estimate
    fit_and_residuals    
    forecast
    forecast_evaluation
    marginal_likelihood

    """
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------    
    
    
    def __init__(self, endogenous, exogenous, constant = True, trend = False, 
                 quadratic_trend = False, b_exogenous = 0, V_exogenous = 1, 
                 b_constant = 0, V_constant = 1, b_trend = 0, V_trend = 1,
                 b_quadratic_trend = 0, V_quadratic_trend = 1, 
                 hyperparameter_optimization = False, optimization_type = 1,
                 credibility_level = 0.95, verbose = False):
        
        """
        constructor for the SimpleBayesianRegression class
        """
        
        self.endogenous = endogenous
        self.exogenous = exogenous
        self.constant = constant
        self.trend = trend
        self.quadratic_trend = quadratic_trend
        self.b_exogenous = b_exogenous
        self.V_exogenous = V_exogenous
        self.b_constant = b_constant
        self.V_constant = V_constant
        self.b_trend = b_trend
        self.V_trend = V_trend
        self.b_quadratic_trend = b_quadratic_trend
        self.V_quadratic_trend = V_quadratic_trend   
        self.hyperparameter_optimization = hyperparameter_optimization
        self.optimization_type = optimization_type
        self.credibility_level = credibility_level
        self.verbose = verbose  
        # make regressors
        self._make_regressors()    
    
    
    def estimate(self):
        
        """
        estimate()
        generates posterior estimates for linear regression model parameters beta and sigma
        
        parameters:
        none
        
        returns:
        none    
        """
        
        # obtain sigma estimates
        self.__get_sigma()
        # optimize hyperparameters, if applicable
        self.__optimize_hyperparameters()
        # define prior values
        self._prior()
        # define posterior values
        self.__posterior()
        # obtain posterior estimates for regression parameters
        self.__parameter_estimates()


    def forecast(self, X_hat, credibility_level):
        
        """
        forecast(self, X_hat, credibility_level)
        predictions for the linear regression model using (3.10.4)

        parameters:
        X_hat : ndarray of shape (m,k)
            array of predictors
        credibility_level : float
            credibility level for predictions (between 0 and 1)

        returns:
        estimates_forecasts : ndarray of shape (m,3)
            posterior estimates for predictions
            column 1: interval lower bound; column 2: median; 
            column 3: interval upper bound
        """
        
        # unpack
        verbose = self.verbose
        b_bar = self.b_bar
        sigma = self.sigma
        V_bar = self.V_bar    
        n = self.n
        constant = self.constant
        trend = self.trend
        quadratic_trend = self.quadratic_trend
        # add constant and trends if included
        X_hat = ru.add_intercept_and_trends(X_hat, constant, trend, quadratic_trend, n)
        # obtain forecast mean and variance, defined in (3.10.4)
        m = X_hat.shape[0]
        mean = X_hat @ b_bar
        variance = sigma * np.identity(m) + X_hat @ V_bar @ X_hat.T
        standard_deviation = np.sqrt(np.diag(variance))
        # if verbose, display progress bar
        if verbose:
            cu.progress_bar_complete('Predictions:')
        # initiate estimate storage; 3 columns: lower bound, median, upper bound
        forecast_estimates = np.zeros((m,3))
        # critical value of normal distribution for credibility level
        Z = su.normal_icdf((1 + credibility_level) / 2)
        # fill estimates
        forecast_estimates[:,0] = mean
        forecast_estimates[:,1] = mean - Z * standard_deviation
        forecast_estimates[:,2] = mean + Z * standard_deviation
        # save as attributes
        self.X_hat = X_hat
        self.m = m
        self.__forecast_mean = mean
        self.__forecast_variance = variance
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
        forecast_mean = self.__forecast_mean
        forecast_variance = self.__forecast_variance
        forecast_estimates = self.forecast_estimates
        m = self.m
        # obtain regular forecast evaluation criteria
        y_hat = forecast_estimates[:,0]
        standard_evaluation_criteria = ru.forecast_evaluation_criteria(y_hat, y)        
        # obtain Bayesian forecast evaluation criteria
        bayesian_evaluation_criteria = self.__bayesian_forecast_evaluation_criteria(y, forecast_mean, forecast_variance, m)   
        # merge dictionaries
        forecast_evaluation_criteria = iu.concatenate_dictionaries(standard_evaluation_criteria, bayesian_evaluation_criteria)        
        # save as attributes
        self.forecast_evaluation_criteria = forecast_evaluation_criteria    
        

    def marginal_likelihood(self):
        
        """ 
        log10 marginal likelihood, defined in (3.10.20)
        
        parameters:
        none
        
        returns:
        m_y: float
            log10 marginal likelihood
        """

        # unpack
        sigma = self.sigma
        n = self.n
        y = self.y
        b = self.b
        V = self.V
        inv_sigma_XX = self.__inv_sigma_XX
        inv_V_b = self._inv_V_b
        inv_V_bar = self.__inv_V_bar
        b_bar = self.b_bar
        # evaluate the log marginal likelihood from equation (3.10.20)
        term_1 = -(n/2) * np.log(2 * np.pi * sigma)
        term_2 = -0.5 * la.stable_determinant(V @ inv_sigma_XX)
        term_3 = -0.5 * (y @ y / sigma + b @ inv_V_b - b_bar @ inv_V_bar @ b_bar)
        log_f_y = term_1 + term_2 + term_3
        # convert to log10
        m_y = log_f_y / np.log(10)
        # save as attributes
        self.m_y = m_y
        return m_y  

    
    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------   
    
   
    def __get_sigma(self):    

        """ generates sigma defined in (3.9.7) along with associated parameters """   
        
        # unpack
        XX = self._XX
        Xy = self._Xy
        # obtain maximum likelihood estimates for sigma, and derived hyperparameters
        _, sigma = self._ols_regression()
        inv_sigma_XX = XX / sigma
        inv_sigma_Xy = Xy / sigma 
        # save as attributes
        self.sigma = sigma
        self.__inv_sigma_XX = inv_sigma_XX
        self.__inv_sigma_Xy = inv_sigma_Xy
    
   
    def __optimize_hyperparameters(self):
        
        """ optimize hyperparameter V with marginal likelihood, either scalar v or vector V """
        
        if self.hyperparameter_optimization:
            # unpack
            optimization_type = self.optimization_type
            constant = self.constant
            trend = self.trend
            quadratic_trend = self.quadratic_trend       
            verbose = self.verbose
            k = self.k
            # estimate prior elements to get proper estimate of b
            self._prior()
            # optimize: in the simplest case,  V = vI so only scalar v is optimized
            if optimization_type == 1:
                # initial value of optimizer
                v_0 = np.array([1])
                # bounds for parameter values
                bound = [(mu.eps(), 1000)]
                # optimize
                result = sco.minimize(self.__negative_likelihood_simple_V, \
                                      v_0, bounds = bound, method='L-BFGS-B')
                # update hyperparameters
                self.V_constant = result.x[0]
                self.V_trend = result.x[0]
                self.V_quadratic = result.x[0]
                self.V_exogenous = result.x[0]
            # in the second case, V is diagonal from vector v, so vector v is optimized
            elif optimization_type == 2:
                # initial value of optimizer
                V_0 = np.ones(k)
                # bounds for parameter values
                bound = [(mu.eps(), 1000)] * k
                # optimize
                result = sco.minimize(self.__negative_likelihood_full_V, \
                          V_0, bounds = bound, method='L-BFGS-B')
                # save as attribute
                self.V_constant = result.x[0]
                self.V_trend = result.x[0 + constant]
                self.V_quadratic_trend = result.x[0 + constant + trend]
                self.V_exogenous = result.x[0 + constant + trend + quadratic_trend:]
            # if verbose, display progress bar and success/failure of optimization
            if verbose:
                cu.progress_bar_complete('hyperparameter optimization:')
                cu.optimization_completion(result.success)
  
         
    def __posterior(self):
        
        """ creates posterior parameters b_bar and V_bar defined in (3.9.14) """
        
        # unpack
        inv_sigma_XX = self.__inv_sigma_XX
        inv_sigma_Xy = self.__inv_sigma_Xy
        inv_V = self._inv_V
        inv_V_b = self._inv_V_b
        verbose = self.verbose       
        # V_bar, defined in (3.9.14)
        inv_V_bar = inv_V + inv_sigma_XX
        V_bar = la.invert_spd_matrix(inv_V_bar)
        # b_bar, defined in (3.9.14)
        b_bar = V_bar @ (inv_V_b + inv_sigma_Xy)
        # if verbose, display progress bar
        if verbose:
            cu.progress_bar_complete('Model parameters:')
        # save as attributes
        self.__inv_V_bar = inv_V_bar
        self.V_bar = V_bar
        self.b_bar = b_bar        


    def __parameter_estimates(self):
        
        """
        point estimates and credibility intervals for model parameters
        use the multivariate normal distribution defined in (3.9.17)
        """
        
        # unpack
        V_bar = self.V_bar
        b_bar = self.b_bar
        credibility_level = self.credibility_level
        k = self.k
        # initiate storage: 4 columns: lower bound, median, upper bound, standard deviation
        beta_estimates = np.zeros((k,4))
        # critical value of normal distribution for credibility level
        Z = su.normal_icdf((1 + credibility_level) / 2)
        # mean and standard deviation of posterior distribution
        mean = b_bar
        standard_deviation = np.sqrt(np.diag(V_bar))
        # fill estimates
        beta_estimates[:,0] = mean
        beta_estimates[:,1] = mean - Z * standard_deviation
        beta_estimates[:,2] = mean + Z * standard_deviation
        beta_estimates[:,3] = standard_deviation
        # save as attributes
        self.beta_estimates = beta_estimates
        
        
    def __negative_likelihood_simple_V(self, v):
        
        """negative log marginal likelihood for scalar V (common variance)"""
        
        # unpack
        k = self.k
        b = self.b
        inv_sigma_XX = self.__inv_sigma_XX
        inv_sigma_Xy = self.__inv_sigma_Xy
        # build elements for equation (3.10.10)
        V = v * np.identity(k)
        inv_V = np.identity(k) / v
        inv_V_b = b / v
        # compute log of marginal likelihood, omitting irrelevant terms
        inv_V_bar = inv_V + inv_sigma_XX
        V_bar = la.invert_spd_matrix(inv_V_bar)
        b_bar = V_bar @ (inv_V_b + inv_sigma_Xy)
        term_1 = -0.5 * la.stable_determinant(V @ inv_sigma_XX)
        term_2 = -0.5 * (b @ inv_V_b - b_bar @ inv_V_bar @ b_bar )
        # take negative (minimize negative to maximize)
        negative_log_f_y = -(term_1 + term_2)
        return negative_log_f_y        
        
        
    def __negative_likelihood_full_V(self, v):
        
        """negative log marginal likelihood for vector V (individual variances)"""
        
        # unpack
        b = self.b
        inv_sigma_XX = self.__inv_sigma_XX
        inv_sigma_Xy = self.__inv_sigma_Xy
        # build elements for equation (3.10.10)
        V = np.diag(v)
        inv_V = np.diag(1/v)
        inv_V_b = b / v
        # compute log of marginal likelihood, omitting irrelevant terms
        inv_V_bar = inv_V + inv_sigma_XX
        V_bar = la.invert_spd_matrix(inv_V_bar)
        b_bar = V_bar @ (inv_V_b + inv_sigma_Xy)
        term_1 = -0.5 * la.stable_determinant(V @ inv_sigma_XX)
        term_2 = -0.5 * (b @ inv_V_b - b_bar @ inv_V_bar @ b_bar)
        # take negative (minimize negative to maximize)
        negative_log_f_y = -(term_1 + term_2)
        return negative_log_f_y        
        
        
    def __bayesian_forecast_evaluation_criteria(self, y, forecast_mean, forecast_variance, m):
        
        """ Bayesian forecast evaluation criteria from equations from equations (3.10.13) and (3.10.15) """   

        log_score = np.zeros(m)
        crps = np.zeros(m)
        for i in range(m):
            # get actual, prediction mean, prediction variance
            y_i = y[i]
            mu_i = forecast_mean[i]
            sigma_i = forecast_variance[i,i]
            # get log score from equation (3.10.13)
            log_pdf, _ = su.normal_pdf(y_i, mu_i, sigma_i)
            log_score[i] = - log_pdf
            # get CRPS from equation (3.10.15)
            s_i = np.sqrt(sigma_i)
            y_tld = (y_i - mu_i) / s_i
            _, pdf = su.normal_pdf(y_tld, 0, 1)
            cdf = su.normal_cdf(y_tld, 0, 1)
            crps[i] = s_i * (y_tld * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
        log_score = np.mean(log_score)
        crps = np.mean(crps)    
        bayesian_forecast_evaluation_criteria = {}
        bayesian_forecast_evaluation_criteria['log_score'] = log_score
        bayesian_forecast_evaluation_criteria['crps'] = crps
        return bayesian_forecast_evaluation_criteria
    
    