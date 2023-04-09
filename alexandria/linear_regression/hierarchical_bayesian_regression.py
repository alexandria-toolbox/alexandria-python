# imports
import numpy as np
import numpy.linalg as nla
import scipy.optimize as sco
import scipy.special as ssp
import alexandria.math.linear_algebra as la
import alexandria.console.console_utilities as cu
import alexandria.math.math_utilities as mu
import alexandria.math.stat_utilities as su
from alexandria.linear_regression.linear_regression import LinearRegression



class HierarchicalBayesianRegression(LinearRegression):
    
    
    """
    Hierarchical Bayesian linear regression, developed in section 9.3
    
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
   
    alpha : float, default = 1e-4
        prior shape, defined in (3.9.21)
   
    delta : float, default = 1e-4
        prior scale, defined in (3.9.21)
    
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
        prior mean for quadratic_trend
   
    V_quadratic_trend : float
        prior variance for quadratic_trend (positive)
        
    b : ndarray of shape (k,)
        prior mean, defined in (3.9.10)
   
    V : ndarray of shape (k,k)
        prior variance, defined in (3.9.10) 
   
    alpha : float
        prior shape, defined in (3.9.21)
   
    delta : float
        prior scale, defined in (3.9.21)
    
    credibility_level : float
        credibility level (between 0 and 1)
    
    verbose : bool
        if True, displays a progress bar during MCMC algorithms
   
    y : ndarray of shape (n,)
        explained variables, defined in (3.9.3)
    
    X : ndarray of shape (n,k)
        explanatory variables, defined in (3.9.3)
   
    n : int
        number of observations, defined in (3.9.1)
    
    k : int
        dimension of beta, defined in (3.9.1)
   
    b_bar : ndarray of shape (k,)
        posterior mean, defined in (3.9.14)
   
    V_bar : ndarray of shape (k,k)
        posterior variance, defined in (3.9.14)  
   
    alpha_bar : float
        posterior shape, defined in (3.9.24)
   
    delta_bar : float
        posterior scale, defined in (3.9.24)    
    
    location : ndarray of shape (k,)
        location for the student posterior of beta, defined in (3.9.28)
   
    scale : ndarray of shape (k,k)
        scale for the student posterior of beta, defined in (3.9.28)
   
    df : float
        degrees of freedom for the student posterior of beta, defined in (3.9.28)
   
    estimates_beta : ndarray of shape (k,4)
        posterior estimates for beta
        column 1: interval lower bound; column 2: median; 
        column 3: interval upper bound; column 4: standard deviation
   
    estimates_sigma : float
        posterior estimate for sigma
   
    X_hat : ndarray of shape (m,k)
        predictors for the model 
   
    m : int
        number of predicted observations, defined in (3.10.1)
   
    estimates_forecasts : ndarray of shape (m,3)
        posterior estimates for predictions   
        column 1: interval lower bound; column 2: median; 
        column 3: interval upper bound
    
    estimates_fit : ndarray of shape (n,)
        posterior estimates (median) for in sample-fit
   
    estimates_residuals : ndarray of shape (n,)
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
    forecast
    fit_and_residuals
    forecast_evaluation
    marginal_likelihood
    optimize_hyperparameters
    """


    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self, endogenous, exogenous, constant = True, trend = False, 
                 quadratic_trend = False, b_exogenous = 0, V_exogenous = 1,
                 b_constant = 0, V_constant = 1, b_trend = 0, V_trend = 1,
                 b_quadratic_trend = 0, V_quadratic_trend = 1, alpha = 1e-4, 
                 delta = 1e-4, credibility_level = 0.95, verbose = False):
        
        """
        constructor for the HierarchicalBayesianRegression class
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
        self.alpha = alpha
        self.delta = delta
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
        
        # define prior values
        self.__prior()
        # define posterior values
        self.__posterior()
        # obtain posterior estimates for regression parameters
        self.__parameter_estimates()
        
        
    def forecast(self, X_hat, credibility_level):
        
        """
        forecast(X_hat, credibility_level)
        predictions for the linear regression model using (3.10.6)
        
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
        V_bar = self.V_bar
        alpha_bar = self.alpha_bar
        delta_bar = self.delta_bar     
        # add constant and trends if included
        X_hat = self._add_intercept_and_trends(X_hat, False)
        # obtain prediction location, scale and degrees of freedom from (3.10.6)
        m = X_hat.shape[0]
        location = X_hat @ b_bar
        scale = (delta_bar / alpha_bar) * (np.identity(m) + X_hat @ V_bar @ X_hat.T)
        df = alpha_bar
        # if verbose, display progress bar
        if verbose:
            cu.progress_bar_complete('Predictions:')
        # initiate estimate storage; 3 columns: lower bound, median, upper bound
        estimates_forecasts = np.zeros((m,3))
        # critical value of Student distribution for credibility level
        Z = su.student_icdf((1 + credibility_level) / 2, df)
        # scale in textbook is square of scale for Python: take square root
        sqrt_scale = np.sqrt(np.diag(scale))
        # fill estimates
        estimates_forecasts[:,0] = location - Z * sqrt_scale
        estimates_forecasts[:,1] = location
        estimates_forecasts[:,2] = location + Z * sqrt_scale  
        # save as attributes
        self.X_hat = X_hat
        self.m = m
        self.__forecast_location = location
        self.__forecast_scale = scale
        self.estimates_forecasts = estimates_forecasts
        return estimates_forecasts    
    
    
    def fit_and_residuals(self):

        """
        fit_and_residuals()
        estimates of in-sample fit and regression residuals
        
        parameters:
        none
        
        returns:
        none
        """
        
        # unpack
        y = self.y        
        X = self.X
        beta = self.estimates_beta[:,1]
        k = self.k
        n = self.n      
        # estimate fits and residuals
        estimates_fit = X @ beta
        estimates_residuals = y - X @ beta
        # estimate in-sample prediction criteria from equation (3.10.8)
        res = estimates_residuals
        ssr = res @ res
        tss = (y - np.mean(y)) @ (y - np.mean(y))
        r2 = 1 - ssr / tss
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)
        insample_evaluation = {}
        insample_evaluation['ssr'] = ssr
        insample_evaluation['r2'] = r2
        insample_evaluation['adj_r2'] = adj_r2   
        # save as attributes
        self.estimates_fit = estimates_fit
        self.estimates_residuals = estimates_residuals
        self.insample_evaluation = insample_evaluation  
        
        
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
        forecast_location = self.__forecast_location
        forecast_scale = self.__forecast_scale
        nu_i = self.alpha_bar
        estimates_forecasts = self.estimates_forecasts
        m = self.m
        # calculate forecast error
        y_hat = estimates_forecasts[:,1]
        err = y - y_hat        
        # compute forecast evaluation from equation (3.10.11)
        rmse = np.sqrt(err @ err / m)
        mae = np.sum(np.abs(err)) / m
        mape = 100 * np.sum(np.abs(err / y)) / m
        theil_u = np.sqrt(err @ err) / (np.sqrt(y @ y) + np.sqrt(y_hat @ y_hat))
        bias = np.sum(err) / np.sum(np.abs(err))        
        # loop over the m predictions
        log_score = np.zeros(m)
        crps = np.zeros(m)        
        for i in range(m):
            # get actual, prediction mean, prediction variance    
            y_i = y[i]
            mu_i = forecast_location[i]
            sigma_i = forecast_scale[i,i]
            # get log score from equation (3.10.13)
            log_pdf, _ = su.student_pdf(y_i, mu_i, sigma_i, nu_i)
            log_score[i] = - log_pdf
            # get CRPS from equation (3.10.16)
            s_i = np.sqrt(sigma_i)
            y_tld = (y_i - mu_i) / s_i
            _, pdf = su.student_pdf(y_tld, 0, 1, nu_i)
            cdf = su.student_cdf(y_tld, 0, 1, nu_i)
            term_1 = y_tld * (2 * cdf - 1)
            term_2 = 2 * pdf * (nu_i + y_tld**2) / (nu_i - 1)
            term_3 = - (2 * np.sqrt(nu_i) * \
            ssp.beta(0.5, nu_i - 0.5)) / ((nu_i - 1) * ssp.beta(0.5, nu_i / 2)**2)
            crps[i] = s_i * (term_1 + term_2 + term_3)
        log_score = np.mean(log_score)
        crps = np.mean(crps)
        forecast_evaluation_criteria = {}
        forecast_evaluation_criteria['rmse'] = rmse
        forecast_evaluation_criteria['mae'] = mae
        forecast_evaluation_criteria['mape'] = mape
        forecast_evaluation_criteria['theil_u'] = theil_u
        forecast_evaluation_criteria['bias'] = bias
        forecast_evaluation_criteria['log_score'] = log_score
        forecast_evaluation_criteria['crps'] = crps
        # save as attributes
        self.forecast_evaluation_criteria = forecast_evaluation_criteria
        
        
        
    def marginal_likelihood(self):
        
        """ 
        marginal_likelihood()
        log10 marginal likelihood, defined in (3.10.24)
        
        parameters:
        none
        
        returns:
        m_y: float
            log10 marginal likelihood
        """

        # unpack
        n = self.n
        alpha = self.alpha
        delta = self.delta        
        V = self.V
        alpha_bar = self.alpha_bar
        delta_bar = self.delta_bar
        XX = self._XX
        # evaluate the log marginal likelihood from equation (3.10.24)
        term_1 = -(n / 2) * np.log(np.pi)
        term_2 = -0.5 * la.stable_determinant(V @ XX)
        term_3 = (alpha / 2) * np.log(delta) - (alpha_bar / 2) * np.log(delta_bar)
        term_4 = np.log(mu.gamma(alpha_bar / 2)) - np.log(mu.gamma(alpha / 2))
        log_f_y = term_1 + term_2 + term_3 + term_4
        # convert to log10
        m_y = log_f_y / np.log(10)
        # save as attributes
        self.m_y = m_y
        return m_y  
    
    
    def optimize_hyperparameters(self, type):

        """
        optimize_hyperparameters(type)
        optimize V and delta by maximizing the marginal likelihood
        
        parameters
        type : int 
            optimization type (1 or 2)
            1 = optimize scalar v; 2 = optimize vector V
        """
        
        # unpack
        constant = self.constant
        trend = self.trend
        quadratic_trend = self.quadratic_trend
        verbose = self.verbose
        k = self.k
        # estimate prior elements to get proper estimate of b
        self.__prior()
        # optimize: in the simplest case,  V = vI so only scalar v is optimized
        if type == 1:
            # initial value of optimizer
            x_0 = np.ones(2)
            # bounds for parameter values
            bound = [(mu.eps(), 1000)] * 2
            # optimize
            result = sco.minimize(self.__negative_likelihood_simple_V, \
                                  x_0, bounds = bound, method='L-BFGS-B')
            # save as attribute
            self.V_constant = result.x[0]
            self.V_trend = result.x[0]
            self.V_quadratic_trend = result.x[0]
            self.V_exogenous = result.x[0]
            self.delta = result.x[1]
        # in the second case, V is diagonal from vector v, so vector v is optimized
        elif type == 2:
            # initial value of optimizer
            x_0 = np.ones(k + 1)
            # bounds for parameter values
            bound = [(mu.eps(), 1000)] * (k + 1)
            # optimize
            result = sco.minimize(self.__negative_likelihood_full_V, \
                      x_0, bounds = bound, method='L-BFGS-B')
            # save as attribute
            if constant:
                self.V_constant = result.x[0]
            if trend:
                self.V_trend = result.x[0 + constant]
            if quadratic_trend:
                self.V_quadratic_trend = result.x[0 + constant + trend] 
            self.V_exogenous = result.x[0 + constant + trend + quadratic_trend:-1]
            self.delta = result.x[-1]
        # if verbose, display progress bar and success/failure of optimization
        if verbose:
            cu.progress_bar_complete('hyperparameter optimization:')
            cu.optimization_completion(result.success)


    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------


    def __prior(self):
        
        """creates prior elements b and V defined in (3.9.10)"""
        
        # generate b
        self.__generate_b()
        # generate V
        self.__generate_V()
    

    def __generate_b(self):
        
        """creates prior element b"""
        
        # unpack
        constant = self.constant
        trend = self.trend
        quadratic_trend = self.quadratic_trend
        b_exogenous = self.b_exogenous
        b_constant = self.b_constant
        b_trend = self.b_trend
        b_quadratic_trend = self.b_quadratic_trend
        n_exogenous = self._n_exogenous
        # if b_exogenous is a scalar, turn it into a vector replicating the value
        if isinstance(b_exogenous, (int,float)):
            b_exogenous = b_exogenous * np.ones(n_exogenous)
        b = b_exogenous
        # if quadratic trend is included, add to prior mean
        if quadratic_trend:
            b = np.hstack((b_quadratic_trend, b))
        # if trend is included, add to prior mean
        if trend:
            b = np.hstack((b_trend, b))
        # if constant is included, add to prior mean
        if constant:
            b = np.hstack((b_constant, b))
        # save as attribute
        self.b = b 


    def __generate_V(self):
        
        """creates prior element V"""
        
        # unpack
        b = self.b        
        constant = self.constant
        trend = self.trend
        quadratic_trend = self.quadratic_trend
        V_exogenous = self.V_exogenous
        V_constant = self.V_constant
        V_trend = self.V_trend
        V_quadratic_trend = self.V_quadratic_trend  
        n_exogenous = self._n_exogenous
        # if V_exogenous is a scalar, turn it into a vector replicating the value
        if isinstance(V_exogenous, (int,float)):
            V_exogenous = V_exogenous * np.ones(n_exogenous)
        V = V_exogenous
        # if quadratic trend is included, add to prior mean
        if quadratic_trend:
            V = np.hstack((V_quadratic_trend, V))
        # if trend is included, add to prior mean
        if trend:
            V = np.hstack((V_trend, V))
        # if constant is included, add to prior mean
        if constant:
            V = np.hstack((V_constant, V))        
        # convert the vector V into an array
        inv_V_b = b / V
        inv_V = np.diag(1/V)
        V = np.diag(V)
        # save as attributes
        self.V = V
        self.__inv_V = inv_V
        self.__inv_V_b = inv_V_b


    def __posterior(self):
        
        """creates posterior parameters b_bar and V_bar defined in (3.9.14)"""
        
        # unpack
        XX = self._XX
        Xy = self._Xy
        yy = self._yy
        n = self.n
        b = self.b
        inv_V = self.__inv_V
        inv_V_b = self.__inv_V_b
        alpha = self.alpha
        delta = self.delta
        verbose = self.verbose
        # V_bar, defined in (3.9.24)
        inv_V_bar = inv_V + XX
        V_bar = la.invert_spd_matrix(inv_V_bar)
        # b_bar, defined in (3.9.24)
        b_bar = V_bar @ (inv_V_b + Xy)
        # alpha_bar, defined in (3.9.24)
        alpha_bar = alpha + n
        # delta_bar, defined in (3.9.24)
        delta_bar = delta + yy + b @ inv_V_b - b_bar @ inv_V_bar @ b_bar
        # posterior location, defined in (3.9.27)
        location = b_bar
        # posterior scale, defined in (3.9.27)
        scale = (delta_bar / alpha_bar) * V_bar
        # posterior degrees of freedom, defined in (3.9.27)
        df = alpha_bar
        # if verbose, display progress bar
        if verbose:
            cu.progress_bar_complete('Model parameters:')
        # save as attributes
        self.__inv_V_bar = inv_V_bar
        self.V_bar = V_bar
        self.b_bar = b_bar
        self.alpha_bar = alpha_bar
        self.delta_bar = delta_bar
        self.location = location
        self.scale = scale
        self.df = df


    def __parameter_estimates(self):
        
        """
        point estimates and credibility intervals for model parameters
        use the multivariate Student distribution defined in (3.9.28)
        """
        
        # unpack
        alpha_bar = self.alpha_bar
        delta_bar = self.delta_bar
        location = self.location
        scale = self.scale
        df = self.df
        credibility_level = self.credibility_level
        k = self.k
        # initiate storage: 4 columns: lower bound, median, upper bound, standard deviation
        estimates_beta = np.zeros((k,4))
        # critical value of Student distribution for credibility level
        Z = su.student_icdf((1 + credibility_level) / 2, df)
        # scale in textbook is square of scale for Python: take square root
        sqrt_scale = np.sqrt(np.diag(scale))
        # fill estimates
        estimates_beta[:,0] = location - Z * sqrt_scale
        estimates_beta[:,1] = location
        estimates_beta[:,2] = location + Z * sqrt_scale
        estimates_beta[:,3] = np.sqrt(df / (df - 2)) * sqrt_scale
        # get point estimate for sigma (approximate median from Table d.17)
        shape = alpha_bar / 2
        scale = delta_bar / 2
        estimates_sigma = (scale * (3 * shape + 0.2)) / (shape * (3 * shape - 0.8))
        # save as attributes
        self.estimates_beta = estimates_beta
        self.estimates_sigma = estimates_sigma


    def __negative_likelihood_simple_V(self, x):
        
        """negative log marginal likelihood for scalar V and delta"""
        
        # unpack
        v = x[0]
        delta = x[1]
        k = self.k
        n = self.n
        b = self.b
        alpha = self.alpha
        XX = self._XX
        yy = self._yy
        Xy = self._Xy
        # build elements for equation (3.10.14)
        V = v * np.identity(k)
        inv_V = np.identity(k) / v
        inv_V_b = b / v
        inv_V_bar = inv_V + XX
        V_bar = la.invert_spd_matrix(inv_V_bar)
        b_bar = V_bar @ (inv_V_b + Xy)
        alpha_bar= alpha + n
        delta_bar = delta + yy + b @ inv_V_b - b_bar @ inv_V_bar @ b_bar
        # compute log of marginal likelihood, omitting irrelevant terms
        term_1 = -0.5 * la.stable_determinant(V @ XX)
        term_2 = (alpha / 2) * np.log(delta) - (alpha_bar / 2) * np.log(delta_bar)
        # take negative (minimize negative to maximize)
        negative_log_f_y = -(term_1 + term_2)
        return negative_log_f_y
    
    
    def __negative_likelihood_full_V(self, x):
        
        """negative log marginal likelihood for vector V and delta"""
        
        # unpack
        v = x[:-1]
        delta = x[-1]
        n = self.n
        b = self.b
        alpha = self.alpha
        XX = self._XX
        yy = self._yy
        Xy = self._Xy
        # build elements for equation (3.10.14)
        V = np.diag(v)
        inv_V = np.diag(1/v)
        inv_V_b = b / v
        inv_V_bar = inv_V + XX
        V_bar = la.invert_spd_matrix(inv_V_bar)
        b_bar = V_bar @ (inv_V_b + Xy)
        alpha_bar = alpha + n
        delta_bar = delta + yy + b @ inv_V_b - b_bar @ inv_V_bar @ b_bar
        # compute log of marginal likelihood, omitting irrelevant terms
        term_1 = -0.5 * la.stable_determinant(V @ XX)
        term_2 = (alpha / 2) * np.log(delta) - (alpha_bar / 2) * np.log(delta_bar)
        # take negative (minimize negative to maximize)
        negative_log_f_y = -(term_1 + term_2)
        return negative_log_f_y


