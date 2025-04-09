# imports
import numpy as np
import numpy.random as nrd
import numpy.linalg as nla
from scipy.stats import chi2
import alexandria.math.linear_algebra as la
import alexandria.math.math_utilities as mu
import alexandria.math.stat_utilities as su
import alexandria.processor.input_utilities as iu
import alexandria.console.console_utilities as cu
import alexandria.math.random_number_generators as rng
import alexandria.linear_regression.regression_utilities as ru
from alexandria.linear_regression.linear_regression import LinearRegression
from alexandria.linear_regression.bayesian_regression import BayesianRegression


class AutocorrelatedBayesianRegression(LinearRegression, BayesianRegression):
    
    
    """
    Autocorrelated Bayesian linear regression, developed in section 9.6
    
    Parameters:
    -----------
    endogenous : ndarray of shape (n,)
        endogenous or explained variable, defined in (3.9.3)
    
    exogenous : ndarray of shape (n,k)
        exogenous or explanatory variables, defined in (3.9.3)
        
    q : int, default = 1
        order of autocorrelation (number of residual lags)
            
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
   
    p : float or ndarray of shape (q,), default = 0
        prior mean, defined in (3.9.62)
   
    H : float or ndarray of shape (q,), default = 100
        prior variance, defined in (3.9.62) 
   
    iterations : int, default = 2000
        post burn-in iterations for MCMC algorithm
   
    burn : int, default = 1000
        burn-in iterations for MCMC algorithm
   
    credibility_level : float, default = 0.95
        credibility level (between 0 and 1)
    
    verbose : bool, default = False
        if True, displays a progress bar  
    
    
    Attributes
    ----------
    endogenous : ndarray of shape (n,)
        endogenous or explained variable
    
    exogenous : ndarray of shape (n,k)
        exogenous or explanatory variables, defined in (3.9.3)
   
    q : int
        order of autocorrelation (number of residual lags)
    
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
   
    p : ndarray of shape (q,)
        prior mean, defined in (3.9.62)
   
    H : ndarray of shape (q,q)
        prior variance, defined in (3.9.62)  
   
    iterations : int
        post burn-in iterations for MCMC algorithm
   
    burn : int
        burn-in iterations for MCMC algorithm
    
    credibility_level : float
        credibility level (between 0 and 1)
    
    verbose : bool
        if True, displays a progress bar during MCMC algorithms
   
    y : ndarray of shape (n,)
        explained variables, defined in (3.9.3)
    
    X : ndarray of shape (n,k)
        explanatory variables, defined in (3.9.3)
   
    T : int
        number of observations, defined in (3.9.52)
    
    k : int
        dimension of beta, defined in (3.9.1)   
   
    alpha_bar : float
        posterior scale, defined in (3.9.35)
   
    mcmc_beta : ndarray of shape (k,iterations)
        storage of mcmc values for beta
   
    mcmc_sigma : ndarray of shape (iterations,)
        storage of mcmc values for sigma
   
    mcmc_phi : ndarray of shape (q,iterations)
        storage of mcmc values for phi
   
    beta_estimates : ndarray of shape (k,4)
        posterior estimates for beta
        column 1: point estimate; column 2: interval lower bound; 
        column 3: interval upper bound; column 4: standard deviation
   
    sigma_estimates : float
        posterior estimate for sigma
   
    phi_estimates : ndarray of shape (q,3)
        posterior estimates for phi
        column 1: interval lower bound; column 2: median; 
        column 3: interval upper bound
   
    X_hat : ndarray of shape (m,k)
        predictors for the model 
   
    m : int
        number of predicted observations, defined in (3.10.1)   
   
    mcmc_forecasts : ndarray of shape (m,iterations)
        storage of mcmc values for forecasts
   
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
    forecast
    fit_and_residuals
    forecast_evaluation
    marginal_likelihood
    """    

    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------
    
    
    def __init__(self, endogenous, exogenous, q = 1,
                constant = True, trend = False, quadratic_trend = False,    
                b_exogenous = 0, V_exogenous = 1, b_constant = 0, V_constant = 1,  
                b_trend = 0, V_trend = 1, b_quadratic_trend = 0, V_quadratic_trend = 1,
                alpha = 1e-4, delta = 1e-4, p = 0, H = 100, iterations = 2000, 
                burn = 1000, credibility_level = 0.95, verbose = False):    
    
        """
        constructor for the AutocorrelatedBayesianRegression class
        """    
    
        self.endogenous = endogenous
        self.exogenous = exogenous
        self.q = q
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
        self.p = p
        self.H = H
        self.iterations = iterations
        self.burn = burn
        self.credibility_level = credibility_level
        self.verbose = verbose
        # make regressors
        self._make_regressors()
        self.__make_autocorrelated_regressors()


    def estimate(self):
        
        """
        estimate()
        generates posterior estimates for linear regression model parameters beta, sigma and gamma
        
        parameters:
        none
        
        returns:
        none
        """
        
        # define prior values
        self._prior()
        # define autocorrelated prior values
        self.__autocorrelated_prior()
        # define posterior values
        self.__posterior()
        # run MCMC algorithm for regression parameters
        self.__parameter_mcmc()
        # obtain posterior estimates for regression parameters
        self.__parameter_estimates()
        
        
    def forecast(self, X_hat, credibility_level):
        
        """
        predictions for the linear regression model
        predictive density along with point estimates and credibility intervals

        parameters:
        X_hat : ndarray of shape (m,k)
            array of predictors
        credibility_level : float
            credibility level for predictions (between 0 and 1)

        returns:
        forecast_estimates : ndarray of shape (m,3)
            posterior estimates for predictions
            column 1: interval lower bound; column 2: median; 
            column 3: interval upper bound
        """
        
        # run mcmc algorithm for predictive density
        mcmc_forecasts, m = self.__forecast_mcmc(X_hat)
        # obtain posterior estimates
        forecast_estimates = self.__make_forecast_estimates(mcmc_forecasts, credibility_level)
        # save as attributes
        self.m = m
        self.mcmc_forecasts = mcmc_forecasts
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
        mcmc_forecasts = self.mcmc_forecasts
        forecast_estimates = self.forecast_estimates
        m = self.m
        iterations = self.iterations
        # obtain regular forecast evaluation criteria
        y_hat = forecast_estimates[:,0]
        standard_evaluation_criteria = ru.forecast_evaluation_criteria(y_hat, y)  
        # obtain Bayesian forecast evaluation criteria
        bayesian_evaluation_criteria = self.__bayesian_forecast_evaluation_criteria(y, mcmc_forecasts, iterations, m)   
        # merge dictionaries
        forecast_evaluation_criteria = iu.concatenate_dictionaries(standard_evaluation_criteria, bayesian_evaluation_criteria)        
        # save as attributes
        self.forecast_evaluation_criteria = forecast_evaluation_criteria            
        
        
    def marginal_likelihood(self):
        
        """ 
        log10 marginal likelihood, defined in (3.10.32)
        
        parameters:
        none
        
        returns:
        m_y: float
            log10 marginal likelihood
        """
        
        # unpack
        y = self.y
        X = self.X
        T = self.T
        k = self.k
        q = self.q
        b = self.b
        inv_V = self._inv_V
        V = self.V
        p = self.p
        H = self.H
        inv_H = self.__inv_H
        alpha = self.alpha
        delta = self.delta
        mcmc_beta = self.mcmc_beta
        mcmc_sigma = self.mcmc_sigma
        mcmc_phi = self.mcmc_phi
        iterations = self.iterations
        # generate theta_hat and Sigma_hat
        mcmc_theta = np.vstack((mcmc_beta, mcmc_sigma, mcmc_phi))
        theta_hat = np.mean(mcmc_theta, 1)
        Sigma_hat = np.cov(mcmc_theta)
        inv_Sigma_hat = la.invert_spd_matrix(Sigma_hat)
        # generate parameters for truncation of the Chi2
        omega = 0.5
        bound = chi2.ppf(omega, k + q + 1)
        # compute the log of first row of (3.10.32)
        J = iterations
        term_1 = - np.log(omega * J)
        term_2 = (T - 1) / 2 * np.log(2 * np.pi)
        term_3 = -0.5 * np.log(nla.det(Sigma_hat))
        term_4 = 0.5 * np.log(nla.det(V))
        term_5 = 0.5 * np.log(nla.det(H))
        term_6 = np.log(mu.gamma(alpha / 2))
        term_7 = - alpha / 2 * np.log(delta / 2)
        row_1 = - (term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7)
        # for second row of (3.10.32), compute the log of each term in summation
        summation = np.zeros(iterations)
        for i in range(iterations):
            theta = mcmc_theta[:,i]
            beta = mcmc_beta[:,i]
            sigma = mcmc_sigma[i]
            phi = mcmc_phi[:,i]
            quadratic_form = (theta - theta_hat) @ inv_Sigma_hat @ (theta - theta_hat)
            if quadratic_form > bound:
                summation[i] = -1000;
            else:
                term_1 = ((alpha + T) / 2 + 1) * np.log(sigma)
                residuals = y - X @ beta
                epsilon, E = la.lag_matrix(residuals, q)
                inv_sigma = 1 / sigma
                u = epsilon - E @ phi
                term_2 = 0.5 * u @ u * inv_sigma
                term_3 = 0.5 * (beta - b) @ inv_V @ (beta - b)
                term_4 = 0.5 * delta * inv_sigma
                term_5 = 0.5 * (phi - p) @ inv_H @ (phi - p)
                term_6 = - 0.5 * quadratic_form
                summation[i] = term_1 + term_2 + term_3 + term_4 + term_5 + term_6
        # turn sum of the logs into log of the sum
        row_2 = - mu.log_sum_exp(summation)
        # sum the two rows and convert to log10
        log_f_y = row_1 + row_2
        m_y = log_f_y / np.log(10)
        self.m_y = m_y
        return m_y  
    
    
    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------


    def __make_autocorrelated_regressors(self):
        
        """generates T defined in (3.9.52)"""
      
        # unpack
        n = self.n
        q = self.q
        # get dimensions
        T = n - q
        # save as attributes
        self.T = T


    def __autocorrelated_prior(self):
        
        """ creates prior elements p and Z defined in (3.9.62) """
        
        # generate p
        self.__generate_p()
        # generate H
        self.__generate_H()


    def __generate_p(self):
        
        """creates prior element p"""
        
        # unpack
        p = self.p
        # if p is a scalar, turn it into a vector with p as first value, 0 afterwards
        if isinstance(p, (int,float)):
            q = self.q
            temp = np.zeros(q)
            temp[0] = p
            p = temp.copy()
        # else, p is a vector, don't change it
        self.p = p


    def __generate_H(self):
        
        """creates prior element H"""
        
        # unpack
        p = self.p
        H = self.H
        # if H is a scalar, turn it into a vector replicating the value
        if isinstance(H, (int,float)):
            q = self.q
            inv_H_p = p / H
            inv_H = np.identity(q) / H
            H = np.identity(q) * H
        # if H is a vector, turn it into a diagonal array
        else:
            inv_H_p = p / H
            inv_H = np.diag(1/H)
            H = np.diag(H)
        self.H = H
        self.__inv_H = inv_H
        self.__inv_H_p = inv_H_p


    def __posterior(self):
        
        """creates constant posterior element alpha_bar defined in (3.9.67)"""
        
        # unpack
        alpha = self.alpha
        T = self.T
        # set value
        alpha_bar = alpha + T
        # save as attribute
        self.alpha_bar = alpha_bar 


    def __parameter_mcmc(self):
        
        """posterior distribution for parameters from algorithm 9.3"""
        
        y = self.y
        X = self.X
        k = self.k
        q = self.q
        inv_V = self._inv_V
        inv_V_b = self._inv_V_b
        alpha_bar = self.alpha_bar
        delta = self.delta
        inv_H = self.__inv_H
        inv_H_p = self.__inv_H_p        
        iterations = self.iterations
        burn = self.burn
        verbose = self.verbose
        # preallocate storage space
        mcmc_beta = np.zeros((k,iterations))
        mcmc_sigma = np.zeros(iterations)
        mcmc_phi = np.zeros((q, iterations))
        total_iterations = iterations + burn
        # set initial values
        beta = np.zeros(k)
        inv_sigma = 1
        phi = np.zeros(q)
        X_star = la.lag_polynomial(X, phi)
        y_star = la.lag_polynomial(y, phi)
        # run algorithm 9.3 (Gibbs sampling for the model parameters)
        for i in range(total_iterations):
            # draw beta from its conditional posterior
            beta, res, res_star = self.__draw_beta(inv_V, inv_V_b, inv_sigma, X, y, X_star, y_star)
            # draw sigma from its conditional posterior
            sigma, inv_sigma = self.__draw_sigma(delta, alpha_bar, res_star)
            # draw phi from its conditional posterior
            phi, X_star, y_star = self.__draw_phi(inv_H, inv_H_p, inv_sigma, res, q, X, y)
            # if burn-in sample is over, record value
            if i >= burn:
                mcmc_beta[:,i-burn] = beta
                mcmc_sigma[i-burn] = sigma
                mcmc_phi[:,i-burn] = phi
            # if verbose, display progress bar
            if verbose:
                cu.progress_bar(i, total_iterations, 'Model parameters:')
        # save as attributes
        self.mcmc_beta = mcmc_beta
        self.mcmc_sigma = mcmc_sigma
        self.mcmc_phi = mcmc_phi


    def __draw_beta(self, inv_V, inv_V_b, inv_sigma,X, y, X_star, y_star):
        
        """draw beta from its conditional posterior defined in (3.9.45)"""
        
        # posterior parameters for beta, defined in (3.9.65)
        inv_V_bar = inv_V + inv_sigma * X_star.T @ X_star
        # posterior b_bar
        b_bar_temp = inv_V_b + inv_sigma * X_star.T @ y_star
        # efficient sampling of beta (algorithm 9.4)
        beta = rng.efficient_multivariate_normal(b_bar_temp, inv_V_bar)
        # compute residuals, as defined in (3.9.53) and (3.9.57)
        res = y - X @ beta
        res_star = y_star - X_star @ beta
        return beta, res, res_star


    def __draw_sigma(self, delta, alpha_bar, res_star):
        
        """draw sigma from its conditional posterior defined in (3.9.66)"""

        # compute delta_bar, defined in (3.9.67)
        delta_bar = delta + res_star @ res_star
        # sample sigma
        sigma = rng.inverse_gamma(alpha_bar / 2, delta_bar / 2)
        inv_sigma = 1 / sigma
        return sigma, inv_sigma
 

    def __draw_phi(self, inv_H, inv_H_p, inv_sigma, res, q, X, y):
        
        """draw phi from its conditional posterior defined in (3.9.69)"""

        # compute epsilon and E, defined in (3.9.60)
        epsilon, E = la.lag_matrix(res, q)
        # posterior parameters for phi, defined in (3.9.70)
        inv_H_bar = inv_H + inv_sigma * E.T @ E
        # posterior p_bar
        p_bar_temp = inv_H_p + inv_sigma * E.T @ epsilon
        # efficient sampling of phi (algorithm 9.4)
        phi = rng.efficient_multivariate_normal(p_bar_temp, inv_H_bar)
        # update X_star and Y_star
        X_star = la.lag_polynomial(X, phi)
        y_star = la.lag_polynomial(y, phi)
        return phi, X_star, y_star


    def __parameter_estimates(self):
        
        """
        point estimates and credibility intervals for model parameters
        uses quantiles of the empirical posterior distribution
        """
        
        # unpack
        mcmc_beta = self.mcmc_beta
        mcmc_sigma = self.mcmc_sigma
        mcmc_phi = self.mcmc_phi
        credibility_level = self.credibility_level
        k = self.k
        q = self.q
        # initiate storage: 4 columns: lower bound, median, upper bound, standard deviation
        beta_estimates = np.zeros((k,4))
        # fill estimates
        beta_estimates[:,0] = np.quantile(mcmc_beta, 0.5, 1)
        beta_estimates[:,1] = np.quantile(mcmc_beta, (1-credibility_level)/2, 1)
        beta_estimates[:,2] = np.quantile(mcmc_beta, (1+credibility_level)/2, 1)
        beta_estimates[:,3] = np.std(mcmc_beta, 1)
        # get median for sigma
        sigma_estimates = np.quantile(mcmc_sigma, 0.5)
        # get estimates for phi
        phi_estimates = np.zeros((q,4))
        phi_estimates[:,0] = np.quantile(mcmc_phi, 0.5, 1)
        phi_estimates[:,1] = np.quantile(mcmc_phi, (1-credibility_level)/2, 1)
        phi_estimates[:,2] = np.quantile(mcmc_phi, (1+credibility_level)/2, 1)
        phi_estimates[:,3] = np.std(mcmc_phi, 1)
        # save as attributes
        self.beta_estimates = beta_estimates
        self.sigma_estimates = sigma_estimates
        self.phi_estimates = phi_estimates


    def __forecast_mcmc(self, X_hat):
        
        """posterior predictive distribution from algorithm 10.3""" 
        
        # unpack
        X = self.X
        y = self.y
        q = self.q
        mcmc_beta = self.mcmc_beta
        mcmc_sigma = self.mcmc_sigma
        mcmc_phi = self.mcmc_phi
        iterations = self.iterations
        verbose = self.verbose
        n = self.n
        constant = self.constant
        trend = self.trend
        quadratic_trend = self.quadratic_trend
        # add constant if necessary
        m = X_hat.shape[0]
        X_hat = ru.add_intercept_and_trends(X_hat, constant, trend, quadratic_trend, n)
        # initiate storage, loop over simulations and simulate predictions
        mcmc_forecasts = np.zeros((m, iterations))        
        for i in range(iterations):
            beta = mcmc_beta[:,i]
            sigma = mcmc_sigma[i]
            phi = mcmc_phi[:,i]
            # get in-sample residuals
            residuals = y - X @ beta
            # set e_t, defined in (3.9.53), for first out-of-sample period
            e_t = np.flip(residuals[-q:])
            # loop over periods and build epsilon_t, defined in (3.9.53)
            epsilon = np.zeros(m)
            for t in range(m):
                # generate epsilon_t
                u_t = np.sqrt(sigma) * nrd.randn()
                epsilon_t = e_t @ phi + u_t
                # update e_t for next period
                e_t = np.hstack((epsilon_t, e_t[:-1]))
                # record value
                epsilon[t] = epsilon_t
            # form prediction
            y_hat = X_hat @ beta + epsilon
            mcmc_forecasts[:,i] = y_hat
            if verbose:
                cu.progress_bar(i, iterations, 'Predictions:')
        return mcmc_forecasts, m


    def __make_forecast_estimates(self, mcmc_forecasts, credibility_level):
        
        """point estimates and credibility intervals for predictions""" 
        
        m = mcmc_forecasts.shape[0]
        # initiate estimate storage; 3 columns: lower bound, median, upper bound
        forecast_estimates = np.zeros((m,3))
        # fill estimates
        forecast_estimates[:,0] = np.quantile(mcmc_forecasts, 0.5, 1)
        forecast_estimates[:,1] = np.quantile(mcmc_forecasts, (1-credibility_level)/2, 1)
        forecast_estimates[:,2] = np.quantile(mcmc_forecasts, (1+credibility_level)/2, 1)
        return forecast_estimates


    def __bayesian_forecast_evaluation_criteria(self, y, mcmc_forecasts, iterations, m):
        
        """ Bayesian forecast evaluation criteria from equations from equations (3.10.13) and (3.10.17) """   

        log_score = np.zeros(m)
        crps = np.zeros(m)
        for i in range(m):
            # get actual, prediction mean, prediction variance    
            y_i = y[i]
            forecasts = mcmc_forecasts[i,:]
            mu_i = np.mean(forecasts)
            sigma_i = np.var(forecasts)
            # get log score from equation (3.10.14)
            log_pdf, _ = su.normal_pdf(y_i, mu_i, sigma_i)
            log_score[i] = - log_pdf
            # get CRPS from equation (3.10.17)
            term_1 = np.sum(np.abs(forecasts - y_i))
            term_2 = 0
            for j in range(iterations):
                term_2 += np.sum(np.abs(forecasts[j] - forecasts))
            crps[i] = term_1 / iterations - term_2 / (2 * iterations**2)
        log_score = np.mean(log_score)
        crps = np.mean(crps)
        bayesian_forecast_evaluation_criteria = {}
        bayesian_forecast_evaluation_criteria['log_score'] = log_score
        bayesian_forecast_evaluation_criteria['crps'] = crps
        return bayesian_forecast_evaluation_criteria

