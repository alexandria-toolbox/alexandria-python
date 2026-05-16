# imports
import numpy as np
import pandas as pd
from datetime import datetime
from os.path import isdir, join
import alexandria.processor.input_utilities as iu
import alexandria.console.console_utilities as cu


class NowcastingResults(object):
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass


    #---------------------------------------------------
    # Methods (Access = private)
    #--------------------------------------------------- 


    def _complete_nowcasting_information(self):  
        # endogenous and exogenous variables
        if 'endogenous_variables' not in self.complementary_information:
            if self.complementary_information['model_type'] == 3:
                self.complementary_information['endogenous_variables'] = ['y1']   
            else:
                n_endo = self.model.endogenous.shape[1]
                self.complementary_information['endogenous_variables'] = ['y' + str(i+1) for i in range(n_endo)]      
        if 'exogenous_variables' not in self.complementary_information:
            if self.complementary_information['model_type'] == 2 or len(self.model.exogenous) == 0:
                self.complementary_information['exogenous_variables'] = ['none']
            else:
                n_exo = self.model.exogenous.shape[1]
                self.complementary_information['exogenous_variables'] = ['x' + str(i+1) for i in range(n_exo)]
        # sample dates                
        if 'dates' not in self.complementary_information:
            T = self.model.T
            if self.complementary_information['model_type'] == 2:
                p = 0
            else:
                p = self.model.p
            self.complementary_information['dates'] = np.arange(1-p,T+1)
        # forecast dates
        if 'forecast_dates' not in self.complementary_information:
            if hasattr(self.model, 'forecast_estimates'):
                f_periods = len(self.model.forecast_estimates)
                T = self.model.T
                self.complementary_information['forecast_dates'] = np.arange(T+1,T+f_periods+1)
            else:
                self.complementary_information['forecast_dates'] = []           
        # conditional forecast dates
        if 'conditional_forecast_dates' not in self.complementary_information:
            if hasattr(self.model, 'conditional_forecast_estimates'):
                f_periods = self.model.conditional_forecast_estimates.shape[0]
                T = self.model.T
                self.complementary_information['conditional_forecast_dates'] = np.arange(T+1,T+f_periods+1)
            else:
                self.complementary_information['conditional_forecast_dates'] = []                  
             
            
    def _add_nowcasting_tab_2_inputs(self):
        # initiate lines
        lines = []
        # header for tab 2
        lines.append('Specification')
        lines.append('-----------------')
        lines.append(' ') 
        # model
        model = self.complementary_information['model_name']   
        lines.append('model: ' + model)
        # iterations
        iterations = str(self.model.iterations)
        lines.append('iterations: ' + iterations)
        # burn-in
        burnin = str(self.model.burnin)
        lines.append('burn-in: ' + burnin)          
        # credibility level
        model_credibility = str(self.model.credibility_level)
        lines.append('credibility level: ' + model_credibility)  
        # get model type, midas regression, mfbvar or bdfm
        model_type = self.complementary_information['model_type']        
        # mfbvar hyperparmeters
        if model_type == 1:
            # constant, trend and quadratic trend
            constant = cu.bool_to_string(self.model.constant) 
            lines.append('constant: ' + constant)        
            trend = cu.bool_to_string(self.model.trend)   
            lines.append('trend: ' + trend)        
            quadratic_trend = cu.bool_to_string(self.model.quadratic_trend)    
            lines.append('quadratic trend: ' + quadratic_trend) 
            # decomposition
            decomposition = cu.bool_to_string(self.model.decomposition) 
            lines.append('decomposition: ' + decomposition)               
            # hyperparameters
            lags = str(self.model.p)
            lines.append('lags: ' + lags)     
            if iu.is_numeric(self.model.ar_coefficients):
                ar_coefficients = str(self.model.ar_coefficients)
            else:
                ar_coefficients = iu.list_to_string(self.model.ar_coefficients)
            lines.append('AR coefficients: ' + ar_coefficients)
            pi1 = str(self.model.pi1)
            lines.append('pi1 (overall tightness): ' + pi1)
            pi2 = str(self.model.pi2)
            lines.append('pi2 (cross-variable shrinkage): ' + pi2)
            pi3 = str(self.model.pi3)
            lines.append('pi3 (lag decay): ' + pi3)
            pi4 = str(self.model.pi4)
            lines.append('pi4 (exogenous slackness): ' + pi4)
            decomposition_file = self.complementary_information['decomposition_file']
            lines.append('decomposition file: ' + decomposition_file) 
        # bdfm hyperparameters
        elif model_type == 2:
            m = str(self.model.m)
            lines.append('m (factors): ' + m)             
            q = str(self.model.q)
            lines.append('q (loadings lags): ' + q)       
            p = str(self.model.p)
            lines.append('p (factor lags): ' + p)       
            r = str(self.model.r)
            lines.append('r (residual lags): ' + r) 
            sigma = str(self.model.sigma)
            lines.append('sigma (residual variance): ' + sigma) 
            omega = str(self.model.omega)
            lines.append('omega (factor variance): ' + omega)      
            delta1 = str(self.model.delta1)
            lines.append('delta1 (loadings tightness): ' + delta1)             
            pi1 = str(self.model.pi1)
            lines.append('pi1 (factor tightness): ' + pi1)
            pi2 = str(self.model.pi2)
            lines.append('pi2 (cross-variable shrinkage): ' + pi2)
            pi3 = str(self.model.pi3)
            lines.append('pi3 (lag decay): ' + pi3)            
            omega1 = str(self.model.omega1)
            lines.append('omega1 (residual tightness): ' + omega1)            
        # midas regression hyperparameters
        elif model_type == 3:     
            representation = self.model.representation
            lines.append('representation: ' + representation)
            prior_type = self.model.prior_type
            lines.append('prior: ' + prior_type)
            endogenous_lags = str(self.model.endogenous_lags)
            lines.append('endogenous lags: ' + endogenous_lags)
            if iu.is_numeric(self.model.exogenous_lags):
                exogenous_lags = str(self.model.exogenous_lags)
            else:
                exogenous_lags = iu.list_to_string(self.model.exogenous_lags)
            lines.append('exogenous lags: ' + exogenous_lags)            
            polynomial_order = str(self.model.polynomial_order)
            lines.append('polynomial order: ' + polynomial_order)          
            omega1 = str(self.model.omega1)
            lines.append('omega1 (endogenous tightness): ' + omega1)         
            omega2 = str(self.model.omega2)
            lines.append('omega2 (endogenous lag decay): ' + omega2)         
            upsilon1 = str(self.model.upsilon1)
            lines.append('upsilon1 (exogenous tightness): ' + upsilon1)         
            upsilon2 = str(self.model.upsilon2)
            lines.append('upsilon2 (exogenous lag decay): ' + upsilon2)         
        lines.append(' ')
        lines.append(' ')              
        self.input_summary += lines             
        
        
    def _add_nowcasting_tab_3_inputs(self):
        # get model type, midas regression, mfbvar or bdfm
        model_type = self.complementary_information['model_type'] 
        # initiate lines
        lines = []
        # header for tab 1
        lines.append('Applications')
        lines.append('---------')
        lines.append(' ')
        # forecasts
        if type(self.complementary_information['forecast']) == bool:
            forecast = cu.bool_to_string(self.complementary_information['forecast'])
        else:
            forecast = self.complementary_information['forecast']
        lines.append('forecast: ' + forecast) 
        forecast_credibility = str(self.complementary_information['forecast_credibility'])
        lines.append('credibility level, forecasts: ' + forecast_credibility)
        # conditional forecasts
        if model_type == 1:
            if type(self.complementary_information['conditional_forecast']) == bool:
                conditional_forecast = cu.bool_to_string(self.complementary_information['conditional_forecast'])
            else:
                conditional_forecast = self.complementary_information['conditional_forecast']
            lines.append('conditional forecast: ' + conditional_forecast) 
            conditional_forecast_credibility = str(self.complementary_information['conditional_forecast_credibility'])
            lines.append('credibility level, conditional forecasts: ' + conditional_forecast_credibility)
        # impulse response function
        if model_type == 1 or model_type == 2:
            if type(self.complementary_information['irf']) == bool:
                irf = cu.bool_to_string(self.complementary_information['irf'])
            else:
                irf = self.complementary_information['irf']
            lines.append('impulse response function: ' + irf) 
            irf_credibility = str(self.complementary_information['irf_credibility'])
            lines.append('credibility level, impulse response function: ' + irf_credibility)         
        # forecast error variance decomposition
        if model_type == 1 or model_type == 2:
            if type(self.complementary_information['fevd']) == bool:
                fevd = cu.bool_to_string(self.complementary_information['fevd'])
            else:
                fevd = self.complementary_information['fevd']
            lines.append('forecast error variance decomposition: ' + fevd)   
            fevd_credibility = str(self.complementary_information['fevd_credibility'])
            lines.append('credibility level, forecast error variance decomposition: ' + fevd_credibility)
        # historical decomposition
        if model_type == 1 or model_type == 2:
            if type(self.complementary_information['hd']) == bool:
                hd = cu.bool_to_string(self.complementary_information['hd'])
            else:
                hd = self.complementary_information['hd']     
            lines.append('historical decomposition: ' + hd)  
            hd_credibility = str(self.complementary_information['hd_credibility'])
            lines.append('credibility level, historical decomposition: ' + hd_credibility)
        # forecast periods
        if iu.is_numeric(self.complementary_information['forecast_periods']):
            forecast_periods = str(self.complementary_information['forecast_periods'])
        else:
            forecast_periods = self.complementary_information['forecast_periods']
        lines.append('forecast periods: ' + forecast_periods)
        # conditional forecast type
        if model_type == 1:
            if iu.is_numeric(self.complementary_information['conditional_forecast_type']):
                if self.complementary_information['conditional_forecast_type'] == 1:
                    conditional_forecast_type = 'agnostic'
                elif self.complementary_information['conditional_forecast_type'] == 2:
                    conditional_forecast_type = 'structural shocks'
            else:
                conditional_forecast_type = self.complementary_information['conditional_forecast_type']
            lines.append('conditional forecast type: ' + conditional_forecast_type)
        # forecast file
        forecast_file = self.complementary_information['forecast_file']
        lines.append('forecast file: ' + forecast_file)
        # conditional forecast file
        if model_type == 1:
            conditional_forecast_file = self.complementary_information['conditional_forecast_file']
            lines.append('conditional forecast file: ' + conditional_forecast_file)        
        # forecast evaluation
        if type(self.complementary_information['forecast_evaluation']) == bool:
            forecast_evaluation = cu.bool_to_string(self.complementary_information['forecast_evaluation'])
        else:
            forecast_evaluation = self.complementary_information['forecast_evaluation']
        lines.append('forecast evaluation: ' + forecast_evaluation)       
        # irf periods
        if model_type == 1 or model_type == 2:
            if iu.is_numeric(self.complementary_information['irf_periods']):
                irf_periods = str(self.complementary_information['irf_periods'])
            else:
                irf_periods = self.complementary_information['irf_periods']
            lines.append('IRF periods: ' + irf_periods)        
        # structural identification
        if model_type == 1:
            if iu.is_numeric(self.complementary_information['structural_identification']):
                structural_identification = str(self.complementary_information['structural_identification'])
            else:
                structural_identification = self.complementary_information['structural_identification']
            lines.append('structural identification: ' + structural_identification)         
        # structural identification file 
        if model_type == 1:
            structural_identification_file = self.complementary_information['structural_identification_file']
            lines.append('structural identification file: ' + structural_identification_file)         
        lines.append(' ')
        lines.append(' ') 
        self.input_summary += lines 
        
        
    def _make_nowcasting_summary(self):
        # initiate string list
        self.estimation_summary = []
        # add model header
        self.__add_nowcasting_header() 
        # add estimation header
        self.__add_nowcasting_estimation_header()
        # get model type, midas regression, mfbvar or bdfm
        model_type = self.complementary_information['model_type'] 
        # mfbvar summary
        if model_type == 1:
            self.__add_mfbvar_summary()
        # bdfm summary
        elif model_type == 2:
            self.__add_bdfm_summary()
        # midas summary
        elif model_type == 3:
            self.__add_midas_summary()        
         
        
    def __add_nowcasting_header(self):
        # recover model name and create header
        model_name = self.complementary_information['model_name']
        self.estimation_summary += cu.model_header(model_name)  
    
  
    def __add_nowcasting_estimation_header(self):
        # initiate lines
        lines = []
        # first row: estimation sample and estimation start
        sample_start = self.complementary_information['sample_start']
        sample_end = self.complementary_information['sample_end']
        if len(sample_start) == 0 or len(sample_end) == 0:
            sample = '—'
        else:
            sample = sample_start + '  ' + sample_end
        estimation_start = self.complementary_information['estimation_start']
        left_element = '{:7}{:>31}'.format('Sample:', sample)  
        right_element = '{:11}{:>27}'.format('Est. start:', estimation_start)  
        lines.append(left_element + '    ' + right_element)        
        # second row: observations and estimation complete   
        T = str(self.model.T)
        estimation_end = self.complementary_information['estimation_end']
        left_element = '{:17}{:>21}'.format('No. observations:', T)
        right_element = '{:14}{:>24}'.format('Est. complete:', estimation_end)
        lines.append(left_element + '    ' + right_element)                
        # third row: frequency and lags
        frequency = self.complementary_information['frequency']
        left_element = '{:10}{:>28}'.format('Frequency:', frequency)
        right_element = '                                      '
        lines.append(left_element + '    ' + right_element)   
        self.estimation_summary += lines   
    
    
    def __add_mfbvar_summary(self):
        # make list of regressors
        self.__mfbvar_regressors_and_index()
        # loop over equations
        for i in range(self.model.n):
            # add coefficient summary
            self.__add_mfbvar_coefficient_summary(i)     
            # residual and shock variance
            self.__add_mfbvar_shock_variance_summary(i)
            # in-sample fit criteria
            self.__add_mfbvar_insample_evaluation(i)    
        # residual variance-covariance matrix
        self.__add_mfbvar_residual_matrix_summary()
        # structural shocks variance-covariance matrix
        self.__add_mfbvar_shock_matrix_summary()
        # structural identification matrix
        self.__add_mfbvar_structural_identification_matrix_summary()
        # add forecast evaluation criteria, if relevant
        self.__add_mfbvar_forecast_evaluation()
    

    def __mfbvar_regressors_and_index(self):
        endogenous = self.complementary_information['endogenous_variables']
        exogenous = self.complementary_information['exogenous_variables']
        constant = self.model.constant
        trend = self.model.trend
        quadratic_trend = self.model.quadratic_trend
        n = self.model.n
        m = self.model.m
        p = self.model.p
        k = self.model.k
        regressors = cu.make_regressors(endogenous, exogenous, constant, trend, quadratic_trend, n, p)
        coefficient_index = cu.make_index(n, m, p, k)
        self.__mfbvar_regressors = regressors
        self.__mfbvar_coefficient_index = coefficient_index


    def __add_mfbvar_coefficient_summary(self, i):
        lines = []
        endogenous_variables = self.complementary_information['endogenous_variables']
        credibility_level = self.model.credibility_level
        lines += cu.equation_header('Equation: ' + endogenous_variables[i])
        lines += cu.coefficient_header(credibility_level)
        lines.append(cu.string_line('VAR coefficients beta:'))
        # loop over equation coefficients
        coefficient_index = self.__mfbvar_coefficient_index
        regressors = self.__mfbvar_regressors
        for j in range(self.model.k):
            regressor = regressors[j]
            index = int(coefficient_index[j])
            coefficient = self.model.beta_estimates[index,i,0]
            standard_deviation = self.model.beta_estimates[index,i,3]
            lower_bound = self.model.beta_estimates[index,i,1]
            upper_bound = self.model.beta_estimates[index,i,2]
            lines.append(cu.parameter_estimate_line(regressor, coefficient,\
                          standard_deviation, lower_bound, upper_bound))
        lines += [cu.hyphen_dashed_line()]
        self.estimation_summary += lines    
            
            
    def __add_mfbvar_shock_variance_summary(self, i):
        lines = []
        residual_variance = self.model.Sigma_estimates[i,i]
        if hasattr(self.model, 'Gamma_estimates'):
            shock_variance = self.model.Gamma_estimates[i]
        else:
            shock_variance = ''
        lines.append(cu.variance_line(residual_variance, shock_variance))
        self.estimation_summary += lines    
        

    def __add_mfbvar_insample_evaluation(self, i):
        # initiate lines
        lines = []
        # check if in-sample evaluation has been conducted
        if hasattr(self.model, 'insample_evaluation'):
            lines += [cu.hyphen_dashed_line()]
            ssr = self.model.insample_evaluation['ssr'][i]
            r2 = self.model.insample_evaluation['r2'][i]
            adj_r2 = self.model.insample_evaluation['adj_r2'][i]
            aic = []
            bic = []
            m_y = []
            lines += cu.insample_evaluation_lines(ssr, r2, adj_r2, m_y, aic, bic)
            self.estimation_summary += lines             
            
            
    def __add_mfbvar_residual_matrix_summary(self):
        Sigma = self.model.Sigma_estimates
        n = self.model.n
        endogenous_variables = self.complementary_information['endogenous_variables']        
        lines = []
        lines += cu.equation_header('Residual variance-covariance Sigma')
        lines += cu.variance_covariance_summary(Sigma, n, endogenous_variables, 'var.Sigma_estimates')
        self.estimation_summary += lines    
        
        
    def __add_mfbvar_shock_matrix_summary(self):
        if hasattr(self.model, 'Gamma_estimates'):
            Gamma = np.diag(self.model.Gamma_estimates)
            n = self.model.n
            endogenous_variables = self.complementary_information['endogenous_variables']        
            lines = []
            lines += cu.intermediate_header('Structural shocks variance-covariance Gamma')
            lines += cu.variance_covariance_summary(Gamma, n, endogenous_variables, 'var.Gamma_estimates')
            self.estimation_summary += lines          
        
        
    def __add_mfbvar_structural_identification_matrix_summary(self):  
        lines = []
        if hasattr(self.model, 'H_estimates'):
            H = self.model.H_estimates
            n = self.model.n
            endogenous_variables = self.complementary_information['endogenous_variables']        
            lines += cu.intermediate_header('Structural identification matrix H')
            lines += cu.variance_covariance_summary(H, n, endogenous_variables, 'var.H_estimates')
        self.estimation_summary += lines             
        
     
    def __add_mfbvar_forecast_evaluation(self):     
        lines = []
        if hasattr(self.model, 'forecast_evaluation_criteria'):  
            endogenous_variables = self.complementary_information['endogenous_variables']    
            forecast_evaluation_criteria = self.model.forecast_evaluation_criteria
            # regular forecast evaluation criteria
            lines += cu.equation_header('Forecast evaluation criteria')
            lines.append('                 RMSE        MAE       MAPE    Theil-U       Bias               ')
            rmse = forecast_evaluation_criteria['rmse']
            mae = forecast_evaluation_criteria['mae']
            mape = forecast_evaluation_criteria['mape']
            theil_u = forecast_evaluation_criteria['theil_u']
            bias = forecast_evaluation_criteria['bias']
            for i in range(self.model.n):
                lines.append(cu.forecast_evaluation_line(endogenous_variables[i], \
                             rmse[i], mae[i], mape[i], theil_u[i], bias[i]))
            # Bayesian criteria: log score
            if 'log_score' in forecast_evaluation_criteria:
                lines += cu.intermediate_header('Log score')
                log_score = forecast_evaluation_criteria['log_score']
                joint_log_score = forecast_evaluation_criteria['joint_log_score']
                lines += cu.forecast_evaluation_summary(log_score, joint_log_score, \
                         endogenous_variables, "var.forecast_evaluation_criteria['log_score']")
            # Bayesian criteria: CRPS
            if 'crps' in forecast_evaluation_criteria:
                lines += cu.intermediate_header('CRPS')
                crps = forecast_evaluation_criteria['crps']
                joint_crps = forecast_evaluation_criteria['joint_crps']
                lines += cu.forecast_evaluation_summary(crps, joint_crps, \
                         endogenous_variables, "var.forecast_evaluation_criteria['crps']")  
        lines.append(cu.equal_dashed_line())
        self.estimation_summary += lines     
 
 
    def __add_bdfm_summary(self):
        # make list of regressors
        self.__bdfm_regressors_and_index()
        # add loadings summary
        self.__add_bdfm_loadings_summary()
        # add factor summary
        self.__add_bdfm_factor_summary()
        # add residual summary
        self.__add_bdfm_residual_summary()
        # add shock variance summary
        self.__add_bdfm_shock_variance_summary()
        # add forecast evaluation criteria, if relevant
        self.__add_bdfm_forecast_evaluation()    

    
    def __bdfm_regressors_and_index(self):
        k = self.model.k
        l = self.model.l
        m = self.model.m
        q = self.model.q
        p = self.model.p
        r = self.model.r
        loadings_regressors, factor_regressors, residual_regressors = cu.make_dfm_regressors(m, q, p, r)
        loadings_index, factor_index = cu.make_dfm_index(k, l, m, q, p)
        self.__bdfm_loadings_regressors = loadings_regressors
        self.__bdfm_factor_regressors = factor_regressors
        self.__bdfm_residual_regressors = residual_regressors
        self.__bdfm_loadings_index = loadings_index   
        self.__bdfm_factor_index = factor_index   
 
    
    def __add_bdfm_loadings_summary(self):
        lines = []
        endogenous_variables = self.complementary_information['endogenous_variables']
        credibility_level = self.model.credibility_level
        loadings_regressors = self.__bdfm_loadings_regressors
        loadings_index = self.__bdfm_loadings_index
        if hasattr(self.model, 'insample_evaluation'):
            evaluation = True
            insample_evaluation = self.model.insample_evaluation
        else:
            evaluation = False
        for i in range(self.model.n):
            lines += cu.equation_header('Loadings equation: ' + endogenous_variables[i])
            lines += cu.coefficient_header(credibility_level)
            lines.append(cu.string_line('Loadings coefficients lambda:'))
            # loop over equation coefficients
            for j in range(self.model.l):
                regressor = loadings_regressors[j]
                index = int(loadings_index[j])
                coefficient = self.model.lambda_estimates[i,index,0]
                standard_deviation = self.model.lambda_estimates[i,index,3]
                lower_bound = self.model.lambda_estimates[i,index,1]
                upper_bound = self.model.lambda_estimates[i,index,2]
                lines.append(cu.parameter_estimate_line(regressor, coefficient,\
                              standard_deviation, lower_bound, upper_bound))
            if evaluation:
                lines += [cu.hyphen_dashed_line()]
                ssr = insample_evaluation['ssr'][i]
                r2 = insample_evaluation['r2'][i]
                adj_r2 = insample_evaluation['adj_r2'][i]
                lines += cu.insample_evaluation_lines(ssr, r2, adj_r2, [], [], []) 
        self.estimation_summary += lines   
    
 
    def __add_bdfm_factor_summary(self):
        lines = []
        variables = ['factor ' + str(i+1) for i in range(self.model.m)]
        credibility_level = self.model.credibility_level
        factor_regressors = self.__bdfm_factor_regressors
        factor_index = self.__bdfm_factor_index
        for i in range(self.model.m):
            lines += cu.equation_header('Factor equation: ' + variables[i])
            lines += cu.coefficient_header(credibility_level)
            lines.append(cu.string_line('VAR coefficients beta:'))
            # loop over equation coefficients
            for j in range(self.model.k):
                regressor = factor_regressors[j]
                index = int(factor_index[j])
                coefficient = self.model.beta_estimates[index,i,0]
                standard_deviation = self.model.beta_estimates[index,i,3]
                lower_bound = self.model.beta_estimates[index,i,1]
                upper_bound = self.model.beta_estimates[index,i,2]
                lines.append(cu.parameter_estimate_line(regressor, coefficient,\
                              standard_deviation, lower_bound, upper_bound))
        self.estimation_summary += lines      
 
 
    def __add_bdfm_residual_summary(self):
        if self.model.r > 0:
            lines = []
            endogenous_variables = self.complementary_information['endogenous_variables']
            credibility_level = self.model.credibility_level
            residual_regressors = self.__bdfm_residual_regressors
            for i in range(self.model.n):
                lines += cu.equation_header('Residual equation: ' + endogenous_variables[i])
                lines += cu.coefficient_header(credibility_level)
                lines.append(cu.string_line('AR coefficients gamma:'))
                # loop over equation coefficients
                for j in range(self.model.r):
                    regressor = residual_regressors[j]
                    index = j
                    coefficient = self.model.gamma_estimates[i,index,0]
                    standard_deviation = self.model.gamma_estimates[i,index,3]
                    lower_bound = self.model.gamma_estimates[i,index,1]
                    upper_bound = self.model.gamma_estimates[i,index,2]
                    lines.append(cu.parameter_estimate_line(regressor, coefficient,\
                                  standard_deviation, lower_bound, upper_bound))
            self.estimation_summary += lines      
     
           
    def __add_bdfm_shock_variance_summary(self):
        lines = []
        lines += cu.equation_header('Structural shock variance')
        residual_variance = self.model.sigma
        shock_variance = self.model.omega
        lines.append(cu.variance_line(residual_variance, shock_variance))
        self.estimation_summary += lines     
 
    
    def __add_bdfm_forecast_evaluation(self):  
        lines = []
        if hasattr(self.model, 'forecast_evaluation_criteria'):  
            endogenous_variables = self.complementary_information['endogenous_variables']    
            forecast_evaluation_criteria = self.model.forecast_evaluation_criteria
            # regular forecast evaluation criteria
            lines += cu.equation_header('Forecast evaluation criteria')
            lines.append('                 RMSE        MAE       MAPE    Theil-U       Bias               ')
            rmse = forecast_evaluation_criteria['rmse']
            mae = forecast_evaluation_criteria['mae']
            mape = forecast_evaluation_criteria['mape']
            theil_u = forecast_evaluation_criteria['theil_u']
            bias = forecast_evaluation_criteria['bias']
            for i in range(self.model.n):
                lines.append(cu.forecast_evaluation_line(endogenous_variables[i], \
                             rmse[i], mae[i], mape[i], theil_u[i], bias[i]))
            # Bayesian criteria: log score
            if 'log_score' in forecast_evaluation_criteria:
                lines += cu.intermediate_header('Log score')
                log_score = forecast_evaluation_criteria['log_score']
                joint_log_score = forecast_evaluation_criteria['joint_log_score']
                lines += cu.forecast_evaluation_summary(log_score, joint_log_score, \
                         endogenous_variables, "var.forecast_evaluation_criteria['log_score']")
            # Bayesian criteria: CRPS
            if 'crps' in forecast_evaluation_criteria:
                lines += cu.intermediate_header('CRPS')
                crps = forecast_evaluation_criteria['crps']
                joint_crps = forecast_evaluation_criteria['joint_crps']
                lines += cu.forecast_evaluation_summary(crps, joint_crps, \
                         endogenous_variables, "var.forecast_evaluation_criteria['crps']")  
        lines.append(cu.equal_dashed_line())
        self.estimation_summary += lines       
 
    
    def __add_midas_summary(self):
        
        # add constant summary
        self.__add_midas_constant_summary()         
        # add endogenous lags summary
        self.__add_midas_endogenous_summary()        
        # add exogenous lags summary
        self.__add_midas_exogenous_summary()          
        # add residual variance summary
        self.__add_midas_residual_variance_summary()  
        # add in-sample evaluation criteria
        self.__add_midas_insample_evaluation()         
        # add forecast evaluation criteria, if relevant
        self.__add_midas_forecast_evaluation() 
    
    
    def __add_midas_constant_summary(self):

        lines = []
        lines += cu.equation_header('Constant')
        lines += cu.coefficient_header(self.model.credibility_level)
        coefficient = self.model.beta_estimates[0,0]
        standard_deviation = self.model.beta_estimates[0,3]
        lower_bound = self.model.beta_estimates[0,1]
        upper_bound = self.model.beta_estimates[0,2]        
        lines.append(cu.parameter_estimate_line('Constant c:', coefficient,\
                      standard_deviation, lower_bound, upper_bound))         
        self.estimation_summary += lines         
        
 
    def __add_midas_endogenous_summary(self):       
        if self.model.p > 0:
            lines = []
            variable = self.complementary_information['endogenous_variables'][0]
            lines += cu.equation_header('Endogenous lags: ' + variable)
            lines += cu.coefficient_header(self.model.credibility_level)
            lines.append(cu.string_line('Autoregressive coefficients alpha:'))
            # loop over equation coefficients
            for i in range(self.model.p):      
                regressor = variable + ' (-' + str(i+1) + ')'
                coefficient = self.model.beta_estimates[1+i,0]
                standard_deviation = self.model.beta_estimates[1+i,3]
                lower_bound = self.model.beta_estimates[1+i,1]
                upper_bound = self.model.beta_estimates[1+i,2]      
                lines.append(cu.parameter_estimate_line(regressor, coefficient,\
                              standard_deviation, lower_bound, upper_bound))        
            self.estimation_summary += lines    
        
        
    def __add_midas_exogenous_summary(self):
        lines = []
        index = self.model.p
        exogenous_variables = self.complementary_information['exogenous_variables']
        for i in range(self.model.n):
            lines += cu.equation_header('Exogenous lags: ' + exogenous_variables[i])
            lines += cu.coefficient_header(self.model.credibility_level)
            lines.append(cu.string_line('Autoregressive coefficients beta:'))            
            # loop over equation coefficients
            for j in range(self.model.p_[i]+1):
                index += 1
                if j == 0:
                    regressor = exogenous_variables[i]
                else:    
                    regressor = exogenous_variables[i] + ' (-' + str(j) + ')'
                coefficient = self.model.beta_estimates[index,0]
                standard_deviation = self.model.beta_estimates[index,3]
                lower_bound = self.model.beta_estimates[index,1]
                upper_bound = self.model.beta_estimates[index,2]    
                lines.append(cu.parameter_estimate_line(regressor, coefficient,\
                              standard_deviation, lower_bound, upper_bound))        
        self.estimation_summary += lines         

 
    def __add_midas_residual_variance_summary(self):
        lines = []
        lines += cu.equation_header('Residual variance sigma')
        residual_variance = self.model.sigma_estimates[0]
        shock_variance = ' '
        lines.append(cu.variance_line(residual_variance, shock_variance))
        self.estimation_summary += lines
 
    
    def __add_midas_insample_evaluation(self):
        lines = []
        lines += cu.equation_header('In-sample evaluation criteria')
        if hasattr(self.model,'insample_evaluation'):
            ssr = self.model.insample_evaluation['ssr']
            r2 = self.model.insample_evaluation['r2']
            adj_r2 = self.model.insample_evaluation['adj_r2']
            lines += cu.insample_evaluation_lines(ssr, r2, adj_r2,[], [], [])
            self.estimation_summary += lines 
    
    
    def __add_midas_forecast_evaluation(self):
        lines = []
        if hasattr(self.model,'forecast_evaluation_criteria'):  
            rmse = self.model.forecast_evaluation_criteria['rmse']
            mae = self.model.forecast_evaluation_criteria['mae']
            mape = self.model.forecast_evaluation_criteria['mape']
            theil_u = self.model.forecast_evaluation_criteria['theil_u']
            bias = self.model.forecast_evaluation_criteria['bias']
            log_score = self.model.forecast_evaluation_criteria['log_score']
            crps = self.model.forecast_evaluation_criteria['crps']
            lines += cu.equation_header('Forecast evaluation criteria')
            lines += cu.forecast_evaluation_lines(rmse, mae, mape, theil_u, bias, log_score, crps)
        lines.append(cu.equal_dashed_line())
        self.estimation_summary += lines  

    
    def _make_nowcasting_application_summary(self):
        # get model type, midas regression, mfbvar or bdfm
        model_type = self.complementary_information['model_type'] 
        # mfbvar application summary
        if model_type == 1:
            self.__make_mfbvar_application_summary()
        # bdfm summary
        elif model_type == 2:
            self.__make_bdfm_application_summary()
        # midas summary
        elif model_type == 3:
            self.__make_midas_application_summary()       
 

    def _save_nowcasting_application(self, path):
        # get model type, midas regression, mfbvar or bdfm
        model_type = self.complementary_information['model_type'] 
        # mfbvar application summary
        if model_type == 1:
            self.__save_mfbvar_application(path)
        # bdfm summary
        elif model_type == 2:
            self.__save_bdfm_application(path)
        # midas summary
        elif model_type == 3:
            self.__save_midas_application(path)       

    
    def __make_mfbvar_application_summary(self):
        # in-sample fit measures
        self.__make_mfbvar_insample_fit_summary()
        # forecasts
        self.__make_mfbvar_forecast_summary()        
        # conditional forecasts
        self.__make_mfbvar_conditional_forecast_summary()  
        # impulse response function
        self.__make_mfbvar_irf_summary()        
        # forecast error variance decomposition
        self.__make_mfbvar_fevd_summary()    
        # historical decomposition
        self.__make_mfbvar_hd_summary()        
        
        
    def __save_mfbvar_application(self, path):
        # save in-sample fit
        self.__save_mfbvar_insample_fit_summary(path)
        # save forecasts
        self.__save_mfbvar_forecast_summary(path)      
        # save conditional forecasts
        self.__save_mfbvar_conditional_forecast_summary(path)         
        # save impulse response function
        self.__save_mfbvar_irf_summary(path)  
        # save forecast error variance decomposition
        self.__save_mfbvar_fevd_summary(path)  
        # save historical decomposition
        self.__save_mfbvar_hd_summary(path)     
 
 
    def __make_mfbvar_insample_fit_summary(self):
        # run only if in-sample fit has been run
        if hasattr(self.model, 'fitted_estimates'):
            fitted_dataframe = []
            Y = self.model.Y
            endogenous_variables = self.complementary_information['endogenous_variables']
            fitted = self.model.fitted_estimates
            residuals = self.model.residual_estimates
            n = self.model.n
            p = self.model.p
            index = self.complementary_information['dates'][p:]
            for i in range (n):
                variable = endogenous_variables[i]
                header = [variable+'_actual', variable+'_fit_med', variable+'_fit_low', variable+'_fit_upp', \
                          variable+'_res_med', variable+'_res_low', variable+'_res_upp']
                data = np.vstack((Y[:,i], fitted[:,i,0], fitted[:,i,1], fitted[:,i,2], \
                       residuals[:,i,0], residuals[:,i,1], residuals[:,i,2])).T
                variable_dataframe = pd.DataFrame(index=index, columns=header, data=data)
                fitted_dataframe.append(variable_dataframe)
            fitted_dataframe = pd.concat(fitted_dataframe,axis=1)
            self.application_summary['insample_fit'] = fitted_dataframe 
            
            
    def __make_mfbvar_forecast_summary(self):
        # run only if forecast has been run
        if hasattr(self.model, 'forecast_estimates'):
            forecast_dataframe = []
            endogenous_variables = self.complementary_information['endogenous_variables']
            n = self.model.n
            p = self.model.p
            Y = self.model.Y
            insample_index = self.complementary_information['dates'][p:]
            forecasts = self.model.forecast_estimates
            forecast_index = self.complementary_information['forecast_dates']
            for i in range (n):
                variable = endogenous_variables[i]
                header = [variable+'_actual', variable+'_med', variable+'_low', variable+'_upp']
                insample_dataframe = pd.DataFrame(index=insample_index,columns=header)
                insample_dataframe.iloc[:,0] = Y[:,i]
                insample_dataframe.iloc[-1,:] = insample_dataframe.iloc[-1,0]
                prediction_dataframe = pd.DataFrame(index=forecast_index,columns=header)
                prediction_dataframe.iloc[:,1:4] = forecasts[:,i,:]
                variable_dataframe = pd.concat([insample_dataframe,prediction_dataframe],axis=0)
                forecast_dataframe.append(variable_dataframe)
            forecast_dataframe = pd.concat(forecast_dataframe,axis=1)
            self.application_summary['forecast'] = forecast_dataframe 
            
    
    def __make_mfbvar_conditional_forecast_summary(self):
        # run only if forecast has been run
        if hasattr(self.model, 'conditional_forecast_estimates') and len(self.model.conditional_forecast_estimates) != 0:
            forecast_dataframe = []
            endogenous_variables = self.complementary_information['endogenous_variables']
            n = self.model.n
            p = self.model.p
            Y = self.model.Y
            insample_index = self.complementary_information['dates'][p:]
            forecast_index = self.complementary_information['conditional_forecast_dates']
            forecasts = self.model.conditional_forecast_estimates
            for i in range (n):
                variable = endogenous_variables[i]
                header = [variable+'_actual', variable+'_med', variable+'_low', variable+'_upp']
                insample_dataframe = pd.DataFrame(index=insample_index,columns=header)
                insample_dataframe.iloc[:,0] = Y[:,i]
                insample_dataframe.iloc[-1,:] = insample_dataframe.iloc[-1,0]
                prediction_dataframe = pd.DataFrame(index=forecast_index,columns=header)
                prediction_dataframe.iloc[:,1:4] = forecasts[:,i,:]
                variable_dataframe = pd.concat([insample_dataframe,prediction_dataframe],axis=0)
                forecast_dataframe.append(variable_dataframe)
            forecast_dataframe = pd.concat(forecast_dataframe,axis=1)
            self.application_summary['conditional_forecast'] = forecast_dataframe  
    
    
    def __make_mfbvar_irf_summary(self):
        endogenous_variables = self.complementary_information['endogenous_variables']
        n = self.model.n
        # run only if IRF has been run
        if hasattr(self.model, 'irf_estimates'):        
            irf_dataframe = []    
            irf = self.model.irf_estimates
            index = np.arange(1,irf.shape[2]+1)
            for i in range(n):
                for j in range(n):
                    variable = endogenous_variables[i]
                    shock = 'shock' + str(j+1)
                    header = [variable+'_'+shock+'_med', variable+'_'+shock+'_low', variable+'_'+shock+'_upp']
                    data = irf[i,j,:,:]
                    variable_dataframe = pd.DataFrame(index=index, columns=header, data=data)
                    irf_dataframe.append(variable_dataframe)
            irf_dataframe = pd.concat(irf_dataframe,axis=1)
            self.application_summary['irf'] = irf_dataframe
        # run only if exogenous IRF have been computed
        if hasattr(self.model, 'exo_irf_estimates') and len(self.model.exo_irf_estimates) != 0:
            exo_irf_dataframe = []
            exogenous_variables = self.complementary_information['exogenous_variables']
            n_exo = len(exogenous_variables)
            exo_irf = self.model.exo_irf_estimates
            index = np.arange(1,exo_irf.shape[2]+1)
            for i in range(n):
                for j in range(n_exo):
                    variable = endogenous_variables[i]
                    shock = exogenous_variables[j]
                    header = [variable+'_'+shock+'_med', variable+'_'+shock+'_low', variable+'_'+shock+'_upp']
                    data = exo_irf[i,j,:,:]
                    variable_dataframe = pd.DataFrame(index=index, columns=header, data=data)
                    exo_irf_dataframe.append(variable_dataframe)
            exo_irf_dataframe = pd.concat(exo_irf_dataframe,axis=1)
            self.application_summary['exo_irf'] = exo_irf_dataframe
                 

    def __make_mfbvar_fevd_summary(self):  
        # run only if FEVD has been run
        if hasattr(self.model, 'fevd_estimates') and len(self.model.fevd_estimates) != 0:    
            fevd_dataframe = []
            endogenous_variables = self.complementary_information['endogenous_variables']
            n = self.model.n
            fevd = self.model.fevd_estimates
            index = np.arange(1,fevd.shape[2]+1)
            for i in range(n):
                for j in range(n):
                    variable = endogenous_variables[i]
                    shock = 'shock' + str(j+1)
                    header = [variable+'_'+shock+'_med', variable+'_'+shock+'_low', variable+'_'+shock+'_upp']
                    data = fevd[i,j,:,:]
                    variable_dataframe = pd.DataFrame(index=index, columns=header, data=data)
                    fevd_dataframe.append(variable_dataframe)
            fevd_dataframe = pd.concat(fevd_dataframe,axis=1)
            self.application_summary['fevd'] = fevd_dataframe


    def __make_mfbvar_hd_summary(self):
        # run only if HD has been run
        if hasattr(self.model, 'hd_estimates') and len(self.model.hd_estimates) != 0:  
            hd_dataframe = []
            endogenous_variables = self.complementary_information['endogenous_variables']
            n = self.model.n
            p = self.model.p
            hd = self.model.hd_estimates
            index = self.complementary_information['dates'][p:]        
            for i in range(n):
                for j in range(n):
                    variable = endogenous_variables[i]
                    shock = 'shock' + str(j+1)  
                    header = [variable+'_'+shock+'_med', variable+'_'+shock+'_low', variable+'_'+shock+'_upp']
                    data = hd[i,j,:,:]
                    variable_dataframe = pd.DataFrame(index=index, columns=header, data=data)
                    hd_dataframe.append(variable_dataframe)
            hd_dataframe = pd.concat(hd_dataframe,axis=1)
            self.application_summary['hd'] = hd_dataframe

        
    def __save_mfbvar_insample_fit_summary(self, path):
        if 'insample_fit' in self.application_summary:
            insample_fit_summary = self.application_summary['insample_fit']
            full_path = join(path, 'insample_fit.csv')
            insample_fit_summary.to_csv(path_or_buf = full_path)         
        
        
    def __save_mfbvar_forecast_summary(self, path):
        if 'forecast' in self.application_summary:
            forecast_summary = self.application_summary['forecast']
            full_path = join(path, 'forecast.csv')
            forecast_summary.to_csv(path_or_buf = full_path)         
        
        
    def __save_mfbvar_conditional_forecast_summary(self, path):
        if 'conditional_forecast' in self.application_summary:
            conditional_forecast_summary = self.application_summary['conditional_forecast']
            full_path = join(path, 'conditional_forecast.csv')
            conditional_forecast_summary.to_csv(path_or_buf = full_path)         
        
        
    def __save_mfbvar_irf_summary(self, path):
        if 'irf' in self.application_summary:
            irf_summary = self.application_summary['irf']
            full_path = join(path, 'irf.csv')
            irf_summary.to_csv(path_or_buf = full_path)          
        
        
    def __save_mfbvar_fevd_summary(self, path):
        if 'fevd' in self.application_summary:
            fevd_summary = self.application_summary['fevd']
            full_path = join(path, 'fevd.csv')
            fevd_summary.to_csv(path_or_buf = full_path)          
        
        
    def __save_mfbvar_hd_summary(self, path):
        if 'hd' in self.application_summary:
            hd_summary = self.application_summary['hd']
            full_path = join(path, 'hd.csv')
            hd_summary.to_csv(path_or_buf = full_path)          
            
            
    def __make_bdfm_application_summary(self):
        # in-sample fit measures
        self.__make_bdfm_insample_fit_summary()
        # forecasts
        self.__make_bdfm_forecast_summary()        
        # impulse response function
        self.__make_bdfm_irf_summary()
        # forecast error variance decomposition
        self.__make_bdfm_fevd_summary()    
        # historical decomposition
        self.__make_bdfm_hd_summary()              
            
            
    def __save_bdfm_application(self, path):
        # save in-sample fit
        self.__save_bdfm_insample_fit_summary(path)
        # save forecasts
        self.__save_bdfm_forecast_summary(path)            
        # save impulse response function
        self.__save_bdfm_irf_summary(path)  
        # save forecast error variance decomposition
        self.__save_bdfm_fevd_summary(path)  
        # save historical decomposition
        self.__save_bdfm_hd_summary(path)             
            
            
    def __make_bdfm_insample_fit_summary(self):
        if hasattr(self.model, 'fitted_estimates'):
            fitted_dataframe = []
            endogenous_variables = self.complementary_information['endogenous_variables']
            fitted = self.model.fitted_estimates
            residuals = self.model.residual_estimates
            n = self.model.n
            index = self.complementary_information['dates']
            for i in range (n):
                variable = endogenous_variables[i]
                header = [variable+'_fit_med', variable+'_fit_low', variable+'_fit_upp', \
                          variable+'_res_med', variable+'_res_low', variable+'_res_upp']
                data = np.vstack((fitted[:,i,0], fitted[:,i,1], fitted[:,i,2], \
                       residuals[:,i,0], residuals[:,i,1], residuals[:,i,2])).T
                variable_dataframe = pd.DataFrame(index=index, columns=header, data=data)
                fitted_dataframe.append(variable_dataframe)
            fitted_dataframe = pd.concat(fitted_dataframe,axis=1)
            self.application_summary['insample_fit'] = fitted_dataframe    
        factor_dataframe = []
        factors = self.model.f_estimates
        m = self.model.m
        for i in range (m):
            variable = 'factor_' + str(i+1)
            header = [variable+'_med', variable+'_low', variable+'_upp']
            data = np.vstack((factors[:,i,0], factors[:,i,1], factors[:,i,2])).T                    
            variable_dataframe = pd.DataFrame(index=index, columns=header, data=data)
            factor_dataframe.append(variable_dataframe)
        factor_dataframe = pd.concat(factor_dataframe,axis=1)
        self.application_summary['factors'] = factor_dataframe    
             
        
    def __make_bdfm_forecast_summary(self):
        if hasattr(self.model, 'forecast_estimates'):
            forecast_dataframe = []
            endogenous_variables = self.complementary_information['endogenous_variables']
            n = self.model.n
            Y = self.model.fitted_estimates[:,:,0]  
            insample_index = self.complementary_information['dates']
            forecasts = self.model.forecast_estimates
            forecast_index = self.complementary_information['forecast_dates']  
            for i in range (n):
                variable = endogenous_variables[i]
                header = [variable+'_actual', variable+'_med', variable+'_low', variable+'_upp']
                insample_dataframe = pd.DataFrame(index=insample_index,columns=header)
                insample_dataframe.iloc[:,0] = Y[:,i]        
                insample_dataframe.iloc[-1,:] = insample_dataframe.iloc[-1,0]        
                prediction_dataframe = pd.DataFrame(index=forecast_index,columns=header)
                prediction_dataframe.iloc[:,1:4] = forecasts[:,i,:]
                variable_dataframe = pd.concat([insample_dataframe,prediction_dataframe],axis=0)
                forecast_dataframe.append(variable_dataframe)       
            forecast_dataframe = pd.concat(forecast_dataframe,axis=1)
            self.application_summary['forecast'] = forecast_dataframe              
        
        
    def __make_bdfm_irf_summary(self):
        if hasattr(self.model, 'irf_estimates'): 
            endogenous_variables = self.complementary_information['endogenous_variables']
            n = self.model.n
            m = self.model.m
            irf_dataframe = []    
            irf = self.model.irf_estimates
            index = np.arange(1,irf.shape[2]+1)
            for i in range(n):     
                variable = endogenous_variables[i]
                for j in range(m+1):
                    if j < m:
                        shock = 'factor' + str(j+1) + '_shock'
                    elif j == m:
                        shock = 'own_shock'
                    header = [variable+'_'+shock+'_med', variable+'_'+shock+'_low', variable+'_'+shock+'_upp']
                    data = irf[i,j,:,:]     
        
                    variable_dataframe = pd.DataFrame(index=index, columns=header, data=data)
                    irf_dataframe.append(variable_dataframe)     
            irf_dataframe = pd.concat(irf_dataframe,axis=1)
            self.application_summary['irf'] = irf_dataframe
        
        
    def __make_bdfm_fevd_summary(self):  
        if hasattr(self.model, 'fevd_estimates'): 
            fevd_dataframe = []
            endogenous_variables = self.complementary_information['endogenous_variables']
            n = self.model.n
            m = self.model.m
            fevd = self.model.fevd_estimates
            index = np.arange(1,fevd.shape[2]+1)
            for i in range(n):
                variable = endogenous_variables[i]
                for j in range(m+1):
                    if j < m:
                        shock = 'factor' + str(j+1) + '_shock'
                    elif j == m:
                        shock = 'own_shock'
                    header = [variable+'_'+shock+'_med', variable+'_'+shock+'_low', variable+'_'+shock+'_upp']
                    data = fevd[i,j,:,:]
                    variable_dataframe = pd.DataFrame(index=index, columns=header, data=data)
                    fevd_dataframe.append(variable_dataframe)
            fevd_dataframe = pd.concat(fevd_dataframe,axis=1)
            self.application_summary['fevd'] = fevd_dataframe
        

    def __make_bdfm_hd_summary(self):
        # run only if HD has been run
        if hasattr(self.model, 'hd_estimates'):
            hd_dataframe = []
            endogenous_variables = self.complementary_information['endogenous_variables']
            n = self.model.n
            m = self.model.m
            hd = self.model.hd_estimates
            index = self.complementary_information['dates']              
            for i in range(n):
                variable = endogenous_variables[i]
                for j in range(m+1):
                    if j < m:
                        shock = 'factor' + str(j+1) + '_shock'
                    elif j == m:
                        shock = 'own_shock'            
                    header = [variable+'_'+shock+'_med', variable+'_'+shock+'_low', variable+'_'+shock+'_upp']
                    data = hd[i,j,:,:]
                    variable_dataframe = pd.DataFrame(index=index, columns=header, data=data)
                    hd_dataframe.append(variable_dataframe)
            hd_dataframe = pd.concat(hd_dataframe,axis=1)
            self.application_summary['hd'] = hd_dataframe

            
    def __save_bdfm_insample_fit_summary(self, path):
        if 'insample_fit' in self.application_summary:
            insample_fit_summary = self.application_summary['insample_fit']
            full_path = join(path, 'insample_fit.csv')
            insample_fit_summary.to_csv(path_or_buf = full_path)         
        
        
    def __save_bdfm_forecast_summary(self, path):
        if 'forecast' in self.application_summary:
            forecast_summary = self.application_summary['forecast']
            full_path = join(path, 'forecast.csv')
            forecast_summary.to_csv(path_or_buf = full_path)         
        
        
    def __save_bdfm_irf_summary(self, path):
        if 'irf' in self.application_summary:
            irf_summary = self.application_summary['irf']
            full_path = join(path, 'irf.csv')
            irf_summary.to_csv(path_or_buf = full_path)          
        
        
    def __save_bdfm_fevd_summary(self, path):
        if 'fevd' in self.application_summary:
            fevd_summary = self.application_summary['fevd']
            full_path = join(path, 'fevd.csv')
            fevd_summary.to_csv(path_or_buf = full_path)          
        
        
    def __save_bdfm_hd_summary(self, path):
        if 'hd' in self.application_summary:
            hd_summary = self.application_summary['hd']
            full_path = join(path, 'hd.csv')
            hd_summary.to_csv(path_or_buf = full_path)             
            
            
    def __make_midas_application_summary(self):
        # in-sample fit measures
        self.__make_midas_insample_fit_summary()
        # forecasts
        self.__make_midas_forecast_summary()        

        
    def __save_midas_application(self, path):
        # save in-sample fit
        self.__save_midas_insample_fit_summary(path)
        # save forecasts
        self.__save_midas_forecast_summary(path)      
            
            
    def __make_midas_insample_fit_summary(self):
        if hasattr(self.model, 'fitted_estimates'):
            index = self.complementary_information['dates'][-self.model.y.shape[0]:]
            columns = ['actual', 'fitted_med', 'fitted_low', 'fitted_upp', 'residual_med', 'residual_low', 'residual_upp']
            data = np.hstack((self.model.y.reshape(-1,1), self.model.fitted_estimates, self.model.residual_estimates))
            fitted_dataframe = pd.DataFrame(index = index, columns = columns, data = data)
            self.application_summary['insample_fit'] = fitted_dataframe            
            
            
    def __make_midas_forecast_summary(self):
        if len(self.model.forecast_estimates) > 1 or len(self.model.forecast_estimates[0]) > 0:
            forecasts = self.model.forecast_estimates
            insample_index = self.complementary_information['dates'][-self.model.y.shape[0]:]
            forecast_index = self.complementary_information['forecast_dates']
            variable = self.complementary_information['endogenous_variables'][0]
            header = [variable+'_actual', variable+'_med', variable+'_low', variable+'_upp']
            insample_dataframe = pd.DataFrame(index=insample_index,columns=header)
            insample_dataframe.iloc[:,0] = self.model.y
            insample_dataframe.iloc[-1,:] = insample_dataframe.iloc[-1,0]    
            prediction_dataframe = pd.DataFrame(index=forecast_index,columns=header)
            for t in range(len(forecasts)):
                prediction_dataframe.iloc[t,1:4] = forecasts[t]
            forecast_dataframe = pd.concat([insample_dataframe,prediction_dataframe],axis=0)
            self.application_summary['forecast'] = forecast_dataframe            
            
            
    def __save_midas_insample_fit_summary(self, path):
        if 'insample_fit' in self.application_summary:
            insample_fit_summary = self.application_summary['insample_fit']
            full_path = join(path, 'insample_fit.csv')
            insample_fit_summary.to_csv(path_or_buf = full_path)         
        
        
    def __save_midas_forecast_summary(self, path):
        if 'forecast' in self.application_summary:
            forecast_summary = self.application_summary['forecast']
            full_path = join(path, 'forecast.csv')
            forecast_summary.to_csv(path_or_buf = full_path) 
    
            