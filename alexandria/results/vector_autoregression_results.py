# imports
import numpy as np
import pandas as pd
from datetime import datetime
from os.path import isdir, join
import alexandria.processor.input_utilities as iu
import alexandria.console.console_utilities as cu


class VectorAutoregressionResults(object):
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass


    #---------------------------------------------------
    # Methods (Access = private)
    #--------------------------------------------------- 

        
    def _complete_var_information(self):  
        # endogenous and exogenous variables
        if 'endogenous_variables' not in self.complementary_information:
            n_endo = self.model.endogenous.shape[1]
            self.complementary_information['endogenous_variables'] = ['y' + str(i+1) for i in range(n_endo)]      
        if 'exogenous_variables' not in self.complementary_information:
            if len(self.model.exogenous) == 0:
                self.complementary_information['exogenous_variables'] = ['none']
            else:
                n_exo = self.model.exogenous.shape[1]
                self.complementary_information['exogenous_variables'] = ['x' + str(i+1) for i in range(n_exo)]        
        # sample dates
        if 'dates' not in self.complementary_information:
            T = self.model.T
            p = self.model.p
            self.complementary_information['dates'] = np.arange(1-p,T+1)
        # forecast dates
        if 'forecast_dates' not in self.complementary_information:
            if hasattr(self.model, 'forecast_estimates'):
                f_periods = self.model.forecast_estimates.shape[0]
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
        # proxy variables
        model_type = self.complementary_information['model_type']
        if model_type == 7 and 'proxy_variables' not in self.complementary_information:
            self.complementary_information['proxy_variables'] = ['proxy_' + str(i+1) for i in range(self.model.proxys.shape[1])]         
        # VAR options
        if 'insample_fit' not in self.complementary_information:
            self.complementary_information['insample_fit'] = '—'
        if 'constrained_coefficients' not in self.complementary_information:
            self.complementary_information['constrained_coefficients'] = '—'
        if 'sums_of_coefficients' not in self.complementary_information:
            self.complementary_information['sums_of_coefficients'] = '—'
        if 'initial_observation' not in self.complementary_information:
            self.complementary_information['initial_observation'] = '—'
        if 'long_run' not in self.complementary_information:
            self.complementary_information['long_run'] = '—'
        if 'stationary' not in self.complementary_information:
            self.complementary_information['stationary'] = '—'
        if 'marginal_likelihood' not in self.complementary_information:
            self.complementary_information['marginal_likelihood'] = '—'
        if 'hyperparameter_optimization' not in self.complementary_information:
            self.complementary_information['hyperparameter_optimization'] = '—'
        if 'coefficients_file' not in self.complementary_information:
            self.complementary_information['coefficients_file'] = '—'
        if 'long_run_file' not in self.complementary_information:
            self.complementary_information['long_run_file'] = '—'            
            
            
    def _add_var_tab_2_inputs(self):
        # initiate lines
        lines = []
        # header for tab 2
        lines.append('Specification')
        lines.append('-----------------')
        lines.append(' ')            
        # VAR type
        var_type = self.complementary_information['model_type']
        if var_type == 1:
            model = 'Maximum Likelihood VAR'
        elif var_type == 2:
            model = 'Minnesota Bayesian VAR'            
        elif var_type == 3:
            model = 'Normal-Wishart Bayesian VAR'          
        elif var_type == 4:
            model = 'Independent Bayesian VAR'
        elif var_type == 5:
            model = 'Dummy Observation Bayesian VAR'              
        elif var_type == 6:
            model = 'Large Bayesian VAR'
        elif var_type == 7:
            model = 'Bayesian Proxy-SVAR'            
        lines.append('VAR type: ' + model)
        # burn-in
        if var_type in [4,6,7]:
            burnin = str(self.model.burnin)
            lines.append('burn-in: ' + burnin)
        # iterations
        if var_type != 1:
            iterations = str(self.model.iterations)
            lines.append('iterations: ' + iterations)           
        # credibility level
        model_credibility = str(self.model.credibility_level)
        lines.append('credibility level: ' + model_credibility)  
        # constant, trend and quadratic trend
        constant = cu.bool_to_string(self.model.constant) 
        lines.append('constant: ' + constant)        
        trend = cu.bool_to_string(self.model.trend)   
        lines.append('trend: ' + trend)        
        quadratic_trend = cu.bool_to_string(self.model.quadratic_trend)    
        lines.append('quadratic trend: ' + quadratic_trend)
        # hyperparameters
        lags = str(self.model.p)
        lines.append('lags: ' + lags)
        if var_type != 1:
            if iu.is_numeric(self.model.ar_coefficients):
                ar_coefficients = str(self.model.ar_coefficients)
            else:
                ar_coefficients = iu.list_to_string(self.model.ar_coefficients)
            lines.append('AR coefficients: ' + ar_coefficients)
            pi1 = str(self.model.pi1)
            lines.append('pi1 (overall tightness): ' + pi1)
            if var_type in [2,4,6]:
                pi2 = str(self.model.pi2)
                lines.append('pi2 (cross-variable shrinkage): ' + pi2)
            pi3 = str(self.model.pi3)
            lines.append('pi3 (lag decay): ' + pi3)
            pi4 = str(self.model.pi4)
            lines.append('pi4 (exogenous slackness): ' + pi4)
            if var_type != 7:
                pi5 = str(self.model.pi5)
                lines.append('pi5 (sums-of-coefficients tightness): ' + pi5)
                pi6 = str(self.model.pi6)
                lines.append('pi6 (initial observation tightness): ' + pi6)
                pi7 = str(self.model.pi7)
                lines.append('pi7 (long-run tightness): ' + pi7)
        if var_type == 7:
            proxy_variables = iu.list_to_string(self.complementary_information['proxy_variables'])
            lines.append('proxy variables: ' + proxy_variables)
            lamda = str(self.model.lamda)
            lines.append('lambda (relevance): ' + lamda)
            proxy_prior = self.model.proxy_prior
            if proxy_prior == 1:
                prior_scheme = 'uninformative'
            elif proxy_prior == 2:
                prior_scheme = 'Minnesota'  
            lines.append('prior scheme: ' + prior_scheme)
        # VAR options: in-sample fit
        if type(self.complementary_information['insample_fit']) == bool:
            insample_fit = cu.bool_to_string(self.complementary_information['insample_fit'])   
        else:
            insample_fit = self.complementary_information['insample_fit']
        lines.append('in-sample fit: ' + insample_fit)         
        # VAR options: constrained coefficients
        if var_type in [2,4,6]:
            constrained_coefficients = cu.bool_to_string(self.model.constrained_coefficients)   
            lines.append('constrained coefficients: ' + constrained_coefficients) 
        # VAR options: sums-of-coefficients
        if var_type in [2,3,4,5,6]:
            sums_of_coefficients = cu.bool_to_string(self.model.sums_of_coefficients)   
            lines.append('sums-of-coefficients: ' + sums_of_coefficients)  
        # VAR options: dummy initial observation
        if var_type in [2,3,4,5,6]:
            initial_observation = cu.bool_to_string(self.model.dummy_initial_observation)   
            lines.append('dummy initial observation: ' + initial_observation)         
        # VAR options: long-run prior
        if var_type in [2,3,4,5,6]:
            long_run_prior = cu.bool_to_string(self.model.long_run_prior)   
            lines.append('long-run prior: ' + long_run_prior)        
        # VAR options: stationary prior
        if var_type in [2,3,4,5,6]:
            stationary_prior = cu.bool_to_string(self.model.stationary_prior)   
            lines.append('stationary prior: ' + stationary_prior)         
        # VAR options: marginal likelihood
        if var_type in [2,3,4]:
            if type(self.complementary_information['marginal_likelihood']) == bool:
                marginal_likelihood = cu.bool_to_string(self.complementary_information['marginal_likelihood'])   
            else:
                marginal_likelihood = self.complementary_information['marginal_likelihood']
            lines.append('marginal likelihood: ' + marginal_likelihood)         
        # VAR options: hyperparameter optimization
        if var_type == 2 or var_type == 3:
            hyperparameter_optimization = cu.bool_to_string(self.model.hyperparameter_optimization)
            lines.append('hyperparameter optimization: ' + hyperparameter_optimization)  
        # VAR options: constrained coefficient file
        if var_type in [2,4,6]:
            constrained_coefficient_file = self.complementary_information['coefficients_file']
            lines.append('constrained coefficients file: ' + constrained_coefficient_file) 
        # VAR options: long-run prior file
        if var_type in [2,3,4,5,6]:
            long_run_file = self.complementary_information['long_run_file']
            lines.append('long-run prior file: ' + long_run_file) 
        lines.append(' ')
        lines.append(' ')              
        self.input_summary += lines          
        
        
    def _add_var_tab_3_inputs(self):
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
        if type(self.complementary_information['conditional_forecast']) == bool:
            conditional_forecast = cu.bool_to_string(self.complementary_information['conditional_forecast'])
        else:
            conditional_forecast = self.complementary_information['conditional_forecast']
        lines.append('conditional forecast: ' + conditional_forecast) 
        conditional_forecast_credibility = str(self.complementary_information['conditional_forecast_credibility'])
        lines.append('credibility level, conditional forecasts: ' + conditional_forecast_credibility)
        # impulse response function
        if type(self.complementary_information['irf']) == bool:
            irf = cu.bool_to_string(self.complementary_information['irf'])
        else:
            irf = self.complementary_information['irf']
        lines.append('impulse response function: ' + irf) 
        irf_credibility = str(self.complementary_information['irf_credibility'])
        lines.append('credibility level, impulse response function: ' + irf_credibility)         
        # forecast error variance decomposition
        if type(self.complementary_information['fevd']) == bool:
            fevd = cu.bool_to_string(self.complementary_information['fevd'])
        else:
            fevd = self.complementary_information['fevd']
        lines.append('forecast error variance decomposition: ' + fevd)   
        fevd_credibility = str(self.complementary_information['fevd_credibility'])
        lines.append('credibility level, forecast error variance decomposition: ' + fevd_credibility)
        # historical decomposition
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
        conditional_forecast_file = self.complementary_information['conditional_forecast_file']
        lines.append('conditional forecast file: ' + conditional_forecast_file)        
        # forecast evaluation
        if type(self.complementary_information['forecast_evaluation']) == bool:
            forecast_evaluation = cu.bool_to_string(self.complementary_information['forecast_evaluation'])
        else:
            forecast_evaluation = self.complementary_information['forecast_evaluation']
        lines.append('forecast evaluation: ' + forecast_evaluation)       
        # irf periods
        if iu.is_numeric(self.complementary_information['irf_periods']):
            irf_periods = str(self.complementary_information['irf_periods'])
        else:
            irf_periods = self.complementary_information['irf_periods']
        lines.append('IRF periods: ' + irf_periods)        
        # structural identification
        if iu.is_numeric(self.complementary_information['structural_identification']):
            structural_identification = str(self.complementary_information['structural_identification'])
        else:
            structural_identification = self.complementary_information['structural_identification']
        lines.append('structural identification: ' + structural_identification)         
        # structural identification file        
        structural_identification_file = self.complementary_information['structural_identification_file']
        lines.append('structural identification file: ' + structural_identification_file)         
        lines.append(' ')
        lines.append(' ') 
        self.input_summary += lines 
        
        
    def _make_var_summary(self):
        # initiate string list
        self.estimation_summary = []
        # add model header
        self.__add_var_header() 
        # add estimation header
        self.__add_var_estimation_header()
        # make list of regressors
        self.__regressors_and_index()
        # loop over equations
        for i in range(self.model.n):
            # add coefficient summary
            self.__add_var_coefficient_summary(i) 
            # residual and shock variance
            self.__add_shock_variance_summary(i)
            # in-sample fit criteria
            self.__add_var_insample_evaluation(i)
        # residual variance-covariance matrix
        self.__add_residual_matrix_summary()
        # structural shocks variance-covariance matrix
        self.__add_shock_matrix_summary()
        # structural identification matrix
        self.__add_structural_identification_matrix_summary()
        # add forecast evaluation criteria, if relevant
        self.__add_var_forecast_evaluation() 
         
        
    def _make_var_application_summary(self):
        # in-sample fit measures
        self.__make_var_insample_fit_summary()
        # forecasts
        self.__make_var_forecast_summary()        
        # conditional forecasts
        self.__make_var_conditional_forecast_summary()  
        # impulse response function
        self.__make_var_irf_summary()        
        # forecast error variance decomposition
        self.__make_var_fevd_summary()    
        # historical decomposition
        self.__make_var_hd_summary()        
        
        
    def _save_var_application(self, path):
        # save in-sample fit
        self.__save_var_insample_fit_summary(path)
        # save forecasts
        self.__save_var_forecast_summary(path)      
        # save conditional forecasts
        self.__save_var_conditional_forecast_summary(path)         
        # save impulse response function
        self.__save_var_irf_summary(path)  
        # save forecast error variance decomposition
        self.__save_var_fevd_summary(path)  
        # save historical decomposition
        self.__save_var_hd_summary(path)          
        
        
    def __add_var_header(self):
        # recover model name and create header
        model_name = self.complementary_information['model_name']
        self.estimation_summary += cu.model_header(model_name)  
    
  
    def __add_var_estimation_header(self):
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
        lags = str(self.model.p)
        left_element = '{:10}{:>28}'.format('Frequency:', frequency)
        right_element = '{:5}{:>33}'.format('Lags:', lags)
        lines.append(left_element + '    ' + right_element)   
        self.estimation_summary += lines   
    

    def __regressors_and_index(self):
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
        self.__regressors = regressors
        self.__coefficient_index = coefficient_index


    def __add_var_coefficient_summary(self, i):
        lines = []
        endogenous_variables = self.complementary_information['endogenous_variables']
        credibility_level = self.model.credibility_level
        lines += cu.equation_header('Equation: ' + endogenous_variables[i])
        lines += cu.coefficient_header(credibility_level)
        lines.append(cu.string_line('VAR coefficients beta:'))
        # loop over equation coefficients
        coefficient_index = self.__coefficient_index
        regressors = self.__regressors
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
            
            
    def __add_shock_variance_summary(self, i):
        lines = []
        residual_variance = self.model.Sigma_estimates[i,i]
        if hasattr(self.model, 'Gamma_estimates'):
            shock_variance = self.model.Gamma_estimates[i]
        else:
            shock_variance = ''
        lines.append(cu.variance_line(residual_variance, shock_variance))
        # lines += [cu.hyphen_dashed_line()]
        self.estimation_summary += lines    
        

    def __add_var_insample_evaluation(self, i):
        # initiate lines
        lines = []
        # check if in-sample evaluation has been conducted
        if hasattr(self.model, 'insample_evaluation'):
            lines += [cu.hyphen_dashed_line()]
            ssr = self.model.insample_evaluation['ssr'][i]
            r2 = self.model.insample_evaluation['r2'][i]
            adj_r2 = self.model.insample_evaluation['adj_r2'][i]
            model_type = self.complementary_information['model_type']
            if model_type == 1:
                aic = self.model.insample_evaluation['aic']
                bic = self.model.insample_evaluation['bic']
            else:
                aic = []
                bic = []
            if hasattr(self.model,'m_y'): 
                m_y = self.model.m_y
            else:
                m_y = []
            lines += cu.insample_evaluation_lines(ssr, r2, adj_r2, m_y, aic, bic)
            self.estimation_summary += lines             
            
            
    def __add_residual_matrix_summary(self):
        Sigma = self.model.Sigma_estimates
        n = self.model.n
        endogenous_variables = self.complementary_information['endogenous_variables']        
        lines = []
        lines += cu.equation_header('Residual variance-covariance Sigma')
        lines += cu.variance_covariance_summary(Sigma, n, endogenous_variables, 'var.Sigma_estimates')
        self.estimation_summary += lines    
        
        
    def __add_shock_matrix_summary(self):
        if hasattr(self.model, 'Gamma_estimates'):
            Gamma = np.diag(self.model.Gamma_estimates)
            n = self.model.n
            endogenous_variables = self.complementary_information['endogenous_variables']        
            lines = []
            lines += cu.intermediate_header('Structural shocks variance-covariance Gamma')
            lines += cu.variance_covariance_summary(Gamma, n, endogenous_variables, 'var.Gamma_estimates')
            self.estimation_summary += lines          
        
        
    def __add_structural_identification_matrix_summary(self):  
        lines = []
        if hasattr(self.model, 'H_estimates'):
            H = self.model.H_estimates
            n = self.model.n
            endogenous_variables = self.complementary_information['endogenous_variables']        
            lines += cu.intermediate_header('Structural identification matrix H')
            lines += cu.variance_covariance_summary(H, n, endogenous_variables, 'var.H_estimates')
        self.estimation_summary += lines             
        
     
    def __add_var_forecast_evaluation(self):     
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
 
    
    def __make_var_insample_fit_summary(self):
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
            
            
    def __make_var_forecast_summary(self):
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
                insample_dataframe.iloc[:,0] = Y[:,0]
                insample_dataframe.iloc[-1,:] = insample_dataframe.iloc[-1,0]
                prediction_dataframe = pd.DataFrame(index=forecast_index,columns=header)
                prediction_dataframe.iloc[:,1:4] = forecasts[:,i,:]
                variable_dataframe = pd.concat([insample_dataframe,prediction_dataframe],axis=0)
                forecast_dataframe.append(variable_dataframe)
            forecast_dataframe = pd.concat(forecast_dataframe,axis=1)
            self.application_summary['forecast'] = forecast_dataframe 
            
    
    def __make_var_conditional_forecast_summary(self):
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
    
    
    def __make_var_irf_summary(self):
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
                 

    def __make_var_fevd_summary(self):  
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


    def __make_var_hd_summary(self):
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

        
    def __save_var_insample_fit_summary(self, path):
        if 'insample_fit' in self.application_summary:
            insample_fit_summary = self.application_summary['insample_fit']
            full_path = join(path, 'insample_fit.csv')
            insample_fit_summary.to_csv(path_or_buf = full_path)         
        
        
    def __save_var_forecast_summary(self, path):
        if 'forecast' in self.application_summary:
            forecast_summary = self.application_summary['forecast']
            full_path = join(path, 'forecast.csv')
            forecast_summary.to_csv(path_or_buf = full_path)         
        
        
    def __save_var_conditional_forecast_summary(self, path):
        if 'conditional_forecast' in self.application_summary:
            conditional_forecast_summary = self.application_summary['conditional_forecast']
            full_path = join(path, 'conditional_forecast.csv')
            conditional_forecast_summary.to_csv(path_or_buf = full_path)         
        
        
    def __save_var_irf_summary(self, path):
        if 'irf' in self.application_summary:
            irf_summary = self.application_summary['irf']
            full_path = join(path, 'irf.csv')
            irf_summary.to_csv(path_or_buf = full_path)          
        
        
    def __save_var_fevd_summary(self, path):
        if 'fevd' in self.application_summary:
            fevd_summary = self.application_summary['fevd']
            full_path = join(path, 'fevd.csv')
            fevd_summary.to_csv(path_or_buf = full_path)          
        
        
    def __save_var_hd_summary(self, path):
        if 'hd' in self.application_summary:
            hd_summary = self.application_summary['hd']
            full_path = join(path, 'hd.csv')
            hd_summary.to_csv(path_or_buf = full_path)          
        
        