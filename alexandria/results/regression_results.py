# imports
import numpy as np
import pandas as pd
from datetime import datetime
from os.path import isdir, join
import alexandria.processor.input_utilities as iu
import alexandria.console.console_utilities as cu


class RegressionResults(object):
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass


    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------
        
    
    def _complete_regression_information(self):  
        # endogenous and exogenous variables
        if 'endogenous_variables' not in self.complementary_information:
            self.complementary_information['endogenous_variables'] = ['y']
        if 'exogenous_variables' not in self.complementary_information:
            n_exo = self.model.exogenous.shape[1]
            self.complementary_information['exogenous_variables'] = ['x' + str(i+1) for i in range(n_exo)]
        # sample dates
        if 'dates' not in self.complementary_information:
            n = self.model.n
            self.complementary_information['dates'] = np.arange(1,n+1)
        # heteroscedastic variables
        model_type = self.complementary_information['model_type']
        if model_type == 5 and 'heteroscedastic_variables' not in self.complementary_information:
            self.complementary_information['heteroscedastic_variables'] = ['z' + str(i+1) for i in range(self.model.Z.shape[1])] 
        # regression options
        if 'insample_fit' not in self.complementary_information:
            self.complementary_information['insample_fit'] = '—'
        if 'marginal_likelihood' not in self.complementary_information:
            self.complementary_information['marginal_likelihood'] = '—'
        if 'hyperparameter_optimization' not in self.complementary_information:
            self.complementary_information['hyperparameter_optimization'] = '—'      
        if 'optimization_type' not in self.complementary_information:
            self.complementary_information['optimization_type'] = '—'      
    

    def _add_regression_tab_2_inputs(self):
        # initiate lines
        lines = []
        # header for tab 2
        lines.append('Specification')
        lines.append('-----------------')
        lines.append(' ')
        # regression type
        regression_type = self.complementary_information['model_type']
        if regression_type == 1:
            model = 'Maximum Likelihood Regression'
        elif regression_type == 2:
            model = 'Simple Bayesian Regression'            
        elif regression_type == 3:
            model = 'Hierarchical Bayesian Regression'          
        elif regression_type == 4:
            model = 'Independent Bayesian Regression'  
        elif regression_type == 5:
            model = 'Heteroscedastic Bayesian Regression'              
        elif regression_type == 6:
            model = 'Autocorrelated Bayesian Regression'
        lines.append('regression type: ' + model)
        # burn-in and iterations
        if regression_type == 4 or regression_type == 5 or regression_type == 6:
            iterations = str(self.model.iterations)
            burn = str(self.model.burn)
            lines.append('iterations: ' + iterations)
            lines.append('burn-in: ' + burn)     
        # credibility level
        model_credibility = str(self.model.credibility_level)
        lines.append('credibility level: ' + model_credibility)    
        # hyperparameters: b and V
        if regression_type != 1:
            b = iu.list_to_string(self.model.b)
            V = iu.list_to_string(np.diag(self.model.V))
            lines.append('b: ' + b)          
            lines.append('V: ' + V)
        # hyperparameters: alpha and delta
        if regression_type != 1 and regression_type != 2:
            alpha = str(self.model.alpha)
            delta = str(self.model.delta)
            lines.append('alpha: ' + alpha)
            lines.append('delta: ' + delta)    
        # hyperparameters: heteroscedastic regression
        if regression_type == 5:       
            g = iu.list_to_string(self.model.g)
            Q = iu.list_to_string(np.diag(self.model.Q))
            tau = str(self.model.tau)
            thinning = cu.bool_to_string(self.model.thinning)
            thinning_frequency = str(self.model.thinning_frequency)
            Z_variables = iu.list_to_string(self.complementary_information['heteroscedastic_variables'])
            lines.append('g: ' + g)
            lines.append('Q: ' + Q)        
            lines.append('tau: ' + tau)  
            lines.append('thinning: ' + thinning)  
            lines.append('thinning frequency: ' + thinning_frequency)  
            lines.append('Z variables: ' + Z_variables)  
        # hyperparameters: autocorrelated regression
        if regression_type == 6:          
            q = str(self.model.q)
            p = iu.list_to_string(self.model.p)
            H = iu.list_to_string(np.diag(self.model.H))
            lines.append('q: ' + q)
            lines.append('p: ' + p)        
            lines.append('H: ' + H)         
        # constant and constant prior  
        constant = cu.bool_to_string(self.model.constant)           
        lines.append('constant: ' + constant)
        if regression_type != 1:   
            b_constant = str(self.model.b_constant) 
            V_constant = str(self.model.V_constant)  
            lines.append('b (constant): ' + b_constant)            
            lines.append('V (constant): ' + V_constant)  
        # trend and trend prior  
        trend = cu.bool_to_string(self.model.trend)                 
        lines.append('trend: ' + trend)
        if regression_type != 1: 
            b_trend = str(self.model.b_trend) 
            V_trend = str(self.model.V_trend) 
            lines.append('b (trend): ' + b_trend)            
            lines.append('V (trend): ' + V_trend) 
        # quadratic trend and quadratic trend prior  
        quadratic_trend = cu.bool_to_string(self.model.quadratic_trend)                 
        lines.append('quadratic trend: ' + quadratic_trend)
        if regression_type != 1:  
            b_quadratic_trend = str(self.model.b_quadratic_trend) 
            V_quadratic_trend = str(self.model.V_quadratic_trend) 
            lines.append('b (quadratic trend): ' + b_quadratic_trend)            
            lines.append('V (quadratic trend): ' + V_quadratic_trend) 
        # in-sample fit
        if type(self.complementary_information['insample_fit']) == bool:
            insample_fit = cu.bool_to_string(self.complementary_information['insample_fit'])   
        else:
            insample_fit = self.complementary_information['insample_fit']
        lines.append('in-sample fit: ' + insample_fit) 
        # marginal likelihood
        if regression_type != 1:
            if type(self.complementary_information['marginal_likelihood']) == bool:
                marginal_likelihood = cu.bool_to_string(self.complementary_information['marginal_likelihood'])   
            else:
                marginal_likelihood = self.complementary_information['marginal_likelihood']
            lines.append('marginal likelihood: ' + marginal_likelihood)
        # hyperparameter optimization
        if regression_type == 2 or regression_type == 3:  
            if type(self.complementary_information['hyperparameter_optimization']) == bool:
                hyperparameter_optimization = cu.bool_to_string(self.complementary_information['hyperparameter_optimization'])   
            else:
                hyperparameter_optimization = self.complementary_information['hyperparameter_optimization']
            lines.append('hyperparameter optimization: ' + hyperparameter_optimization)
            # optimization type
            optimization_type = self.complementary_information['optimization_type']
            lines.append('optimization type: ' + optimization_type) 
        lines.append(' ')
        lines.append(' ')              
        self.input_summary += lines                
            
        
    def _add_regression_tab_3_inputs(self):
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
        # forecast options
        forecast_file = self.complementary_information['forecast_file']
        lines.append('forecast file: ' + forecast_file)    
        if type(self.complementary_information['forecast_evaluation']) == bool:
            forecast_evaluation = cu.bool_to_string(self.complementary_information['forecast_evaluation'])
        else:
            forecast_evaluation = self.complementary_information['forecast_evaluation']
        lines.append('forecast evaluation: ' + forecast_evaluation)
        lines.append(' ')
        lines.append(' ') 
        self.input_summary += lines  
        
        
    def _make_regression_summary(self):
        # initiate string list
        self.estimation_summary = []
        # add model header
        self.__add_regression_header()     
        # add estimation header
        self.__add_regression_estimation_header()
        # add coefficient header
        self.__add_regression_coefficient_summary()
        # add heteroscedastic coefficients, if relevant
        self.__add_heteroscedastic_coefficient() 
        # add autocorrelated coefficients, if relevant
        self.__add_autocorrelated_coefficient()         
        # add sigma coefficient
        self.__add_sigma_coefficient()
        # add in-sample evaluation criteria, if relevant
        self.__add_regression_insample_evaluation()  
        # add forecast evaluation criteria, if relevant
        self.__add_regression_forecast_evaluation()    
        
        
    def _make_regression_application_summary(self):
        # in-sample fit measures
        self.__make_regression_insample_fit_summary()
        # forecasts
        self.__make_regression_forecast_summary()
        
        
    def _save_regression_application(self, path):
        # save in-sample fit
        self.__save_regression_insample_fit_summary(path)
        # save forecasts
        self.__save_regression_forecast_summary(path)                
        
        
    def __add_regression_header(self):
        # recover model name and create header
        model_name = self.complementary_information['model_name']
        self.estimation_summary += cu.model_header(model_name)
        
        
    def __add_regression_estimation_header(self): 
        # initiate lines
        lines = []
        # first row: dependent variable and frequency
        endogenous_variables = self.complementary_information['endogenous_variables'][0]
        frequency = self.complementary_information['frequency']
        left_element = '{:14}{:>24}'.format('Dep. variable:', cu.shorten_string(endogenous_variables, 20))
        right_element = '{:10}{:>28}'.format('Frequency:', frequency)
        lines.append(left_element + '    ' + right_element)
        # second row: estimation sample and estimation start
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
        # third row: observations and estimation complete   
        n = str(self.model.n)
        estimation_end = self.complementary_information['estimation_end']
        left_element = '{:17}{:>21}'.format('No. observations:', n)
        right_element = '{:14}{:>24}'.format('Est. complete:', estimation_end)
        lines.append(left_element + '    ' + right_element)
        # equal dashed line
        lines.append(cu.equal_dashed_line())
        self.estimation_summary += lines 

        
    def __add_regression_coefficient_summary(self):
        # initiate lines
        lines = []
        # coefficients header
        credibility_level = self.model.credibility_level
        lines += cu.coefficient_header(credibility_level)
        lines.append(cu.string_line('regression coefficients beta:'))
        # coefficient summary, coefficient by coefficient
        regressors = []
        if self.model.constant:
            regressors.append('constant')
        if self.model.trend:
            regressors.append('trend')
        if self.model.quadratic_trend:
            regressors.append('quadratic trend')   
        regressors += self.complementary_information['exogenous_variables']
        for i in range(self.model.k):
            regressor = regressors[i]
            coefficient = self.model.beta_estimates[i,0]
            standard_deviation = self.model.beta_estimates[i,3]
            lower_bound = self.model.beta_estimates[i,1]
            upper_bound = self.model.beta_estimates[i,2]
            lines.append(cu.parameter_estimate_line(regressor, coefficient,\
                         standard_deviation, lower_bound, upper_bound))  
        lines.append(cu.hyphen_dashed_line())
        self.estimation_summary += lines 
        
        
    def __add_heteroscedastic_coefficient(self):        
        model_type = self.complementary_information['model_type']
        if model_type == 5:
            lines = []
            heteroscedastic_variables = self.complementary_information['heteroscedastic_variables']
            gamma_estimates = self.model.gamma_estimates
            lines.append(cu.string_line('heteroscedastic coefficients gamma:'))    
            for i in range(self.model.h):
                lines.append(cu.parameter_estimate_line(heteroscedastic_variables[i], \
                gamma_estimates[i,0], gamma_estimates[i,3], gamma_estimates[i,1], gamma_estimates[i,2]))
            lines.append(cu.hyphen_dashed_line())
            self.estimation_summary += lines          
        

    def __add_autocorrelated_coefficient(self):
        model_type = self.complementary_information['model_type']
        if model_type == 6:
            lines = []
            phi_estimates = self.model.phi_estimates
            lines.append(cu.string_line('autocorrelation coefficients phi:'))            
            for i in range(self.model.q):
                lines.append(cu.parameter_estimate_line('resid[-' + str(i+1) + ']', 
                phi_estimates[i,0], phi_estimates[i,3], phi_estimates[i,1], phi_estimates[i,2]))
            lines.append(cu.hyphen_dashed_line())
            self.estimation_summary += lines


    def __add_sigma_coefficient(self):
        # initiate lines
        lines = []
        # coefficients header
        lines.append(cu.string_line('residual variance sigma:'))
        # coefficient summary
        model_type = self.complementary_information['model_type']
        if model_type == 1 or model_type == 2:
            sigma = self.model.sigma
        else:
            sigma = self.model.sigma_estimates
        lines.append('resid' + ' ' * 20 + cu.format_number(sigma) + ' ' * 45)
        lines.append(cu.equal_dashed_line())
        self.estimation_summary += lines        


    def __add_regression_insample_evaluation(self):
        # initiate lines
        lines = []
        # check if in-sample evaluation has been conducted
        if hasattr(self.model,'insample_evaluation'):
            ssr = self.model.insample_evaluation['ssr']
            r2 = self.model.insample_evaluation['r2']
            adj_r2 = self.model.insample_evaluation['adj_r2']
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
            lines.append(cu.equal_dashed_line())
            self.estimation_summary += lines 


    def __add_regression_forecast_evaluation(self):
        # initiate lines
        lines = []
        # check if forecast evaluation has been conducted
        if hasattr(self.model,'forecast_evaluation_criteria'):  
            rmse = self.model.forecast_evaluation_criteria['rmse']
            mae = self.model.forecast_evaluation_criteria['mae']
            mape = self.model.forecast_evaluation_criteria['mape']
            theil_u = self.model.forecast_evaluation_criteria['theil_u']
            bias = self.model.forecast_evaluation_criteria['bias']
            model_type = self.complementary_information['model_type']
            if model_type != 1:
                log_score = self.model.forecast_evaluation_criteria['log_score']
                crps = self.model.forecast_evaluation_criteria['crps']
            else:
                log_score = []
                crps = []
            lines += cu.forecast_evaluation_lines(rmse, mae, mape, theil_u, bias, log_score, crps)
            lines.append(cu.equal_dashed_line())
            self.estimation_summary += lines 
 

    def __make_regression_insample_fit_summary(self):
        # run only if in-sample fit has been run
        if hasattr(self.model, 'fitted_estimates'):
            # create index, column labels and data
            index = self.complementary_information['dates']
            columns = ['actual', 'fitted', 'residuals']
            data = np.vstack((self.model.endogenous, self.model.fitted_estimates, self.model.residual_estimates)).T
            fitted_dataframe = pd.DataFrame(index = index, columns = columns, data = data)
            self.application_summary['insample_fit'] = fitted_dataframe
        

    def __make_regression_forecast_summary(self):
        # run only if forecast has been run
        if hasattr(self.model, 'forecast_estimates'):        
        # create index, column labels and data
            index = np.arange(1,self.model.forecast_estimates.shape[0]+1)
            columns = ['median', 'lower_bound', 'upper_bound']
            data = self.model.forecast_estimates
            forecast_dataframe = pd.DataFrame(index = index, columns = columns, data = data)
            self.application_summary['forecast'] = forecast_dataframe
        
        
    def __save_regression_insample_fit_summary(self, path):
        if 'insample_fit' in self.application_summary:
            insample_fit_summary = self.application_summary['insample_fit']
            full_path = join(path, 'insample_fit.csv')
            insample_fit_summary.to_csv(path_or_buf = full_path)   
            

    def __save_regression_forecast_summary(self, path):
        if 'forecast' in self.application_summary:
            forecast_summary = self.application_summary['forecast']
            full_path = join(path, 'forecast.csv')
            forecast_summary.to_csv(path_or_buf = full_path) 


