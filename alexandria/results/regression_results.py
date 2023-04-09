# imports
import numpy as np
import pandas as pd
from datetime import datetime
from os.path import isdir, join
import alexandria.processor.input_utilities as iu
import alexandria.console.console_utilities as cu
from alexandria.results.results import Results


class RegressionResults(Results):
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        # save estimation start time as attribute
        self.estimation_start = datetime.now()
        # display Alexandria header
        self._print_alexandria_header()
        # display Alexandria initialization
        self._print_start_message()
        
        
    def result_summary(self, ip, lr):
        # save estimation end time as attribute
        self.estimation_complete = datetime.now()
        # gather information from input processor 
        self.__input_information(ip)
        # then gather information from regression model
        self.__regression_information(lr)
        # print completion message
        self._print_completion_message()
        # build the string list for result summary
        self.__generate_result_summary()
        # print and save regression result summary
        self._print_and_save_summary()


    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------
        
        
    def __input_information(self, ip):                 
        # input processor information: endogenous
        self.endogenous = ip.endogenous_variables
        # input processor information: exogenous
        self.exogenous = ip.exogenous_variables
        # input processor information: data frequency
        self.frequency = ip.frequency
        # input processor information: sample dates
        self.start_date = ip.start_date 
        self.end_date = ip.end_date
        # input processor information: project folder
        self.project_path = ip.project_path
        # input processor information: data file
        self.data_file = ip.data_file
        # input processor information: progress bar
        self.progress_bar = ip.progress_bar
        # input processor information: graphics and figures
        self.create_graphics = ip.create_graphics     
        # input processor information: save results
        self.save_results = ip.save_results
        # input processor information: model
        self.regression_type = ip.regression_type  
        # input processor information: iterations
        self.iterations = ip.iterations  
        # input processor information: burn-in
        self.burnin = ip.burnin
        # input processor information: credibility level
        self.model_credibility = ip.model_credibility
        # input processor information: hyperparameter b
        self.b = ip.b
        # input processor information: hyperparameter V
        self.V = ip.V
        # input processor information: hyperparameter alpha
        self.alpha = ip.alpha
        # input processor information: hyperparameter delta
        self.delta = ip.delta  
        # input processor information: hyperparameter g
        self.g = ip.g     
        # input processor information: hyperparameter Q
        self.Q = ip.Q          
        # input processor information: hyperparameter tau
        self.tau = ip.tau          
        # input processor information: thinning
        self.thinning = ip.thinning
        # input processor information: thinning frequency
        self.thinning_frequency = ip.thinning_frequency
        # input processor information: Z regressors
        self.Z_variables = ip.Z_variables
        # input processor information: q
        self.q = ip.q
        # input processor information: p        
        self.p = ip.p
        # input processor information: H
        self.H = ip.H
        # input processor information: constant
        self.constant = ip.constant
        # input processor information: b constant        
        self.b_constant = ip.b_constant
        # input processor information: V constant        
        self.V_constant = ip.V_constant
        # input processor information: trend
        self.trend = ip.trend
        # input processor information: b trend        
        self.b_trend = ip.b_trend
        # input processor information: V trend        
        self.V_trend = ip.V_trend
        # input processor information: quadratic trend
        self.quadratic_trend = ip.quadratic_trend
        # input processor information: b trend        
        self.b_quadratic_trend = ip.b_quadratic_trend
        # input processor information: V trend        
        self.V_quadratic_trend = ip.V_quadratic_trend
        # input processor information: in-sample fit
        self.insample_fit = ip.insample_fit
        # input processor information: marginal likelihood
        self.marginal_likelihood = ip.marginal_likelihood
        # input processor information: hyperparameter optimization
        self.hyperparameter_optimization = ip.hyperparameter_optimization
        # input processor information: optimization type
        self.optimization_type = ip.optimization_type
        # input processor information: forecasts
        self.forecast = ip.forecast
        # input processor information: forecasts credibility level
        self.forecast_credibility = ip.forecast_credibility
        # input processor information: forecast file      
        self.forecast_file = ip.forecast_file
        # input processor information: forecast evaluation     
        self.forecast_evaluation = ip.forecast_evaluation
        # input processor information: actual
        self.actual = ip.endogenous
        # input processor information: in-sample dates
        self.insample_dates = ip.dates
        # input processor information: forecast dates
        self.forecast_dates = ip.forecast_dates
      
        
    def __regression_information(self, lr):
        # regression information: dimensions
        self.k = lr.k
        if self.regression_type == 6:
            self.n = lr.T
        else:
            self.n = lr.n
        # regression information: coefficients
        self.beta = lr.estimates_beta
        if self.regression_type == 1 or self.regression_type == 2:
            self.sigma = lr.sigma
        else:
            self.sigma = lr.estimates_sigma
        if self.regression_type == 5:
            self.h = lr.h
            self.gamma = lr.estimates_gamma
        if self.regression_type == 6:
            self.q = lr.q
            self.phi = lr.estimates_phi 
        # regression information: in-sample evaluation
        if self.insample_fit:
            self.fitted = lr.estimates_fit
            self.residuals = lr.estimates_residuals
            self.insample_evaluation = lr.insample_evaluation
        # regression information: marginal likelihood
        if self.marginal_likelihood and self.regression_type != 1:
            self.m_y = lr.m_y
        # regression information: forecasts
        if self.forecast:
            self.estimates_forecasts = lr.estimates_forecasts
        # regression information: forecast evaluation
        if self.forecast and self.forecast_evaluation:
            self.forecast_evaluation_criteria = lr.forecast_evaluation_criteria

            
    def __generate_result_summary(self):
        # initiate string list
        self.summary = []
        # add model header, depending on regression type
        self.__add_model_header()
        # add estimation header
        self.__add_estimation_header()
        # add coefficient header
        self.__add_coefficient_header()
        # add beta coefficients
        self.__add_beta_coefficients()
        # add sigma coefficient
        self.__add_sigma_coefficient()
        # add gamma coefficients, if relevant
        self.__add_gamma_coefficient()        
        # add phi coefficients, if relevant
        self.__add_phi_coefficient()         
        # add equal line for separation
        self.__add_equal_line()
        # add in-sample evaluation criteria, if relevant
        self.__add_insample_evaluation()
        # add forecast evaluation criteria, if relevant
        self.__add_forecast_evaluation()
        
        
    def __add_model_header(self):
        if self.regression_type == 1:
            model = 'Maximum Likelihood Regression'
        elif self.regression_type == 2:
            model = 'Simple Bayesian Regression'            
        elif self.regression_type == 3:
            model = 'Hierarchical Bayesian Regression'          
        elif self.regression_type == 4:
            model = 'Independent Bayesian Regression'  
        elif self.regression_type == 5:
            model = 'Heteroscedastic Bayesian Regression'              
        elif self.regression_type == 6:
            model = 'Autocorrelated Bayesian Regression'   
        self.summary += cu.model_header(model)        


    def __add_estimation_header(self): 
        estimation_start = self.estimation_start
        estimation_complete = self.estimation_complete
        n = self.n
        endogenous = self.endogenous[0]
        if self.frequency == 1:
            frequency = 'cross-sectional/undated'
        elif self.frequency == 2:
            frequency = 'annual'            
        elif self.frequency == 3:
            frequency = 'quarterly'   
        elif self.frequency == 4:
            frequency = 'monthly'   
        elif self.frequency == 5:
            frequency = 'weekly'   
        elif self.frequency == 6:
            frequency = 'daily'        
        sample = self.start_date + ' ' + self.end_date
        self.summary += cu.estimation_header(estimation_start, estimation_complete, \
                        n, endogenous, frequency, sample)
        self.summary.append(cu.equal_dashed_line())
        
        
    def __add_coefficient_header(self):
        credibility = self.model_credibility
        self.summary += cu.coefficient_header(credibility)


    def __add_beta_coefficients(self):
        lines = [cu.string_line('regression coefficients beta:')]
        regressors = []
        if self.constant:
            regressors.append('constant')
        if self.trend:
            regressors.append('trend')
        if self.quadratic_trend:
            regressors.append('quadratic trend')   
        regressors += self.exogenous
        for i in range(self.k):
            regressor = regressors[i]
            coefficient = self.beta[i,1]
            standard_deviation = self.beta[i,3]
            lower_bound = self.beta[i,0]
            upper_bound = self.beta[i,2]
            lines.append(cu.parameter_estimate_line(regressor, coefficient,\
                         standard_deviation, lower_bound, upper_bound))           
        self.summary += lines
            
            
    def __add_sigma_coefficient(self):
        lines = [cu.hyphen_dashed_line()]
        lines.append(cu.string_line('residual variance sigma:'))
        lines.append('resid' + ' ' * 20 + cu.format_number(self.sigma) + ' ' * 45)
        self.summary += lines


    def __add_gamma_coefficient(self):
        if self.regression_type == 5:
            lines = [cu.hyphen_dashed_line()]
            lines.append(cu.string_line('heteroscedastic coefficients gamma:'))    
            for i in range(self.h):
                lines.append(cu.parameter_estimate_line(\
                            'Z[' + str(i+1) + ']', self.gamma[i,1], \
                             self.gamma[i,3], self.gamma[i,0], self.gamma[i,2]))
            self.summary += lines                
        
    
    def __add_phi_coefficient(self):
        if self.regression_type == 6:
            lines = [cu.hyphen_dashed_line()]
            lines.append(cu.string_line('autocorrelation coefficients phi:'))            
            for i in range(self.q):
                lines.append(cu.parameter_estimate_line(\
                            'resid[-' + str(i+1) + ']', self.phi[i,1], \
                             self.phi[i,3], self.phi[i,0], self.phi[i,2]))
            self.summary += lines 
        
        
    def __add_insample_evaluation(self):
        if self.insample_fit:
            ssr = self.insample_evaluation['ssr']
            r2 = self.insample_evaluation['r2']
            adj_r2 = self.insample_evaluation['adj_r2']
            if self.regression_type == 1:
                aic = self.insample_evaluation['aic']
                bic = self.insample_evaluation['bic']
                m_y = []
            else:
                aic = []
                bic = []
                if self.marginal_likelihood:
                    m_y = self.m_y
                else:
                    m_y = []
            lines = cu.insample_evaluation_lines(ssr, r2, adj_r2, m_y, aic, bic)
            self.summary += lines 
            # add equal line for separation
            self.__add_equal_line()            


    def __add_forecast_evaluation(self):
        if self.forecast and self.forecast_evaluation:
            rmse = self.forecast_evaluation_criteria['rmse']
            mae = self.forecast_evaluation_criteria['mae']
            mape = self.forecast_evaluation_criteria['mape']
            theil_u = self.forecast_evaluation_criteria['theil_u']
            bias = self.forecast_evaluation_criteria['bias']
            if self.regression_type != 1:
                log_score = self.forecast_evaluation_criteria['log_score']
                crps = self.forecast_evaluation_criteria['crps']
            else:
                log_score = []
                crps = []
            lines = cu.forecast_evaluation_lines(rmse, mae, mape, theil_u, bias, log_score, crps)
            self.summary += lines 
            # add equal line for separation
            self.__add_equal_line()                 
                

    def __add_equal_line(self):
        self.summary.append(cu.equal_dashed_line())
        
        
    def settings_summary(self):
        if self.save_results:
            # initiate string list
            self.settings = []
            # add settings header
            self._add_settings_header()
            # add tab 1 settings
            self._add_tab_1_settings()
            # add tab 2 settings
            self.__add_tab_2_settings()
            # add tab 3 settings
            self.__add_tab_3_settings()
            # print and save regression result summary
            self._save_settings()
            
            
    def __add_tab_2_settings(self):
        # initiate lines
        lines = []
        # header for tab 2
        lines.append('Specifications')
        lines.append('-----------------')
        lines.append(' ')
        # recover elements
        regression_type = self.regression_type
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
        iterations = str(self.iterations)
        burnin = str(self.burnin)
        model_credibility = str(self.model_credibility)
        b = iu.list_to_string(self.b)
        V = iu.list_to_string(self.V)
        alpha = str(self.alpha) 
        delta = str(self.delta)
        g = iu.list_to_string(self.g)
        Q = iu.list_to_string(self.Q)
        tau = str(self.tau)  
        thinning = cu.bool_to_string(self.thinning)
        thinning_frequency = str(self.thinning_frequency)
        Z_variables = iu.list_to_string(self.Z_variables)
        q = str(self.q)
        p = iu.list_to_string(self.p)
        H = iu.list_to_string(self.H)
        constant = self.constant
        constant_string = cu.bool_to_string(constant)
        b_constant = str(self.b_constant) 
        V_constant = str(self.V_constant) 
        trend = self.trend
        trend_string = cu.bool_to_string(self.trend)      
        b_trend = str(self.b_trend)       
        V_trend = str(self.V_trend) 
        quadratic_trend = self.quadratic_trend
        quadratic_trend_string = cu.bool_to_string(self.quadratic_trend)      
        b_quadratic_trend = str(self.b_quadratic_trend)       
        V_quadratic_trend = str(self.V_quadratic_trend)         
        insample_fit = cu.bool_to_string(self.insample_fit)
        marginal_likelihood = cu.bool_to_string(self.marginal_likelihood)
        hyperparameter_optimization = self.hyperparameter_optimization
        hyperparameter_optimization_string = cu.bool_to_string(hyperparameter_optimization)
        optimization_type = str(self.optimization_type)
        # other elements for tab 2
        lines.append('regression type: ' + model)
        if regression_type == 4 or regression_type == 5 or regression_type == 6:
            lines.append('iterations: ' + iterations)
            lines.append('burn-in: ' + burnin)    
        lines.append('credibility level: ' + model_credibility) 
        if regression_type != 1:
            lines.append('b: ' + b)          
            lines.append('V: ' + V)    
        if regression_type != 1 and regression_type != 2:            
            lines.append('alpha: ' + alpha)
            lines.append('delta: ' + delta)
        if regression_type == 5:             
            lines.append('g: ' + g)
            lines.append('Q: ' + Q)        
            lines.append('tau: ' + tau)  
            lines.append('thinning: ' + thinning)  
            lines.append('thinning frequency: ' + thinning_frequency)  
            lines.append('Z variables: ' + Z_variables)  
        if regression_type == 6:             
            lines.append('q: ' + q)
            lines.append('p: ' + p)        
            lines.append('H: ' + H)             
        lines.append('constant: ' + constant_string)
        if constant and regression_type != 1:        
            lines.append('b (constant): ' + b_constant)            
            lines.append('V (constant): ' + V_constant)        
        lines.append('trend: ' + trend_string)
        if trend and regression_type != 1:
            lines.append('b (trend): ' + b_trend)            
            lines.append('V (trend): ' + V_trend)          
        lines.append('quadratic trend: ' + quadratic_trend_string)
        if quadratic_trend and regression_type != 1:
            lines.append('b (quadratic trend): ' + b_quadratic_trend)            
            lines.append('V (quadratic trend): ' + V_quadratic_trend)
        lines.append('in-sample fit: ' + insample_fit)
        if regression_type != 1:        
            lines.append('marginal likelihood: ' + marginal_likelihood)
        if regression_type == 2 or regression_type == 3:        
            lines.append('hyperparameter optimization: ' + hyperparameter_optimization_string)
            if hyperparameter_optimization:
                lines.append('optimization type: ' + optimization_type) 
        lines.append(' ')
        lines.append(' ')
        self.settings += lines
        
            
    def __add_tab_3_settings(self):
        # initiate lines
        lines = []            
        # header for tab 3
        lines.append('Applications')
        lines.append('-----------------')
        lines.append(' ')        
        # recover elements
        forecast = self.forecast
        forecast_string = cu.bool_to_string(forecast)
        forecast_credibility = str(self.forecast_credibility)
        forecast_file = self.forecast_file
        forecast_evaluation = cu.bool_to_string(self.forecast_evaluation)
        # other elements for tab 3
        lines.append('forecast: ' + forecast_string)
        if forecast:
            lines.append('forecast credibility: ' + forecast_credibility)       
            lines.append('forecast file: ' + forecast_file)
            lines.append('forecast evaluation: ' + forecast_evaluation)
        # other elements for tab 3          
        self.settings += lines            
            
        
    def application_summary(self):
        if self.save_results:
            # actual, fitted and residuals
            if self.insample_fit:
                self.__save_fitted_and_residuals()
            # forecasts
            if self.forecast:
                self.__save_forecasts()
            
                       
    def __save_fitted_and_residuals(self):
        # create index, column labels and data
        index = self.insample_dates
        columns = ['actual', 'fitted', 'residuals']
        data = np.vstack((self.actual, self.fitted, self.residuals)).T
        fitted_dataframe = pd.DataFrame(index = index, columns = columns, data = data)
        # create path to file and save
        fitted_file_path = join(self.project_path, 'results', 'fitted_and_residuals.csv')
        fitted_dataframe.to_csv(path_or_buf = fitted_file_path)              
            

    def __save_forecasts(self):
        # create index, column labels and data
        index = self.forecast_dates
        columns = ['lower_bound', 'median', 'upper_bound']
        data = self.estimates_forecasts
        forecast_dataframe = pd.DataFrame(index = index, columns = columns, data = data)
        # create path to file and save
        forecast_file_path = join(self.project_path, 'results', 'forecasts.csv')
        forecast_dataframe.to_csv(path_or_buf = forecast_file_path)          
        
        