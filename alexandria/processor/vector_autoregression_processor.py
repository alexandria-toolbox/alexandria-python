# imports
import numpy as np
import pandas as pd
import alexandria.math.linear_algebra as la
import alexandria.processor.input_utilities as iu
import alexandria.console.console_utilities as cu


class VectorAutoregressionProcessor(object):

    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass


    def _vector_autoregression_inputs(self):
        # recover vector autoregression type
        self.var_type = self.__get_vector_autoregression_type()
        # recover iterations
        self.var_iterations = self.__get_var_iterations()
        # recover burn-in
        self.var_burnin = self.__get_var_burnin() 
        # recover credibility level for model estimates
        self.var_model_credibility = self.__get_var_model_credibility()        
        # recover constant
        self.var_constant = self.__get_var_constant() 
        # recover trend
        self.var_trend = self.__get_var_trend() 
        # recover quadratic trend
        self.var_quadratic_trend = self.__get_var_quadratic_trend()         
        # recover lags
        self.lags = self.__get_lags()  
        # recover AR coefficients
        self.ar_coefficients = self.__get_ar_coefficients()          
        # recover pi1
        self.pi1 = self.__get_pi1()  
        # recover pi2
        self.pi2 = self.__get_pi2() 
        # recover pi3
        self.pi3 = self.__get_pi3() 
        # recover pi4
        self.pi4 = self.__get_pi4() 
        # recover pi5
        self.pi5 = self.__get_pi5() 
        # recover pi6
        self.pi6 = self.__get_pi6() 
        # recover pi7
        self.pi7 = self.__get_pi7() 
        # recover proxys
        self.proxy_variables = self.__get_proxy_variables()
        # recover lamda
        self.lamda = self.__get_lamda() 
        # recover proxy prior type
        self.proxy_prior = self.__get_proxy_prior()
        # recover in-sample fit
        self.var_insample_fit = self.__get_var_insample_fit()
        # recover constrained_coefficients
        self.constrained_coefficients = self.__get_constrained_coefficients()
        # recover sums-of-coefficients
        self.sums_of_coefficients = self.__get_sums_of_coefficients()     
        # recover dummy initial observation
        self.initial_observation = self.__get_initial_observation()
        # recover long-run prior
        self.long_run = self.__get_long_run()
        # recover stationary prior
        self.stationary = self.__get_stationary()
        # recover marginal likelihood
        self.var_marginal_likelihood = self.__get_var_marginal_likelihood()
        # recover hyperparameter optimization
        self.var_hyperparameter_optimization = self.__get_var_hyperparameter_optimization()
        # recover constrained coefficients file
        self.coefficients_file = self.__get_coefficients_file()
        # recover long-run prior file
        self.long_run_file = self.__get_long_run_file()
        
        
    def _vector_autoregression_data(self):
        # print loading message
        if self.progress_bar:
            cu.print_message_to_overwrite('Data loading:')
            # cu.print_message('Data loading:')
        # recover in-sample endogenous and exogenous
        self.var_endogenous, self.var_exogenous, self.var_dates = self.__get_var_insample_data()
        # recover proxy variables
        self.proxys = self.__get_proxy_data()
        # recover constrained coefficients
        self.constrained_coefficients_table = self.__get_constrained_coefficients_table()
        # recover long run prior
        self.long_run_table = self.__get_long_run_table()
        # recover forecast data
        self.var_Z_p, self.var_Y_p, self.var_forecast_dates = self.__get_var_forecast_data()
        # recover conditional forecast data
        self.condition_table, self.shock_table = self.__get_condition_table()
        # recover sign restrictions data
        self.restriction_table = self.__get_restriction_table()
        # print loading done message
        if self.progress_bar:
            cu.print_message('Data loading:  —  done')
        

    def _make_var_information(self):
        # get sample dates
        self.results_information['dates'] = self.var_dates 
        # get forecast dates
        self.results_information['forecast_dates'] = self.var_forecast_dates 
        # get proxy variables for proxy SVAR
        if self.var_type == 7:
            self.results_information['proxy_variables'] = self.proxy_variables
        else:
            self.results_information['proxy_variables'] = []        
        # get var option: in-sample fit
        self.results_information['insample_fit'] = self.var_insample_fit
        # get var option: constrained coefficients
        self.results_information['constrained_coefficients'] = self.constrained_coefficients      
        # get var option: sums-of-coefficients
        self.results_information['sums_of_coefficients'] = self.sums_of_coefficients
        # get var option: dummy initial observation
        self.results_information['initial_observation'] = self.initial_observation
        # get var option: long-run prior
        self.results_information['long_run'] = self.long_run
        # get var option: stationary prior
        self.results_information['stationary'] = self.stationary
        # get var option: marginal likelihood
        self.results_information['marginal_likelihood'] = self.var_marginal_likelihood
        # get var option: hyperparameter optimization
        self.results_information['hyperparameter_optimization'] = self.var_hyperparameter_optimization
        # get constrained coefficients file
        self.results_information['coefficients_file'] = self.coefficients_file
        # get long run prior file
        self.results_information['long_run_file'] = self.long_run_file


    def _make_var_graphics_information(self):
        # get sample dates
        self.graphics_information['dates'] = self.var_dates
        # get forecast dates
        self.graphics_information['forecast_dates'] = self.var_forecast_dates
        # get actual data for forecast evaluation, if available
        self.graphics_information['Y_p'] = self.var_Y_p


    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------  


    def __get_vector_autoregression_type(self):
        var_type = self.user_inputs['tab_2_var']['var_type']
        if var_type not in [1, 2, 3, 4, 5, 6, 7]:
            raise TypeError('Value error for vector autoregression type. Should be integer between 1 and 7.')  
        return var_type 


    def __get_var_iterations(self):
        iterations = self.user_inputs['tab_2_var']['iterations']       
        if not isinstance(iterations, (int, str)):
            raise TypeError('Type error for iterations. Should be integer.')
        if iterations and isinstance(iterations, str):
            if iterations.isdigit():
                iterations = int(iterations)
            else:
                raise TypeError('Type error for iterations. Should be positive integer.')
        if isinstance(iterations, int) and iterations <= 0:
            raise TypeError('Value error for iterations. Should be positive integer.')
        return iterations


    def __get_var_burnin(self):
        burnin = self.user_inputs['tab_2_var']['burnin']       
        if not isinstance(burnin, (int, str)):
            raise TypeError('Type error for burn-in. Should be integer.')
        if burnin and isinstance(burnin, str):
            if burnin.isdigit():
                burnin = int(burnin)
            else:
                raise TypeError('Type error for burn-in. Should be positive integer.')
        if isinstance(burnin, int) and burnin <= 0:
            raise TypeError('Value error for burn-in. Should be positive integer.')
        return burnin


    def __get_var_model_credibility(self):
        model_credibility = self.user_inputs['tab_2_var']['model_credibility']
        if not isinstance(model_credibility, (str, float)):
            raise TypeError('Type error for model credibility level. Should be float between 0 and 1.')
        if isinstance(model_credibility, str):
            if not model_credibility.replace('.','',1).isdigit():
                raise TypeError('Type error for model credibility level. Should be float between 0 and 1.')
            else:
                model_credibility = float(model_credibility)
        if model_credibility <= 0 or model_credibility >= 1:
            raise TypeError('Value error for model credibility level. Should be float between 0 and 1 (not included).')
        return model_credibility 


    def __get_var_constant(self):
        constant = self.user_inputs['tab_2_var']['constant']
        if not isinstance(constant, bool):
            raise TypeError('Type error for constant. Should be boolean.') 
        return constant


    def __get_var_trend(self):
        trend = self.user_inputs['tab_2_var']['trend']
        if not isinstance(trend, bool):
            raise TypeError('Type error for trend. Should be boolean.') 
        return trend


    def __get_var_quadratic_trend(self):
        quadratic_trend = self.user_inputs['tab_2_var']['quadratic_trend']
        if not isinstance(quadratic_trend, bool):
            raise TypeError('Type error for quadratic trend. Should be boolean.') 
        return quadratic_trend


    def __get_lags(self):
        lags = self.user_inputs['tab_2_var']['lags']
        if not isinstance(lags, (int, str)):
            raise TypeError('Type error for lags. Should be integer.')
        if lags and isinstance(lags, str):
            if lags.isdigit():
                lags = int(lags)
            else:
                raise TypeError('Type error for lags. Should be positive integer.')
        if isinstance(lags, int) and lags <= 0:
            raise TypeError('Value error for lags. Should be positive integer.')
        return lags


    def __get_ar_coefficients(self):
        ar_coefficients = self.user_inputs['tab_2_var']['ar_coefficients']
        if not isinstance(ar_coefficients, (str, list, float, int)):
            raise TypeError('Type error for AR coefficients. Should be scalar or list of scalars.')
        if isinstance(ar_coefficients, str):
            ar_coefficients = iu.string_to_list(ar_coefficients)
            if not all([ar_entry.replace('.','',1).replace('-','',1).isdigit() for ar_entry in ar_coefficients]):
                raise TypeError('Type error for AR coefficients. All elements should be scalars.')
            else:
                ar_coefficients = [float(ar_entry) for ar_entry in ar_coefficients]
        if isinstance(ar_coefficients, list):
            if len(ar_coefficients) != len(self.endogenous_variables) and len(ar_coefficients) != 1:
                raise TypeError('Dimension error for AR coefficients. Dimension of AR coefficients and endogenous don\'t match.')
            if not all([isinstance(ar_entry, (int, float)) for ar_entry in ar_coefficients]):
                raise TypeError('Type error for AR coefficients. All elements should be scalars.')
            else:
                ar_coefficients = np.array(ar_coefficients)
            if len(ar_coefficients) == 1:
                ar_coefficients = ar_coefficients[0]
        return ar_coefficients


    def __get_pi1(self):
        pi1 = self.user_inputs['tab_2_var']['pi1']
        if not isinstance(pi1, (str, float, int)):
            raise TypeError('Type error for pi1. Should be float or integer.')
        if isinstance(pi1, str):
            if not pi1.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for pi1. Should be float or integer.')
            else:
                pi1 = float(pi1)
        if pi1 <= 0:
            raise TypeError('Value error for pi1. Should be strictly positive.')
        return pi1


    def __get_pi2(self):
        pi2 = self.user_inputs['tab_2_var']['pi2']
        if not isinstance(pi2, (str, float, int)):
            raise TypeError('Type error for pi2. Should be float or integer.')
        if isinstance(pi2, str):
            if not pi2.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for pi2. Should be float or integer.')
            else:
                pi2 = float(pi2)
        if pi2 <= 0:
            raise TypeError('Value error for pi2. Should be strictly positive.')
        return pi2


    def __get_pi3(self):
        pi3 = self.user_inputs['tab_2_var']['pi3']
        if not isinstance(pi3, (str, float, int)):
            raise TypeError('Type error for pi3. Should be float or integer.')
        if isinstance(pi3, str):
            if not pi3.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for pi3. Should be float or integer.')
            else:
                pi3 = float(pi3)
        if pi3 <= 0:
            raise TypeError('Value error for pi3. Should be strictly positive.')
        return pi3


    def __get_pi4(self):
        pi4 = self.user_inputs['tab_2_var']['pi4']
        if not isinstance(pi4, (str, float, int)):
            raise TypeError('Type error for pi4. Should be float or integer.')
        if isinstance(pi4, str):
            if not pi4.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for pi4. Should be float or integer.')
            else:
                pi4 = float(pi4)
        if pi4 <= 0:
            raise TypeError('Value error for pi4. Should be strictly positive.')
        return pi4


    def __get_pi5(self):
        pi5 = self.user_inputs['tab_2_var']['pi5']
        if not isinstance(pi5, (str, float, int)):
            raise TypeError('Type error for pi5. Should be float or integer.')
        if isinstance(pi5, str):
            if not pi5.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for pi5. Should be float or integer.')
            else:
                pi5 = float(pi5)
        if pi5 <= 0:
            raise TypeError('Value error for pi5. Should be strictly positive.')
        return pi5


    def __get_pi6(self):
        pi6 = self.user_inputs['tab_2_var']['pi6']
        if not isinstance(pi6, (str, float, int)):
            raise TypeError('Type error for pi6. Should be float or integer.')
        if isinstance(pi6, str):
            if not pi6.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for pi6. Should be float or integer.')
            else:
                pi6 = float(pi6)
        if pi6 <= 0:
            raise TypeError('Value error for pi6. Should be strictly positive.')
        return pi6


    def __get_pi7(self):
        pi7 = self.user_inputs['tab_2_var']['pi7']
        if not isinstance(pi7, (str, float, int)):
            raise TypeError('Type error for pi7. Should be float or integer.')
        if isinstance(pi7, str):
            if not pi7.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for pi7. Should be float or integer.')
            else:
                pi7 = float(pi7)
        if pi7 <= 0:
            raise TypeError('Value error for pi7. Should be strictly positive.')
        return pi7


    def __get_proxy_variables(self):
        proxy_variables = self.user_inputs['tab_2_var']['proxy_variables']
        if self.var_type == 7:
            if not proxy_variables or not isinstance(proxy_variables, (str, list)):
                raise TypeError('Type error for proxys. Should be non-empty list of strings.')
            proxy_variables = iu.string_to_list(proxy_variables)
            if not all(isinstance(element, str) for element in proxy_variables):
                raise TypeError('Type error for proxys. Should be list of strings.')        
        return proxy_variables


    def __get_lamda(self):
        lamda = self.user_inputs['tab_2_var']['lamda']
        if not isinstance(lamda, (str, float, int)):
            raise TypeError('Type error for lambda. Should be float or integer.')
        if isinstance(lamda, str):
            if not lamda.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for lamda. Should be float or integer.')
            else:
                lamda = float(lamda)
        if (lamda <= 0 or lamda > 1):
            raise TypeError('Value error for lambda. Should be float between 0 and 1.')
        return lamda


    def __get_proxy_prior(self):
        proxy_prior = self.user_inputs['tab_2_var']['proxy_prior']
        if proxy_prior not in [1, 2]:
            raise TypeError('Value error for proxy prior. Should be 1 or 2.')  
        return proxy_prior      


    def __get_var_insample_fit(self):
        var_insample_fit = self.user_inputs['tab_2_var']['insample_fit']
        if not isinstance(var_insample_fit, bool):
            raise TypeError('Type error for in-sample fit. Should be boolean.') 
        return var_insample_fit
    
    
    def __get_constrained_coefficients(self):
        constrained_coefficients = self.user_inputs['tab_2_var']['constrained_coefficients']
        if not isinstance(constrained_coefficients, bool):
            raise TypeError('Type error for constrained coefficients. Should be boolean.') 
        return constrained_coefficients


    def __get_sums_of_coefficients(self):
        sums_of_coefficients = self.user_inputs['tab_2_var']['sums_of_coefficients']
        if not isinstance(sums_of_coefficients, bool):
            raise TypeError('Type error for sums-of-coefficients. Should be boolean.') 
        return sums_of_coefficients


    def __get_initial_observation(self):
        initial_observation = self.user_inputs['tab_2_var']['initial_observation']
        if not isinstance(initial_observation, bool):
            raise TypeError('Type error for dummy initial observation. Should be boolean.') 
        return initial_observation


    def __get_long_run(self):
        long_run = self.user_inputs['tab_2_var']['long_run']
        if not isinstance(long_run, bool):
            raise TypeError('Type error for long-run prior. Should be boolean.') 
        return long_run


    def __get_stationary(self):
        stationary = self.user_inputs['tab_2_var']['stationary']
        if not isinstance(stationary, bool):
            raise TypeError('Type error for stationary prior. Should be boolean.') 
        return stationary


    def __get_var_marginal_likelihood(self):
        marginal_likelihood = self.user_inputs['tab_2_var']['marginal_likelihood']
        if not isinstance(marginal_likelihood, bool):
            raise TypeError('Type error for marginal likelihood. Should be boolean.') 
        return marginal_likelihood


    def __get_var_hyperparameter_optimization(self):
        hyperparameter_optimization = self.user_inputs['tab_2_var']['hyperparameter_optimization']
        if not isinstance(hyperparameter_optimization, bool):
            raise TypeError('Type error for hyperparameter optimization. Should be boolean.') 
        return hyperparameter_optimization


    def __get_coefficients_file(self):
        coefficients_file = self.user_inputs['tab_2_var']['coefficients_file']
        if not isinstance(coefficients_file, str):
            raise TypeError('Type error for constrained coefficients file. Should be string.')
        coefficients_file = iu.fix_string(coefficients_file)
        return coefficients_file


    def __get_long_run_file(self):
        long_run_file = self.user_inputs['tab_2_var']['long_run_file']
        if not isinstance(long_run_file, str):
            raise TypeError('Type error for long-run prior file. Should be string.')
        long_run_file = iu.fix_string(long_run_file)
        return long_run_file


    def __get_var_insample_data(self):
        # check that data path and files are valid
        iu.check_file_path(self.project_path, self.data_file)
        # then load data file
        data = iu.load_data(self.project_path, self.data_file)
        # check that endogenous and exogenous variables are found in data
        iu.check_variables(data, self.data_file, self.endogenous_variables, 'Endogenous variable')
        iu.check_variables(data, self.data_file, self.exogenous_variables, 'Exogenous variable(s)')
        # check that the start and end dates can be found in the file        
        iu.check_dates(data, self.data_file, self.start_date, self.end_date)
        # recover endogenous and exogenous data
        endogenous = iu.fetch_data(data, self.data_file, self.start_date, \
        self.end_date, self.endogenous_variables, 'Endogenous variables')
        exogenous = iu.fetch_data(data, self.data_file, self.start_date, \
        self.end_date, self.exogenous_variables, 'Exogenous variables')            
        # infer date format, then recover sample dates
        date_format = iu.infer_date_format(self.frequency, self.data_file, \
                                        self.start_date, self.end_date)
        dates = iu.generate_dates(data, date_format, self.frequency, self.data_file, \
                               self.start_date, self.end_date)
        return endogenous, exogenous, dates


    def __get_proxy_data(self):
        # load data only if specified model is proxy-SVAR
        if self.var_type == 7:
            # check that data path and files are valid
            iu.check_file_path(self.project_path, self.data_file)            
            # then load data file
            data = iu.load_data(self.project_path, self.data_file)
            # check that proxy variables are found in data
            iu.check_variables(data, self.data_file, self.proxy_variables, 'Proxy variables')
            # check that the start and end dates can be found in the file        
            iu.check_dates(data, self.data_file, self.start_date, self.end_date)
            # recover proxy variables
            proxys = iu.fetch_data(data, self.data_file, self.start_date, \
            self.end_date, self.proxy_variables, 'Proxy variables')
        # if model is not proxy-SVAR, return empty list
        else:
            proxys = []           
        return proxys


    def __get_constrained_coefficients_table(self):
        # load data only if constrained coefficients is selected
        if self.constrained_coefficients and self.var_type != 1:
            # check that data path and files are valid
            iu.check_file_path(self.project_path, self.coefficients_file)  
            # then load data file
            data = iu.load_data(self.project_path, self.coefficients_file)
            # check data format
            iu.check_coefficients_table(data, self.endogenous_variables, \
                self.exogenous_variables, self.lags, self.var_constant, \
                self.var_trend, self.var_quadratic_trend, self.coefficients_file)
            # if format is correct, convert to numpy array
            constrained_coefficients_table = iu.get_constrained_coefficients_table(data,\
                self.endogenous_variables,self.exogenous_variables)
        else:
            constrained_coefficients_table = []
        return constrained_coefficients_table
                
    
    def __get_long_run_table(self):
        # load data only if long run prior is selected
        if self.long_run and self.var_type != 1:
            # check that data path and files are valid
            iu.check_file_path(self.project_path, self.long_run_file)
            # then load data file
            data = iu.load_data(self.project_path, self.long_run_file)
            # check data format
            iu.check_long_run_table(data, self.endogenous_variables, self.long_run_file)
            # if format is correct, convert to numpy array
            long_run_table = data.values
        else:
            long_run_table = []
        return long_run_table


    def __get_var_forecast_data(self):
        # default values for endogenous and exogenous
        Z_p, Y_p = [], []
        # if forecast is selected, recover forecast dates
        if self.forecast or self.conditional_forecast:
            end_date = self.var_dates[-1]
            forecast_dates = iu.generate_forecast_dates(end_date, self.forecast_periods, self.frequency) 
        # if forecasts is not selected, return empty dates
        else:
            forecast_dates = []
        # if forecast is selected, further recover endogenous and exogenous, if relevant
        if self.forecast and (self.forecast_evaluation or self.exogenous_variables):
            # check that data path and files are valid
            iu.check_file_path(self.project_path, self.forecast_file)
            # then load data file
            data = iu.load_data(self.project_path, self.forecast_file)
            # if forecast evaluation is selected
            if self.forecast_evaluation:
                # check that endogenous variables are found in data
                iu.check_variables(data, self.forecast_file, self.endogenous_variables, 'endogenous variables')
                # load endogenous variables
                Y_p = iu.fetch_forecast_data(data, [], self.endogenous_variables, \
                self.forecast_file, self.forecast_evaluation, self.forecast_periods, 'endogenous variable')      
            # if there are exogenous variables in the model
            if self.exogenous_variables:
                # check that exogenous variables are found in data
                iu.check_variables(data, self.forecast_file, self.exogenous_variables, 'exogenous variables')
                # load exogenous data
                Z_p = iu.fetch_forecast_data(data, [], self.exogenous_variables, 
                self.forecast_file, True, self.forecast_periods, 'exogenous variable')   
        return Z_p, Y_p, forecast_dates
    
    
    def __get_condition_table(self):
        # if conditional forecast is selected, load data
        if self.conditional_forecast and self.var_type != 1:
            # check that data path and files are valid
            iu.check_file_path(self.project_path, self.conditional_forecast_file)
            # then load data file
            data = iu.load_data(self.project_path, self.conditional_forecast_file)
            # check data format
            iu.check_condition_table(data, self.endogenous_variables, self.forecast_periods, self.conditional_forecast_file)
            # if format is correct, recover conditions
            condition_table, shock_table = iu.get_condition_table(data, self.endogenous_variables)
        # if conditional forecast is not selected, return empty lists
        else:
            condition_table, shock_table = [], []
        return condition_table, shock_table
    
    
    def __get_restriction_table(self):
        # if sign restriction is selected, load data
        if self.structural_identification == 4 and self.var_type != 1:
            # check that data path and files are valid
            iu.check_file_path(self.project_path, self.structural_identification_file)
            # then load data file
            data = iu.load_data(self.project_path, self.structural_identification_file)  
            # get raw sample dates
            raw_dates = iu.get_raw_sample_dates(self.project_path, self.data_file, self.start_date, self.end_date)
            # check data format
            iu.check_restriction_table(data, raw_dates, self.endogenous_variables, self.proxy_variables, \
                                self.var_type, self.irf_periods, self.structural_identification_file)
            # if format is correct, recover restrictions
            restriction_table = iu.get_restriction_table(data, raw_dates, self.endogenous_variables, self.proxy_variables)  
        # if sign restriction is not selected, return empty list
        else:
            restriction_table = []
        return restriction_table
        
        
        