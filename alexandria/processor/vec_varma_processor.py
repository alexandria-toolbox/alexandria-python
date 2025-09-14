# imports
import numpy as np
import pandas as pd
import alexandria.math.linear_algebra as la
import alexandria.processor.input_utilities as iu
import alexandria.console.console_utilities as cu


class VecVarmaProcessor(object):

    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass


    def _vec_varma_inputs(self):
        # recover extension model
        self.ext_model = self.__get_extension_model()
        # recover iterations
        self.ext_iterations = self.__get_ext_iterations()        
        # recover burn-in
        self.ext_burnin = self.__get_ext_burnin()         
        # recover credibility level for model estimates
        self.ext_model_credibility = self.__get_ext_model_credibility()     
        # recover constant
        self.ext_constant = self.__get_ext_constant() 
        # recover trend
        self.ext_trend = self.__get_ext_trend() 
        # recover quadratic trend
        self.ext_quadratic_trend = self.__get_ext_quadratic_trend()          
        # recover VEC lags
        self.vec_lags = self.__get_vec_lags()       
        # recover vec pi1
        self.vec_pi1 = self.__get_vec_pi1()  
        # recover vec pi2
        self.vec_pi2 = self.__get_vec_pi2() 
        # recover vec pi3
        self.vec_pi3 = self.__get_vec_pi3() 
        # recover vec pi4
        self.vec_pi4 = self.__get_vec_pi4() 
        # recover prior type
        self.vec_prior_type = self.__get_prior_type() 
        # recover error correction type
        self.error_correction_type = self.__get_error_correction_type() 
        # recover max cointegration rank
        self.max_cointegration_rank = self.__get_max_cointegration_rank() 
        # recover VARMA lags
        self.varma_lags = self.__get_varma_lags()          
        # recover AR coefficients
        self.varma_ar_coefficients = self.__get_varma_ar_coefficients()      
        # recover varma pi1
        self.varma_pi1 = self.__get_varma_pi1()  
        # recover varma pi2
        self.varma_pi2 = self.__get_varma_pi2() 
        # recover varma pi3
        self.varma_pi3 = self.__get_varma_pi3() 
        # recover varma pi4
        self.varma_pi4 = self.__get_varma_pi4()         
        # recover residual lags
        self.residual_lags = self.__get_residual_lags()        
        # recover lambda1
        self.lambda1 = self.__get_lambda1()  
        # recover lambda2
        self.lambda2 = self.__get_lambda2() 
        # recover lambda3
        self.lambda3 = self.__get_lambda3()        
            
        
    def _vec_varma_data(self):
        # print loading message
        if self.progress_bar:
            cu.print_message_to_overwrite('Data loading:')
        # recover in-sample endogenous and exogenous
        self.ext_endogenous, self.ext_exogenous, self.ext_dates = self.__get_ext_insample_data()
        # recover forecast data
        self.ext_Z_p, self.ext_Y_p, self.ext_forecast_dates = self.__get_ext_forecast_data()
        # recover conditional forecast data
        self.ext_condition_table, self.ext_shock_table = self.__get_ext_condition_table()
        # recover sign restrictions data
        self.ext_restriction_table = self.__get_ext_restriction_table()
        # print loading done message
        if self.progress_bar:
            cu.print_message('Data loading:  —  done')


    def _make_vec_varma_information(self):
        # get sample dates
        self.results_information['dates'] = self.ext_dates 
        # get forecast dates
        self.results_information['forecast_dates'] = self.ext_forecast_dates 
        self.results_information['conditional_forecast_dates'] = self.ext_forecast_dates         


    def _make_vec_varma_graphics_information(self):
        # get sample dates
        self.graphics_information['dates'] = self.ext_dates
        # get forecast dates
        self.graphics_information['forecast_dates'] = self.ext_forecast_dates
        self.graphics_information['conditional_forecast_dates'] = self.ext_forecast_dates        
        # get actual data for forecast evaluation, if available
        self.graphics_information['Y_p'] = self.ext_Y_p


    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------  


    def __get_extension_model(self):
        model = self.user_inputs['tab_2_ext']['model']
        if model not in [1, 2]:
            raise TypeError('Value error for VAR extension type. Should be 1 or 2.')  
        return model


    def __get_ext_iterations(self):
        iterations = self.user_inputs['tab_2_ext']['iterations']       
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


    def __get_ext_burnin(self):
        burnin = self.user_inputs['tab_2_ext']['burnin']       
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


    def __get_ext_model_credibility(self):
        model_credibility = self.user_inputs['tab_2_ext']['model_credibility']
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


    def __get_ext_constant(self):
        constant = self.user_inputs['tab_2_ext']['constant']
        if not isinstance(constant, bool):
            raise TypeError('Type error for constant. Should be boolean.') 
        return constant


    def __get_ext_trend(self):
        trend = self.user_inputs['tab_2_ext']['trend']
        if not isinstance(trend, bool):
            raise TypeError('Type error for trend. Should be boolean.') 
        return trend


    def __get_ext_quadratic_trend(self):
        quadratic_trend = self.user_inputs['tab_2_ext']['quadratic_trend']
        if not isinstance(quadratic_trend, bool):
            raise TypeError('Type error for quadratic trend. Should be boolean.') 
        return quadratic_trend


    def __get_vec_lags(self):
        lags = self.user_inputs['tab_2_ext']['vec_lags']
        if not isinstance(lags, (int, str)):
            raise TypeError('Type error for VEC lags. Should be integer.')
        if lags and isinstance(lags, str):
            if lags.isdigit():
                lags = int(lags)
            else:
                raise TypeError('Type error for VEC lags. Should be positive integer.')
        if isinstance(lags, int) and lags <= 0:
            raise TypeError('Value error for VEC lags. Should be positive integer.')
        return lags


    def __get_vec_pi1(self):
        pi1 = self.user_inputs['tab_2_ext']['vec_pi1']
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


    def __get_vec_pi2(self):
        pi2 = self.user_inputs['tab_2_ext']['vec_pi2']
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


    def __get_vec_pi3(self):
        pi3 = self.user_inputs['tab_2_ext']['vec_pi3']
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


    def __get_vec_pi4(self):
        pi4 = self.user_inputs['tab_2_ext']['vec_pi4']
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


    def __get_prior_type(self):
        prior = self.user_inputs['tab_2_ext']['prior_type']
        if prior not in [1, 2, 3]:
            raise TypeError('Value error for VEC prior type. Should be 1, 2 or 3.')  
        return prior


    def __get_error_correction_type(self):
        error_correction_type = self.user_inputs['tab_2_ext']['error_correction_type']
        if error_correction_type not in [1, 2]:
            raise TypeError('Value error for VEC error correction type. Should be 1 or 2.')  
        return error_correction_type


    def __get_max_cointegration_rank(self):
        rank = self.user_inputs['tab_2_ext']['max_cointegration_rank']
        if not isinstance(rank, (int, str)):
            raise TypeError('Type error for max cointegration rank. Should be integer.')
        if rank and isinstance(rank, str):
            if rank.isdigit():
                rank = int(rank)
            else:
                raise TypeError('Type error for max cointegration rank. Should be positive integer.')
        if isinstance(rank, int) and rank <= 0:
            raise TypeError('Value error for max cointegration rank. Should be positive integer.')
        return rank


    def __get_varma_lags(self):
        lags = self.user_inputs['tab_2_ext']['varma_lags']
        if not isinstance(lags, (int, str)):
            raise TypeError('Type error for VARMA lags. Should be integer.')
        if lags and isinstance(lags, str):
            if lags.isdigit():
                lags = int(lags)
            else:
                raise TypeError('Type error for VARMA lags. Should be positive integer.')
        if isinstance(lags, int) and lags <= 0:
            raise TypeError('Value error for VARMA lags. Should be positive integer.')
        return lags


    def __get_varma_ar_coefficients(self):
        ar_coefficients = self.user_inputs['tab_2_ext']['ar_coefficients']
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


    def __get_varma_pi1(self):
        pi1 = self.user_inputs['tab_2_ext']['varma_pi1']
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


    def __get_varma_pi2(self):
        pi2 = self.user_inputs['tab_2_ext']['varma_pi2']
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


    def __get_varma_pi3(self):
        pi3 = self.user_inputs['tab_2_ext']['varma_pi3']
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


    def __get_varma_pi4(self):
        pi4 = self.user_inputs['tab_2_ext']['varma_pi4']
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


    def __get_residual_lags(self):
        lags = self.user_inputs['tab_2_ext']['residual_lags']
        if not isinstance(lags, (int, str)):
            raise TypeError('Type error for residual lags. Should be integer.')
        if lags and isinstance(lags, str):
            if lags.isdigit():
                lags = int(lags)
            else:
                raise TypeError('Type error for residual lags. Should be positive integer.')
        if isinstance(lags, int) and lags <= 0:
            raise TypeError('Value error for residual lags. Should be positive integer.')
        return lags


    def __get_lambda1(self):
        lambda1 = self.user_inputs['tab_2_ext']['lambda1']
        if not isinstance(lambda1, (str, float, int)):
            raise TypeError('Type error for pi1. Should be float or integer.')
        if isinstance(lambda1, str):
            if not lambda1.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for lambda1. Should be float or integer.')
            else:
                lambda1 = float(lambda1)
        if lambda1 <= 0:
            raise TypeError('Value error for lambda1. Should be strictly positive.')
        return lambda1


    def __get_lambda2(self):
        lambda2 = self.user_inputs['tab_2_ext']['lambda2']
        if not isinstance(lambda2, (str, float, int)):
            raise TypeError('Type error for lambda2. Should be float or integer.')
        if isinstance(lambda2, str):
            if not lambda2.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for lambda2. Should be float or integer.')
            else:
                lambda2 = float(lambda2)
        if lambda2 <= 0:
            raise TypeError('Value error for lambda2. Should be strictly positive.')
        return lambda2


    def __get_lambda3(self):
        lambda3 = self.user_inputs['tab_2_ext']['lambda3']
        if not isinstance(lambda3, (str, float, int)):
            raise TypeError('Type error for lambda3. Should be float or integer.')
        if isinstance(lambda3, str):
            if not lambda3.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for lambda3. Should be float or integer.')
            else:
                lambda3 = float(lambda3)
        if lambda3 <= 0:
            raise TypeError('Value error for lambda3. Should be strictly positive.')
        return lambda3


    def __get_ext_insample_data(self):
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


    def __get_ext_forecast_data(self):
        # default values for endogenous and exogenous
        Z_p, Y_p = [], []
        # if forecast is selected, recover forecast dates
        if self.forecast or self.conditional_forecast:
            end_date = self.ext_dates[-1]
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
    
    
    def __get_ext_condition_table(self):
        # if conditional forecast is selected, load data
        if self.conditional_forecast:
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
    
    
    def __get_ext_restriction_table(self):
        # if sign restriction is selected, load data
        if self.structural_identification == 4:
            # check that data path and files are valid
            iu.check_file_path(self.project_path, self.structural_identification_file)
            # then load data file
            data = iu.load_data(self.project_path, self.structural_identification_file)  
            # get raw sample dates
            raw_dates = iu.get_raw_sample_dates(self.project_path, self.data_file, self.start_date, self.end_date)
            # check data format
            iu.check_restriction_table(data, raw_dates, self.endogenous_variables, [], \
                                       2, self.irf_periods, self.structural_identification_file)
            # if format is correct, recover restrictions
            restriction_table = iu.get_restriction_table(data, raw_dates, self.endogenous_variables, [])  
        # if sign restriction is not selected, return empty list
        else:
            restriction_table = []
        return restriction_table
        
        
        