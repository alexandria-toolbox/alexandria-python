# imports
import numpy as np
import pandas as pd
import alexandria.processor.input_utilities as iu
import alexandria.console.console_utilities as cu


class NowcastingProcessor(object):

    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass


    def _nowcasting_inputs(self):
        # recover extension model
        self.now_model = self.__get_nowcasting_model()
        # recover iterations
        self.now_iterations = self.__get_now_iterations()        
        # recover burn-in
        self.now_burnin = self.__get_now_burnin() 
        # recover credibility level for model estimates
        self.now_model_credibility = self.__get_now_model_credibility()  
        # recover endogenous lags for midas
        self.midas_endogenous_lags = self.__get_midas_endogenous_lags()   
        # recover exogenous lags for midas
        self.midas_exogenous_lags = self.__get_midas_exogenous_lags() 
        # recover polynomial order for midas 
        self.midas_polynomial_order = self.__get_midas_polynomial_order() 
        # recover prior type for midas 
        self.midas_representation, self.midas_prior_type = self.__get_midas_model() 
        # recover omega1 for midas 
        self.midas_omega1 = self.__get_midas_omega1()
        # recover omega2 for midas 
        self.midas_omega2 = self.__get_midas_omega2()
        # recover upsilon1 for midas 
        self.midas_upsilon1 = self.__get_midas_upsilon1()
        # recover upsilon2 for midas 
        self.midas_upsilon2 = self.__get_midas_upsilon2()
        # recover mfbvar constant
        self.mfbvar_constant = self.__get_mfbvar_constant() 
        # recover mfbvar trend
        self.mfbvar_trend = self.__get_mfbvar_trend() 
        # recover mfbvar quadratic trend
        self.mfbvar_quadratic_trend = self.__get_mfbvar_quadratic_trend()  
        # recover mfbvar decomposition
        self.mfbvar_decomposition = self.__get_mfbvar_decomposition() 
        # recover mfbvar lags
        self.mfbvar_lags = self.__get_mfbvar_lags()      
        # recover AR coefficients
        self.mfbvar_ar_coefficients = self.__get_mfbvar_ar_coefficients()
        # recover mfbvar pi1
        self.mfbvar_pi1 = self.__get_mfbvar_pi1()  
        # recover mfbvar pi2
        self.mfbvar_pi2 = self.__get_mfbvar_pi2() 
        # recover mfbvar pi3
        self.mfbvar_pi3 = self.__get_mfbvar_pi3() 
        # recover mfbvar pi4
        self.mfbvar_pi4 = self.__get_mfbvar_pi4() 
        # get long run prior file
        self.mfbvar_decomposition_file = self.__get_mfbvar_decomposition_file()
        # get dfm factors
        self.dfm_factors = self.__get_dfm_factors()
        # get dfm loadings lags
        self.dfm_loadings_lags = self.__get_dfm_loadings_lags()
        # get dfm factor lags
        self.dfm_factor_lags = self.__get_dfm_factor_lags()
        # get dfm residual lags
        self.dfm_residual_lags = self.__get_dfm_residual_lags()
        # get dfm sigma
        self.dfm_sigma = self.__get_dfm_sigma()
        # get dfm omega
        self.dfm_omega = self.__get_dfm_omega()
        # get dfm delta1
        self.dfm_delta1 = self.__get_dfm_delta1()
        # get dfm pi1
        self.dfm_pi1 = self.__get_dfm_pi1()
        # get dfm pi2
        self.dfm_pi2 = self.__get_dfm_pi2()
        # get dfm pi3
        self.dfm_pi3 = self.__get_dfm_pi3()
        # get dfm omega1
        self.dfm_omega1 = self.__get_dfm_omega1()


    def _nowcasting_data(self):
        # print loading message
        if self.progress_bar:
            cu.print_message_to_overwrite('Data loading:')
        # recover in-sample data
        self.now_endogenous, self.now_exogenous, self.now_dates = self.__get_now_insample_data()
        # recover forecast data
        self.now_Z_p, self.now_Y_p, self.now_forecast_dates = self.__get_now_forecast_data()
        # recover decomposition data
        self.now_decomposition_table = self.__get_now_decomposition_table()
        # recover conditional forecast data
        self.now_condition_table, self.now_shock_table = self.__get_now_condition_table()        
        # recover sign restrictions data
        self.now_restriction_table = self.__get_now_restriction_table()  
        # print loading done message
        if self.progress_bar:
            cu.print_message('Data loading:  —  done')
        
        
    def _make_nowcasting_information(self):
        # get sample dates
        self.results_information['dates'] = self.now_dates
        # get forecast dates
        self.results_information['forecast_dates'] = self.now_forecast_dates 
        self.results_information['conditional_forecast_dates'] = self.now_forecast_dates  
        # get decomposition file
        self.results_information['decomposition_file'] = self.mfbvar_decomposition_file
        
    
    def _make_nowcasting_graphics_information(self):
        # get sample dates
        self.graphics_information['dates'] = self.now_dates
        # get forecast dates
        self.graphics_information['forecast_dates'] = self.now_forecast_dates
        self.graphics_information['conditional_forecast_dates'] = self.now_forecast_dates        
        # get actual data for forecast evaluation, if available
        self.graphics_information['Y_p'] = self.now_Y_p


    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------  


    def __get_nowcasting_model(self):
        model = self.user_inputs['tab_2_now']['model']
        if model not in [1, 2, 3]:
            raise TypeError('Value error for nowcasting model. Should be 1, 2 or 3.')  
        return model        
        
        
    def __get_now_iterations(self):
        iterations = self.user_inputs['tab_2_now']['iterations']       
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
        
        
    def __get_now_burnin(self):
        burnin = self.user_inputs['tab_2_now']['burnin']       
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
        
        
    def __get_now_model_credibility(self):
        model_credibility = self.user_inputs['tab_2_now']['model_credibility']
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
        
      
    def __get_midas_endogenous_lags(self):
        lags = self.user_inputs['tab_2_now']['midas_endogenous_lags']
        if not isinstance(lags, (int, str)):
            raise TypeError('Type error for MIDAS endogenous lags. Should be integer.')
        if lags and isinstance(lags, str):
            if lags.isdigit():
                lags = int(lags)
            else:
                raise TypeError('Type error for MIDAS endogenous lags. Should be positive integer.')
        if isinstance(lags, int) and lags < 0:
            raise TypeError('Value error for MIDAS endogenous lags. Should be positive integer.')
        return lags        
    
    
    def __get_midas_exogenous_lags(self):
        lags = self.user_inputs['tab_2_now']['midas_exogenous_lags']   
        if not isinstance(lags, (str, list, float, int)):
            raise TypeError('Type error for MIDAS exogenous lags. Should be integer or list of integers.')
        if isinstance(lags, str):
            lags = iu.string_to_list(lags)
            if not all([lag.replace('.','',1).isdigit() for lag in lags]):
                raise TypeError('Type error for exogenous lags. All elements should be integers.')  
            else:
                lags = [int(lag) for lag in lags] 
        if isinstance(lags, list):
            if len(lags) != len(self.exogenous_variables) and len(lags) != 1:
                raise TypeError('Dimension error for exogenous lags. Dimension of exogenous lags and exogenous variables don\'t match.')    
            if not all([isinstance(lag, (int, float)) for lag in lags]):
                raise TypeError('Type error for exogenous lags. All elements should be integers.')    
            else:
                lags = np.array(lags)    
            if len(lags) == 1:
                lags = lags[0]   
        return lags
    
        
    def __get_midas_polynomial_order(self):
        order = self.user_inputs['tab_2_now']['midas_polynomial_order']
        if not isinstance(order, (int, str)):
            raise TypeError('Type error for MIDAS polynomial order. Should be integer.')
        if order and isinstance(order, str):
            if order.isdigit():
                order = int(order)
            else:
                raise TypeError('Type error for MIDAS polynomial order. Should be positive integer.')
        if isinstance(order, int) and order <= 0:
            raise TypeError('Value error for MIDAS polynomial order. Should be positive integer.')
        return order    
      
        
    def __get_midas_model(self):
        prior = self.user_inputs['tab_2_now']['midas_model']
        if prior not in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            raise TypeError('Value error for MIDAS prior. Should be integer between 1 and 9.')
        if prior == 1:
            representation = 'unrestricted'
            prior_type = 'minnesota'
        elif prior == 2:
            representation = 'unrestricted'            
            prior_type = 'horseshoe'
        elif prior == 3:
            representation = 'unrestricted'            
            prior_type = 'lasso'
        elif prior == 4:
            representation = 'almon'
            prior_type = 'minnesota'
        elif prior == 5:
            representation = 'almon'            
            prior_type = 'horseshoe'     
        elif prior == 6:
            representation = 'almon'            
            prior_type = 'lasso'                
        elif prior == 7:
            representation = 'fourier'            
            prior_type = 'minnesota'                
        elif prior == 8:
            representation = 'fourier'            
            prior_type = 'horseshoe'                 
        elif prior == 9:
            representation = 'fourier'            
            prior_type = 'lasso'               
        return representation, prior_type 
        
        
    def __get_midas_omega1(self):
        omega1 = self.user_inputs['tab_2_now']['midas_omega1']
        if not isinstance(omega1, (str, float, int)):
            raise TypeError('Type error for MIDAS omega1. Should be float or integer.')
        if isinstance(omega1, str):
            if not omega1.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for MIDAS omega1. Should be float or integer.')
            else:
                omega1 = float(omega1)
        if omega1 <= 0:
            raise TypeError('Value error for MIDAS omega1. Should be strictly positive.')
        return omega1      
        
      
    def __get_midas_omega2(self):
        omega2 = self.user_inputs['tab_2_now']['midas_omega2']
        if not isinstance(omega2, (str, float, int)):
            raise TypeError('Type error for MIDAS omega2. Should be float or integer.')
        if isinstance(omega2, str):
            if not omega2.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for MIDAS omega2. Should be float or integer.')
            else:
                omega2 = float(omega2)
        if omega2 <= 0:
            raise TypeError('Value error for MIDAS omega2. Should be strictly positive.')
        return omega2          
      
        
    def __get_midas_upsilon1(self):
        upsilon1 = self.user_inputs['tab_2_now']['midas_upsilon1']
        if not isinstance(upsilon1, (str, float, int)):
            raise TypeError('Type error for MIDAS upsilon1. Should be float or integer.')
        if isinstance(upsilon1, str):
            if not upsilon1.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for MIDAS upsilon1. Should be float or integer.')
            else:
                upsilon1 = float(upsilon1)
        if upsilon1 <= 0:
            raise TypeError('Value error for MIDAS upsilon1. Should be strictly positive.')
        return upsilon1


    def __get_midas_upsilon2(self):
        upsilon2 = self.user_inputs['tab_2_now']['midas_upsilon2']
        if not isinstance(upsilon2, (str, float, int)):
            raise TypeError('Type error for MIDAS upsilon2. Should be float or integer.')
        if isinstance(upsilon2, str):
            if not upsilon2.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for MIDAS upsilon2. Should be float or integer.')
            else:
                upsilon2 = float(upsilon2)
        if upsilon2 <= 0:
            raise TypeError('Value error for MIDAS upsilon2. Should be strictly positive.')
        return upsilon2 


    def __get_mfbvar_constant(self):
        constant = self.user_inputs['tab_2_now']['mfbvar_constant']
        if not isinstance(constant, bool):
            raise TypeError('Type error for constant. Should be boolean.') 
        return constant


    def __get_mfbvar_trend(self):
        trend = self.user_inputs['tab_2_now']['mfbvar_trend']
        if not isinstance(trend, bool):
            raise TypeError('Type error for trend. Should be boolean.') 
        return trend


    def __get_mfbvar_quadratic_trend(self):
        quadratic_trend = self.user_inputs['tab_2_now']['mfbvar_quadratic_trend']
        if not isinstance(quadratic_trend, bool):
            raise TypeError('Type error for quadratic trend. Should be boolean.') 
        return quadratic_trend


    def __get_mfbvar_decomposition(self):
        decomposition = self.user_inputs['tab_2_now']['mfbvar_decomposition']
        if not isinstance(decomposition, bool):
            raise TypeError('Type error for decomposition. Should be boolean.') 
        return decomposition


    def __get_mfbvar_lags(self):
        lags = self.user_inputs['tab_2_now']['mfbvar_lags']
        if not isinstance(lags, (int, str)):
            raise TypeError('Type error for MF-BVAR lags. Should be integer.')
        if lags and isinstance(lags, str):
            if lags.isdigit():
                lags = int(lags)
            else:
                raise TypeError('Type error for MF-BVAR lags. Should be positive integer.')
        if isinstance(lags, int) and lags <= 0:
            raise TypeError('Value error for MF-BVAR lags. Should be positive integer.')
        return lags


    def __get_mfbvar_ar_coefficients(self):
        ar_coefficients = self.user_inputs['tab_2_now']['mfbvar_ar_coefficients']
        if not isinstance(ar_coefficients, (str, list, float, int)):
            raise TypeError('Type error for MF-BVAR AR coefficients. Should be scalar or list of scalars.')
        if isinstance(ar_coefficients, str):
            ar_coefficients = iu.string_to_list(ar_coefficients)
            if not all([ar_entry.replace('.','',1).replace('-','',1).isdigit() for ar_entry in ar_coefficients]):
                raise TypeError('Type error for MF-BVAR AR coefficients. All elements should be scalars.')
            else:
                ar_coefficients = [float(ar_entry) for ar_entry in ar_coefficients]
        if isinstance(ar_coefficients, list):
            if len(ar_coefficients) != len(self.endogenous_variables) and len(ar_coefficients) != 1:
                raise TypeError('Dimension error for MF-BVAR AR coefficients. Dimension of AR coefficients and endogenous don\'t match.')
            if not all([isinstance(ar_entry, (int, float)) for ar_entry in ar_coefficients]):
                raise TypeError('Type error for MF-BVAR AR coefficients. All elements should be scalars.')
            else:
                ar_coefficients = np.array(ar_coefficients)
            if len(ar_coefficients) == 1:
                ar_coefficients = ar_coefficients[0]
        return ar_coefficients


    def __get_mfbvar_pi1(self):
        pi1 = self.user_inputs['tab_2_now']['mfbvar_pi1']
        if not isinstance(pi1, (str, float, int)):
            raise TypeError('Type error for MF-BVAR pi1. Should be float or integer.')
        if isinstance(pi1, str):
            if not pi1.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for MF-BVAR pi1. Should be float or integer.')
            else:
                pi1 = float(pi1)
        if pi1 <= 0:
            raise TypeError('Value error for MF-BVAR pi1. Should be strictly positive.')
        return pi1


    def __get_mfbvar_pi2(self):
        pi2 = self.user_inputs['tab_2_now']['mfbvar_pi2']
        if not isinstance(pi2, (str, float, int)):
            raise TypeError('Type error for MF-BVAR pi2. Should be float or integer.')
        if isinstance(pi2, str):
            if not pi2.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for MF-BVAR pi2. Should be float or integer.')
            else:
                pi2 = float(pi2)
        if pi2 <= 0:
            raise TypeError('Value error for MF-BVAR pi2. Should be strictly positive.')
        return pi2


    def __get_mfbvar_pi3(self):
        pi3 = self.user_inputs['tab_2_now']['mfbvar_pi3']
        if not isinstance(pi3, (str, float, int)):
            raise TypeError('Type error for MF-BVAR pi3. Should be float or integer.')
        if isinstance(pi3, str):
            if not pi3.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for MF-BVAR pi3. Should be float or integer.')
            else:
                pi3 = float(pi3)
        if pi3 <= 0:
            raise TypeError('Value error for MF-BVAR pi3. Should be strictly positive.')
        return pi3


    def __get_mfbvar_pi4(self):
        pi4 = self.user_inputs['tab_2_now']['mfbvar_pi4']
        if not isinstance(pi4, (str, float, int)):
            raise TypeError('Type error for MF-BVAR pi4. Should be float or integer.')
        if isinstance(pi4, str):
            if not pi4.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for MF-BVAR pi4. Should be float or integer.')
            else:
                pi4 = float(pi4)
        if pi4 <= 0:
            raise TypeError('Value error for MF-BVAR pi4. Should be strictly positive.')
        return pi4


    def __get_mfbvar_decomposition_file(self):
        decomposition_file = self.user_inputs['tab_2_now']['mfbvar_decomposition_file']
        if not isinstance(decomposition_file, str):
            raise TypeError('Type error for MF-BVAR decomposition file. Should be string.')
        decomposition_file = iu.fix_string(decomposition_file)
        return decomposition_file 


    def __get_dfm_factors(self):
        factors = self.user_inputs['tab_2_now']['dfm_factors']       
        if not isinstance(factors, (int, str)):
            raise TypeError('Type error for DFM factors. Should be integer.')
        if factors and isinstance(factors, str):
            if factors.isdigit():
                factors = int(factors)
            else:
                raise TypeError('Type error for DFM factors. Should be positive integer.')
        if isinstance(factors, int) and factors <= 0:
            raise TypeError('Value error for DFM factors. Should be positive integer.')
        return factors


    def __get_dfm_loadings_lags(self):
        lags = self.user_inputs['tab_2_now']['dfm_loadings_lags']
        if not isinstance(lags, (int, str)):
            raise TypeError('Type error for DFM loadings lags. Should be integer.')
        if lags and isinstance(lags, str):
            if lags.isdigit():
                lags = int(lags)
            else:
                raise TypeError('Type error for DFM loadings lags. Should be positive integer.')
        if isinstance(lags, int) and lags < 0:
            raise TypeError('Value error for DFM loadings lags. Should be positive integer.')
        return lags


    def __get_dfm_factor_lags(self):
        lags = self.user_inputs['tab_2_now']['dfm_factor_lags']
        if not isinstance(lags, (int, str)):
            raise TypeError('Type error for DFM factor lags. Should be integer.')
        if lags and isinstance(lags, str):
            if lags.isdigit():
                lags = int(lags)
            else:
                raise TypeError('Type error for DFM factor lags. Should be positive integer.')
        if isinstance(lags, int) and lags <= 0:
            raise TypeError('Value error for DFM factor lags. Should be positive integer.')
        return lags


    def __get_dfm_residual_lags(self):
        lags = self.user_inputs['tab_2_now']['dfm_residual_lags']
        if not isinstance(lags, (int, str)):
            raise TypeError('Type error for DFM residual lags. Should be integer.')
        if lags and isinstance(lags, str):
            if lags.isdigit():
                lags = int(lags)
            else:
                raise TypeError('Type error for DFM residual lags. Should be positive integer.')
        if isinstance(lags, int) and lags < 0:
            raise TypeError('Value error for DFM residual lags. Should be positive integer.')
        return lags


    def __get_dfm_sigma(self):
        sigma = self.user_inputs['tab_2_now']['dfm_sigma']
        if not isinstance(sigma, (str, float, int)):
            raise TypeError('Type error for DFM sigma. Should be float or integer.')
        if isinstance(sigma, str):
            if not sigma.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for DFM sigma. Should be float or integer.')
            else:
                sigma = float(sigma)
        if sigma <= 0:
            raise TypeError('Value error for DFM sigma. Should be strictly positive.')
        return sigma


    def __get_dfm_omega(self):
        omega = self.user_inputs['tab_2_now']['dfm_omega']
        if not isinstance(omega, (str, float, int)):
            raise TypeError('Type error for DFM omega. Should be float or integer.')
        if isinstance(omega, str):
            if not omega.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for DFM omega. Should be float or integer.')
            else:
                omega = float(omega)
        if omega <= 0:
            raise TypeError('Value error for DFM omega. Should be strictly positive.')
        return omega


    def __get_dfm_delta1(self):
        delta1 = self.user_inputs['tab_2_now']['dfm_delta1']
        if not isinstance(delta1, (str, float, int)):
            raise TypeError('Type error for DFM delta1. Should be float or integer.')
        if isinstance(delta1, str):
            if not delta1.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for DFM delta1. Should be float or integer.')
            else:
                delta1 = float(delta1)
        if delta1 <= 0:
            raise TypeError('Value error for DFM delta1. Should be strictly positive.')
        return delta1


    def __get_dfm_pi1(self):
        pi1 = self.user_inputs['tab_2_now']['dfm_pi1']
        if not isinstance(pi1, (str, float, int)):
            raise TypeError('Type error for DFM pi1. Should be float or integer.')
        if isinstance(pi1, str):
            if not pi1.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for DFM pi1. Should be float or integer.')
            else:
                pi1 = float(pi1)
        if pi1 <= 0:
            raise TypeError('Value error for DFM pi1. Should be strictly positive.')
        return pi1


    def __get_dfm_pi2(self):
        pi2 = self.user_inputs['tab_2_now']['dfm_pi2']
        if not isinstance(pi2, (str, float, int)):
            raise TypeError('Type error for DFM pi2. Should be float or integer.')
        if isinstance(pi2, str):
            if not pi2.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for DFM pi2. Should be float or integer.')
            else:
                pi2 = float(pi2)
        if pi2 <= 0:
            raise TypeError('Value error for DFM pi2. Should be strictly positive.')
        return pi2


    def __get_dfm_pi3(self):
        pi3 = self.user_inputs['tab_2_now']['dfm_pi3']
        if not isinstance(pi3, (str, float, int)):
            raise TypeError('Type error for DFM pi3. Should be float or integer.')
        if isinstance(pi3, str):
            if not pi3.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for DFM pi3. Should be float or integer.')
            else:
                pi3 = float(pi3)
        if pi3 <= 0:
            raise TypeError('Value error for DFM pi3. Should be strictly positive.')
        return pi3


    def __get_dfm_omega1(self):
        omega1 = self.user_inputs['tab_2_now']['dfm_omega1']
        if not isinstance(omega1, (str, float, int)):
            raise TypeError('Type error for DFM omega1. Should be float or integer.')
        if isinstance(omega1, str):
            if not omega1.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for DFM omega1. Should be float or integer.')
            else:
                omega1 = float(omega1)
        if omega1 <= 0:
            raise TypeError('Value error for DFM omega1. Should be strictly positive.')
        return omega1


    def __get_now_insample_data(self):
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
        endogenous = iu.fetch_nan_data(data, self.data_file, self.start_date, \
        self.end_date, self.endogenous_variables, 'Endogenous variables')
        exogenous = iu.fetch_nan_data(data, self.data_file, self.start_date, \
        self.end_date, self.exogenous_variables, 'Exogenous variables')            
        # infer date format, then recover sample dates
        date_format = iu.infer_date_format(self.frequency, self.data_file, \
                                        self.start_date, self.end_date)
        dates = iu.generate_dates(data, date_format, self.frequency, self.data_file, \
                               self.start_date, self.end_date)
        if self.now_model == 3:
            endogenous = endogenous.flatten()
            dates = pd.to_datetime(data.loc[self.start_date:self.end_date, \
                    self.endogenous_variables[0]].dropna().index)
        return endogenous, exogenous, dates


    def __get_now_forecast_data(self):
        # default values for endogenous and exogenous
        Z_p, Y_p = [], []
        # if forecast is selected, recover forecast dates
        if self.forecast or self.conditional_forecast:
            end_date = self.now_dates[-1]
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
            # if model if MFBVAR and there are exogenous variables in the model
            if self.now_model == 1 and self.exogenous_variables: 
                # load in-sample data to fill missing variables
                in_sample_data = iu.load_data(self.project_path, self.data_file)
                in_sample_data = in_sample_data[self.exogenous_variables].iloc[-2:].values
                # load exogenous data
                Z_p = iu.fetch_forecast_data(data, in_sample_data, self.exogenous_variables, 
                self.forecast_file, True, self.forecast_periods, 'exogenous variable')                  
        return Z_p, Y_p, forecast_dates


    def __get_now_decomposition_table(self):
        # if model is MFBVAR and decomposition is selected, load data
        if self.now_model == 1 and self.mfbvar_decomposition:
            # check that data path and files are valid
            iu.check_file_path(self.project_path, self.mfbvar_decomposition_file)
            # then load data file
            data = iu.load_data(self.project_path, self.mfbvar_decomposition_file)
            # check data format
            iu.check_decomposition_table(data, self.endogenous_variables, self.mfbvar_decomposition_file)
            # turn to array
            decomposition_table = data.values[0]
        else:
            decomposition_table = []
        return decomposition_table
            

    def __get_now_condition_table(self):
        # if model is MFBVAR and conditional forecast is selected, load data
        if self.now_model == 1 and self.conditional_forecast:
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


    def __get_now_restriction_table(self):
        # if model is MFBVAR and sign restriction is selected, load data
        if self.now_model == 1 and self.structural_identification == 4:
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






