# imports
import numpy as np
import pandas as pd
import alexandria.math.linear_algebra as la
import alexandria.processor.input_utilities as iu
import alexandria.console.console_utilities as cu


class RegressionProcessor(object):

    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass


    def _regression_inputs(self):
        # recover regression type
        self.regression_type = self.__get_regression_type()
        # recover iterations
        self.iterations = self.__get_iterations()
        # recover burn-in
        self.burnin = self.__get_burnin()     
        # recover credibility level for model estimates
        self.model_credibility = self.__get_model_credibility()
        # recover b
        self.b = self.__get_b()          
        # recover V
        self.V = self.__get_V()         
        # recover alpha
        self.alpha = self.__get_alpha()            
        # recover delta
        self.delta = self.__get_delta()         
        # recover g
        self.g = self.__get_g()            
        # recover Q
        self.Q = self.__get_Q()         
        # recover tau
        self.tau = self.__get_tau()           
        # recover thinning
        self.thinning = self.__get_thinning()          
        # recover thinning frequency
        self.thinning_frequency = self.__get_thinning_frequency()          
        # recover name of file for Z regressors
        self.Z_variables = self.__get_Z_variables()
        # recover q
        self.q = self.__get_q()          
        # recover p
        self.p = self.__get_p()  
        # recover H
        self.H = self.__get_H()  
        # recover constant
        self.constant = self.__get_constant() 
        # recover b (constant)
        self.b_constant = self.__get_b_constant() 
        # recover V (constant)
        self.V_constant = self.__get_V_constant() 
        # recover trend
        self.trend = self.__get_trend() 
        # recover b (trend)
        self.b_trend = self.__get_b_trend() 
        # recover V (trend)
        self.V_trend = self.__get_V_trend() 
        # recover quadratic trend
        self.quadratic_trend = self.__get_quadratic_trend() 
        # recover b (quadratic trend)
        self.b_quadratic_trend = self.__get_b_quadratic_trend() 
        # recover V (quadratic trend)
        self.V_quadratic_trend = self.__get_V_quadratic_trend() 
        # recover in-sample fit
        self.insample_fit = self.__get_insample_fit() 
        # recover marginal likelihood
        self.marginal_likelihood = self.__get_marginal_likelihood() 
        # recover hyperparameter optimization
        self.hyperparameter_optimization = self.__get_hyperparameter_optimization() 
        # recover optimization type
        self.optimization_type = self.__get_optimization_type() 


    def _regression_data(self):
        # print loading message
        if self.progress_bar:
            cu.print_message('data loading:')
        # recover in-sample endogenous and exogenous
        self.endogenous, self.exogenous, self.dates = self.__get_insample_data()
        # recover heteroscedastic data
        self.Z = self.__get_heteroscedastic_data()
        # recover forecast data
        self.X_p, self.y_p, self.Z_p, self.forecast_dates = self.__get_forecast_data()  
        # print loading completion message
        if self.progress_bar:
            cu.print_message('  — / —    ' + '[' + 33 * '=' + ']' + '  —  done')


    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------  


    def __get_regression_type(self):
        regression_type = self.user_inputs['tab_2_lr']['regression_type']
        if regression_type not in [1, 2, 3, 4, 5, 6]:
            raise TypeError('Value error for regression type. Should be integer between 1 and 6.')  
        return regression_type 


    def __get_iterations(self):
        iterations = self.user_inputs['tab_2_lr']['iterations']       
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


    def __get_burnin(self):
        burnin = self.user_inputs['tab_2_lr']['burnin']       
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
    

    def __get_model_credibility(self):
        model_credibility = self.user_inputs['tab_2_lr']['model_credibility']
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
    
    
    def __get_b(self):
        b = self.user_inputs['tab_2_lr']['b']
        if not isinstance(b, (str, list, float, int)):
            raise TypeError('Type error for b. Should be scalar or list of scalars.')
        if isinstance(b, str):
            b = iu.string_to_list(b)
            if not all([b_entry.replace('.','',1).replace('-','',1).isdigit() for b_entry in b]):
                raise TypeError('Type error for b. All elements should be scalars.')
            else:
                b = [float(b_entry) for b_entry in b]
        if isinstance(b, list):
            if len(b) != len(self.exogenous_variables) and len(b) != 1:
                raise TypeError('Dimension error for b. Dimension of b and exogenous don\'t match.')
            if not all([isinstance(b_entry, (int, float)) for b_entry in b]):
                raise TypeError('Type error for b. All elements should be scalars.')
            else:
                b = np.array(b)
            if len(b) == 1:
                b = b[0]
        return b
            
            
    def __get_V(self):
        V = self.user_inputs['tab_2_lr']['V']
        if not isinstance(V, (str, list, float, int)):
            raise TypeError('Type error for V. Should be scalar or list of scalars.')
        if isinstance(V, str):
            V = iu.string_to_list(V)
            if not all([V_entry.replace('.','',1).replace('-','',1).isdigit() for V_entry in V]):
                raise TypeError('Type error for V. All elements should be scalars.')
            else:
                V = [float(V_entry) for V_entry in V]
        if isinstance(V, list):
            if len(V) != len(self.exogenous_variables) and len(V) != 1:
                raise TypeError('Dimension error for V. Dimension of V and exogenous don\'t match.')
            if not all([isinstance(V_entry, (int, float)) for V_entry in V]):
                raise TypeError('Type error for V. All elements should be scalars.')
            else:
                V = np.array(V)
            if not all(V > 0):
                raise TypeError('Value error for V. All elements should be positive scalars.')
            if len(V) == 1:
                V = V[0]
        else:
            if V <= 0:
                raise TypeError('Value error for V. Should be positive scalar.')
        return V      
    
    
    def __get_alpha(self):
        alpha = self.user_inputs['tab_2_lr']['alpha']
        if not isinstance(alpha, (str, float, int)):
            raise TypeError('Type error for alpha. Should be float or integer.')
        if isinstance(alpha, str):
            if not alpha.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for alpha. Should be float or integer.')
            else:
                alpha = float(alpha)
        if alpha <= 0:
            raise TypeError('Value error for alpha. Should be strictly positive.')
        return alpha    
    
    
    def __get_delta(self):
        delta = self.user_inputs['tab_2_lr']['delta']
        if not isinstance(delta, (str, float, int)):
            raise TypeError('Type error for delta. Should be float or integer.')
        if isinstance(delta, str):
            if not delta.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for delta. Should be float or integer.')
            else:
                delta = float(delta)
        if delta <= 0:
            raise TypeError('Value error for delta. Should be strictly positive.')
        return delta      
    

    def __get_g(self):
        g = self.user_inputs['tab_2_lr']['g']
        if not isinstance(g, (str, list, float, int)):
            raise TypeError('Type error for g. Should be scalar or list of scalars.')
        if isinstance(g, str):
            g = iu.string_to_list(g)
            if not all([g_entry.replace('.','',1).replace('-','',1).isdigit() for g_entry in g]):
                raise TypeError('Type error for g. All elements should be scalars.')
            else:
                g = [float(g_entry) for g_entry in g]
        if isinstance(g, list):
            if not all([isinstance(g_entry, (int, float)) for g_entry in g]):
                raise TypeError('Type error for g. All elements should be scalars.')
            else:
                g = np.array(g)
            if len(g) == 1:
                g = g[0]
        return g    
    

    def __get_Q(self):
        Q = self.user_inputs['tab_2_lr']['Q']
        if not isinstance(Q, (str, list, float, int)):
            raise TypeError('Type error for Q. Should be scalar or list of scalars.')
        if isinstance(Q, str):
            Q = iu.string_to_list(Q)
            if not all([Q_entry.replace('.','',1).replace('-','',1).isdigit() for Q_entry in Q]):
                raise TypeError('Type error for Q. All elements should be scalars.')
            else:
                Q = [float(Q_entry) for Q_entry in Q]
        if isinstance(Q, list):
            if not all([isinstance(Q_entry, (int, float)) for Q_entry in Q]):
                raise TypeError('Type error for Q. All elements should be scalars.')
            else:
                Q = np.array(Q)
            if not all(Q > 0):
                raise TypeError('Value error for Q. All elements should be positive scalars.')
            if len(Q) == 1:
                Q = Q[0]
        else:
            if Q <= 0:
                raise TypeError('Value error for Q. Should be positive scalar.')
        return Q       


    def __get_tau(self):
        tau = self.user_inputs['tab_2_lr']['tau']
        if not isinstance(tau, (str, float, int)):
            raise TypeError('Type error for tau. Should be float or integer.')
        if isinstance(tau, str):
            if not tau.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for tau. Should be float or integer.')
            else:
                tau = float(tau)
        if tau <= 0:
            raise TypeError('Value error for tau. Should be strictly positive.')
        return tau 
    
    
    def __get_thinning(self):
        thinning = self.user_inputs['tab_2_lr']['thinning']
        if not isinstance(thinning, bool):
            raise TypeError('Type error for thinning. Should be boolean.') 
        return thinning     
    
    
    def __get_thinning_frequency(self):
        thinning_frequency = self.user_inputs['tab_2_lr']['thinning_frequency']       
        if not isinstance(thinning_frequency, (int, str)):
            raise TypeError('Type error for thinning frequency. Should be integer.')
        if thinning_frequency and isinstance(thinning_frequency, str):
            if thinning_frequency.isdigit():
                thinning_frequency = int(thinning_frequency)
            else:
                raise TypeError('Type error for thinning frequency. Should be positive integer.')
        if isinstance(thinning_frequency, int) and thinning_frequency <= 0:
            raise TypeError('Value error for thinning frequency. Should be positive integer.')
        return thinning_frequency    
    

    def __get_Z_variables(self):
        Z_variables = self.user_inputs['tab_2_lr']['Z_variables']
        if self.regression_type == 5:
            if not Z_variables or not isinstance(Z_variables, (str, list)):
                raise TypeError('Type error for Z variables. Should be list of strings.')
            Z_variables = iu.string_to_list(Z_variables)
            if not all(isinstance(element, str) for element in Z_variables):
                raise TypeError('Type error for Z variables. Should be list of strings.')
        else:
            Z_variables = ' '
        return Z_variables


    def __get_q(self):
        q = self.user_inputs['tab_2_lr']['q']       
        if not isinstance(q, (int, str)):
            raise TypeError('Type error for q. Should be integer.')
        if q and isinstance(q, str):
            if q.isdigit():
                q = int(q)
            else:
                raise TypeError('Type error for q. Should be positive integer.')
        if isinstance(q, int) and q <= 0:
            raise TypeError('Value error for q. Should be positive integer.')
        return q


    def __get_p(self):
        p = self.user_inputs['tab_2_lr']['p']
        if not isinstance(p, (str, list, float, int)):
            raise TypeError('Type error for p. Should be scalar or list of scalars.')
        if isinstance(p, str):
            p = iu.string_to_list(p)
            if not all([p_entry.replace('.','',1).replace('-','',1).isdigit() for p_entry in p]):
                raise TypeError('Type error for p. All elements should be scalars.')
            else:
                p = [float(p_entry) for p_entry in p]
        if isinstance(p, list):
            if len(p) != self.q and len(p) != 1:
                raise TypeError('Dimension error for p. Dimension of p and lag length q don\'t match.')              
            if not all([isinstance(p_entry, (int, float)) for p_entry in p]):
                raise TypeError('Type error for p. All elements should be scalars.')
            else:
                p = np.array(p)
            if len(p) == 1:
                p = p[0]
        return p   


    def __get_H(self):
        H = self.user_inputs['tab_2_lr']['H']
        if not isinstance(H, (str, list, float, int)):
            raise TypeError('Type error for H. Should be scalar or list of scalars.')
        if isinstance(H, str):
            H = iu.string_to_list(H)
            if not all([H_entry.replace('.','',1).replace('-','',1).isdigit() for H_entry in H]):
                raise TypeError('Type error for H. All elements should be scalars.')
            else:
                H = [float(H_entry) for H_entry in H]
        if isinstance(H, list):
            if len(H) != self.q and len(H) != 1:
                raise TypeError('Dimension error for H. Dimension of H and lag length q don\'t match.')            
            if not all([isinstance(H_entry, (int, float)) for H_entry in H]):
                raise TypeError('Type error for H. All elements should be scalars.')
            else:
                H = np.array(H)
            if not all(H > 0):
                raise TypeError('Value error for H. All elements should be positive scalars.')
            if len(H) == 1:
                H = H[0]
        else:
            if H <= 0:
                raise TypeError('Value error for H. Should be positive scalar.')
        return H 


    def __get_constant(self):
        constant = self.user_inputs['tab_2_lr']['constant']
        if not isinstance(constant, bool):
            raise TypeError('Type error for constant. Should be boolean.') 
        return constant


    def __get_b_constant(self):
        b_constant = self.user_inputs['tab_2_lr']['b_constant']
        if not isinstance(b_constant, (str, float, int)):
            raise TypeError('Type error for b (constant). Should be float or integer.')
        if isinstance(b_constant, str):
            if not b_constant.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for b (constant). Should be float or integer.')
            else:
                b_constant = float(b_constant)
        return b_constant 


    def __get_V_constant(self):
        V_constant = self.user_inputs['tab_2_lr']['V_constant']
        if not isinstance(V_constant, (str, float, int)):
            raise TypeError('Type error for V (constant). Should be float or integer.')
        if isinstance(V_constant, str):
            if not V_constant.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for V (constant). Should be float or integer.')
            else:
                V_constant = float(V_constant)
        if V_constant <= 0:
            raise TypeError('Value error for V (constant). Should be strictly positive.')
        return V_constant 


    def __get_trend(self):
        trend = self.user_inputs['tab_2_lr']['trend']
        if not isinstance(trend, bool):
            raise TypeError('Type error for trend. Should be boolean.') 
        return trend


    def __get_b_trend(self):
        b_trend = self.user_inputs['tab_2_lr']['b_trend']
        if not isinstance(b_trend, (str, float, int)):
            raise TypeError('Type error for b (trend). Should be float or integer.')
        if isinstance(b_trend, str):
            if not b_trend.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for b (trend). Should be float or integer.')
            else:
                b_trend = float(b_trend)
        return b_trend 


    def __get_V_trend(self):
        V_trend = self.user_inputs['tab_2_lr']['V_trend']
        if not isinstance(V_trend, (str, float, int)):
            raise TypeError('Type error for V (trend). Should be float or integer.')
        if isinstance(V_trend, str):
            if not V_trend.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for V (trend). Should be float or integer.')
            else:
                V_trend = float(V_trend)
        if V_trend <= 0:
            raise TypeError('Value error for V (trend). Should be strictly positive.')
        return V_trend 
     

    def __get_quadratic_trend(self):
        quadratic_trend = self.user_inputs['tab_2_lr']['quadratic_trend']
        if not isinstance(quadratic_trend, bool):
            raise TypeError('Type error for quadratic trend. Should be boolean.') 
        return quadratic_trend


    def __get_b_quadratic_trend(self):
        b_quadratic_trend = self.user_inputs['tab_2_lr']['b_quadratic_trend']
        if not isinstance(b_quadratic_trend, (str, float, int)):
            raise TypeError('Type error for b (quadratic trend). Should be float or integer.')
        if isinstance(b_quadratic_trend, str):
            if not b_quadratic_trend.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for b (quadratic trend). Should be float or integer.')
            else:
                b_quadratic_trend = float(b_quadratic_trend)
        return b_quadratic_trend 


    def __get_V_quadratic_trend(self):
        V_quadratic_trend = self.user_inputs['tab_2_lr']['V_quadratic_trend']
        if not isinstance(V_quadratic_trend, (str, float, int)):
            raise TypeError('Type error for V (quadratic trend). Should be float or integer.')
        if isinstance(V_quadratic_trend, str):
            if not V_quadratic_trend.replace('.','',1).replace('-','',1).isdigit():
                raise TypeError('Type error for V (quadratic trend). Should be float or integer.')
            else:
                V_quadratic_trend = float(V_quadratic_trend)
        if V_quadratic_trend <= 0:
            raise TypeError('Value error for V (quadratic trend). Should be strictly positive.')
        return V_quadratic_trend    


    def __get_insample_fit(self):
        insample_fit = self.user_inputs['tab_2_lr']['insample_fit']
        if not isinstance(insample_fit, bool):
            raise TypeError('Type error for in-sample fit. Should be boolean.') 
        return insample_fit  


    def __get_marginal_likelihood(self):
        marginal_likelihood = self.user_inputs['tab_2_lr']['marginal_likelihood']
        if not isinstance(marginal_likelihood, bool):
            raise TypeError('Type error for marginal likelihood. Should be boolean.') 
        return marginal_likelihood  
    
    
    def __get_hyperparameter_optimization(self):
        hyperparameter_optimization = self.user_inputs['tab_2_lr']['hyperparameter_optimization']
        if not isinstance(hyperparameter_optimization, bool):
            raise TypeError('Type error for hyperparameter optimization. Should be boolean.') 
        return hyperparameter_optimization      
    

    def __get_optimization_type(self):
        optimization_type = self.user_inputs['tab_2_lr']['optimization_type']
        if optimization_type not in [1, 2]:
            raise TypeError('Value error for optimization type. Should be 1 or 2.')  
        return optimization_type 
    

    def __get_insample_data(self):
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
        endogenous = la.vec(endogenous)
        exogenous = iu.fetch_data(data, self.data_file, self.start_date, \
        self.end_date, self.exogenous_variables, 'Exogenous variables')            
        # infer date format, then recover sample dates
        date_format = iu.infer_date_format(self.frequency, self.data_file, \
                                        self.start_date, self.end_date)
        dates = iu.generate_dates(data, date_format, self.frequency, self.data_file, \
                               self.start_date, self.end_date)
        return endogenous, exogenous, dates


    def __get_heteroscedastic_data(self):
        # load data only if specified model is heteroscedastic regression
        if self.regression_type == 5:
            # check that data path and files are valid
            iu.check_file_path(self.project_path, self.data_file)            
            # then load data file
            data = iu.load_data(self.project_path, self.data_file)
            # check that Z variables are found in data
            iu.check_variables(data, self.data_file, self.Z_variables, 'Z variables')
            # check that the start and end dates can be found in the file        
            iu.check_dates(data, self.data_file, self.start_date, self.end_date)
            # recover Z variables
            Z = iu.fetch_data(data, self.data_file, self.start_date, \
            self.end_date, self.Z_variables, 'Z variables')
            # if dimensions are not consistent with g or Q, raise error
            h = Z.shape[1]
            if (isinstance(self.g, np.ndarray) and h != self.g.shape[0]) \
                or (isinstance(self.Q, np.ndarray) and h != self.Q.shape[0]):
                raise TypeError('Dimension error for heteroscedastic regressors Z. The dimensions of g, Q and Z are not consistent.') 
        # if model is not heteroscedastic regression, return empty list
        else:
            Z = []           
        return Z


    def __get_forecast_data(self):
        # if forecast is selected, load data
        if self.forecast:
            # check that data path and files are valid
            iu.check_file_path(self.project_path, self.forecast_file)
            # then load data file
            data = iu.load_data(self.project_path, self.forecast_file)
            # define the number of forecast periods
            periods = len(data.index)
            # check that exogenous variables are found in data
            iu.check_variables(data, self.forecast_file, self.exogenous_variables, 'Exogenous variables')
            # recover endogenous and exogenous data
            y_p = iu.fetch_forecast_data(data, [], self.endogenous_variables, \
            self.forecast_file, self.forecast_evaluation, periods, 'Endogenous variable')            
            if len(y_p) != 0:
                y_p = la.vec(y_p)
            X_p = iu.fetch_forecast_data(data, [], self.exogenous_variables, \
            self.forecast_file, True, periods, 'Exogenous variable')            
            # if model is heteroscedastic, Z variables must be obtained as well
            if self.regression_type == 5:
                iu.check_variables(data, self.forecast_file, self.Z_variables, 'Z variables')
                # recover Z variables
                Z_p = iu.fetch_forecast_data(data, [], self.Z_variables, \
                self.forecast_file, True, periods, 'Z variable')  
            else:
                Z_p = []
            # recover forecast dates
            end_date = self.dates[-1]
            forecast_dates = iu.generate_forecast_dates(end_date, periods, self.frequency)          
        # if forecasts is not selected, return empty data
        else:
            X_p, y_p, Z_p, forecast_dates = [], [], [], []
        return X_p, y_p, Z_p, forecast_dates


    
    
