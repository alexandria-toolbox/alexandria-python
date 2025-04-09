# imports
from os import getcwd



class DefaultInputInterface(object):

    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass    

    
    def create_default_inputs(self):
    
        # overall dictionary
        user_inputs = {}
        
        # sub dictionaries for each of the three tabs taking user inputs
        user_inputs['tab_1'] = {}
        user_inputs['tab_2_lr'] = {}
        user_inputs['tab_2_var'] = {}
        user_inputs['tab_3'] = {}
        
        # default values for tab 1
        user_inputs['tab_1']['model'] = 1
        user_inputs['tab_1']['endogenous_variables'] = ''
        user_inputs['tab_1']['exogenous_variables'] = ''
        user_inputs['tab_1']['frequency'] = 1
        user_inputs['tab_1']['sample'] = ''
        user_inputs['tab_1']['project_path'] = getcwd()
        user_inputs['tab_1']['data_file'] = ''
        user_inputs['tab_1']['progress_bar'] = True
        user_inputs['tab_1']['create_graphics'] = True
        user_inputs['tab_1']['save_results'] = True
        
        # default values for tab 2, linear regression
        user_inputs['tab_2_lr']['regression_type'] = 1
        user_inputs['tab_2_lr']['iterations'] = '2000'
        user_inputs['tab_2_lr']['burnin'] = '1000'
        user_inputs['tab_2_lr']['model_credibility'] = '0.95'
        user_inputs['tab_2_lr']['b'] = '0'
        user_inputs['tab_2_lr']['V'] = '1'
        user_inputs['tab_2_lr']['alpha'] = '0.0001'
        user_inputs['tab_2_lr']['delta'] = '0.0001'
        user_inputs['tab_2_lr']['g'] = '0'
        user_inputs['tab_2_lr']['Q'] = '100'
        user_inputs['tab_2_lr']['tau'] = '0.001'
        user_inputs['tab_2_lr']['thinning'] = False
        user_inputs['tab_2_lr']['thinning_frequency'] = '10'
        user_inputs['tab_2_lr']['Z_variables'] = ''
        user_inputs['tab_2_lr']['q'] = '1'
        user_inputs['tab_2_lr']['p'] = '0'
        user_inputs['tab_2_lr']['H'] = '100'
        user_inputs['tab_2_lr']['constant'] = True
        user_inputs['tab_2_lr']['b_constant'] = '0'
        user_inputs['tab_2_lr']['V_constant'] = '1'
        user_inputs['tab_2_lr']['trend'] = False
        user_inputs['tab_2_lr']['b_trend'] = '0'
        user_inputs['tab_2_lr']['V_trend'] = '1'
        user_inputs['tab_2_lr']['quadratic_trend'] = False
        user_inputs['tab_2_lr']['b_quadratic_trend'] = '0'
        user_inputs['tab_2_lr']['V_quadratic_trend'] = '1'
        user_inputs['tab_2_lr']['insample_fit'] = False
        user_inputs['tab_2_lr']['marginal_likelihood'] = False
        user_inputs['tab_2_lr']['hyperparameter_optimization'] = False
        user_inputs['tab_2_lr']['optimization_type'] = 1

        # default values for tab 2, vector autoregression
        user_inputs['tab_2_var']['var_type'] = 1
        user_inputs['tab_2_var']['iterations'] = '3000'
        user_inputs['tab_2_var']['burnin'] = '1000'
        user_inputs['tab_2_var']['model_credibility'] = '0.95'
        user_inputs['tab_2_var']['constant'] = True
        user_inputs['tab_2_var']['trend'] = False
        user_inputs['tab_2_var']['quadratic_trend'] = False
        user_inputs['tab_2_var']['lags'] = '4'
        user_inputs['tab_2_var']['ar_coefficients'] = '0.9'
        user_inputs['tab_2_var']['pi1'] = '0.1'
        user_inputs['tab_2_var']['pi2'] = '0.5'
        user_inputs['tab_2_var']['pi3'] = '1'
        user_inputs['tab_2_var']['pi4'] = '100'
        user_inputs['tab_2_var']['pi5'] = '1'
        user_inputs['tab_2_var']['pi6'] = '0.1'
        user_inputs['tab_2_var']['pi7'] = '0.1'
        user_inputs['tab_2_var']['proxy_variables'] = ''
        user_inputs['tab_2_var']['lamda'] = '0.2'
        user_inputs['tab_2_var']['proxy_prior'] = 1
        user_inputs['tab_2_var']['insample_fit'] = False
        user_inputs['tab_2_var']['constrained_coefficients'] = False
        user_inputs['tab_2_var']['sums_of_coefficients'] = False
        user_inputs['tab_2_var']['initial_observation'] = False
        user_inputs['tab_2_var']['long_run'] = False
        user_inputs['tab_2_var']['stationary'] = False
        user_inputs['tab_2_var']['marginal_likelihood'] = False
        user_inputs['tab_2_var']['hyperparameter_optimization'] = False
        user_inputs['tab_2_var']['coefficients_file'] = ''
        user_inputs['tab_2_var']['long_run_file'] = ''
        
        # default values for tab 3
        user_inputs['tab_3']['forecast'] = False
        user_inputs['tab_3']['conditional_forecast'] = False
        user_inputs['tab_3']['irf'] = False
        user_inputs['tab_3']['fevd'] = False
        user_inputs['tab_3']['hd'] = False
        user_inputs['tab_3']['forecast_credibility'] = '0.95'
        user_inputs['tab_3']['conditional_forecast_credibility'] = '0.95'
        user_inputs['tab_3']['irf_credibility'] = '0.95'
        user_inputs['tab_3']['fevd_credibility'] = '0.95'
        user_inputs['tab_3']['hd_credibility'] = '0.95'
        user_inputs['tab_3']['forecast_periods'] = ''
        user_inputs['tab_3']['conditional_forecast_type'] = 1
        user_inputs['tab_3']['forecast_file'] = ''
        user_inputs['tab_3']['conditional_forecast_file'] = ''
        user_inputs['tab_3']['forecast_evaluation'] = False
        user_inputs['tab_3']['irf_periods'] = ''
        user_inputs['tab_3']['structural_identification'] = 1
        user_inputs['tab_3']['structural_identification_file'] = ''
        
        # save as attribute
        self.user_inputs = user_inputs 
    
        
    def reset_default_inputs(self):    
        
        # tab 1
        self.t1_mnu1.setCurrentIndex(self.user_inputs['tab_1']['model'] - 1)
        self.t1_edt1.setText(self.user_inputs['tab_1']['endogenous_variables']) 
        self.t1_edt2.setText(self.user_inputs['tab_1']['exogenous_variables']) 
        self.t1_mnu2.setCurrentIndex(self.user_inputs['tab_1']['frequency'] - 1)
        self.t1_edt3.setText(self.user_inputs['tab_1']['sample']) 
        self.t1_edt4.setText(self.user_inputs['tab_1']['project_path']) 
        self.t1_edt5.setText(self.user_inputs['tab_1']['data_file'])
        self.t1_rdb1.setChecked(True) 
        self.t1_rdb3.setChecked(True)  
        self.t1_rdb5.setChecked(True) 
        
        # tab 2, linear regression
        self.t2_lr_rdb1.setChecked(True)
        self.t2_lr_edt1.setText(self.user_inputs['tab_2_lr']['iterations'])
        self.t2_lr_edt2.setText(self.user_inputs['tab_2_lr']['burnin'])
        self.t2_lr_edt3.setText(self.user_inputs['tab_2_lr']['model_credibility'])
        self.t2_lr_edt4.setText(self.user_inputs['tab_2_lr']['b'])
        self.t2_lr_edt5.setText(self.user_inputs['tab_2_lr']['V'])
        self.t2_lr_edt6.setText(self.user_inputs['tab_2_lr']['alpha'])
        self.t2_lr_edt7.setText(self.user_inputs['tab_2_lr']['delta'])
        self.t2_lr_edt8.setText(self.user_inputs['tab_2_lr']['g'])
        self.t2_lr_edt9.setText(self.user_inputs['tab_2_lr']['Q'])
        self.t2_lr_edt10.setText(self.user_inputs['tab_2_lr']['tau'])
        self.t2_lr_cbx1.setChecked(False)
        self.t2_lr_edt11.setText(self.user_inputs['tab_2_lr']['thinning_frequency'])
        self.t2_lr_edt12.setText(self.user_inputs['tab_2_lr']['Z_variables'])
        self.t2_lr_edt13.setText(self.user_inputs['tab_2_lr']['q'])
        self.t2_lr_edt14.setText(self.user_inputs['tab_2_lr']['p'])
        self.t2_lr_edt15.setText(self.user_inputs['tab_2_lr']['H'])
        self.t2_lr_cbx2.setChecked(True)
        self.t2_lr_edt16.setText(self.user_inputs['tab_2_lr']['b_constant'])
        self.t2_lr_edt17.setText(self.user_inputs['tab_2_lr']['V_constant'])
        self.t2_lr_cbx3.setChecked(False)
        self.t2_lr_edt18.setText(self.user_inputs['tab_2_lr']['b_trend'])
        self.t2_lr_edt19.setText(self.user_inputs['tab_2_lr']['V_trend'])
        self.t2_lr_cbx4.setChecked(False)
        self.t2_lr_edt20.setText(self.user_inputs['tab_2_lr']['b_quadratic_trend'])
        self.t2_lr_edt21.setText(self.user_inputs['tab_2_lr']['V_quadratic_trend'])
        self.t2_lr_cbx5.setChecked(False)
        self.t2_lr_cbx6.setChecked(False)
        self.t2_lr_cbx7.setChecked(False)
        self.t2_lr_rdb7.setChecked(True)
        
        # tab 3
        self.t3_rdb2.setChecked(True)
        self.t3_edt1.setText(self.user_inputs['tab_3']['forecast_credibility'])
        self.t3_rdb4.setChecked(True)
        self.t3_edt2.setText(self.user_inputs['tab_3']['conditional_forecast_credibility'])
        self.t3_rdb6.setChecked(True)
        self.t3_edt3.setText(self.user_inputs['tab_3']['irf_credibility'])
        self.t3_rdb8.setChecked(True)
        self.t3_edt4.setText(self.user_inputs['tab_3']['fevd_credibility'])
        self.t3_rdb10.setChecked(True)
        self.t3_edt5.setText(self.user_inputs['tab_3']['hd_credibility'])
        self.t3_edt6.setText(self.user_inputs['tab_3']['forecast_periods'])
        self.t3_mnu1.setCurrentIndex(self.user_inputs['tab_3']['conditional_forecast_type'] - 1)
        self.t3_edt7.setText(self.user_inputs['tab_3']['forecast_file'])
        self.t3_edt8.setText(self.user_inputs['tab_3']['conditional_forecast_file'])
        self.t3_cbx1.setChecked(False)
        self.t3_edt9.setText(self.user_inputs['tab_3']['irf_periods'])
        self.t3_mnu2.setCurrentIndex(self.user_inputs['tab_3']['structural_identification'] - 1)
        self.t3_edt10.setText(self.user_inputs['tab_3']['structural_identification_file'])
        
            
    
    