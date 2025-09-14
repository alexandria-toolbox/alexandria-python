# imports
from datetime import datetime
from alexandria.results.regression_results import RegressionResults
from alexandria.results.vector_autoregression_results import VectorAutoregressionResults
from alexandria.results.vec_varma_results import VecVarmaResults
import alexandria.console.console_utilities as cu
import alexandria.processor.input_utilities as iu
from os.path import join


class Results(RegressionResults, VectorAutoregressionResults, VecVarmaResults):
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self, model, complementary_information = {}):
        # save attributes
        self.model = model
        self.complementary_information = complementary_information
        # complement information with possible missing elements
        self.__complete_information()
        
        
    def make_input_summary(self):
        # initiate string list
        self.input_summary = []
        # add settings header
        self.__add_input_header()
        # add tab 1 settings
        self.__add_tab_1_inputs()       
        # add tab 2 settings
        self.__add_tab_2_inputs()
        # add tab 3 settings
        self.__add_tab_3_inputs()
        
        
    def show_input_summary(self):
        # display input summary in console
        cu.print_string_list(self.input_summary)
        
        
    def save_input_summary(self, path):
        # check if path exists, and create directory if needed
        cu.check_path(path)
        # generate full path
        full_path = join(path, 'input_summary.txt')
        # write txt file on disk
        input_summary = cu.alexandria_header() + self.input_summary
        cu.write_string_list(input_summary, full_path)
        
        
    def make_estimation_summary(self):
        model_class = self.complementary_information['model_class']
        # if model is linear regression, make regression summary
        if model_class == 1:
            self._make_regression_summary()         
        # if model is vector autoregression, make VAR summary
        elif model_class == 2:
            self._make_var_summary() 
        # if model is VEC/VARMA, make VAR extension summary
        elif model_class == 3:
            self._make_vec_varma_summary() 
            
        
    def show_estimation_summary(self):
        # display estimation summary in console
        cu.print_string_list(self.estimation_summary)     
        
        
    def save_estimation_summary(self, path):
        # check if path exists, and create directory if needed
        cu.check_path(path)
        # generate full path
        full_path = join(path, 'estimation_summary.txt')
        # write txt file on disk
        estimation_summary = cu.alexandria_header() + self.estimation_summary
        cu.write_string_list(estimation_summary, full_path)        
        

    def make_application_summary(self):
        # initiate application_summary
        self.application_summary = {}
        # identify model to run relevant application summary
        model_class = self.complementary_information['model_class']
        # if model is linear regression, make regression summary
        if model_class == 1:
            self._make_regression_application_summary()        
        # if model is vector autoregression, make VAR summary
        elif model_class == 2:
            self._make_var_application_summary()  
        # if model is VEC/VARMA, make VAR extension summary
        elif model_class == 3:
            self._make_vec_varma_application_summary() 
            
        
    def save_application_summary(self, path):
        # check if path exists, and create directory if needed
        cu.check_path(path) 
        # identify model to run relevant application summary
        model_class = self.complementary_information['model_class']
        # if model is linear regression, save regression summary
        if model_class == 1:
            self._save_regression_application(path)           
        # if model is vector autoregression, save regression summary
        elif model_class == 2:
            self._save_var_application(path)          
        # if model is VEC/VARMA, save VAR extension summary
        elif model_class == 3:
            self._save_vec_varma_application(path)  
            
    
    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------


    def __complete_information(self):
        # add general model information
        self.__complete_model_information()
        # if model is linear regression, add regression elements
        if self.complementary_information['model_class'] == 1:
            self._complete_regression_information()
        # if model is vector autoregression, add VAR elements
        elif self.complementary_information['model_class'] == 2:
            self._complete_var_information()
        # if model is VEC/VARMA, add extension elements
        elif self.complementary_information['model_class'] == 3:
            self._complete_vec_varma_information()            
        # add application information
        self.__complete_application_information()


    def __complete_model_information(self):
        # recover and add common model elements
        model_name, model_class, model_type = iu.identify_model(self.model)
        self.complementary_information['model_name'] = model_name
        self.complementary_information['model_class'] = model_class
        self.complementary_information['model_type'] = model_type
        if 'estimation_start' not in self.complementary_information:
            self.complementary_information['estimation_start'] = '—'
        if 'estimation_end' not in self.complementary_information:
            self.complementary_information['estimation_end'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if 'sample_start' not in self.complementary_information:
            self.complementary_information['sample_start'] = ''
        if 'sample_end' not in self.complementary_information:
            self.complementary_information['sample_end'] = ''  
        if 'frequency' not in self.complementary_information:
            self.complementary_information['frequency'] = '—'
        if 'project_path' not in self.complementary_information:
            self.complementary_information['project_path'] = '—'
        if 'data_file' not in self.complementary_information:
            self.complementary_information['data_file'] = '—'     
        if 'progress_bar' not in self.complementary_information:
            self.complementary_information['progress_bar'] = '—'  
        if 'create_graphics' not in self.complementary_information:
            self.complementary_information['create_graphics'] = '—'  
        if 'save_results' not in self.complementary_information:
            self.complementary_information['save_results'] = '—'  


    def __complete_application_information(self):
        if 'forecast' not in self.complementary_information:
            self.complementary_information['forecast'] = '—'
        if 'forecast_credibility' not in self.complementary_information:
            self.complementary_information['forecast_credibility'] = '—'
        if 'conditional_forecast' not in self.complementary_information:
            self.complementary_information['conditional_forecast'] = '—'
        if 'conditional_forecast_credibility' not in self.complementary_information:
            self.complementary_information['conditional_forecast_credibility'] = '—'
        if 'irf' not in self.complementary_information:
            self.complementary_information['irf'] = '—'
        if 'irf_credibility' not in self.complementary_information:
            self.complementary_information['irf_credibility'] = '—'
        if 'fevd' not in self.complementary_information:
            self.complementary_information['fevd'] = '—'
        if 'fevd_credibility' not in self.complementary_information:
            self.complementary_information['fevd_credibility'] = '—'
        if 'hd' not in self.complementary_information:
            self.complementary_information['hd'] = '—'            
        if 'hd_credibility' not in self.complementary_information:
            self.complementary_information['hd_credibility'] = '—'
        if 'forecast_periods' not in self.complementary_information:
            self.complementary_information['forecast_periods'] = '—'
        if 'conditional_forecast_type' not in self.complementary_information:
            self.complementary_information['conditional_forecast_type'] = '—'            
        if 'forecast_file' not in self.complementary_information:
            self.complementary_information['forecast_file'] = '—'            
        if 'conditional_forecast_file' not in self.complementary_information:
            self.complementary_information['conditional_forecast_file'] = '—'            
        if 'forecast_evaluation' not in self.complementary_information:
            self.complementary_information['forecast_evaluation'] = '—'        
        if 'irf_periods' not in self.complementary_information:
            self.complementary_information['irf_periods'] = '—'        
        if 'structural_identification' not in self.complementary_information:
            self.complementary_information['structural_identification'] = '—'
        if 'structural_identification_file' not in self.complementary_information:
            self.complementary_information['structural_identification_file'] = '—'            


    def __add_input_header(self):
        # Alexandria header and estimation date
        lines = []
        lines.append('Estimation date:  ' + self.complementary_information['estimation_end'])
        lines.append(' ')
        lines.append(' ')
        self.input_summary += lines 
        

    def __add_tab_1_inputs(self):
        # initiate lines
        lines = []
        # header for tab 1
        lines.append('Model')
        lines.append('---------')
        lines.append(' ')
        # model class
        model_class = self.complementary_information['model_class']
        if model_class == 1:
            model = 'linear regression'
        elif model_class == 2:
            model = 'vector autoregression'    
        elif model_class == 3:
            model = 'vec / varma'    
        lines.append('selected model: ' + model)
        # endogenous variables
        endogenous_variables = iu.list_to_string(self.complementary_information['endogenous_variables'])
        lines.append('endogenous variables: ' + endogenous_variables) 
        # exogenous variables
        exogenous_variables = iu.list_to_string(self.complementary_information['exogenous_variables'])
        lines.append('exogenous variables: ' + exogenous_variables)
        # data frequency
        frequency = self.complementary_information['frequency']
        lines.append('data frequency: ' + frequency)
        # sample dates
        sample_start = self.complementary_information['sample_start']
        sample_end = self.complementary_information['sample_end']
        sample_dates = sample_start + ' ' + sample_end
        lines.append('estimation sample: ' + sample_dates)
        # project path and data file
        project_path = self.complementary_information['project_path']
        data_file = self.complementary_information['data_file']
        lines.append('path to project folder: ' + project_path)
        lines.append('data file: ' + data_file)
        # progress bar, graphics and result saves
        if type(self.complementary_information['progress_bar']) == bool:
            progress_bar = cu.bool_to_string(self.complementary_information['progress_bar'])
        else:
            progress_bar = self.complementary_information['progress_bar']
        if type(self.complementary_information['create_graphics']) == bool:
            create_graphics = cu.bool_to_string(self.complementary_information['create_graphics'])
        else:
            create_graphics = self.complementary_information['create_graphics']            
        if type(self.complementary_information['save_results']) == bool:
            save_results = cu.bool_to_string(self.complementary_information['save_results'])
        else:
            save_results = self.complementary_information['save_results'] 
        lines.append('progress bar: ' + progress_bar)
        lines.append('create graphics: ' + create_graphics)
        lines.append('save_results: ' + save_results)  
        lines.append(' ')
        lines.append(' ')        
        self.input_summary += lines 


    def __add_tab_2_inputs(self):
        # tab 2 elements for linear regression
        if self.complementary_information['model_class'] == 1:
            self._add_regression_tab_2_inputs()
        elif self.complementary_information['model_class'] == 2:
            self._add_var_tab_2_inputs()
        elif self.complementary_information['model_class'] == 3:
            self._add_vec_varma_tab_2_inputs()
            

    def __add_tab_3_inputs(self):
        # tab 3 elements for linear regression
        if self.complementary_information['model_class'] == 1:
            self._add_regression_tab_3_inputs()
        elif self.complementary_information['model_class'] == 2:
            self._add_var_tab_3_inputs()
        elif self.complementary_information['model_class'] == 3:
            self._add_vec_varma_tab_3_inputs()            

