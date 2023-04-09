# import
from alexandria.processor.regression_processor import RegressionProcessor
import alexandria.processor.input_utilities as iu


class InputProcessor(RegressionProcessor):


    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self, user_inputs):
        # save inputs as attributes
        self.user_inputs = user_inputs


    def process_input(self):
        # get inputs of tab 1 of graphical interface (common to all models)
        self.__tab_1_inputs()
        # get inputs from tab 2 of graphical interface (model-specific)
        self.__tab_2_inputs()
        # then get inputs of tab 3 of graphical interface (again common to all models)
        self.__tab_3_inputs()
        # finally, load all relevant data files
        self.__data_inputs()
        

    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------  

        
    def __tab_1_inputs(self):
        # recover model
        self.model = self.__get_model()
        # get list of endogenous variables
        self.endogenous_variables = self.__get_endogenous_variables()
        # get list of exogenous variables
        self.exogenous_variables = self.__get_exogenous_variables()
        # get data frequency
        self.frequency = self.__get_frequency()
        # get sampe periods
        self.start_date, self.end_date = self.__get_sample_dates()
        # get path to project folder
        self.project_path = self.__get_project_path()
        # get name of data file
        self.data_file = self.__get_data_file()
        # get progress bar decision
        self.progress_bar = self.__get_progress_bar()
        # get graphic creation decision
        self.create_graphics = self.__get_create_graphics()
        # get result saving decision
        self.save_results = self.__get_save_results()
        
        
    def __tab_2_inputs(self):
        # if model is model 1, get user inputs for linear regression
        if self.model == 1:
            self._regression_inputs()

        
    def __tab_3_inputs(self):
        # recover forecast decision
        self.forecast = self.__get_forecast()
        # recover forecast credibility interval
        self.forecast_credibility = self.__get_forecast_credibility()
        # recover conditional forecast decision
        self.conditional_forecast = self.__get_conditional_forecast()
        # recover conditional forecast credibility interval
        self.conditional_forecast_credibility = self.__get_conditional_forecast_credibility()        
        # recover irf decision
        self.irf = self.__get_irf()
        # recover irf credibility interval
        self.irf_credibility = self.__get_irf_credibility()
        # recover fevd decision
        self.fevd = self.__get_fevd()
        # recover fevd credibility interval
        self.fevd_credibility = self.__get_fevd_credibility()
        # recover hd decision
        self.hd = self.__get_hd()
        # recover hd credibility interval
        self.hd_credibility = self.__get_hd_credibility()
        # recover number of forecast periods
        self.forecast_periods = self.__get_forecast_periods()
        # recover type of conditional forecast
        self.conditional_forecast_type = self.__get_conditional_forecast_type()
        # recover name of forecast input file
        self.forecast_file = self.__get_forecast_file()
        # recover forecast evaluation decision
        self.forecast_evaluation = self.__get_forecast_evaluation()
        # recover number of irf periods
        self.irf_periods = self.__get_irf_periods()        
        # recover type of structural identification
        self.structural_identification = self.__get_structural_identification()
        # recover type of structural identification
        self.structural_identification_file = self.__get_structural_identification_file()


    def __data_inputs(self):
        # if model is model 1, get data for linear regression
        if self.model == 1:
            self._regression_data()
        
             
    def __get_model(self):
        model = self.user_inputs['tab_1']['model']
        if model not in [1]:
            raise TypeError('Value error for model. Should be equal to 1.') 
        return model
        
    
    def __get_endogenous_variables(self):
        endogenous_variables = self.user_inputs['tab_1']['endogenous_variables']
        if not endogenous_variables or not isinstance(endogenous_variables, (str, list)):
            raise TypeError('Type error for endogenous variables. Should be non-empty list of strings.')
        endogenous_variables = iu.string_to_list(endogenous_variables)
        if not all(isinstance(element, str) for element in endogenous_variables):
            raise TypeError('Type error for endogenous variables. Should be list of strings.')
        if self.model == 1 and len(endogenous_variables) != 1:
            raise TypeError('Dimension error for endogenous variable. Linear regression model shoud specify one and only one endogenous variable.')          
        return endogenous_variables
        
    
    def __get_exogenous_variables(self):
        exogenous_variables = self.user_inputs['tab_1']['exogenous_variables']
        if not isinstance(exogenous_variables, (str, list)):
            raise TypeError('Type error for exogenous variables. Should be list of strings.')
        exogenous_variables = iu.string_to_list(exogenous_variables)        
        if not all(isinstance(element, str) for element in exogenous_variables):
            raise TypeError('Type error for exogenous variables. Should be list of strings.')    
        return exogenous_variables


    def __get_frequency(self):
        frequency = self.user_inputs['tab_1']['frequency']
        if frequency not in [1, 2, 3, 4, 5, 6]:
            raise TypeError('Value error for frequency. Should be int between 1 and 6.')    
        return frequency
    
    
    def __get_sample_dates(self):
        sample_dates = self.user_inputs['tab_1']['sample']
        if not sample_dates or not isinstance(sample_dates, (str, list)):
            raise TypeError('Type error for sample dates. Should non-empty list of strings.')        
        sample_dates = iu.string_to_list(sample_dates) 
        if (not all(isinstance(element, str) for element in sample_dates)) or len(sample_dates) != 2:
            raise TypeError('Value error for sample dates. Should be pair of strings for sample start - sample end.')
        start_date, end_date = sample_dates[0], sample_dates[1]
        return start_date, end_date
    
    
    def __get_project_path(self):
        project_path = self.user_inputs['tab_1']['project_path']
        if not isinstance(project_path, str):
            raise TypeError('Type error for project folder path. Should be string.')
        project_path = iu.fix_string(project_path)
        if not project_path:
            raise TypeError('Value error for project folder path. Should be non-empty string.')
        return project_path


    def __get_data_file(self):
        data_file = self.user_inputs['tab_1']['data_file']
        if not isinstance(data_file, str):
            raise TypeError('Type error for data file. Should be string.')
        data_file = iu.fix_string(data_file)
        if not data_file:
            raise TypeError('Value error for data file. Should be non-empty string.')
        return data_file


    def __get_progress_bar(self):
        progress_bar = self.user_inputs['tab_1']['progress_bar']
        if not isinstance(progress_bar, bool):
            raise TypeError('Type error for progress bar. Should be boolean.')          
        return progress_bar        
    
    
    def __get_create_graphics(self):
        create_graphics = self.user_inputs['tab_1']['create_graphics']
        if not isinstance(create_graphics, bool):
            raise TypeError('Type error for create graphics. Should be boolean.') 
        return create_graphics   
    
    
    def __get_save_results(self):
        save_results = self.user_inputs['tab_1']['save_results']
        if not isinstance(save_results, bool):
            raise TypeError('Type error for save results. Should be boolean.') 
        return save_results   
    
    
    def __get_forecast(self):
        forecast = self.user_inputs['tab_3']['forecast']
        if not isinstance(forecast, bool):
            raise TypeError('Type error for forecasts. Should be boolean.') 
        return forecast     


    def __get_forecast_credibility(self):
        forecast_credibility = self.user_inputs['tab_3']['forecast_credibility']
        if not isinstance(forecast_credibility, (str, float)):
            raise TypeError('Type error for forecasts credibility level. Should be float between 0 and 1.')
        if isinstance(forecast_credibility, str):
            if not forecast_credibility.replace('.','',1).isdigit():
                raise TypeError('Type error for forecasts credibility level. Should be float between 0 and 1.')
            else:
                forecast_credibility = float(forecast_credibility)
        if forecast_credibility <= 0 or forecast_credibility >= 1:
            raise TypeError('Value error for forecasts credibility level. Should be float between 0 and 1 (not included).')
        return forecast_credibility
        
    
    def __get_conditional_forecast(self):
        conditional_forecast = self.user_inputs['tab_3']['conditional_forecast']
        if not isinstance(conditional_forecast, bool):
            raise TypeError('Type error for conditional forecast. Should be boolean.') 
        return conditional_forecast     


    def __get_conditional_forecast_credibility(self):
        conditional_forecast_credibility = self.user_inputs['tab_3']['conditional_forecast_credibility']
        if not isinstance(conditional_forecast_credibility, (str, float)):
            raise TypeError('Type error for conditional forecasts credibility level. Should be float between 0 and 1.')
        if isinstance(conditional_forecast_credibility, str):
            if not conditional_forecast_credibility.replace('.','',1).isdigit():
                raise TypeError('Type error for conditional forecasts credibility level. Should be float between 0 and 1.')
            else:
                conditional_forecast_credibility = float(conditional_forecast_credibility)
        if conditional_forecast_credibility <= 0 or conditional_forecast_credibility >= 1:
            raise TypeError('Value error for conditional forecasts credibility level. Should be float between 0 and 1 (not included).')
        return conditional_forecast_credibility


    def __get_irf(self):
        irf = self.user_inputs['tab_3']['irf']
        if not isinstance(irf, bool):
            raise TypeError('Type error for impulse response function. Should be boolean.') 
        return irf  
    
    
    def __get_irf_credibility(self):
        irf_credibility = self.user_inputs['tab_3']['irf_credibility']
        if not isinstance(irf_credibility, (str, float)):
            raise TypeError('Type error for irf credibility level. Should be float between 0 and 1.')
        if isinstance(irf_credibility, str):
            if not irf_credibility.replace('.','',1).isdigit():
                raise TypeError('Type error for irf credibility level. Should be float between 0 and 1.')
            else:
                irf_credibility = float(irf_credibility)
        if irf_credibility <= 0 or irf_credibility >= 1:
            raise TypeError('Value error for irf credibility level. Should be float between 0 and 1 (not included).')
        return irf_credibility   


    def __get_fevd(self):
        fevd = self.user_inputs['tab_3']['fevd']
        if not isinstance(fevd, bool):
            raise TypeError('Type error for forecast error variance decomposition. Should be boolean.') 
        return fevd


    def __get_fevd_credibility(self):
        fevd_credibility = self.user_inputs['tab_3']['fevd_credibility']
        if not isinstance(fevd_credibility, (str, float)):
            raise TypeError('Type error for fevd credibility level. Should be float between 0 and 1.')
        if isinstance(fevd_credibility, str):
            if not fevd_credibility.replace('.','',1).isdigit():
                raise TypeError('Type error for fevd credibility level. Should be float between 0 and 1.')
            else:
                fevd_credibility = float(fevd_credibility)
        if fevd_credibility <= 0 or fevd_credibility >= 1:
            raise TypeError('Value error for fevd credibility level. Should be float between 0 and 1 (not included).')
        return fevd_credibility  
    
    
    def __get_hd(self):
        hd = self.user_inputs['tab_3']['hd']
        if not isinstance(hd, bool):
            raise TypeError('Type error for historical decomposition. Should be boolean.') 
        return hd


    def __get_hd_credibility(self):
        hd_credibility = self.user_inputs['tab_3']['hd_credibility']
        if not isinstance(hd_credibility, (str, float)):
            raise TypeError('Type error for historical decomposition credibility level. Should be float between 0 and 1.')
        if isinstance(hd_credibility, str):
            if not hd_credibility.replace('.','',1).isdigit():
                raise TypeError('Type error for historical decomposition credibility level. Should be float between 0 and 1.')
            else:
                hd_credibility = float(hd_credibility)
        if hd_credibility <= 0 or hd_credibility >= 1:
            raise TypeError('Value error for historical decomposition credibility level. Should be float between 0 and 1 (not included).')
        return hd_credibility  


    def __get_forecast_periods(self):
        forecast_periods = self.user_inputs['tab_3']['forecast_periods']       
        if not isinstance(forecast_periods, (int, str)):
            raise TypeError('Type error for forecast periods. Should be integer.')
        if forecast_periods and isinstance(forecast_periods, str):
            if forecast_periods.isdigit():
                forecast_periods = int(forecast_periods)
            else:
                raise TypeError('Type error for forecast periods. Should be positive integer.')
        if isinstance(forecast_periods, int) and forecast_periods <= 0:
            raise TypeError('Value error for forecast periods. Should be positive integer.')
        return forecast_periods
        
    
    def __get_conditional_forecast_type(self):
        conditional_forecast_type = self.user_inputs['tab_3']['conditional_forecast_type']
        if conditional_forecast_type not in [1, 2]:
            raise TypeError('Value error for conditional forecast type. Should be 1 or 2.')    
        return conditional_forecast_type    
    
    
    def __get_forecast_file(self):
        forecast_file = self.user_inputs['tab_3']['forecast_file']
        if not isinstance(forecast_file, str):
            raise TypeError('Type error for forecast file. Should be string.')
        forecast_file = iu.fix_string(forecast_file)
        return forecast_file    
    
    
    def __get_forecast_evaluation(self):
        forecast_evaluation = self.user_inputs['tab_3']['forecast_evaluation']
        if not isinstance(forecast_evaluation, bool):
            raise TypeError('Type error for forecast evaluation. Should be boolean.') 
        return forecast_evaluation  
    

    def __get_irf_periods(self):
        irf_periods = str(self.user_inputs['tab_3']['irf_periods'])        
        if not isinstance(irf_periods, (int, str)):
            raise TypeError('Type error for irf periods. Should be integer.')
        if irf_periods and isinstance(irf_periods, str):
            if irf_periods.isdigit():
                irf_periods = int(irf_periods)
            else:
                raise TypeError('Type error for irf periods. Should be positive integer.')
        if isinstance(irf_periods, int) and irf_periods <= 0:
            raise TypeError('Value error for irf periods. Should be positive integer.')
        return irf_periods    
    
    
    def __get_structural_identification(self):
        structural_identification = self.user_inputs['tab_3']['structural_identification']
        if structural_identification not in [1, 2]:
            raise TypeError('Value error for structural identification. Should be 1 or 2.')  
        return structural_identification    
    
    
    def __get_structural_identification_file(self):
        structural_identification_file = self.user_inputs['tab_3']['structural_identification_file']
        if not isinstance(structural_identification_file, str):
            raise TypeError('Type error for structural identification file. Should be string.')
        structural_identification_file = iu.fix_string(structural_identification_file)
        return structural_identification_file
    

