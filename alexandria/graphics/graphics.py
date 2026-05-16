# imports
from os import getcwd, mkdir
from os.path import isdir
from shutil import rmtree
import alexandria.processor.input_utilities as iu
from alexandria.graphics.regression_graphics import RegressionGraphics
from alexandria.graphics.vector_autoregression_graphics import VectorAutoregressionGraphics
from alexandria.graphics.nowcasting_graphics import NowcastingGraphics


class Graphics(RegressionGraphics, VectorAutoregressionGraphics, NowcastingGraphics):
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self, model, complementary_information = {}, path = [], clear_folder = False):
        # save attributes
        self.model = model
        self.complementary_information = complementary_information
        self.path = path
        self.clear_folder = clear_folder
        # initialize save folder
        self.__initialize_folder()
        # complement information with possible missing elements
        self.__complete_information()
        
        
    def insample_fit_graphics(self, show, save):
        model_class = self.complementary_information['model_class']
        # if model is linear regression, make regression insample graphics
        if model_class == 1:
            self._regression_fitted(show, save)
            self._regression_residuals(show, save)
        # if model is vector autoregression, make VAR insample graphics
        elif model_class == 2 or model_class == 3:
            self._var_fitted(show, save)
            self._var_residuals(show, save)
            self._var_shocks(show, save)
            self._var_steady_state(show, save)
        # if model is nowcasting, make nowcasting insample graphics
        elif model_class == 4:
            self._nowcasting_fitted(show, save)
            self._nowcasting_residuals(show, save)
            self._nowcasting_shocks(show, save)
            self._nowcasting_steady_state(show, save)
            self._nowcasting_factors(show, save)
            
        
    def forecast_graphics(self, show, save):
        model_class = self.complementary_information['model_class']
        # if model is linear regression, make regression forecast graphics
        if model_class == 1:
            self._regression_forecasts(show, save)
        # if model is vector autoregression, make VAR forecast graphics
        elif model_class == 2 or model_class == 3:
            self._var_forecasts(show, save)
        # if model is nowcasting, make nowcasting forecast graphics
        elif model_class == 4:
            self._nowcasting_forecasts(show, save)


    def conditional_forecast_graphics(self, show, save):
        model_class = self.complementary_information['model_class']
        # if model is vector autoregression, make VAR conditional forecast graphics
        if model_class == 2 or model_class == 3:
            self._var_conditional_forecasts(show, save)
        # if model is nowcasting, make nowcasting conditional forecast graphics
        elif model_class == 4:
            self._nowcasting_conditional_forecasts(show, save)
            

    def irf_graphics(self, show, save):
        model_class = self.complementary_information['model_class']
        # if model is vector autoregression, make VAR IRF graphics
        if model_class == 2 or model_class == 3:
            self._var_irf(show, save)        
        # if model is nowcasting, make nowcasting IRF graphics
        elif model_class == 4:
            self._nowcasting_irf(show, save)
            

    def fevd_graphics(self, show, save):
        model_class = self.complementary_information['model_class']
        # if model is vector autoregression, make VAR FEVD graphics
        if model_class == 2 or model_class == 3:
            self._var_fevd(show, save) 
        # if model is nowcasting, make nowcasting IRF graphics
        elif model_class == 4:
            self._nowcasting_fevd(show, save)
            

    def hd_graphics(self, show, save):
        model_class = self.complementary_information['model_class']
        # if model is vector autoregression, make VAR HD graphics
        if model_class == 2 or model_class == 3:
            self._var_hd(show, save) 
        # if model is nowcasting, make nowcasting HD graphics
        elif model_class == 4:
            self._nowcasting_hd(show, save)
            

    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------
    
    
    def  __initialize_folder(self):   
        # check if path to save folder is defined, otherwise default is current directory
        if not self.path:
            self.path = getcwd()
        # clear save folder and re-initialize if activated
        if self.clear_folder and isdir(self.path):
            rmtree(self.path, ignore_errors = True)
        # create path if it does not exist
        if not isdir(self.path):
            mkdir(self.path)
        
            
    def __complete_information(self): 
        # add general model information
        self.__complete_model_information()        
        # if model is linear regression, add regression elements
        if self.complementary_information['model_class'] == 1:
            self._complete_regression_information()        
        # if model is vector autoregression, add var elements
        elif self.complementary_information['model_class'] == 2:
            self._complete_var_information() 
        # if model is VEC/VARMA, add var elements (vec and varma just recycle VAR functions)
        elif self.complementary_information['model_class'] == 3:
            self._complete_var_information() 
        # if model is nowcasting, add nowcasting elements
        elif self.complementary_information['model_class'] == 4:
            self._complete_nowcasting_information() 
            

    def __complete_model_information(self):
        # recover and add common model elements
        model_name, model_class, model_type = iu.identify_model(self.model)
        self.complementary_information['model_name'] = model_name
        self.complementary_information['model_class'] = model_class
        self.complementary_information['model_type'] = model_type        
        
        
