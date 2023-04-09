# imports
import alexandria.graphics.graphics_utilities as gu
from alexandria.graphics.graphics import Graphics


class RegressionGraphics(Graphics):
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self, ip, lr):

        # determine which applications will produce graphics
        self.__graphics_information(ip)
        # gather information from input processor 
        self.__input_information(ip)
        # then gather information from regression model
        self.__regression_information(lr)
        
        
    def make_graphics(self):
        # delete existing graphics folder, if any
        self._delete_graphics_folder()
        # create graphics for in-sample fit, if selected
        self.__insample_fit_graphics()
        # create graphics for forecasts, if selected
        self.__forecasts_graphics()


    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------


    def __graphics_information(self, ip):
        # input processor information: in-sample fit
        self.insample_fit = ip.insample_fit
        # input processor information: forecast decision
        self.forecast = ip.forecast        
        

    def __input_information(self, ip):     
        # input processor information: project folder
        self.project_path = ip.project_path
        # input processor information: endogenous
        self.endogenous = ip.endogenous_variables
        # data specific to in-sample fit
        if self.insample_fit:
            # input processor information: in-sample dates
            self.insample_dates = ip.dates    
        # data specific to forecasts
        if self.forecast:
            # input processor information: endogenous, actual values for predictions
            self.y_p = ip.y_p
            # input processor information: forecast dates
            self.forecast_dates = ip.forecast_dates


    def __regression_information(self, lr):
        # data specific to in-sample fit
        if self.insample_fit:
            # regression information: endogenous values
            self.actual = lr.y
            # regression information: fitted
            self.fitted = lr.estimates_fit
            # regression information: residual estimates
            self.residuals = lr.estimates_residuals 
        # data specific to forecasts
        if self.forecast:
            # regression information: forecast estimates
            self.estimates_forecasts = lr.estimates_forecasts   
        
            
    def __insample_fit_graphics(self):
        if self.insample_fit:
            actual = self.actual
            fitted = self.fitted
            residuals = self.residuals
            dates = self.insample_dates
            name = self.endogenous[0]
            path = self.project_path
            gu.fit_single_variable(actual, fitted, dates, name, path)
            gu.residual_single_variable(residuals, dates, name, path)

                
    def __forecasts_graphics(self):
        if self.forecast:
            forecasts = self.estimates_forecasts
            y_p = self.y_p
            dates = self.forecast_dates
            name = self.endogenous[0]
            path = self.project_path
            gu.ols_forecasts_single_variable(forecasts, y_p, dates, name, path)











        