# imports
import alexandria.graphics.graphics_utilities as gu
import numpy as np



class RegressionGraphics(object):
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass


    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------
        
    
    def _complete_regression_information(self):  
        # endogenous and exogenous variables
        if 'endogenous_variables' not in self.complementary_information:
            self.complementary_information['endogenous_variables'] = ['y']
        if 'exogenous_variables' not in self.complementary_information:
            n_exo = self.model.exogenous.shape[1]
            self.complementary_information['exogenous_variables'] = ['x' + str(i+1) for i in range(n_exo)]
        # sample dates
        if 'dates' not in self.complementary_information:
            n = self.model.n
            self.complementary_information['dates'] = np.arange(1,n+1)    
        # forecast_dates
        if hasattr(self.model, 'forecast_estimates'):
            if self.complementary_information['model_type'] == 6 and hasattr(self.complementary_information, 'forecast_dates'):
                forecast_dates = self.complementary_information['forecast_dates'] 
            else:
                forecast_dates = np.arange(1,self.model.forecast_estimates.shape[0]+1)
            self.complementary_information['forecast_dates'] = forecast_dates
        # actual
        if 'y_p' not in self.complementary_information:
            self.complementary_information['y_p'] = []
                

    def _regression_fitted(self, show, save):
        if hasattr(self.model, 'fitted_estimates'):
            actual = self.model.y
            fitted = self.model.fitted_estimates
            dates = self.complementary_information['dates']
            path = self.path
            name = self.complementary_information['endogenous_variables'][0]
            file_name = 'fit-' + name + '.png'
            fig = gu.fit_single_variable(actual, fitted, dates, name)
            gu.show_and_save(fig, show, save, path, file_name)
            
            
    def _regression_residuals(self, show, save):
        if hasattr(self.model, 'residual_estimates'):
            residuals = self.model.residual_estimates
            dates = self.complementary_information['dates']
            path = self.path
            name = self.complementary_information['endogenous_variables'][0]
            file_name = 'residuals-' + name + '.png'
            fig = gu.residual_single_variable(residuals, dates, name)
            gu.show_and_save(fig, show, save, path, file_name)            
            

    def _regression_forecasts(self, show, save):
        if hasattr(self.model, 'forecast_estimates'):
            forecasts = self.model.forecast_estimates
            y_p = self.complementary_information['y_p']
            dates = self.complementary_information['forecast_dates']
            path = self.path
            name = self.complementary_information['endogenous_variables'][0]
            file_name = 'forecasts-' + name + '.png'
            fig = gu.ols_forecasts_single_variable(forecasts, y_p, dates, name)
            gu.show_and_save(fig, show, save, path, file_name)


        