# imports
import alexandria.graphics.graphics_utilities as gu
import numpy as np


class NowcastingGraphics(object):
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass


    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------


    def _complete_nowcasting_information(self):  
        # endogenous and exogenous variables
        if 'endogenous_variables' not in self.complementary_information:
            if self.complementary_information['model_type'] == 3:
                self.complementary_information['endogenous_variables'] = ['y'] 
            else:
                n = self.model.n
                self.complementary_information['endogenous_variables'] = ['y' + str(i+1) for i in range(n)]          
        if 'exogenous_variables' not in self.complementary_information:
            if self.complementary_information['model_type'] == 2 or len(self.model.exogenous) == 0:
                exogenous = []
            else:
                n_exo = self.model.exogenous.shape[1]
                exogenous = ['x' + str(i+1) for i in range(n_exo)]
            self.complementary_information['exogenous_variables'] = exogenous
        # sample dates
        if 'dates' not in self.complementary_information:
            T = self.model.T
            if self.complementary_information['model_type'] == 1:
                p = self.model.p
                self.complementary_information['dates'] = np.arange(-p+1,T+1)
            else:
                self.complementary_information['dates'] = np.arange(1,T+1)
        # forecast_dates
        if hasattr(self.model, 'forecast_estimates') and 'forecast_dates' not in self.complementary_information:
            T = self.model.T
            if self.complementary_information['model_type'] == 3:
                forecast_periods = len(self.model.forecast_estimates)
            else:
                forecast_periods = self.model.forecast_estimates.shape[0]
            forecast_dates = np.arange(T+1,T+forecast_periods+1)
            self.complementary_information['forecast_dates'] = forecast_dates
        # conditional forecast_dates
        if self.complementary_information['model_type'] == 1 \
            and hasattr(self.model, 'conditional_forecast_estimates') \
            and 'conditional_forecast_dates' not in self.complementary_information:
            T = self.model.T
            forecast_periods = self.model.conditional_forecast_estimates.shape[0]
            forecast_dates = np.arange(T+1,T+forecast_periods+1)
            self.complementary_information['conditional_forecast_dates'] = forecast_dates     
        # actual
        if 'Y_p' not in self.complementary_information:
            self.complementary_information['Y_p'] = []
        # structural shocks
        if 'shocks' not in self.complementary_information:
            if self.complementary_information['model_type'] == 1:
                n = self.model.n
                self.complementary_information['shocks'] = ['shock' + str(i+1) for i in range(n)]  
            elif self.complementary_information['model_type'] == 2:
                m = self.model.m
                self.complementary_information['shocks'] = \
                ['factor' + str(i+1) + '_shock'  for i in range(m)] + ['own_shock']


    def _nowcasting_fitted(self, show, save):
        if self.complementary_information['model_type'] == 1:
            self.__mfbvar_fitted(show, save)
        elif self.complementary_information['model_type'] == 2:
            self.__bdfm_fitted(show, save)
        elif self.complementary_information['model_type'] == 3:
            self.__midas_fitted(show, save)
            

    def __mfbvar_fitted(self, show, save):
        if hasattr(self.model, 'fitted_estimates'):
            # recover graphics elements
            n = self.model.n
            p = self.model.p
            actual = self.model.endogenous[p:]
            fitted = self.model.fitted_estimates
            dates = self.complementary_information['dates'][p:]
            path = self.path
            endogenous = self.complementary_information['endogenous_variables']
            # produce individual graphs
            for i in range(n):
                file_name = 'fit-' + endogenous[i] + '.png'
                fig = gu.nowcasting_fit_single_variable(actual[:,[i]], fitted[:,i,:], dates, endogenous[i])
                gu.show_and_save(fig, show, save, path, file_name)
            # joint graph
            fig = gu.nowcasting_fit_all(actual, fitted, dates, endogenous, n)
            gu.show_and_save(fig, show, save, path, 'fit-all.png')  


    def __bdfm_fitted(self, show, save):
        if hasattr(self.model, 'fitted_estimates'):
            # recover graphics elements
            n = self.model.n
            actual = self.model.endogenous
            fitted = self.model.fitted_estimates
            dates = self.complementary_information['dates']
            path = self.path
            endogenous = self.complementary_information['endogenous_variables']
            # produce individual graphs
            for i in range(n):
                file_name = 'fit-' + endogenous[i] + '.png'
                fig = gu.nowcasting_fit_single_variable(actual[:,[i]], fitted[:,i,:3], dates, endogenous[i])
                gu.show_and_save(fig, show, save, path, file_name)
            # joint graph
            fig = gu.nowcasting_fit_all(actual, fitted, dates, endogenous, n)
            gu.show_and_save(fig, show, save, path, 'fit-all.png')  


    def __midas_fitted(self, show, save):
        if hasattr(self.model, 'fitted_estimates'):
            # recover graphics elements
            actual = self.model.y.reshape(-1,1)
            fitted = self.model.fitted_estimates
            dates = self.complementary_information['dates'][-self.model.y.shape[0]:]
            path = self.path
            endogenous = self.complementary_information['endogenous_variables'][0]
            # produce individual graphs
            file_name = 'fit-' + endogenous + '.png'
            fig = gu.nowcasting_fit_single_variable(actual, fitted, dates, endogenous)
            gu.show_and_save(fig, show, save, path, file_name)


    def _nowcasting_residuals(self, show, save):
        if self.complementary_information['model_type'] == 1:
            self.__mfbvar_residuals(show, save)
        elif self.complementary_information['model_type'] == 2:
            self.__bdfm_residuals(show, save)
        elif self.complementary_information['model_type'] == 3:
            self.__midas_residuals(show, save)


    def __mfbvar_residuals(self, show, save):
        if hasattr(self.model, 'residual_estimates'):
            # recover graphics elements
            n = self.model.n
            p = self.model.p
            residuals = self.model.residual_estimates
            dates = self.complementary_information['dates'][p:]
            path = self.path
            endogenous = self.complementary_information['endogenous_variables']            
            # produce individual graphs
            for i in range(n):
                file_name = 'residuals-' + endogenous[i] + '.png'
                fig = gu.var_residual_single_variable(residuals[:,i,:], dates, endogenous[i])
                gu.show_and_save(fig, show, save, path, file_name)            
            # joint graph
            fig = gu.var_residual_all(residuals, dates, endogenous, n)
            gu.show_and_save(fig, show, save, path, 'residuals-all.png')
            
            
    def __bdfm_residuals(self, show, save):
        if hasattr(self.model, 'residual_estimates'):
            # recover graphics elements
            n = self.model.n
            residuals = self.model.residual_estimates
            dates = self.complementary_information['dates']
            path = self.path
            endogenous = self.complementary_information['endogenous_variables']            
            # produce individual graphs
            for i in range(n):
                file_name = 'residuals-' + endogenous[i] + '.png'
                fig = gu.var_residual_single_variable(residuals[:,i,:3], dates, endogenous[i])
                gu.show_and_save(fig, show, save, path, file_name)            
            # joint graph
            fig = gu.var_residual_all(residuals, dates, endogenous, n)
            gu.show_and_save(fig, show, save, path, 'residuals-all.png')            
            
            
    def __midas_residuals(self, show, save):
        if hasattr(self.model, 'residual_estimates'):
            # recover graphics elements
            residuals = self.model.residual_estimates
            dates = self.complementary_information['dates'][-self.model.y.shape[0]:]
            path = self.path
            endogenous = self.complementary_information['endogenous_variables'][0]    
            # produce individual graphs
            file_name = 'residuals-' + endogenous + '.png'
            fig = gu.var_residual_single_variable(residuals, dates, endogenous)
            gu.show_and_save(fig, show, save, path, file_name)                      
            
            
    def _nowcasting_shocks(self, show, save):    
        if self.complementary_information['model_type'] == 1 and \
        hasattr(self.model, 'structural_shock_estimates'):
            # recover graphics elements
            n = self.model.n
            p = self.model.p
            shocks = self.model.structural_shock_estimates
            dates = self.complementary_information['dates'][p:]
            path = self.path
            endogenous = self.complementary_information['endogenous_variables']            
            # produce individual graphs
            for i in range(n):
                file_name = 'shocks-' + endogenous[i] + '.png'
                fig = gu.var_shocks_single_variable(shocks[:,i,:], dates, endogenous[i])
                gu.show_and_save(fig, show, save, path, file_name)            
            # joint graph
            fig = gu.var_shocks_all(shocks, dates, endogenous, n)
            gu.show_and_save(fig, show, save, path, 'shocks-all.png')              
            
    
    def _nowcasting_steady_state(self, show, save):  
        if self.complementary_information['model_type'] == 1 and \
        hasattr(self.model, 'steady_state_estimates'):
            # recover graphics elements
            n = self.model.n
            p = self.model.p
            actual = self.model.Y
            steady_state = self.model.steady_state_estimates
            dates = self.complementary_information['dates'][p:]
            path = self.path
            endogenous = self.complementary_information['endogenous_variables']            
            # produce individual graphs
            for i in range(n):
                file_name = 'steady_state-' + endogenous[i] + '.png'
                fig = gu.var_steady_state_single_variable(actual[:,[i]],steady_state[:,i,:], dates, endogenous[i])
                gu.show_and_save(fig, show, save, path, file_name)            
            # joint graph
            fig = gu.var_steady_state_all(actual, steady_state, dates, endogenous, n)
            gu.show_and_save(fig, show, save, path, 'steady_state-all.png')     
            
            
    def _nowcasting_factors(self, show, save):            
        if self.complementary_information['model_type'] == 2 and \
        hasattr(self.model, 'f_estimates'):
            # recover graphics elements
            m = self.model.m
            factors = self.model.f_estimates
            dates = self.complementary_information['dates']
            path = self.path         
            # produce individual graphs
            for i in range(m):
                file_name = 'factors-factor' + str(i+1) + '.png'
                fig = gu.factor_single_variable(factors[:,i,:3], dates, str(i+1))
                gu.show_and_save(fig, show, save, path, file_name)            
            # joint graph
            fig = gu.factor_all(factors, dates, m)
            gu.show_and_save(fig, show, save, path, 'factors-all.png')               
            
            
    def _nowcasting_forecasts(self, show, save):
        if self.complementary_information['model_type'] == 1:
            self.__mfbvar_forecasts(show, save)
        elif self.complementary_information['model_type'] == 2:
            self.__bdfm_forecasts(show, save)
        elif self.complementary_information['model_type'] == 3:
            self.__midas_forecasts(show, save)    
    
    
    def __mfbvar_forecasts(self, show, save):
        if hasattr(self.model, 'forecast_estimates'):  
            # recover graphics elements
            n = self.model.n
            p = self.model.p
            actual = self.model.endogenous[p:]
            forecasts = self.model.forecast_estimates
            Y_p = self.complementary_information['Y_p']
            Y_p_i = []
            dates = self.complementary_information['dates'][p:]
            forecast_dates = self.complementary_information['forecast_dates']
            path = self.path
            endogenous = self.complementary_information['endogenous_variables']
            # produce individual graphs
            for i in range(n):
                file_name = 'forecasts-' + endogenous[i] + '.png'
                if len(Y_p) != 0:
                    Y_p_i = Y_p[:,[i]]
                fig = gu.nowcasting_forecasts_single_variable(actual[:,[i]],forecasts[:,i,:], Y_p_i, dates, forecast_dates, endogenous[i])
                gu.show_and_save(fig, show, save, path, file_name)               
            # joint graph
            fig = gu.nowcasting_forecasts_all(actual, forecasts, Y_p, dates, forecast_dates, endogenous, n)
            gu.show_and_save(fig, show, save, path, 'forecasts-all.png')  


    def __bdfm_forecasts(self, show, save):
        if hasattr(self.model, 'forecast_estimates'):  
            # recover graphics elements
            n = self.model.n
            actual = self.model.endogenous
            forecasts = self.model.forecast_estimates
            Y_p = self.complementary_information['Y_p']
            Y_p_i = []
            dates = self.complementary_information['dates']
            forecast_dates = self.complementary_information['forecast_dates']
            path = self.path
            endogenous = self.complementary_information['endogenous_variables']
            # produce individual graphs
            for i in range(n):
                file_name = 'forecasts-' + endogenous[i] + '.png'
                if len(Y_p) != 0:
                    Y_p_i = Y_p[:,[i]]
                fig = gu.nowcasting_forecasts_single_variable(actual[:,[i]],forecasts[:,i,:], Y_p_i, dates, forecast_dates, endogenous[i])
                gu.show_and_save(fig, show, save, path, file_name)               
            # joint graph
            fig = gu.nowcasting_forecasts_all(actual, forecasts, Y_p, dates, forecast_dates, endogenous, n)
            gu.show_and_save(fig, show, save, path, 'forecasts-all.png')  


    def __midas_forecasts(self, show, save):
        if hasattr(self.model, 'forecast_estimates'): 
            # recover graphics elements
            p = self.model.p
            actual = self.model.endogenous[p:]
            forecasts = np.vstack(self.model.forecast_estimates)
            Y_p = self.complementary_information['Y_p']
            dates = self.complementary_information['dates'][p:]
            forecast_dates = self.complementary_information['forecast_dates']
            path = self.path
            endogenous = self.complementary_information['endogenous_variables'][0]
            # produce individual graphs
            file_name = 'forecasts-' + endogenous + '.png'
            fig = gu.nowcasting_forecasts_single_variable(actual,forecasts, Y_p, dates, forecast_dates, endogenous)
            gu.show_and_save(fig, show, save, path, file_name)               
 

    def _nowcasting_conditional_forecasts(self, show, save):
        if self.complementary_information['model_type'] == 1 and \
        hasattr(self.model, 'conditional_forecast_estimates'):   
            # recover graphics elements
            n = self.model.n
            p = self.model.p
            actual = self.model.endogenous[p:]
            forecasts = self.model.conditional_forecast_estimates
            Y_p = self.complementary_information['Y_p']
            Y_p_i = []
            dates = self.complementary_information['dates'][p:]
            forecast_dates = self.complementary_information['forecast_dates']
            path = self.path
            endogenous = self.complementary_information['endogenous_variables']
            # produce individual graphs
            for i in range(n):
                file_name = 'conditional_forecasts-' + endogenous[i] + '.png'
                if len(Y_p) != 0:
                    Y_p_i = Y_p[:,[i]]
                fig = gu.nowcasting_conditional_forecasts_single_variable(actual[:,[i]],\
                      forecasts[:,i,:], Y_p_i, dates, forecast_dates, endogenous[i])
                gu.show_and_save(fig, show, save, path, file_name)               
            # joint graph
            fig = gu.nowcasting_conditional_forecasts_all(actual, forecasts, Y_p, \
                  dates, forecast_dates, endogenous, n)
            gu.show_and_save(fig, show, save, path, 'conditional_forecasts-all.png') 


    def _nowcasting_irf(self, show, save):
        if self.complementary_information['model_type'] == 1:
            self.__mfbvar_irf(show, save)
        elif self.complementary_information['model_type'] == 2:
            self.__bdfm_irf(show, save)


    def __mfbvar_irf(self, show, save):
        if hasattr(self.model, 'irf_estimates'):  
            # recover graphics elements
            n = self.model.n
            irf = self.model.irf_estimates
            path = self.path
            endogenous = self.complementary_information['endogenous_variables']
            shocks = self.complementary_information['shocks']
            # produce individual graphs
            for i in range(n):
                for j in range(n):
                    file_name = 'irf-' + endogenous[i] + '@' + shocks[j] + '.png'
                    fig = gu.var_irf_single_variable(irf[i,j,:,:], endogenous[i], shocks[j])
                    gu.show_and_save(fig, show, save, path, file_name)    
        if hasattr(self.model, 'exo_irf_estimates') and len(self.model.exo_irf_estimates) != 0:
            # recover graphics elements
            n = self.model.n
            n_exo = self.model.exogenous.shape[1]
            exo_irf = self.model.exo_irf_estimates
            path = self.path
            endogenous = self.complementary_information['endogenous_variables']
            shocks = self.complementary_information['exogenous_variables']
            # produce individual graphs
            for i in range(n):
                for j in range(n_exo):
                    file_name = 'irf-' + endogenous[i] + '@' + shocks[j] + '.png'
                    fig = gu.var_irf_single_variable(exo_irf[i,j,:,:], endogenous[i], shocks[j])
                    gu.show_and_save(fig, show, save, path, file_name)                       
        if hasattr(self.model, 'irf_estimates'): 
            # recover graphics elements
            n_endo = self.model.n
            n_shocks = n_endo
            irf = self.model.irf_estimates
            variables = self.complementary_information['endogenous_variables']
            shocks = self.complementary_information['shocks']
            if hasattr(self.model, 'exo_irf_estimates') and len(self.model.exo_irf_estimates) != 0:
                n_exo = self.model.exogenous.shape[1]
                exo_irf = self.model.exo_irf_estimates
                exogenous = self.complementary_information['exogenous_variables']
                n_shocks += n_exo
                irf = np.concatenate((irf,exo_irf), axis=1)
                shocks += exogenous
            # joint graph
            fig = gu.var_irf_all(irf, variables, shocks, n_endo, n_shocks)
            gu.show_and_save(fig, show, save, path, 'irf-all.png')  


    def __bdfm_irf(self, show, save):
        if hasattr(self.model, 'irf_estimates'):  
            # recover graphics elements
            n = self.model.n
            m = self.model.m
            irf = self.model.irf_estimates
            path = self.path
            endogenous = self.complementary_information['endogenous_variables']
            shocks = self.complementary_information['shocks']
            # produce individual graphs
            for i in range(n):
                for j in range(m+1):
                    file_name = 'irf-' + endogenous[i] + '@' + shocks[j] + '.png'
                    fig = gu.var_irf_single_variable(irf[i,j,:,:], endogenous[i], shocks[j])
                    gu.show_and_save(fig, show, save, path, file_name)    
            # joint graph
            fig = gu.var_irf_all(irf, endogenous, shocks, n, m+1)
            gu.show_and_save(fig, show, save, path, 'irf-all.png')  


    def _nowcasting_fevd(self, show, save):
        if self.complementary_information['model_type'] == 1:
            self.__mfbvar_fevd(show, save)
        elif self.complementary_information['model_type'] == 2:
            self.__bdfm_fevd(show, save)


    def __mfbvar_fevd(self, show, save):
        if hasattr(self.model, 'fevd_estimates'):
            # recover graphics elements
            n = self.model.n
            fevd = self.model.fevd_estimates
            path = self.path
            endogenous = self.complementary_information['endogenous_variables']
            shocks = self.complementary_information['shocks']
            # produce individual graphs
            for i in range(n):
                for j in range(n):
                    file_name = 'fevd-' + endogenous[i] + '@' + shocks[j] + '.png'
                    fig = gu.var_fevd_single_variable(fevd[i,j,:,:], endogenous[i], shocks[j])
                    gu.show_and_save(fig, show, save, path, file_name)  
            # partial joint graph
            for i in range(n):
                file_name = 'fevd-' + endogenous[i] + '@all.png'
                fig = gu.var_fevd_joint(fevd[i,:,:,0].T, endogenous[i], shocks, n)
                gu.show_and_save(fig, show, save, path, file_name)
            # joint graph
            fig = gu.var_fevd_all(fevd, endogenous, shocks, n, n)
            gu.show_and_save(fig, show, save, path, 'fevd-all.png') 


    def __bdfm_fevd(self, show, save):
        if hasattr(self.model, 'fevd_estimates'):
            # recover graphics elements
            n = self.model.n
            m = self.model.m
            fevd = self.model.fevd_estimates
            path = self.path
            endogenous = self.complementary_information['endogenous_variables']
            shocks = self.complementary_information['shocks']
            # produce individual graphs
            for i in range(n):
                for j in range(m+1):
                    file_name = 'fevd-' + endogenous[i] + '@' + shocks[j] + '.png'
                    fig = gu.var_fevd_single_variable(fevd[i,j,:,:], endogenous[i], shocks[j])
                    gu.show_and_save(fig, show, save, path, file_name)  
            # partial joint graph
            for i in range(n):
                file_name = 'fevd-' + endogenous[i] + '@all.png'
                fig = gu.var_fevd_joint(fevd[i,:,:,0].T, endogenous[i], shocks, m+1)
                gu.show_and_save(fig, show, save, path, file_name)
            # joint graph
            fig = gu.var_fevd_all(fevd, endogenous, shocks, n, m+1)
            gu.show_and_save(fig, show, save, path, 'fevd-all.png') 


    def _nowcasting_hd(self, show, save):
        if self.complementary_information['model_type'] == 1:
            self.__mfbvar_hd(show, save)
        elif self.complementary_information['model_type'] == 2:
            self.__bdfm_hd(show, save)


    def __mfbvar_hd(self, show, save):
        if hasattr(self.model, 'hd_estimates'):
            # recover graphics elements
            n = self.model.n
            p = self.model.p
            T = self.model.T
            dates = self.complementary_information['dates'][p:]
            hd = self.model.hd_estimates
            path = self.path
            endogenous = self.complementary_information['endogenous_variables']
            shocks = self.complementary_information['shocks']
            # produce individual graphs
            for i in range(n):
                for j in range(n):
                    file_name = 'hd-' + endogenous[i] + '@' + shocks[j] + '.png'
                    fig = gu.var_hd_single_variable(hd[i,j,:,:], endogenous[i], shocks[j], dates)
                    gu.show_and_save(fig, show, save, path, file_name)  
            # partial joint graph
            for i in range(n):
                file_name = 'hd-' + endogenous[i] + '@all.png'
                fig = gu.var_hd_joint(hd[i,:,:,0].T, endogenous[i], shocks, dates, n, T)
                gu.show_and_save(fig, show, save, path, file_name)
            # joint graph
            fig = gu.var_hd_all(hd, endogenous, shocks, dates, n, n, T)
            gu.show_and_save(fig, show, save, path, 'hd-all.png')  
            
            
    def __bdfm_hd(self, show, save):
        if hasattr(self.model, 'hd_estimates'):
            # recover graphics elements
            n = self.model.n
            m = self.model.m
            T = self.model.T
            dates = self.complementary_information['dates']
            hd = self.model.hd_estimates
            path = self.path
            endogenous = self.complementary_information['endogenous_variables']
            shocks = self.complementary_information['shocks']
            # produce individual graphs
            for i in range(n):
                for j in range(m+1):
                    file_name = 'hd-' + endogenous[i] + '@' + shocks[j] + '.png'
                    fig = gu.var_hd_single_variable(hd[i,j,:,:], endogenous[i], shocks[j], dates)
                    gu.show_and_save(fig, show, save, path, file_name)  
            # partial joint graph
            for i in range(n):
                file_name = 'hd-' + endogenous[i] + '@all.png'
                fig = gu.var_hd_joint(hd[i,:,:,0].T, endogenous[i], shocks, dates, m+1, T)
                gu.show_and_save(fig, show, save, path, file_name)
            # joint graph
            fig = gu.var_hd_all(hd, endogenous, shocks, dates, n, m+1, T)
            gu.show_and_save(fig, show, save, path, 'hd-all.png')     

