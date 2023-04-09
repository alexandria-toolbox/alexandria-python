#---------------------------------------------------
# Imports
#--------------------------------------------------- 


import IPython
import os
from warnings import filterwarnings
import matplotlib.pyplot as plt
import alexandria.console.console_utilities as cu

# clear workspace and console (not to be modified)
IPython.get_ipython().magic('reset -sf')
cu.clear_console()
filterwarnings('ignore')
plt.close('all')

# initiate user inputs (not to be modified)
user_inputs = {}
user_inputs['tab_1'] = {}
user_inputs['tab_2_lr'] = {}
user_inputs['tab_3'] = {}


#---------------------------------------------------
# Editable part: tab 1
#---------------------------------------------------  


# model choice (1 = linear regression)
user_inputs['tab_1']['model'] = 1

# endogenous variables, as list of strings (e.g. ['var1', 'var2'])
user_inputs['tab_1']['endogenous_variables'] = ['']

# exogenous variables, as list of strings (e.g. ['var1', 'var2']; leave as empty list [] if no exogenous)
user_inputs['tab_1']['exogenous_variables'] = ['']

# data frequency (1: cross-sectional/undated, 2: yearly, 3: quarterly, 4: monthly, 5: weekly, 6: daily)
user_inputs['tab_1']['frequency'] = 1

# data sample: start and end periods, as list of timestamps strings (e.g. ['1990-03-31', '2020-12-31']) or periods (e.g. ['1990Q1', '2020Q4'])
# either in timestamp format: ['1990-01-31', '2020-12-31'] or in period format: ['1990M1', '2020M12']
user_inputs['tab_1']['sample'] = ['', '']

# path to project folder, as string (e.g. 'D:\my_project')
user_inputs['tab_1']['project_path'] = os.getcwd()

# name of data file, as string (e.g. 'data.csv')
user_inputs['tab_1']['data_file'] = 'data.csv'

# display progress bar during estimation (True: yes, False: no)
user_inputs['tab_1']['progress_bar'] = True

# generate estimation graphics (True: yes, False: no)
user_inputs['tab_1']['create_graphics'] = True

# save estimation results (True: yes, False: no)
user_inputs['tab_1']['save_results'] = True


#---------------------------------------------------
# Editable part: tab 2, linear regression
#--------------------------------------------------- 


# this applies only if the selected model is linear regression (model = 1)
if user_inputs['tab_1']['model'] == 1:
    
    # choice of linear regression model (1: maximum likelihood; 2: simple Bayesian;
    # 3: hierarchical; 4: independent; 5: heteroscedastic; 6: autocorrelated)
    user_inputs['tab_2_lr']['regression_type'] = 1

    # post-burn iterations for MCMC algorithm (integer)
    user_inputs['tab_2_lr']['iterations'] = 2000

    # burnin iterations for MCMC algorithm (integer)
    user_inputs['tab_2_lr']['burnin'] = 1000
    
    # credibility level for model estimates (float between 0 and 1)
    user_inputs['tab_2_lr']['model_credibility'] = 0.95

    # prior mean for regression coefficients beta: either scalar for common mean (e.g. 0),
    # or list of values, one for each coefficient (e.g. [0, 0, 0])
    user_inputs['tab_2_lr']['b'] = 0

    # prior variance for regression coefficients beta: either scalar for common variance (e.g. 1),
    # or list of values, one for each coefficient (e.g. [1, 1, 1])
    user_inputs['tab_2_lr']['V'] = 1
    
    # prior shape for regression variance sigma (positive float)
    user_inputs['tab_2_lr']['alpha'] = 0.0001

    # prior scale for regression variance sigma (positive float)
    user_inputs['tab_2_lr']['delta'] = 0.0001

    # prior mean for heteroscedastic coefficients gamma: either scalar for common mean (e.g. 0),
    # or list of values, one for each coefficient (e.g. [0, 0, 0])
    user_inputs['tab_2_lr']['g'] = 0

    # prior variance for heteroscedastic coefficients gamma: either scalar for common variance (e.g. 1),
    # or list of values, one for each coefficient (e.g. [1, 1, 1])
    user_inputs['tab_2_lr']['Q'] = 100
    
    # variance of transition kernel for Metropolis-Hastings step in heteroscedastic model (positive float)
    user_inputs['tab_2_lr']['tau'] = 0.001

    # apply posterior thinning to MCMC draws in heteroscedastic model (True: yes, False: no)
    user_inputs['tab_2_lr']['thinning'] = False

    # frequency of posterior thinning (positive integer)
    user_inputs['tab_2_lr']['thinning_frequency'] = 10

    # Z variables, as list of strings (e.g. ['var1', 'var2']); can be empty if model is not heteroscedastic regression
    user_inputs['tab_2_lr']['Z_variables'] = ['']
    
    # order of autoregressive process for residuals in autocorrelated models (positive integer)
    user_inputs['tab_2_lr']['q'] = 1
    
    # prior mean for autocorrelation coefficients phi: either scalar for common mean (e.g. 0),
    # or list of values, one value for each coefficient (e.g. [0, 0, 0])
    user_inputs['tab_2_lr']['p'] = 0

    # prior variance for autocorrelation coefficients phi: either scalar for common variance (e.g. 1),
    # or list of values, one value for each coefficient (e.g. [1, 1, 1])
    user_inputs['tab_2_lr']['H'] = 100  
    
    # include constant in regression (True: yes, False: no)
    user_inputs['tab_2_lr']['constant'] = True     
    
    # prior mean for regression constant (float)
    user_inputs['tab_2_lr']['b_constant'] = 0

    # prior variance for constant (positive float)
    user_inputs['tab_2_lr']['V_constant'] = 1   
    
    # include trend in regression (True: yes, False: no)
    user_inputs['tab_2_lr']['trend'] = False
    
    # prior mean for regression trend (float)
    user_inputs['tab_2_lr']['b_trend'] = 0

    # prior variance for trend (positive float)
    user_inputs['tab_2_lr']['V_trend'] = 1      
    
    # include quadratic trend in regression (True: yes, False: no)
    user_inputs['tab_2_lr']['quadratic_trend'] = False
    
    # prior mean for regression quadratic trend (float)
    user_inputs['tab_2_lr']['b_quadratic_trend'] = 0

    # prior variance for quadratic trend (positive float)
    user_inputs['tab_2_lr']['V_quadratic_trend'] = 1
    
    # estimate in-sample fit (True: yes, False: no)
    user_inputs['tab_2_lr']['insample_fit'] = False
    
    # estimate marginal likelihood (True: yes, False: no)
    user_inputs['tab_2_lr']['marginal_likelihood'] = False
    
    # apply hyperparameter optimization (True: yes, False: no)
    user_inputs['tab_2_lr']['hyperparameter_optimization'] = False
    
    # type of hyperparameter optimization (1: common variance, 2: coefficient-specific variances plus residual variance)
    user_inputs['tab_2_lr']['optimization_type'] = 1
    
    
#---------------------------------------------------
# Editable part: tab 3
#--------------------------------------------------- 


# estimate forecasts for the model (True: yes, False: no)
user_inputs['tab_3']['forecast'] = False
 
# credibility level for forecast estimates (float between 0 and 1)
user_inputs['tab_3']['forecast_credibility'] = 0.95

# estimate conditional forecasts for the model (True: yes, False: no)
user_inputs['tab_3']['conditional_forecast'] = False    

# credibility level for conditional forecast estimates (float between 0 and 1)
user_inputs['tab_3']['conditional_forecast_credibility'] = 0.95

# estimate impulse response functions for the model (True: yes, False: no)
user_inputs['tab_3']['irf'] = False

# credibility level for impulse response functions estimates (float between 0 and 1)
user_inputs['tab_3']['irf_credibility'] = 0.95

# estimate forecast error variance decomposition for the model (True: yes, False: no)
user_inputs['tab_3']['fevd'] = False

# credibility level for forecast error variance decomposition estimates (float between 0 and 1)
user_inputs['tab_3']['fevd_credibility'] = 0.95

# estimate historical decomposition for the model (True: yes, False: no)
user_inputs['tab_3']['hd'] = False

# credibility level for historical decomposition estimates (float between 0 and 1)
user_inputs['tab_3']['hd_credibility'] = 0.95

# number of forecast periods (positive integer)
user_inputs['tab_3']['forecast_periods'] = 1

# number of impulse response functions periods (positive integer)
user_inputs['tab_3']['irf_periods'] = 1

# type of conditional forecasts (1: all shocks, 2: shock-specific)
user_inputs['tab_3']['conditional_forecast_type'] = 1

# structural identification scheme (1: none, 2: Cholesky)
user_inputs['tab_3']['structural_identification'] = 1

# estimate forecast evaluation criteria (True: yes, False: no)
user_inputs['tab_3']['forecast_evaluation'] = False

# name of forecast data file, as string (e.g. 'data_forecast.csv')
user_inputs['tab_3']['forecast_file'] = ''

# name of structural identification file, as string (e.g. 'structural_identification.csv')
user_inputs['tab_3']['structural_identification_file'] = ''


#---------------------------------------------------
# Main code (not to be modified)
#---------------------------------------------------  


model = user_inputs['tab_1']['model']

# if model is linear regression, import main code for linear regression, run it, and return model
if model == 1:
    from alexandria.linear_regression.main import linear_regression_main_code
    lr = linear_regression_main_code(user_inputs)
 



