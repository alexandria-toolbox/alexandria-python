# imports
import numpy as np
from os import system, name
from time import sleep
from IPython import get_ipython

    
# module console_utilities
# a module containing methods for console display


#---------------------------------------------------
# Methods
#---------------------------------------------------


def clear_console():
    
    """
    clear_console()
    clears Python console
    
    parameters:
    none

    returns:
    none        
    """
    
    try:
        get_ipython().magic('clear')
    except:
        system('cls' if name == 'nt' else 'clear')
    
   
def progress_bar(iteration, total_iterations, application_tag):
    
    """
    progress_bar(iteration, total_iterations, application_tag)
    progress bar for MCMC algorithms
    
    parameters:
    iteration: int
        current iteration of MCMC algorithm
    total_iterations: int
        total number of iterations of MCMC algorithm
    application_tag: str
        string flagging the application being estimated by MCMC
        
    returns:
    none
    """
    
    if iteration == 0:
        # print application tag (which application is being run)
        print(application_tag)
        # print iteration 0 progress bar
        string = ' ' * (len(str(total_iterations)) - len(str(iteration))) \
        + '1/' + str(total_iterations) + '  ' + '[' + 33 * '.' + ']   0%' 
        print(string, end = '\r')
        # print(string, end='\r')        
    elif iteration == (total_iterations - 1):
        # print final iteration progress bar
        string = str(total_iterations) + '/' + str(total_iterations) + '  ' \
        + '[' + 33 * '=' + ']' + '  —  done'
        print('\r' + string)
        # print(string)        
    else:
        # integer defining the arrow position in the progress bar
        arrow_position = int(np.floor((iteration + 1) * 33.3333 / total_iterations))
        # integer percentage of progress
        percentage = int(np.floor((iteration + 1) * 100 / total_iterations))
        string_percentage = ' ' * (percentage < 10) + str(percentage) + '%'
        # print progress bar
        string = ' ' * (len(str(total_iterations)) - len(str(iteration + 1))) \
        + str(iteration + 1) + '/' + str(total_iterations) + '  ' + '[' \
        + '=' * (arrow_position - 1) + '>' * (arrow_position != 0) \
        + (33 - arrow_position) * '.' + ']  ' + string_percentage
        print('\r' + string, end = '')
        # print(string, end='\r')
        sleep(0.0007)
    
    
def progress_bar_complete(application_tag):
    
    """
    progress_bar_complete(application_tag)
    pseudo progress bar used to show completion when no actual MCMC is run
    
    parameters:
    application_tag: str
        string flagging the application being estimated by MCMC
        
    returns:
    none
    """

    # print application tag (which application is being run)
    print(application_tag)    
    # print pseudo progress bar
    string = '  — / —    ' + '[' + 33 * '=' + ']' + '  —  done'
    print(string)

   
def optimization_completion(success):
    
    """
    optimization_completion(success)
    binary message that indicates success or failure of optimization
    
    parameters:   
    success: bool
        boolean indicating whether optimization has succeeded
        
    returns:
    none
    """
    
    if success:
        print('Optimization conducted succesfully.')
    else:
        print('Warning! Optimization failed. Prior values may be unreliable.')
        
        
def print_message(message):
    
    """
    print_message(message)
    a simple print function that prints any message
    
    parameters:     
    message: str
        the message to print
        
    returns:
    none
    """   
    
    print(message)


def print_string_list(string_list):
    
    """
    print_string_list(string_list)
    a function that prints a list of strings, line by line
    
    parameters:      
    string_list: str list
        list containing the strings to print
        
    returns:
    none
    """  
    
    [print(string) for string in string_list];
    

def write_string_list(string_list, filepath):
    
    """
    write_string_list(string_list, filepath)
    a function that writes a list of strings, line by line, in a specified path
    
    parameters:     
    string_list: str list
        list containing the strings to print
    filepath: str
        full path to file (must include file name)
        
    returns:
    none
    """      
    
    string_list_with_breaks = [string + '\n' for string in string_list]
    file = open(filepath,"a")
    file.writelines(string_list_with_breaks)
    file.close()


def alexandria_header():
    
    """
    alexandria_header()
    a function that creates a list of strings constituting Alexandria's header
    
    parameters:      
    none
        
    returns:
    header: str list
        the list of strings constituting Alexandria's header
    """     
    
    header = []
    header.append('                                                                               ')    
    header.append('      ======================================================================== ')
    header.append('         _____   __                                   _                        ')
    header.append('        / __  / / / ___  __  __   ____    ____    ___/ / ____  (_) ____        ')
    header.append('       / /_/ / / / / _ \ \ \/ /  / __ `/ / __ \ / __  / / __/ / / / __ `/      ')
    header.append('      / __  / / / /  __/  |  |  / /_/ / / / / // /_/ / / /   / / / /_/ /       ')
    header.append('     /_/ /_/ /_/  \___/  /_/\_\ \__,_/ /_/ /_/ \____/ /_/   /_/  \__,_/        ')
    header.append('                                                                               ')
    header.append('     The library of Bayesian time-series models                                ')
    header.append('     V 0.1 - Copyright Ⓒ  Romain Legrand                                      ')       
    header.append('   ========================================================================    ')
    header.append('                                                                               ')        
    header.append('                                                                               ')     
    return header


def format_number(number):
    
    """
    format_number(number)
    formats any number into a 10-character string
    
    parameters:     
    number: float
        number to be formatted
        
    returns:
    formatted_number: str
        string containing the formatted number
    """    
    
    # if number is of regular length, use decimal notation with 3 decimals
    if 0.0001 < abs(number) < 100000:
        formatted_number = '{:10.3f}'.format(number)
    else:
    # if value is too small or too large, switch to exponential notation
        formatted_number = '{:10.3e}'.format(number)
    return formatted_number


def format_name(name):
    
    """
    format_name(name)
    formats any variable name as a string containing a maximum of 20 characters
    
    parameters:      
    name: str
        variable name to be formatted
        
    returns:
    formatted_name: str
        string containing the formatted name
    """    
    
    # if name is less than 20 characters, return it left-justified
    formatted_name = '{:20}'.format(shorten_string(name, 20))
    return formatted_name


def shorten_string(string, n):
    
    """
    shorten_string(string, n)
    allows for a maximum of n characters in string; if longer, string is shortened with ' ...'
    
    parameters:      
    string: str
        string to be formatted
        
    returns:
    string: str
        string containing at most n characters
    """       
    
    if len(string) > n:
        string = string[:n-4] + ' ...'
    return string


def equal_dashed_line():
    
    """
    equal_dashed_line()
    return a line of 80 equal signs ('=')
    
    parameters:       
    none
        
    returns:
    line: str
        string containing the line
    """    
    
    line = 80 * '='
    return line
    
    
def hyphen_dashed_line():
    
    """
    hyphen_dashed_line()
    return a line of 80 hyphen signs ('-')
    
    parameters:      
    none
        
    returns:
    line: str
        string containing the line
    """    
    
    line = 80 * '-'
    return line    


def model_header(model):
    
    """
    model_header(model)
    return a results header with model name and two wrapping equal dashed lines
    
    parameters:      
    model: str
        string containing the model name
        
    returns:
    header: str list
        list of string containing the model header
    """ 
    
    # initiate header, add equal dashed line
    header = []
    header.append(equal_dashed_line())
    # center-justify model name
    header.append('{:^80}'.format(model))
    # add final equal dashed line
    header.append(equal_dashed_line())
    return header


def estimation_header(start, complete, n, endogenous, frequency, sample):
    
    """
    estimation_header(endogenous, start, complete, sample, frequency, n)
    return an estimation header with basic model information
    
    parameters:  
    start: datetime object
        datetime corresponding to date where estimation starts
    complete: datetime object
        datetime corresponding to date where estimation is complete
    n: int
        number of sample observations
    endogenous: str
        explained variable
    frequency: str
        data frequency
    sample: str
        estimation sample, start and end dates
        
    returns:
    header: str list
        list of string containing the estimation header
    """ 
    
    # initiate header
    header = []
    # first row: dependent variable and data frequency
    left_element = '{:14}{:>24}'.format('Dep. variable:', shorten_string(endogenous, 20))
    right_element = '{:7}{:>31}'.format('Sample:', sample)   
    header.append(left_element + '    ' + right_element)
    # second row: estimation start and sample
    left_element = '{:11}{:>27}'.format('Est. start:', start.strftime('%Y-%m-%d %H:%M:%S'))  
    right_element = '{:10}{:>28}'.format('Frequency:', frequency)
    header.append(left_element + '    ' + right_element)       
    # third row: estimation complete and observations
    left_element = '{:14}{:>24}'.format('Est. complete:', complete.strftime('%Y-%m-%d %H:%M:%S'))      
    right_element = '{:17}{:>21}'.format('No. observations:', str(n))
    header.append(left_element + '    ' + right_element) 
    return header
    
    
def coefficient_header(credibility_level):

    """
    coefficient_header(credibility_level)
    return an estimation line with headers for coefficient, standard devations and credibility bounds
    
    parameters:
    credibility_level: float
        credibility level for model
        
    returns:
    line: str
        string containing coefficient header
    """ 
    
    # initiate header
    header = []
    # calculate lower and upper bounds
    lower_bound = '{:5.3f}'.format(0.5 - credibility_level / 2)
    upper_bound = '{:5.3f}'.format(0.5 + credibility_level / 2)
    # generate first line of header
    header.append(' ' * 29 + 'median' + ' ' * 8 + 'std dev' + ' ' * 9 + '[' + lower_bound + ' ' * 9 + upper_bound + ']') 
    # generate second line of header
    header.append(hyphen_dashed_line())
    return header


def parameter_estimate_line(name, coefficient, standard_deviation, lower_bound, upper_bound):
    
    """
    parameter_estimate_line(name, coefficient, standard_deviation, lower_bound, upper_bound)
    return an estimation line with formatted values for coefficient, standard devations and credibility bounds
    
    parameters:  
    name: str
        name of variable to which parameter is related
    coefficient: float
        coefficient value
    standard_deviation: float (positive)
        standard deviation of parameter
    lower_bound: float
        lower bound of parameter
    upper_bound: float
        upper bound of parameter
        
    returns:
    line: str
        string containing coefficient summary
    """     
    
    line = format_name(name) + 5 * ' ' + format_number(coefficient) + 5 * ' ' + \
           format_number(standard_deviation) + 5 * ' ' + format_number(lower_bound) + \
           5 * ' ' + format_number(upper_bound)
    return line


def string_line(string):

    """
    string_line(name)
    returns a line where name is left-aligned, and right padding is added to reach 80 characters
    
    parameters:  
    string: str
        string to left-align
        
    returns:
    line: str
        80-character line with left-aligned string
    """      
    
    # left-justify the name, with pad to reach 80 characters
    line = '{:80}'.format(string)
    return line
    

def insample_evaluation_lines(ssr, r2, adj_r2, m_y, aic, bic):

    """
    insample_evaluation_lines(ssr, r2, adj_r2, m_y, aic, bic)
    returns the set of lines with the results for in-sample evaluation criteria
    
    parameters:  
    ssr: float
        sum of squared residuals
    r2: float
        coefficient of determination
    adj_r2: float
        adjusted coefficient of determination
    m_y: float
        log10 marginal likelihood
    aic: float
        Akaike information criterion
    bic: float
        Bayesian information criterion
        
    returns:
    lines: str list
        set of lines reporting in-sample evaluation criteria
    """ 
    
    # initiate lines
    lines = []
    # first row: ssr and marginal likelihood
    left_element = '{:4}{:>34}'.format('ssr:', format_number(ssr))
    if m_y:
        right_element = '{:17}{:>21}'.format('log10 marg. lik.:', format_number(m_y))
    else:
        right_element = ' ' * 38
    lines.append(left_element + '    ' + right_element)
    # second row: r2 and adjusted r2
    left_element = '{:3}{:>35}'.format('R2:', format_number(r2))  
    right_element = '{:8}{:>30}'.format('adj. R2:', format_number(adj_r2))
    lines.append(left_element + '    ' + right_element)  
    # third row AIC and BIC
    if aic:
        left_element = '{:4}{:>34}'.format('AIC:', format_number(aic))  
        right_element = '{:4}{:>34}'.format('BIC:', format_number(bic))
        lines.append(left_element + '    ' + right_element)
    return lines


def forecast_evaluation_lines(rmse, mae, mape, theil_u, bias, log_score, crps):
    
    """
    forecast_evaluation_lines(rmse, mae, mape, theil_u, bias, log_score, crps)
    returns the set of lines with the results for out-of-sample evaluation criteria
    
    parameters:  
    rmse: float
        root mean squared error for predictions
    mae: float
        mean absolute error for predictions
    mape: float
        mean absolute percentage error for predictions
    theil_u: float
        Theil's U coefficient for predictions
    bias: float
        bias for predictions
    log_score: float
        log score for predictions
    crps: float
        continuous rank probability score for predictions
        
    returns:
    lines: str list
        set of lines reporting out-of-sample evaluation criteria
    """ 
    
    # initiate lines
    lines = []
    # first row: rmse
    left_element = '{:5}{:>33}'.format('rmse:', format_number(rmse))
    right_element = ' '* 38
    lines.append(left_element + '    ' + right_element)
    # second row: mae and mape
    left_element = '{:4}{:>34}'.format('mae:', format_number(mae))  
    right_element = '{:5}{:>33}'.format('mape:', format_number(mape))
    lines.append(left_element + '    ' + right_element) 
    # third row: Theil's U and bias
    left_element = '{:10}{:>28}'.format("Theil's U:", format_number(theil_u))  
    right_element = '{:5}{:>33}'.format('bias:', format_number(bias))
    lines.append(left_element + '    ' + right_element) 
    # fourth row: log score and crps
    if log_score:
        left_element = '{:10}{:>28}'.format("log score:", format_number(log_score))  
        right_element = '{:5}{:>33}'.format('crps:', format_number(crps))
        lines.append(left_element + '    ' + right_element) 
    return lines


def tab_1_settings(model, endogenous, exogenous, frequency, sample, path, file, \
                   progress_bar, create_graphics, save_results):

    """
    tab_1_settings(model, endogenous, exogenous, frequency, sample, path, file, \
                   progress_bar, create_graphics, save_results)
    returns the set of lines with the results for tab 1 settings
    
    parameters: 
    model: str
        selected model (e.g. 'linear regression')
    endogenous: str
        set of endogenous variables, separated by a comma
    exogenous: str
        set of exogenous variables, separated by a comma
    frequency: str
        sample data frequency
    sample: str
        sample dates, separated by a space
    path: str
        path to project folder
    file: str
        name of data file
    progress_bar: bool
        user's choice for progress bar
    create graphics: bool
        user's choice for graphics creation
    save_results: bool
        user's choice for saving results
        
    returns:
    lines: str list
        set of lines reporting settings for tab 1
    """ 
    
    # initiate lines
    lines = []
    # header for tab 1
    lines.append('Models')
    lines.append('---------')
    lines.append(' ')
    # other elements for tab 1
    lines.append('model selection: ' + model)
    lines.append('endogenous variables: ' + endogenous) 
    lines.append('exogenous variables: ' + exogenous)
    lines.append('data frequency: ' + frequency)
    lines.append('estimation sample: ' + sample)
    lines.append('path to project folder: ' + path)
    lines.append('data file: ' + file)
    lines.append('progress bar: ' + bool_to_string(progress_bar))
    lines.append('create graphics: ' + bool_to_string(create_graphics))    
    lines.append('save_results: ' + bool_to_string(save_results))
    lines.append(' ')
    lines.append(' ')
    return lines


def bool_to_string(bool):
    
    """
    bool_to_string(bool)
    converts boolean to 'yes' or 'no'

    parameters:  
    bool: bool
        boolean to convert

    returns:
    string: str
        'yes' or 'no', depending on convered boolean  
    """
    
    if bool:
        string = 'yes'
    else:
        string = 'no'
    return string


def tab_3_settings(forecasts, forecast_credibility, conditional_forecasts, \
                   conditional_forecast_credibility, irf, irf_credibility, \
                   fevd, fevd_credibility, hd, hd_credibility, forecast_periods, \
                   conditional_forecast_type, forecast_file, forecast_evaluation, \
                   irf_periods, structural_identification, structural_identification_file):

    """
    tab_3_settings(forecasts, forecast_credibility, conditional_forecasts, \
                   conditional_forecast_credibility, irf, irf_credibility, \
                   fevd, fevd_credibility, hd, hd_credibility, forecast_periods, \
                   conditional_forecast_type, forecast_file, forecast_evaluation, \
                   irf_periods, structural_identification, structural_identification_file)
    returns the set of lines with the results for tab 3 settings 

    parameters:  
    forecasts: bool
        user's choice for forecasts
    forecast_credibility: float
        credibility level for forecasts
    conditional_forecasts: bool
        user's choice for conditional forecasts
    conditional_forecast_credibility: float
        credibility level for conditional forecasts
    irf: bool
        user's choice for impulse response functions
    irf_credibility: float
        credibility level for impulse response functions
    fevd: bool
        user's choice for forecast error variance decomposition
    fevd_credibility: float
        credibility level for forecast error variance decomposition  
    hd: bool
        user's choice for historical decomposition
    hd_credibility: float
        credibility level for historical decomposition  
    forecast_periods: int
        number of forecast periods
    conditional_forecast_type: str
        type of conditional forecasts
    forecast_file: str
        name of file containing the information for conditional forecasts
    forecast_evaluation: bool
        user's choice for forecast evaluation
    irf_periods: int
        number of impulse repsonse function periods        
    structural_identification: str
        type of structural identification        
    structural_identification_file: str
        name of file containing the information for structural identification     
        
    returns:
    lines: str list
        set of lines reporting settings for tab 3     
    """ 
    
    # initiate lines
    lines = []
    # header for tab 1
    lines.append('Applications')
    lines.append('---------')
    lines.append(' ')
    # other elements for tab 3
    lines.append('forecasts: ' + forecasts)
    lines.append('credibility level, forecasts: ' + forecast_credibility)    
    lines.append('conditional forecasts: ' + conditional_forecasts) 
    lines.append('credibility level, conditional forecasts: ' + conditional_forecast_credibility)     
    lines.append('impulse response functions: ' + irf)
    lines.append('credibility level, impulse response functions: ' + irf_credibility)     
    lines.append('forecast error variance decomposition: ' + fevd)
    lines.append('credibility level,forecast error variance decomposition: ' + fevd_credibility)    
    lines.append('historical decomposition: ' + hd)
    lines.append('credibility level,historical decomposition: ' + hd_credibility)
    lines.append('forecast periods: ' + forecast_periods)    
    lines.append('conditional forecast type: ' + conditional_forecast_type)        
    lines.append('forecast file: ' + forecast_file)    
    lines.append('forecast evaluation: ' + forecast_evaluation)  
    lines.append('impulse response function periods: ' + irf_periods)    
    lines.append('structural identification: ' + structural_identification)  
    lines.append('structural identification file: ' + structural_identification_file)
    lines.append(' ')
    lines.append(' ')
    return lines 























