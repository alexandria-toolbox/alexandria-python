# imports
import numpy as np
from os import system, name, mkdir
from os.path import isdir, join
from time import sleep
from IPython import get_ipython
import alexandria.processor.input_utilities as iu

    
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


def print_message_to_overwrite(message):
    
    """
    print_message_to_overwrite(message)
    a print function that prints a message that will be overwritten by the next message
    
    parameters:     
    message: str
        the message to print
        
    returns:
    none
    """   
    
    print(message, end='\r')


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
    file = open(filepath,'w')
    file.writelines(string_list_with_breaks)
    file.close()


def check_path(path):

    """
    check_path(path)
    checks whether folder given by path exists, and create it if needed
    
    parameters:     
    path: str
        path to folder
        
    returns:
    none
    """ 

    if not isdir(path):
        mkdir(path)
 
    
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
    header.append('     V 2.0 - Copyright Ⓒ  Romain Legrand                                      ')       
    header.append('   ========================================================================    ')
    header.append('                                                                               ')        
    header.append('                                                                               ')     
    return header


def print_alexandria_header():
    
    """
    print_alexandria_header()
    display Alexandria header on console
    
    parameters:     
    none
        
    returns:
    none
    """       
    
    # get Alexandria header
    header = alexandria_header()
    # print header
    print_string_list(header)  



def print_start_message():

    """
    print_start_message()
    display start message
    
    parameters:     
    none
        
    returns:
    none
    """    
    
    print_message('Starting estimation of your model...')
    print_message(' ')



def print_completion_message(progress_bar):

    """
    print_completion_message(progress_bar)
    display completion message
    
    parameters:     
    progress_bar: bool
        if yes, progress bar is displayed
        
    returns:
    none
    """    
    
    if progress_bar:
        print_message(' ')
    print_message('Estimation completed successfully.')
    print_message(' ')  
    print_message(' ')


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
    
    # if number is exactly 0, format it normally
    if number == 0:
        formatted_number = '     0.000'
    # if number is of regular length, use decimal notation with 3 decimals
    elif 0.0001 <= abs(number) < 100000:
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
    allows for a maximum of n characters in string; if longer, string is shortened with '..'
    
    parameters:      
    string: str
        string to be formatted
        
    returns:
    string: str
        string containing at most n characters
    """       
    
    if len(string) > n:
        string = string[:n-2] + '..'
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
    

def equation_header(string):
    
    """
    equation_header(string)
    return a header with string, and two wrapping dashed lines
    
    parameters:      
    string: str
        string containing the model name
        
    returns:
    header: str list
        list of string containing the model header
    """ 
    
    # initiate header, add equal dashed line
    header = []
    header.append(equal_dashed_line())
    # center-justify model name
    header.append('{:^80}'.format(string))
    # add final equal dashed line
    header.append(hyphen_dashed_line())
    return header


def intermediate_header(string):
    
    """
    intermediate_header(string)
    return an intermediate header with string, and two wrapping dashed lines
    
    parameters:      
    string: str
        string containing the model name
        
    returns:
    header: str list
        list of string containing the model header
    """ 
    
    # initiate header, add equal dashed line
    header = []
    header.append(hyphen_dashed_line())
    # center-justify model name
    header.append('{:^80}'.format(string))
    # add final equal dashed line
    header.append(hyphen_dashed_line())
    return header


def make_regressors(endogenous, exogenous, constant, trend, quadratic_trend, n, p):

    """
    make_regressors(endogenous, exogenous, constant, trend, quadratic_trend, n, p)
    return a list of strings of regressors
    
    parameters:      
    endogenous: str list
        list of endogenous variables
    exogenous: str list
        list of exogenous variables   
    constant: bool
        if true, a constant is added to the model
    trend: bool
        if true, a trend is added to the model
    quadratic_trend: bool
        if true, a quadratic trend is added to the model
    n : int
        number of endogenous variables
    p : int
        number of lags
        
    returns:
    regressors: str list
        list of string containing the model regressors
    """

    regressors = []
    if constant:
        regressors.append('constant')
    if trend:
        regressors.append('trend')
    if quadratic_trend:
        regressors.append('quadratic trend')         
    if exogenous != ['none']:
        regressors += exogenous
    for i in range(n):
        for j in range(p):
            regressors.append(endogenous[i] + ' (-' + str(j+1) + ')')
    return regressors


def make_index(n, m, p, k):

    """
    make_index(n, m, p, k)
    return an array of indices for VAR coefficients
    
    parameters:      
    n : int
        number of endogenous variables
    m : int
        number of exogenous variables            
    p : int
        number of lags
    k : int
        number of coefficients per VAR equation
        
    returns:
    index: ndarray of size (k,)
        array of indices
    """

    index = np.zeros(k)
    i = -1
    for j in range(m):
        i = i + 1
        index[i] = j
    for g in range(n):
        for h in range(p):
            i = i + 1
            index[i] = m + h * n + g
    return index


def variance_line(residual_variance, shock_variance):
    
    """
    variance_line(residual_variance, shock_variance)
    return a line with residual and shock variance estimate
    
    parameters:      
    variable: str
        string containing the model name
        
    returns:
    residual_variance: float
        residual variance estimate
    residual_variance: float or empty str
        shock variance estimate    
    """     
    
    formatted_residual_variance = format_number(residual_variance)
    left_element = '{:18}{:>20}'.format('residual variance:', formatted_residual_variance)
    if iu.is_numeric(shock_variance):
        formatted_shock_variance = format_number(shock_variance)
        right_element = '{:15}{:>23}'.format('shock variance:', formatted_shock_variance)
    else:
        right_element = ' ' * 38
    line = left_element + '    ' + right_element
    return line
    
    
def variance_covariance_summary(Sigma, n, endogenous_variables, tag):

    """
    variance_covariance_summary(Sigma, n, endogenous_variables)
    return a set of lines that summarizes a variance-covariance matrix
    
    parameters:      
    Sigma: ndarray of shape (n,n)
        spd variance-covariance matrix
    n : int
        number of endogenous variables
    endogenous_variables: str list
        list of endogenous variables
    tag: str
        string providing the command to use to display full matrix
        
    returns:
    lines: str list
        string containing variance-covariance matrix summary
    """     

    lines = []
    dimension = min(Sigma.shape[0], 6)
    header_line = ' ' * 10
    for i in range(dimension):
        header_line += '{:>11}'.format(shorten_string(endogenous_variables[i], 10))
    header_line = string_line(header_line)
    lines.append(header_line)
    for i in range(dimension):
        current_line = '{:<10}'.format(shorten_string(endogenous_variables[i], 10))
        for j in range(dimension):
            current_line += ' ' + format_number(Sigma[i,j])
        if n > 6:
            current_line += ' ...'
        current_line = string_line(current_line)
        lines.append(current_line)
    if n > 6:
        current_line = '  ⋮               ⋮          ⋮          ⋮          ⋮          ⋮          ⋮   ⋱  '
        lines.append(current_line)
        lines.append(' ' * 80)
        lines.append(string_line('output is too long'))
        lines.append(string_line('use ' + tag + ' to obtain full view'))
    return lines
     
            
def forecast_evaluation_line(variable, value_1, value_2, value_3, value_4, value_5):
    
    """
    forecast_evaluation_line(variable, value_1, value_2, value_3, value_4, value_5)
    return a line with variable name and formatted numerical values
    
    parameters:      
    variable: str
        variable name
    value_1: float
        first value to display on line
    value_2: float
        second value to display on line
    value_3: float
        third value to display on line
    value_4: float
        fourth value to display on line
    value_5: float
        fifth value to display on line
        
    returns:
    line: str
        string containing formatted forecast evaluation criteria summary
    """           
    
    line = '{:<10}'.format(shorten_string(variable, 10))
    line += ' ' + format_number(value_1)
    line += ' ' + format_number(value_2)
    line += ' ' + format_number(value_3)
    line += ' ' + format_number(value_4)
    line += ' ' + format_number(value_5)
    line = string_line(line)
    return line

    
def forecast_evaluation_summary(log_score, joint_log_score, endogenous_variables, tag):
    
    """
    forecast_evaluation_summary(log_score, joint_log_score, endogenous_variables, tag)
    return a set of lines that summarizes Bayesian forecast evaluation criteria
    
    parameters:      
    log_score: ndarray of shape (forecast_periods,n)
        array of log score values
    joint_log_score: ndarray of shape (forecast_periods,)
        array of joint log score values
    endogenous_variables: str list
        list of endogenous variables
    tag: str
        string providing the command to use to display full matrix
        
    returns:
    line: str
        list of strings containing formatted forecast evaluation criteria summary
    """      
    
    lines = []
    periods = log_score.shape[0]
    dimension = min(periods, 6)
    header_line = ' ' * 10
    for i in range(dimension):
        header_line += '{:>11}'.format('(+' + str(i+1) + ')')
    if dimension == 6:
        header_line += ' ...'
    else:
        header_line += '      (all)'
    header_line = string_line(header_line)
    lines.append(header_line)
    for i in range(len(endogenous_variables)):
        line = '{:<10}'.format(shorten_string(endogenous_variables[i], 10))
        for j in range(dimension):
            value = log_score[j,i]
            line += ' ' + format_number(value)
        if dimension == 6:
            line += ' ...'
        else:
            value = joint_log_score[i]
            line += ' ' + format_number(value)
        line = string_line(line)
        lines.append(line)
    if dimension == 6:
        lines.append(' ' * 80)
        lines.append(string_line('output is too long'))
        lines.append(string_line('use ' + tag + ' to obtain full view'))  
    return lines


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
    line = '{:80}'.format(string)[:80]
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
    # first row: r2 and ssr
    left_element = '{:3}{:>35}'.format('R2:', format_number(r2))  
    right_element = '{:4}{:>34}'.format('ssr:', format_number(ssr))
    lines.append(left_element + '    ' + right_element)
    # second row: adjusted r2 and marginal likelihood
    left_element = '{:8}{:>30}'.format('adj. R2:', format_number(adj_r2))
    if m_y:
        right_element = '{:17}{:>21}'.format('log10 marg. lik.:', format_number(m_y))
    else:
        right_element = ' ' * 38    
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


