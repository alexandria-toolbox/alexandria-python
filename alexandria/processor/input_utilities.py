# imports
import numpy as np
import pandas as pd
import re
import os.path as osp
from datetime import datetime
import alexandria.console.console_utilities as cu


# module input_utilities
# a module containing methods to handle user inputs such as string, dates, lists, and so on
    
    
#---------------------------------------------------
# Methods
#---------------------------------------------------


def fix_string(string):
    
    """
    fix_string(string)
    converts tabs and multiple spaces to single spaces, removes leading and final spaces
    
    parameters:
    string : str
        string to fix
        
    returns:
    fixed_string : str
        string with spaces made regular
    """
    # convert tabs into spaces
    string = string.replace('\t', ' ')
    # convert multiple spaces into single spaces, trim initial and final spaces
    fixed_string = ' '.join(string.split())
    return fixed_string


def string_to_list(string):
    
    """
    string_to_list(string)
    converts string into list of string, with elements split at spaces
    
    parameters: 
    string: str or list of str
        string to split and distribute in list
        
    returns:
    string_list: list of str
         list obtained after splitting original string
    """
    
    # if string is actually already a list of strings, ignore and return
    if isinstance(string, list):
        string_list = string
    # else, item is a string: fix it in case it is incorrectly formated
    else:
        string = fix_string(string)
        # split it at spaces to convert to list
        string_list = string.split()
    return string_list


def list_to_string(string_list):
    
    """
    list_to_string(string_list)
    converts list of string into string, with elements separated with commas
    
    parameters: 
    string_list: str list
        list to burst into single string
        
    returns:
    string: str
         string obtained by bursting string list
    """
    
    # if list is actually a scalar, just convert to string
    if isinstance(string_list, (float, int)):
        string = str(string_list)
    # if list is numpy array, convert to string
    elif type(string_list) == np.ndarray:
        string = ", ".join([str(x) for x in string_list])
    # else, item is a list of string: convert to single string
    else:
        string = ", ".join(string_list)
    return string


def is_numeric(x):
    
    """
    is_numeric(x)
    checks whether input is of numeric type
    
    parameters:
    x: input, possibly of numeric type
        element to check for numeric type
        
    returns:
    numeric: bool
        True if element is numeric, False otherwise
    """
    
    type_list = [int, np.intc , np.int16, np.int32, np.int64, float, np.float_, np.float32, np.float64, \
                 complex, np.complex_, np.complex64, np.complex128]
    if type(x) in type_list and not np.isnan(x) and not np.isinf(x):
        numeric = True
    else:
        numeric = False
    return numeric


def is_integer(x):
    
    """
    is_integer(x)
    checks whether input is of integer type (possibly of type float but effectively integer)
    
    parameters:
    x: input, possibly of integer type
        element to check for integer type
        
    returns:
    integer: bool
        True if element is integer, False otherwise
    """
    
    type_list = [int, np.intc , np.int16, np.int32, np.int64]
    if (type(x) in type_list) or (is_numeric(x) and int(x) == x):
        integer = True
    else:
        integer = False
    return integer


def concatenate_dictionaries(dictionary_1, dictionary_2):
    
    """
    concatenate_dictionaries(dictionary_1, dictionary_2)
    concatenate two dictionaries that have different keys
    
    parameters: 
    dictionary_1 : dict
        first dictionary in concatenation
    dictionary_2 : dict
        second dictionary in concatenation
        
    returns:
    dictionary : dict
        concatenated dictionary
    """
    
    dictionary = {**dictionary_1, **dictionary_2}
    return dictionary


def get_timer():

    """
    get_timer()
    simple timer
    
    parameters:
    None

    returns:
    estimation_time : timestamp
        time at the moment the function runs
    """  
    
    estimation_time = datetime.now()
    return estimation_time


def check_file_path(path, file):
    
    """
    check_file_path(path, file)
    checks whether file exists at given path, and is of correct format (csv, xls, xlsx)
    
    parameters: 
    path : str
        path to folder containing data file
    file : str
        name of data file (with extension csv, xls or xlsx)
        
    returns:
    none
    """

    # check if path is valid
    if not osp.exists(path):
        raise TypeError('Path error. Specified path ' + path + ' does not exist.')
    # check if file exists
    file_path = osp.join(path, file)
    if not osp.exists(file_path):
        raise TypeError('File error. File ' + file_path + ' could not be found.')
    # check if data file is of correct format (excel or csv)
    file_type = file.split('.')[-1]
    if file_type not in ['csv', 'xls', 'xlsx']:
        raise TypeError('Type error for file ' + file_path + '. File must be of type csv, xls or xlsx.')         


def load_data(path, file):
    
    """
    load_data(path, file)
    loads data file of type csv, xls or xlsx into pandas dataframe
    
    parameters: 
    path : str
        path to folder containing data file
    file : str
        name of data file (with extension csv, xls or xlsx)
        
    returns:
    data : pandas dataframe
        dataframe containing loaded data
    """
    
    # make sure path and file strings are properly formatted
    path, file = fix_string(path), fix_string(file)
    # get format of data file
    file_type = file.split('.')[-1]
    # load file, depending on format
    file_path = osp.join(path, file)
    if file_type == 'csv':
        data = pd.read_csv(file_path, delimiter = ',', index_col = 0)
    elif file_type == 'xls':
        data = pd.read_excel(file_path, index_col = 0)
    elif file_type == 'xlsx':
        data = pd.read_excel(file_path, index_col = 0, engine = 'openpyxl')
    data.index = data.index.astype(str)
    return data       


def check_variables(data, file, variables, tag):
    
    """
    check_variables(data, file, variables, tag)
    checks whether specified variables exist in dataframe
    
    parameters:
    data : pandas dataframe
        dataframe to examine for data
    file : str
        name of data file (with extension csv, xls or xlsx)        
    variables : list of str
        list of variables to check in dataframe
    tag : str
        tag to apply in error message, if any (e.g. "Endogenous variables")
        
    returns:
    none
    """

    # obtain the list of variables in dataframe
    dataframe_variables = data.columns
    # check that variables exist in this list
    missing_variables = [variable for variable in variables if variable not in dataframe_variables]
    if missing_variables:
        raise TypeError('Data error for file ' + file + '. ' + tag + ' ' + ", ".join(missing_variables) + ' cannot be found.')


def check_dates(data, file, start_date, end_date):
    
    """
    check_dates(data, file, start_date, end_date)
    check whether start and end dates can be found in index of dataframe
    
    parameters:
    data : pandas dataframe
        dataframe to examine for start and end dates
    file : str
        name of data file (with extension csv, xls or xlsx)        
    start_date : str
        sample start date to search in dataframe index
    end_date : str
        sample end date to search in dataframe index
        
    returns:
    none     
    """
    
    # first obtain the list of dates (as strings)
    dates = data.index
    # check for start date
    if start_date not in dates:
        raise TypeError('Date error for file '  + file + '. Start date ' + start_date + ' cannot be found.')
    if end_date not in dates:
        raise TypeError('Date error for file '  + file + '. End date ' + end_date + ' cannot be found.')        


def fetch_data(data, file, start_date, end_date, variables, tag):
    
    """
    fetch_data(data, file, start_date, end_date, variables, tag)
    fetches variables from data at given sample dates
    
    parameters: 
    data : pandas dataframe
        dataframe containing loaded data
    file : str
        name of data file (with extension csv, xls or xlsx)         
    start_date : str
        sample start date to search in dataframe index
    end_date : str
        sample end date to search in dataframe index        
    variables : list of str
        list of endogenous variables to search in dataframe
    tag : str
        tag to apply in error message, if any (e.g. "Endogenous variables")
        
    returns:
    sample : ndarray
        ndarray containing fetched data   
    """
    
    # if variable list is not empty, recover sample for given variables and dates
    if variables:
        sample = data.loc[start_date:end_date, variables]
        # test for non-numerical values (strings), and if any, raise error
        if 'O' in sample.dtypes.tolist():
            raise TypeError('Data error for file ' + file + '. ' + tag + ' contains text entries, which are unhandled.')
        # test for NaNs, and if any, raise error
        elif sample.isnull().values.any():
            raise TypeError('Data error for file ' + file + '. ' + tag + ' contains NaN entries, which are unhandled.')
        # else, data is valid: convert to numpy array
        else:
            sample = sample.copy().values
    # if no variable, return empty list    
    else:
        sample = []
    return sample


def infer_date_format(frequency, file, start_date, end_date):

    """
    infer_date_format(frequency, file, start_date, end_date)
    infer date format for given data file, which can be either periods or timestamps
    
    parameters:
    frequency : int
        data frequency, as int between 1 and 6
    file : str
        name of data file (with extension csv, xls or xlsx)        
    start_date : str
        sample start date to search in dataframe index
    end_date : str
        sample end date to search in dataframe index
        
    returns:
    date_format : str
        date format, either 'periods' or 'timestamps'
    """

    if frequency == 1:
        date_format = _infer_undated(file, start_date, end_date)
    elif frequency == 2:
        date_format = _infer_annual(file, start_date, end_date)
    elif frequency == 3:
        date_format = _infer_quarterly(file, start_date, end_date)
    elif frequency == 4:
        date_format = _infer_monthly(file, start_date, end_date)
    elif frequency == 5:
        date_format = _infer_weekly(file, start_date, end_date)            
    elif frequency == 6:
        date_format = _infer_daily(file, start_date, end_date)
    return date_format


def _infer_undated(file, start_date, end_date):
    if not start_date.isdigit() or not end_date.isdigit():
        raise TypeError('Date error for file ' + file + '. Unrecognized format for cross-sectional/undated sample start and end dates. Should be integers.')
    else:
        date_format = 'periods'
    return date_format


def _infer_annual(file, start_date, end_date):
    if start_date.isdigit() and end_date.isdigit():
        date_format = 'periods'
    elif re.match('\d{4}[-]\d{1,2}[-]\d{1,2}', start_date) and re.match('\d{4}[-]\d{1,2}[-]\d{1,2}', end_date):
        date_format = 'timestamps'
    else:
        raise TypeError('Date error for file ' + file + '. Unrecognized format for annual sample start and end dates. Should be integers (e.g. 1990) or timestamp (e.g. 1990-12-31).')
    return date_format


def _infer_quarterly(file, start_date, end_date):
    if re.match('\d{4}Q[1-4]', start_date) and re.match('\d{4}Q[1-4]', end_date):
        date_format = 'periods'
    elif re.match('\d{4}[-]\d{1,2}[-]\d{1,2}', start_date) and re.match('\d{4}[-]\d{1,2}[-]\d{1,2}', end_date):
        date_format = 'timestamps'
    else:
        raise TypeError('Date error for file ' + file + '. Unrecognized format for quarterly sample start and end dates. Should be period (e.g. 1990Q1) or timestamp (e.g. 1990-03-31).')
    return date_format


def _infer_monthly(file, start_date, end_date):
    if re.match('\d{4}M(0?[1-9]|[1][0-2])$', start_date) and re.match('\d{4}M(0?[1-9]|[1][0-2])$', end_date):
        date_format = 'periods'
    elif re.match('\d{4}[-]\d{1,2}[-]\d{1,2}', start_date) and re.match('\d{4}[-]\d{1,2}[-]\d{1,2}', end_date):
        date_format = 'timestamps'
    else:
        raise TypeError('Date error for file ' + file + '. Unrecognized format for monthly sample start and end dates. Should be period (e.g. 1990M1) or timestamp (e.g. 1990-01-31).')
    return date_format


def _infer_weekly(file, start_date, end_date):
    if re.match('\d{4}W(0?[1-9]|[1-4][0-9]|[5][0-3])$', start_date) and re.match('\d{4}W(0?[1-9]|[1-4][0-9]|[5][0-3])$', end_date):
        date_format = 'periods'
    elif re.match('\d{4}[-]\d{1,2}[-]\d{1,2}', start_date) and re.match('\d{4}[-]\d{1,2}[-]\d{1,2}', end_date):
        date_format = 'timestamps'
    else:
        raise TypeError('Date error for file ' + file + '. Unrecognized format for weekly sample start and end dates. Should be period (e.g. 1990W1) or timestamp (e.g. 1990-01-05).')
    return date_format


def _infer_daily(file, start_date, end_date):
    if re.match('\d{4}D(0?0?[1-9]|0?[1-9]\d|[1-2]\d{2}|3[0-5][0-9]|36[0-6])$', start_date) and re.match('\d{4}D(0?0?[1-9]|0?[1-9]\d|[1-2]\d{2}|3[0-5][0-9]|36[0-6])$', end_date):
        date_format = 'periods'
    elif re.match('\d{4}[-]\d{1,2}[-]\d{1,2}', start_date) and re.match('\d{4}[-]\d{1,2}[-]\d{1,2}', end_date):
        date_format = 'timestamps'
    else:
        raise TypeError('Date error for file ' + file + '. Unrecognized format for daily sample start and end dates. Should be period (e.g. 1990D10) or timestamp (e.g. 1990-01-10).')     
    return date_format
        

def generate_dates(data, date_format, frequency, file, start_date, end_date):
    
    """
    generate_dates(data, date_format, frequency, file, start_date, end_date)
    generates date series, under the form of a pandas index
    
    parameters:
    data : pandas dataframe
        dataframe to extract dates from index
    date_format : str
        date format, either 'periods' or 'timestamps'
    frequency : int
        data frequency, as int between 1 and 6
    file : str
        name of data file (with extension csv, xls or xlsx)        
    start_date : str
        sample start date to search in dataframe index
    end_date : str
        sample end date to search in dataframe index
        
    returns:
    dates : datetime index
        index of Datetime entries
    """
    
    dates = data.loc[start_date:end_date].index
    if frequency == 1:
        dates = _generate_undated_dates(dates, date_format, file)
    elif frequency == 2:
        dates = _generate_annual_dates(dates, date_format, file)
    elif frequency == 3:
        dates = _generate_quarterly_dates(dates, date_format, file)
    elif frequency == 4:
        dates = _generate_monthly_dates(dates, date_format, file)
    elif frequency == 5:
        dates = _generate_weekly_dates(dates, date_format, file)            
    elif frequency == 6:
        dates = _generate_daily_dates(dates, date_format, file)
    return dates


def _generate_undated_dates(dates, date_format, file):
    if date_format == 'periods':
        try:
            dates = dates.astype(int)
        except:
            raise TypeError('Date error for file ' + file + '. Some sample periods seem to be improperly formatted. Please verify the dates in the data file.')
    else:
        raise TypeError('Date error for file ' + file + '. Date format should be periods, not timestamps.')
    return dates


def _generate_annual_dates(dates, date_format, file):
    try:
        if date_format == 'periods':        
            dates = pd.PeriodIndex(dates, freq='Y').to_timestamp().shift(0, freq = 'Y')
        elif date_format == 'timestamps':
            dates = pd.to_datetime(dates).shift(0, freq = 'Y')
    except:
        raise TypeError('Date error for file ' + file + '. Some sample periods seem to be improperly formatted. Please verify the dates in the data file.')
    return dates
   
    
def _generate_quarterly_dates(dates, date_format, file): 
    try:
        if date_format == 'periods':    
            dates = pd.PeriodIndex(dates, freq='Q').to_timestamp().shift(0, freq = 'Q')    
        elif date_format == 'timestamps':
            dates = pd.to_datetime(dates).shift(0, freq = 'Q')
    except:
        raise TypeError('Date error for file ' + file + '. Some sample periods seem to be improperly formatted. Please verify the dates in the data file.')        
    return dates
    

def _generate_monthly_dates(dates, date_format, file): 
    try:
        if date_format == 'periods':    
            dates = pd.to_datetime([date.replace('M', '-') for date in dates]).shift(0, freq = 'M')  
        elif date_format == 'timestamps':
            dates = pd.to_datetime(dates).shift(0, freq = 'M')
    except:
        raise TypeError('Date error for file ' + file + '. Some sample periods seem to be improperly formatted. Please verify the dates in the data file.')  
    return dates


def _generate_weekly_dates(dates, date_format, file): 
    try:
        if date_format == 'periods':
            years = [date.split('W')[0] for date in dates]
            weeks = [date.split('W')[1] for date in dates]
            dates = [pd.to_datetime(1000 * int(years[i]) + 10 * int(weeks[i]), \
                    format = "%Y%W%w") for i in range(len(years))]
            dates = pd.DataFrame(index = dates).index.shift(-1, freq = 'W-FRI')
        elif date_format == 'timestamps':
            dates = pd.to_datetime(dates).shift(0, freq = 'W-FRI')
    except:
        raise TypeError('Date error for file ' + file + '. Some sample periods seem to be improperly formatted. Please verify the dates in the data file.')  
    return dates


def _generate_daily_dates(dates, date_format, file):
    try:
        if date_format == 'periods':
            years = [date.split('D')[0] for date in dates]
            days = [date.split('D')[1] for date in dates]
            dates = [pd.to_datetime(1000 * int(years[i]) + int(days[i]), format = "%Y%j") \
                              for i in range(len(years))]
            dates = pd.DataFrame(index = dates).index 
        elif date_format == 'timestamps':
            dates = pd.to_datetime(dates)
    except:
        raise TypeError('Date error for file ' + file + '. Some sample periods seem to be improperly formatted. Please verify the dates in the data file.')  
    return dates        


def fetch_forecast_data(data, insample_data, variables, file, required, periods, tag):
    
    """
    fetch_forecast_data(data, insample_data, variables, file, required, periods, tag)
    fetches predictor variables from forecast data file
    
    parameters: 
    data : pandas dataframe
        dataframe containing loaded forecast data
    insample_data : ndarray
        ndarray containing in-sample counterparts of forecast variables
    variables : list of str
        list of variables to extract from dataframe
    file : str
        name of data file (with extension csv, xls or xlsx) 
    required : bool
        if True, checks that variables are provided in file
    periods : int, default = None
        number of forecast periods
    tag : str
        tag indicating the varables to be checked

    returns:
    sample_p : ndarray
        ndarray containing forecast data   
    """
    
    # find any missing variable
    missing_variables = [variable for variable in variables if variable not in data]
    # if some variables are missing
    if missing_variables:
        # if variables are required and no in-sample data is provided, raise error
        if required and not insample_data:
            raise TypeError('Data error for file ' + file + '. ' + tag + ' ' + ", ".join(missing_variables) + ' required to conduct forecast (or forecast evaluation), but cannot be found.')
        # if variables are required but some in-sample data is provided, replicate final in-sample value
        elif required and insample_data:
            sample_p = np.tile(insample_data[-1,:], (periods,1))
        # if not required, return empty list
        else:
            sample_p = []
    # if no variable is missing, recover prediction data
    else:
        # if variables is empty, return empty list
        if len(variables) == 0:
            sample_p = []
        # if there are some variables to fetch, get them
        else:
            sample_p = data[variables]
            # if too few periods of data are provided, raise error
            if sample_p.shape[0] < periods:
                raise TypeError('Data error for file ' + file + '. Forecasts must be conducted for ' + str(periods) + ' periods, but ' + tag + ' is provided for fewer periods.')
            # else, reduce data to number of periods
            else:
                sample_p = sample_p.iloc[:periods]
            # test for NaNs, and if any, raise error
            if sample_p.isnull().values.any():
                raise TypeError('Data error for file ' + file + '. ' + tag + ' contains NaN entries, which are unhandled.')
            # convert to numpy array
            sample_p = sample_p.copy().values
    return sample_p


def generate_forecast_dates(end_date, periods, frequency):
    
    """
    generate_forecast_dates(final_sample_date, periods, frequency)
    generates date series for forecasts, under the form of a pandas index, for a given number of periods
    
    parameters:
    end_date : timestamp
        final in-sample date
    periods : int
        number of out-of-sample periods
    frequency : int
        data frequency, as int between 1 and 6
        
    returns:
    forecast_dates : datetime index
        index of Datetime entries
    """

    # if frequency is undated, create a range from 1 to periods as forecast dates
    if frequency == 1:
        forecast_dates = pd.Index(np.arange(1, periods+1))
    # if frequency is annual, expand sample by years equal to periods
    elif frequency == 2:
        forecast_dates = pd.date_range(start = end_date, periods = periods+1, freq = 'Y')[1:]            
    # if frequency is quarterly, expand sample by quarters equal to periods
    elif frequency == 3:
        forecast_dates = pd.date_range(start = end_date, periods = periods+1, freq = 'Q')[1:]              
    # if frequency is monthly, expand sample by months equal to periods
    elif frequency == 4:
        forecast_dates = pd.date_range(start = end_date, periods = periods+1, freq = 'M')[1:]
    # if frequency is weekly, expand sample by weeks equal to periods         
    elif frequency == 5:
        forecast_dates = pd.date_range(start = end_date, periods = periods+1, freq = 'W-FRI')[1:]
    # if frequency is daily, expand sample by days equal to periods
    elif frequency == 6:
        forecast_dates = pd.date_range(start = end_date, periods = periods+1, freq = 'B')[1:]       
    return forecast_dates


def check_coefficients_table(data, endogenous_variables, exogenous_variables, \
                             lags, constant, trend, quadratic_trend, file):
    
    """
    check_coefficients_table(data, endogenous_variables, exogenous_variables, \
                             lags, constant, trend, quadratic_trend, file)
    checks whether constrained coefficient table is of valid format
    
    parameters:
    data : pandas dataframe
        dataframe containing constrained coefficient information
    endogenous_variables : list of strings
        list containing the names of endogenous variables
    exogenous_variables : list of strings
        list containing the names of exogenous variables
    lags : int
        number of lags for the VAR model
    constant : bool
        set to True if a constant is included in the model
    trend : bool
        set to True if a linear trend is included in the model        
    quadratic_trend : bool
        set to True if a quadratic trend is included in the model
    file : str
        name of data file (with extension csv, xls or xlsx)

    returns:
    none
    """
    
    data = data.reset_index()
    columns = data.columns.to_list()
    if columns != ['variable', 'responding_to', 'lag', 'mean', 'variance']:
        raise TypeError('Data error for file '  + file + '. Column names don\'t match the required pattern.')
    types = data.dtypes
    if types['lag'] != 'int64':
        raise TypeError('Data error for file '  + file + '. Some entries in column ' \
        + '\'lag\' are not integers.')        
    if types['mean'] not in ['int64','float64']:
        raise TypeError('Data error for file '  + file + '. Some entries in column ' \
        + '\'mean\' are not numeric.')    
    if types['variance'] not in ['int64','float64']:
        raise TypeError('Data error for file '  + file + '. Some entries in column ' \
        + '\'variance\' are not numeric.') 
    automated_variables = ['constant', 'trend', 'quadratic_trend']
    variables = endogenous_variables + exogenous_variables + automated_variables
    all_exogenous_variables = exogenous_variables + automated_variables
    all_lags = [lag for lag in np.arange(-1,lags+1).tolist()]
    for i in range(data.shape[0]):
        variable_value = data.iloc[i].loc['variable']
        responding_value = data.iloc[i].loc['responding_to']
        lag_value = data.iloc[i].loc['lag']
        mean_value = data.iloc[i].loc['mean']
        var_value = data.iloc[i].loc['variance']
        if variable_value not in endogenous_variables:
            raise TypeError('Data error for file '  + file + '. Entry for column ' \
            + '\'variable\', row ' + str(i+1) + ', does not correspond to an endogenous variable.')
        if responding_value not in variables:
            raise TypeError('Data error for file '  + file + '. Entry for column \'responding_to\', row '
            + str(i+1) + ', does not correspond to any of the model variables.')
        if responding_value == 'constant' and not constant:
            raise TypeError('Data error for file '  + file + '. Entry for column \'responding_to\', row '
            + str(i+1) + ', is \'constant\', but constant is not activated.') 
        if responding_value == 'trend' and not trend:
            raise TypeError('Data error for file '  + file + '. Entry for column \'responding_to\', row '
            + str(i+1) + ', is \'trend\', but trend is not activated.')  
        if responding_value == 'quadratic_trend' and not quadratic_trend:
            raise TypeError('Data error for file '  + file + '. Entry for column \'responding_to\', row '
            + str(i+1) + ', is \'quadratic_trend\', but quadratic trend is not activated.') 
        if not is_numeric(lag_value):
            raise TypeError('Data error for file '  + file + '. Entry for column \'lag\', row '
            + str(i+1) + ' is not numeric.')
        if responding_value not in all_exogenous_variables and lag_value not in all_lags:
            raise TypeError('Data error for file '  + file + '. Entry for column \'lag\', row '
            + str(i+1) + ' should be an integer in the range of specified lags.')
        if not is_numeric(mean_value):
            raise TypeError('Data error for file '  + file + '. Entry for column \'mean\', row '
            + str(i+1) + ' is NaN of inf.')
        if not is_numeric(var_value):
            raise TypeError('Data error for file '  + file + '. Entry for column \'variance\', row '
            + str(i+1) + ' is NaN of inf.')
        if var_value <= 0:
            raise TypeError('Data error for file '  + file + '. Entry for column \'variance\', row '
            + str(i+1) + ' should be strictly positive.')


def get_constrained_coefficients_table(data, endogenous_variables, exogenous_variables):
    
    """
    get_constrained_coefficients_table(data, endogenous_variables, exogenous_variables)
    recover constrained coefficient table in numeric format
    
    parameters:
    data : pandas dataframe
        dataframe containing constrained coefficient information
    endogenous_variables : list of strings
        list containing the names of endogenous variables
    exogenous_variables : list of strings
        list containing the names of exogenous variables

    returns:
    constrained_coefficients_table : ndarray
        ndarray containing numeric values only for constrained coefficients prior
    """
    
    automated_variables = ['constant', 'trend', 'quadratic_trend']
    data = data.reset_index()
    rows = data.shape[0]
    temp = np.zeros((rows,5))
    for i in range(rows):
        variable_value = data.iloc[i].loc['variable']
        responding_value = data.iloc[i].loc['responding_to']
        lag_value = data.iloc[i].loc['lag']
        mean_value = data.iloc[i].loc['mean']
        var_value = data.iloc[i].loc['variance']    
        temp[i,0] = endogenous_variables.index(variable_value) + 1
        if responding_value == 'constant':
            temp[i,1] = 0.1
        elif responding_value == 'trend':
            temp[i,1] = 0.2
        elif responding_value == 'quadratic_trend':
            temp[i,1] = 0.3
        elif responding_value in endogenous_variables:
            temp[i,1] = endogenous_variables.index(responding_value) + 1
        elif responding_value in exogenous_variables:
            temp[i,1] = -(exogenous_variables.index(responding_value) + 1)
        if responding_value not in automated_variables:
            temp[i,2] = lag_value
        temp[i,3] = mean_value
        temp[i,4] = var_value
    constrained_coefficients_table = temp
    return constrained_coefficients_table
        
    
def check_long_run_table(data, endogenous_variables, file):
    
    """
    check_long_run_table(data, endogenous_variables, file)
    checks whether long run prior table is of valid format
    
    parameters:
    data : pandas dataframe
        dataframe containing constrained coefficient information
    endogenous_variables : list of strings
        list containing the names of endogenous variables
    file : str
        name of data file (with extension csv, xls or xlsx)

    returns:
    none
    """

    columns = data.columns.to_list()
    if columns != endogenous_variables:
        raise TypeError('Data error for file '  + file + '. Column names don\'t match the set of endogenous variables.')
    rows = data.index.to_list()
    if rows != endogenous_variables:
        raise TypeError('Data error for file '  + file + '. Index names don\'t match the set of endogenous variables.')
    types = data.dtypes.to_frame().rename(columns={0:'type'})
    numeric_types = ['int64','float64']
    for variable in types.index:
        if types.loc[variable,'type'] not in numeric_types:
            raise TypeError('Data error for file '  + file + '. Some entries in column ' + variable + ' are not numeric.')   
    values = data.values
    dimension = values.shape[0]
    for i in range(dimension):
        for j in range(dimension):
            if not is_numeric(values[i,j]):
                raise TypeError('Data error for file '  + file + '. Entry in row ' + str(i+1) + ', column ' + str(j+1) + ' is not numeric, NaN of inf.')
        
        
def check_condition_table(data, endogenous_variables, periods, file):
    
    """
    check_condition_table(data, endogenous_variables, periods, file)
    checks whether condition table is of valid format
    
    parameters:
    data : pandas dataframe
        dataframe containing conditional forecast information
    endogenous_variables : list of strings
        list containing the names of endogenous variables
    periods : int
        number of forecast periods        
    file : str
        name of data file (with extension csv, xls or xlsx)

    returns:
    none
    """      
    
    data = data.reset_index()
    number_endogenous = len(endogenous_variables)
    shock_list = ['shock' + str(i+1) for i in range(number_endogenous)]
    columns = data.columns.to_list()
    expected_columns = ['variable', 'period', 'mean', 'variance'] + shock_list
    if columns != expected_columns:
        raise TypeError('Data error for file '  + file + '. Column names don\'t match the required pattern.')
    types = data.dtypes
    if types['period'] != 'int64':
        raise TypeError('Data error for file '  + file + '. Some entries in column ' \
        + '\'period\' are not integers.')        
    if types['mean'] not in ['int64','float64']:
        raise TypeError('Data error for file '  + file + '. Some entries in column ' \
        + '\'mean\' are not numeric.')    
    if types['variance'] not in ['int64','float64']:
        raise TypeError('Data error for file '  + file + '. Some entries in column ' \
        + '\'variance\' are not numeric.') 
    for i in range(number_endogenous):
        if types['shock' + str(i+1)] not in ['int64','float64']:
            raise TypeError('Data error for file '  + file + '. Entry in row 1, column "shock' + str(i+1) + '" should be 1 or empty.') 
    rows = data.shape[0]
    for i in range(rows):
        variable = data.iloc[i].loc['variable']
        period = data.iloc[i].loc['period']
        mean = data.iloc[i].loc['mean']
        variance = data.iloc[i].loc['variance']          
        if variable not in endogenous_variables:
            raise TypeError('Data error for file '  + file + '. Entry in row ' + str(i+1) + ', column "variable" does not correspond to one of the model endogenous variables.')
        if not is_numeric(period):
            raise TypeError('Data error for file '  + file + '. Entry in row ' + str(i+1) + ', column "period" is not numeric, NaN of inf.')
        if period > periods:
            raise TypeError('Data error for file '  + file + '. Entry in row ' + str(i+1) + ', column "period" is larger than the number of forecast periods.')
        if period <= 0:
            raise TypeError('Data error for file '  + file + '. Entry in row ' + str(i+1) + ', column "period" should be a positive integer.')
        if not is_numeric(mean):
            raise TypeError('Data error for file '  + file + '. Entry in row ' + str(i+1) + ', column "mean" is not numeric, NaN of inf.')
        if not is_numeric(variance):
            raise TypeError('Data error for file '  + file + '. Entry in row ' + str(i+1) + ', column "variance" is not numeric, NaN of inf.')
        if variance < 0:
            raise TypeError('Data error for file '  + file + '. Entry in row ' + str(i+1) + ', column "variance" should be non-negative.')  
        if i == 0:
            for j in range(number_endogenous):
                shock = data.iloc[i].loc['shock' + str(j+1)]
                if shock not in [0,1]:
                    raise TypeError('Data error for file '  + file + '. Entry in row 1, column "shock' + str(j+1) + '" should be 0 or 1.') 
            

def get_condition_table(data, endogenous_variables):
    
    """
    get_condition_table(data, endogenous_variables)
    recover condition table in numeric format
    
    parameters:
    data : pandas dataframe
        dataframe containing constrained coefficient information
    endogenous_variables : list of strings
        list containing the names of endogenous variables

    returns:
    condition_table : ndarray
        ndarray containing numeric values for conditions
    shock_table : ndarray
        ndarray containing numeric values for shocks
    """       
    
    data = data.reset_index()
    rows = data.shape[0]
    condition_table = np.zeros((rows,4))
    for i in range(rows):
        variable = data.loc[i,'variable']
        condition_table[i,0] = endogenous_variables.index(variable) + 1
        condition_table[i,1] = data.loc[i,'period']
        condition_table[i,2] = data.loc[i,'mean']
        if data.loc[i,'variance'] == 0:
            condition_table[i,3] = 1e-16
        else:
            condition_table[i,3] = data.loc[i,'variance']
    shocks = data.iloc[0,4:].fillna(0)
    shock_table = shocks.values
    return condition_table, shock_table


def get_raw_sample_dates(path, file, start_date, end_date):
    
    """
    get_raw_sample_dates(path, file, start_date, end_date)
    get sample dates, in raw format (as in data file, without any convesion to datetime)
    
    parameters:
    path : str
        path to folder containing data file
    file : str
        name of data file (with extension csv, xls or xlsx)
    start_date : str
        sample start date to search in dataframe index
    end_date : str
        sample end date to search in dataframe index

    returns:
    raw_dates : datetime index
        index of Datetime entries
    """  
    
    data = load_data(path, file)
    raw_dates = data.loc[start_date:end_date].index
    return raw_dates

        
def check_restriction_table(data, raw_dates, endogenous_variables, proxy_variables, var_type, irf_periods, file):

    """
    check_restriction_table(data, raw_dates, endogenous_variables, irf_periods, file)
    checks whether restriction table is of valid format
    
    parameters:
    data : pandas dataframe
        dataframe containing constrained coefficient information
    raw_dates : datetime index
        index of Datetime entries
    endogenous_variables : list of strings
        list containing the names of endogenous variables
    proxy_variables : list of strings
        list containing the names of proxy variables        
    var_type : int
        type of VAR model
    irf_periods : int
        number of IRF periods    
    file : str
        name of data file (with extension csv, xls or xlsx)

    returns:
    none
    """  

    data = data.reset_index()
    number_endogenous = len(endogenous_variables)
    number_proxys = len(proxy_variables)
    shock_list = ['shock' + str(i+1) for i in range(number_endogenous)]
    columns = data.columns.to_list()
    expected_columns = ['type', 'variable', 'period'] + shock_list
    if columns != expected_columns:
        raise TypeError('Data error for file '  + file + '. Column names don\'t match the required pattern.')
    data = data.astype({'period': 'string'})
    rows = data.shape[0]
    for i in range(rows):
        restriction_type = data.iloc[i].loc['type']
        variable = data.iloc[i].loc['variable']    
        period = data.iloc[i].loc['period']
        if restriction_type not in ['sign', 'zero', 'shock', 'historical', 'covariance']:
            raise TypeError('Data error for file '  + file + '. Entry in row ' + str(i+1) + ', column "type" does not correspond to one of the allowed restriction types.')
        if restriction_type == 'sign' or restriction_type == 'zero':
            if not period.isdigit():
                raise TypeError('Data error for file '  + file + '. Entry in row ' + str(i+1) + ', column "period" should be an integer.')
            elif int(period) < 0 or int(period) > irf_periods:
                raise TypeError('Data error for file '  + file + '. Entry in row ' + str(i+1) + ', column "period" should be an integer between 0 and IRF periods.')
        if (restriction_type == 'shock' or restriction_type == 'historical') and period not in raw_dates:
            raise TypeError('Data error for file '  + file + '. Entry in row ' + str(i+1) + ', column "period" should be a sample date.')
        if restriction_type != 'shock' and restriction_type != 'covariance' and variable not in endogenous_variables:
            raise TypeError('Data error for file '  + file + '. Entry in row ' + str(i+1) + ', column "variable" does not correspond to one of the model endogenous variables.')
        for j in range(number_endogenous):
            shock = str(data.iloc[i,3+j])
            try:
                shock = float(shock)
            except:
                raise TypeError('Data error for file '  + file + '. Entry in row ' + str(i+1) + ', column "shock' + str(j+1) + '" is not numeric.')
            if shock not in [-1, 0, 1]:
                raise TypeError('Data error for file '  + file + '. Entry in row ' + str(i+1) + ', column "shock' + str(j+1) + '" is not -1, 0 or 1.')
        if len(np.nonzero(data.iloc[i,3:].values)) not in [1,2]:
            raise TypeError('Data error for file '  + file + '. Ill-defined restrictions in row ' + str(i+1) + ', shock columns must contain either 1 or 2 non-zero coefficients.')
        if restriction_type == 'covariance' and var_type != 7:
            raise TypeError('Data error for file '  + file + '. Covariance restriction found in row ' + str(i+1) + ', but model is not proxy-SVAR.')
        elif restriction_type == 'covariance' and var_type == 7:
            for j in range(number_endogenous-number_proxys):
                shock = data.iloc[i,3+j]
                if shock != 0:
                    raise TypeError('Data error for file '  + file + '. Entry in row ' + str(i+1) + ', column "shock' + str(j+1) + '" has covariance restriction while not correlated with proxys.')
                

def get_restriction_table(data, raw_dates, endogenous_variables, proxy_variables):
    
    """
    get_restriction_table(data, raw_dates, endogenous_variables)
    recover restriction table in numeric format
    
    parameters:
    data : pandas dataframe
        dataframe containing constrained coefficient information
    raw_dates : datetime index
        index of Datetime entries        
    endogenous_variables : list of strings
        list containing the names of endogenous variables
    proxy_variables : list of strings
        list containing the names of proxy variables         

    returns:
    restriction_table : ndarray
        ndarray containing numeric values for restrictions
    """   
    
    data = data.reset_index()
    rows = data.shape[0]
    columns = data.shape[1]
    number_endogenous = len(endogenous_variables)
    restriction_types = ['zero', 'sign', 'shock', 'historical', 'covariance']
    restriction_table = np.zeros((rows,columns))
    for i in range(rows):
        restriction_type = data.loc[i,'type']
        period = data.loc[i,'period']
        variable = data.loc[i,'variable']
        restriction_table[i,0] = restriction_types.index(restriction_type) + 1
        if restriction_type == 'sign' or restriction_type == 'zero' or restriction_type == 'historical':
            restriction_table[i,1] = endogenous_variables.index(variable) + 1
        elif restriction_type == 'covariance':
            restriction_table[i,1] = proxy_variables.index(variable) + 1     
        if restriction_type == 'sign' or restriction_type == 'zero':
            restriction_table[i,2] = int(period)
        elif restriction_type == 'shock' or restriction_type == 'historical':
            restriction_table[i,2] = raw_dates.tolist().index(period) + 1
        for j in range(3,3+number_endogenous):
            coefficient = data.iloc[i,j]
            restriction_table[i,j] = coefficient
    return restriction_table
    

def identify_model(model):
    
    """
    identify_model(model)
    get model and model type
    
    parameters:
    model : class
        class from which model must be extracted

    returns:
    model_name : string
        model name
    model_class : int
        general model class (linear regression, VAR, ...)
    model_type : int
        specific model type (maximum likelihood regression, simple Bayesian regression, ...)
    """

    class_name = model.__class__.__name__
    if class_name == 'MaximumLikelihoodRegression':
        model_name = 'Maximum Likelihood Regression'
        model_class = 1
        model_type = 1
    elif class_name == 'SimpleBayesianRegression':
        model_name = 'Simple Bayesian Regression'
        model_class = 1
        model_type = 2
    elif class_name == 'HierarchicalBayesianRegression':
        model_name = 'Hierarchical Bayesian Regression'
        model_class = 1
        model_type = 3
    elif class_name == 'IndependentBayesianRegression':
        model_name = 'Independent Bayesian Regression'
        model_class = 1
        model_type = 4
    elif class_name == 'HeteroscedasticBayesianRegression':
        model_name = 'Heteroscedastic Bayesian Regression'
        model_class = 1
        model_type = 5
    elif class_name == 'AutocorrelatedBayesianRegression':
        model_name = 'Autocorrelated Bayesian Regression'
        model_class = 1
        model_type = 6  
    elif class_name == 'MaximumLikelihoodVar':
        model_name = 'Maximum Likelihood Var'
        model_class = 2
        model_type = 1        
    elif class_name == 'MinnesotaBayesianVar':
        model_name = 'Minnesota Bayesian Var'
        model_class = 2
        model_type = 2
    elif class_name == 'NormalWishartBayesianVar':
        model_name = 'Normal-Wishart Bayesian Var'
        model_class = 2
        model_type = 3 
    elif class_name == 'IndependentBayesianVar':
        model_name = 'Independent Bayesian Var'
        model_class = 2
        model_type = 4 
    elif class_name == 'DummyObservationBayesianVar':
        model_name = 'Dummy Observation Bayesian Var'
        model_class = 2
        model_type = 5 
    elif class_name == 'LargeBayesianVar':
        model_name = 'Large Bayesian Var'
        model_class = 2
        model_type = 6 
    elif class_name == 'BayesianProxySvar':
        model_name = 'Bayesian Proxy Svar'
        model_class = 2
        model_type = 7       
    elif class_name == 'VectorErrorCorrection':
        model_name = 'Bayesian Vector Error Correction'
        model_class = 3
        model_type = 1 
    elif class_name == 'VectorAutoregressiveMovingAverage':
        model_name = 'Bayesian Vector Autoregressive Moving Average'
        model_class = 3
        model_type = 2
    return model_name, model_class, model_type



