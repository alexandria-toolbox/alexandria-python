# imports
import numpy as np
import pandas as pd
import re
import os.path as osp


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
    check_variables1(data, file, variables, tag)
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
        # test for NaNs, and if any, raise error
        if sample.isnull().values.any():
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


