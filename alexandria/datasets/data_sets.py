# imports
import numpy as np
import pandas as pd
from os.path import join, dirname


# module data_sets
# a module containing methods to load datasets
    
    
#---------------------------------------------------
# Methods
#---------------------------------------------------


def load_taylor_table():
    
    """
    load_taylor_table()
    load the Taylor dataset as a Pandas dataframe
    
    parameters:
    none
    
    returns:
    data : Pandas dataframe
        dataframe containing the Taylor dataset
    """
        
    file_path = _get_file_path('taylor')
    data = pd.read_csv(file_path, delimiter = ',', index_col = 0)
    data.index = pd.to_datetime(data.index)
    return data
    

def load_taylor():
    
    """
    load_taylor()
    load the raw Taylor dataset as a Numpy ndarray
    
    parameters:
    none
    
    returns:
    data : Numpy ndarray
        array containing the raw data for the Taylor dataset
    """
    
    dataframe = load_taylor_table()
    data = dataframe.values
    return data

    
def _get_file_path(file_name):
    dataset_folder_path = dirname(__file__)
    file_path = join(dataset_folder_path, file_name + '.csv')
    return file_path