# imports
import numpy as np
import pandas as pd
from os.path import join, dirname


class DataSets(object):
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass

    
    def load_taylor_table(self):
        
        """
        load_taylor_table()
        load the Taylor dataset as a Pandas dataframe
        
        parameters:
        none
        
        returns:
        data : Pandas dataframe
            dataframe containing the Taylor dataset
        """
            
        file_path = self.__get_file_path('taylor')
        data = pd.read_csv(file_path, delimiter = ',', index_col = 0)
        data.index = pd.to_datetime(data.index)
        return data
        
    
    def load_taylor(self):
        
        """
        load_taylor()
        load the raw Taylor dataset as a Numpy ndarray
        
        parameters:
        none
        
        returns:
        data : Numpy ndarray
            array containing the raw data for the Taylor dataset
        """
        
        dataframe = self.load_taylor_table()
        data = dataframe.values
        return data
    
    
    def load_islm_table(self):
        
        """
        load_islm_table()
        load the Euro Area IS-LM dataset as a Pandas dataframe
        
        parameters:
        none
        
        returns:
        data : Pandas dataframe
            dataframe containing the IS-LM dataset
        """
            
        file_path = self.__get_file_path('islm')
        data = pd.read_csv(file_path, delimiter = ',', index_col = 0)
        data.index = pd.to_datetime(data.index)
        return data
        
    
    def load_islm(self):
        
        """
        load_islm()
        load the raw Euro Area IS-LM dataset as a Numpy ndarray
        
        parameters:
        none
        
        returns:
        data : Numpy ndarray
            array containing the raw data for the IS-LM dataset
        """
        
        dataframe = self.load_islm_table()
        data = dataframe.values
        return data
    

    #---------------------------------------------------
    # Methods (Access = private)
    #--------------------------------------------------- 

    
    def __get_file_path(self, file_name):
        dataset_folder_path = dirname(__file__)
        file_path = join(dataset_folder_path, file_name + '.csv')
        return file_path