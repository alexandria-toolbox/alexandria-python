# imports
from os import mkdir
from os.path import isdir, join
from shutil import rmtree


class Graphics(object):
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass


    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------
    
    
    def _delete_graphics_folder(self):
        # create path to graphics folder
        graphics_folder_path = join(self.project_path, 'graphics')
        # if graphics folder already exists, delete it
        if isdir(graphics_folder_path):
            rmtree(graphics_folder_path, ignore_errors = True)
        # create results folder
        mkdir(graphics_folder_path)   