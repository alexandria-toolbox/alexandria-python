# imports
from os import mkdir
from os.path import isdir, join
from shutil import rmtree
import alexandria.console.console_utilities as cu


class Results(object):
    
    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass
    
    
    def create_result_file(self, project_path, save_results):
        # if save_results is activated
        if save_results:
            # create path to results folder
            results_folder_path = join(project_path, 'results')
            results_file_path = join(results_folder_path, 'results.txt')
            # if results folder already exists, delete it
            if isdir(results_folder_path):
                rmtree(results_folder_path, ignore_errors = True)
            # create results folder
            mkdir(results_folder_path)
            # get Alexandria header
            header = cu.alexandria_header()
            # write header in results file
            cu.write_string_list(header, results_file_path)


    #---------------------------------------------------
    # Methods (Access = private)
    #---------------------------------------------------
    
    
    def _print_alexandria_header(self):
        # get Alexandria header
        header = cu.alexandria_header()
        # print header
        cu.print_string_list(header)    
    
    
    def _print_start_message(self):
        cu.print_message('Starting estimation of your model...')
        cu.print_message(' ')
        
        
    def _print_completion_message(self):
        if self.progress_bar:
            cu.print_message(' ')
        cu.print_message('Estimation completed successfully.')
        cu.print_message(' ')  
        cu.print_message(' ')
    
        
    def _print_and_save_summary(self):
        # display result summary on console
        cu.print_string_list(self.summary)
        # if save_results is activated, save summary in file
        if self.save_results:
            results_file_path = join(self.project_path, 'results', 'results.txt')
            cu.write_string_list(self.summary, results_file_path)
            
    
    def _add_settings_header(self):
        # Alexandria header
        lines = cu.alexandria_header()
        lines.append('Estimation date:  ' + self.estimation_start.strftime('%Y-%m-%d %H:%M:%S'))
        lines.append(' ')
        lines.append(' ')
        self.settings += lines 
    
    
    def _add_tab_1_settings(self):
        # recover tab 1 elements
        endogenous = ", ".join(self.endogenous)  
        exogenous = ", ".join(self.exogenous)  
        if self.frequency == 1:
            frequency = 'cross-sectional/undated'
        elif self.frequency == 2:
            frequency = 'annual'            
        elif self.frequency == 3:
            frequency = 'quarterly'   
        elif self.frequency == 4:
            frequency = 'monthly'   
        elif self.frequency == 5:
            frequency = 'weekly'   
        elif self.frequency == 6:
            frequency = 'daily'        
        sample = self.start_date + ' ' + self.end_date
        project_path = self.project_path
        data_file = self.data_file
        progress_bar = self.progress_bar
        create_graphics = self.create_graphics
        save_results = self.save_results
        lines = cu.tab_1_settings('linear regression', endogenous, exogenous, \
                                  frequency, sample, project_path, data_file, \
                                  progress_bar, create_graphics, save_results)
        self.settings += lines
    
    
    def _save_settings(self):
        # if save_results is activated, save settings in file
        if self.save_results:
            settings_file_path = join(self.project_path, 'results', 'settings.txt')
            cu.write_string_list(self.settings, settings_file_path)    
    
    
