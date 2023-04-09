#---------------------------------------------------
# Imports
#--------------------------------------------------- 


import sys
import IPython
import warnings
import matplotlib.pyplot as plt
import alexandria.console.console_utilities as cu
from alexandria.interface.graphical_user_interface import GraphicalUserInterface


#---------------------------------------------------
# Graphical User Interface
#--------------------------------------------------- 


# clear workspace and console
IPython.get_ipython().magic('reset -sf')
cu.clear_console()
warnings.filterwarnings('ignore')
plt.close('all')

# create and run graphical user interface
gui = GraphicalUserInterface()

# if user closed window manually without validating interface, terminate application
if gui.user_interrupt:
    sys.exit(0)
    
# recover user inputs
user_inputs = gui.user_inputs


#---------------------------------------------------
# Main code
#--------------------------------------------------- 


model = user_inputs['tab_1']['model']


# if model is linear regression, import main code for linear regression, run it, and return model
if model == 1:
    from alexandria.linear_regression.main import linear_regression_main_code
    lr = linear_regression_main_code(user_inputs)
    
