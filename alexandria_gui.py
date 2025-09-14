#---------------------------------------------------
# Imports
#--------------------------------------------------- 

import IPython
IPython.get_ipython().run_line_magic('reset', '-sf')   
import sys
import warnings
import matplotlib.pyplot as plt
import alexandria.console.console_utilities as cu
from alexandria.interface.graphical_user_interface import GraphicalUserInterface


#---------------------------------------------------
# Graphical User Interface
#--------------------------------------------------- 


# clear workspace and console
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
    
# else, if model is vector autoregression, import main code for vector autoregression, run it, and return model
elif model == 2:
    from alexandria.vector_autoregression.main import vector_autoregression_main_code
    var = vector_autoregression_main_code(user_inputs)

# else, if model is vec/varma, import main code for vector autoregression extension, run it, and return model
elif model == 3:
    from alexandria.vec_varma.main import vec_varma_main_code
    model = vec_varma_main_code(user_inputs)









