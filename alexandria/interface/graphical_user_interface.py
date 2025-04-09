# imports
import sys
import json
from platform import system
from os.path import isfile, join, dirname
from PyQt5.QtWidgets import QMainWindow, QApplication, QDesktopWidget, QPushButton
from PyQt5.QtGui import QFont
from alexandria.interface.default_input_interface import DefaultInputInterface
from alexandria.interface.tab1_interface import Tab1Interface
from alexandria.interface.tab2_regression_interface import Tab2RegressionInterface
from alexandria.interface.tab2_vector_autoregression_interface import Tab2VectorAutoregressionInterface
from alexandria.interface.tab3_interface import Tab3Interface
from alexandria.interface.tab4_interface import Tab4Interface
from alexandria.interface.tab5_interface import Tab5Interface



class GraphicalUserInterface(QMainWindow, DefaultInputInterface, Tab1Interface, \
                             Tab2RegressionInterface, Tab2VectorAutoregressionInterface, \
                             Tab3Interface, Tab4Interface, Tab5Interface):


    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------    
    

    def __init__(self, view_graphics = False):
        # save graphics view as attribute
        self.view_graphics = view_graphics
        # initiate GUI application
        app = QApplication(sys.argv)
        # initiate dictionary of user inputs
        self.initiate_inputs()
        # generate overall interface and tabs
        self.create_interface_and_tabs()
        # generate elements of tab 1
        self.create_tab_1()
        # generate elements of tab 2
        self.create_tab_2()
        # generate elements of tab 3
        self.create_tab_3()
        # generate elements of tab 4
        self.create_tab_4()
        # generate elements of tab 5
        self.create_tab_5()
        # check if GUI is switched back to tab 4
        self.switch_tab_4()
        # display GUI and start event loop
        self.show()
        app.exec_()
        
        
    def initiate_inputs(self):
        # get path to interface folder
        self.interface_path = dirname(__file__)
        # if previous user inputs have been saved, load them
        if isfile(join(self.interface_path, "user_inputs.json")):
            user_inputs = json.load(open(join(self.interface_path, "user_inputs.json")))
            self.user_inputs = user_inputs
        # otherwise, implement default inputs
        else:
            self.create_default_inputs()        
        
        
    def create_interface_and_tabs(self):
        # define colors, depending on operating system
        if system() == 'Windows':
            self.background_color = (255, 255, 196)
            self.tab_color = (245, 237, 184)
            self.backtabs_color = (242, 222, 140)            
        else:            
            self.background_color = (255, 247, 224)
            self.tab_color = (225, 217, 184)
            self.backtabs_color = (200, 180, 130)            
        # get screen resolution
        screen_dimensions = QDesktopWidget().screenGeometry(-1)
        screen_width = screen_dimensions.width()
        screen_heigth = screen_dimensions.height()
        # calculate interface position and create main window
        interface_width = 1000
        interface_heigth = 650
        left_shift = (screen_width - interface_width) / 2
        up_shift = (screen_heigth - interface_heigth) / 2
        # main window
        super().__init__()
        self.setGeometry(left_shift, up_shift, interface_width, interface_heigth)      
        self.setWindowTitle("Alexandria")
        self.setStyleSheet("background:rgb" + str(self.background_color) + ";")
        # first tab button
        self.tab_pbt1 = QPushButton(self)
        self.tab_pbt1.move(0, 0)
        self.tab_pbt1.resize(200, 40)
        self.tab_pbt1.setText('Models')
        self.tab_pbt1.setFont(QFont('ZapfDingbats', 16))
        self.tab_pbt1.setStyleSheet("background:rgb" + str(self.tab_color) + ";")
        self.tab_pbt1.clicked.connect(self.cb_tab_pbt1)
        # second tab button
        self.tab_pbt2 = QPushButton(self)
        self.tab_pbt2.move(200, 0)
        self.tab_pbt2.resize(200, 40)
        self.tab_pbt2.setText('Specifications')
        self.tab_pbt2.setFont(QFont('ZapfDingbats', 16))
        self.tab_pbt2.setStyleSheet("background:rgb" + str(self.backtabs_color) + ";")
        self.tab_pbt2.clicked.connect(self.cb_tab_pbt2)
        # third tab button
        self.tab_pbt3 = QPushButton(self)
        self.tab_pbt3.move(400, 0)
        self.tab_pbt3.resize(200, 40)
        self.tab_pbt3.setText('Applications')
        self.tab_pbt3.setFont(QFont('ZapfDingbats', 16))
        self.tab_pbt3.setStyleSheet("background:rgb" + str(self.backtabs_color) + ";")
        self.tab_pbt3.clicked.connect(self.cb_tab_pbt3)
        # fourth tab button
        self.tab_pbt4 = QPushButton(self)
        self.tab_pbt4.move(600, 0)
        self.tab_pbt4.resize(200, 40)
        self.tab_pbt4.setText('Graphics')
        self.tab_pbt4.setFont(QFont('ZapfDingbats', 16))
        self.tab_pbt4.setStyleSheet("background:rgb" + str(self.backtabs_color) + ";")
        self.tab_pbt4.clicked.connect(self.cb_tab_pbt4)
        # fifth tab button
        self.tab_pbt5 = QPushButton(self)
        self.tab_pbt5.move(800, 0)
        self.tab_pbt5.resize(200, 40)
        self.tab_pbt5.setText('Credits')
        self.tab_pbt5.setFont(QFont('ZapfDingbats', 16))
        self.tab_pbt5.setStyleSheet("background:rgb" + str(self.backtabs_color) + ";")
        self.tab_pbt5.clicked.connect(self.cb_tab_pbt5)
        # for incoming lazy evaluation of tab 2, set all possible tabs 2 as not yet created
        self.created_tab_2_lr = False
        self.created_tab_2_var = False
        # initiate user interrupt as True (will become False if interface is properly validated later)
        self.user_interrupt = True
        
        
    def cb_tab_pbt1(self):
        # hide current tab, show tab 1
        self.hide_current_tab()
        self.show_tab_1()
        # set current tab as tab 1, update tab button color
        self.current_tab = 'tab_1'
        self.tab_pbt1.setStyleSheet("background:rgb" + str(self.tab_color) + ";") 


    def cb_tab_pbt2(self):
        # hide current tab
        self.hide_current_tab()
        # tab 2 is created in a lazy fashion: create the tab only when it is called
        # if tab2 is called for linear regression:
        if self.user_inputs['tab_1']['model'] == 1:
            # if tab 2 for linear regression does not exist, create it
            if not self.created_tab_2_lr:
                self.create_tab_2_lr()
            # show tab 2 for linear regression    
            self.show_tab_2_lr()
            # set current tab as tab 2, linear regression
            self.current_tab = 'tab_2_lr' 
        # else, if tab2 is called for vector autoregression:    
        elif self.user_inputs['tab_1']['model'] == 2:
            # if tab 2 for vector autoregression does not exist, create it
            if not self.created_tab_2_var:
                self.create_tab_2_var()
            # show tab 2 for vector autoregression 
            self.show_tab_2_var() 
            # set current tab as tab 2, vector autoregression
            self.current_tab = 'tab_2_var'
        # update tab button color
        self.tab_pbt2.setStyleSheet("background:rgb" + str(self.tab_color) + ";")


    def cb_tab_pbt3(self):
        # hide current tab, show tab 3
        self.hide_current_tab()
        self.show_tab_3()
        # set current tab as tab 3, update tab button color
        self.current_tab = 'tab_3'
        self.tab_pbt3.setStyleSheet("background:rgb" + str(self.tab_color) + ";")


    def cb_tab_pbt4(self):
        # hide current tab, show tab 4
        self.hide_current_tab()
        self.show_tab_4()
        # set current tab as tab 4, update tab button color
        self.current_tab = 'tab_4'
        self.tab_pbt4.setStyleSheet("background:rgb" + str(self.tab_color) + ";")


    def cb_tab_pbt5(self):
        # hide current tab, show tab 5
        self.hide_current_tab()
        self.show_tab_5()
        # set current tab as tab 5, update tab button color
        self.current_tab = 'tab_5'
        self.tab_pbt5.setStyleSheet("background:rgb" + str(self.tab_color) + ";")          
        
        
    def create_tab_2(self):
        if self.user_inputs['tab_1']['model'] == 1:
            self.create_tab_2_lr()   
        elif self.user_inputs['tab_1']['model'] == 2:
            self.create_tab_2_var()
        
        
    def hide_current_tab(self):  
        # if current tab is tab 1, hide it
        if self.current_tab == 'tab_1':
            self.hide_tab_1()
        # if current tab is tab 2 for regression, hide it
        elif self.current_tab == 'tab_2_lr':
            self.hide_tab_2_lr()  
        # if current tab is tab 2 for vector autoregression, hide it
        elif self.current_tab == 'tab_2_var':
            self.hide_tab_2_var()               
        # if current tab is tab 3, hide it
        elif self.current_tab == 'tab_3':
            self.hide_tab_3()
        # if current tab is tab 4, hide it
        elif self.current_tab == 'tab_4':
            self.hide_tab_4()
        # if current tab is tab 5, hide it
        elif self.current_tab == 'tab_5':
            self.hide_tab_5()
            
            
    def switch_tab_4(self):
        # if view graphics is True, move back to tab 4
        if self.view_graphics:
            self.cb_tab_pbt4()
        
        
    def validate_interface(self):
        # save user inputs to drive
        user_inputs = self.user_inputs
        json.dump(user_inputs, open(join(self.interface_path, "user_inputs.json"), 'w'))
        # set user interrupt to False to inidicate proper validation, then close interface
        self.user_interrupt = False
        self.close()               
        

