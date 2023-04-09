# imports
from os import listdir
from os.path import join, dirname, abspath
from PyQt5.QtWidgets import QLabel, QFrame, QSlider, QComboBox, QRadioButton, QButtonGroup, QScrollBar
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap


class Tab4Interface(object):

    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass    


    def create_tab_4(self):
    
        # application label
        self.t4_txt1 = QLabel(self)
        self.t4_txt1.move(30, 70)
        self.t4_txt1.setFixedSize(250, 30)
        self.t4_txt1.setText(' Application') 
        self.t4_txt1.setAlignment(Qt.AlignLeft)
        self.t4_txt1.setStyleSheet('font-size: 14pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t4_txt1.setFont(font)
        self.t4_txt1.setHidden(True)
        
        # frame around application
        self.t4_frm1 = QFrame(self)   
        self.t4_frm1.setGeometry(20, 95, 280, 110)  
        self.t4_frm1.setFrameShape(QFrame.Panel)
        self.t4_frm1.setLineWidth(1)  
        self.t4_frm1.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t4_frm1.setHidden(True)
        
        # application selection label
        self.t4_txt2 = QLabel(self)
        self.t4_txt2.move(30, 105)
        self.t4_txt2.setFixedSize(200, 30)
        self.t4_txt2.setText(' none') 
        self.t4_txt2.setAlignment(Qt.AlignLeft)
        self.t4_txt2.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t4_txt2.setHidden(True)

        # set stylesheet for incoming QscrollBar controls
        stylesheet = """
        QScrollBar:horizontal{
            border: 2px solid rgb(170, 170, 170);
            margin: 0px 14px 0 14px}
        QScrollBar::handle:horizontal{
            border: 1px solid rgb(140, 140, 180);
            background: rgb(215, 230, 255);
            min-width: 16px}
        QScrollBar::add-line:horizontal{
            border: 1px solid rgb(100, 100, 100);
            background: rgb(240, 240, 240);
            width: 14px;
            subcontrol-position: right;
            subcontrol-origin: margin}
        QScrollBar::sub-line:horizontal {
            border: 1px solid rgb(100, 100, 100);
            background: rgb(240, 240, 240);
            width: 14px;
            subcontrol-position: left;
            subcontrol-origin: margin}   
        QScrollBar:left-arrow:horizontal {
            image: url(""" + join(self.interface_path, 'caret_left.png') + """);
            width: 12px;
            height: 12px;
            background: rgb(240, 240, 240)}   
        QScrollBar:right-arrow:horizontal {
            image: url(""" + join(self.interface_path, 'caret_right.png') + """);
            width: 12px;
            height: 12px;
            background: rgb(240, 240, 240)}  
        """
        
        # application slider
        self.t4_sld1 = QScrollBar(Qt.Horizontal, self)
        self.t4_sld1.setGeometry(35, 135, 240, 20) 
        self.t4_sld1.setStyleSheet(stylesheet)
        self.t4_sld1.setMinimum(1)
        self.t4_sld1.setSingleStep(1)   
        self.t4_sld1.setPageStep(3)
        self.t4_sld1.setHidden(True)
        
        # application menu
        self.t4_mnu1 = QComboBox(self)
        self.t4_mnu1.move(35, 165)                                             
        self.t4_mnu1.resize(240, 25)
        self.t4_mnu1.setStyleSheet('QListView{background-color: white}')
        self.t4_mnu1.view().setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.t4_mnu1.addItem('none', 1)
        self.t4_mnu1.setHidden(True)
        
        # display label
        self.t4_txt3 = QLabel(self)
        self.t4_txt3.move(30, 220)
        self.t4_txt3.setFixedSize(250, 30)
        self.t4_txt3.setText(' Display type') 
        self.t4_txt3.setAlignment(Qt.AlignLeft)
        self.t4_txt3.setStyleSheet('font-size: 14pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t4_txt3.setFont(font)
        self.t4_txt3.setHidden(True)
        
        # frame around display type
        self.t4_frm2 = QFrame(self)   
        self.t4_frm2.setGeometry(20, 250, 280, 70)  
        self.t4_frm2.setFrameShape(QFrame.Panel)
        self.t4_frm2.setLineWidth(1)  
        self.t4_frm2.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t4_frm2.setHidden(True)
        
        # display radiobuttons
        self.t4_rdb1 = QRadioButton(' joint plot', self)  
        self.t4_rdb1.setGeometry(30, 255, 200, 30)
        self.t4_rdb1.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')  
        self.t4_rdb1.setHidden(True)       
        self.t4_rdb2 = QRadioButton(' individual graphs', self)   
        self.t4_rdb2.setGeometry(30, 285, 200, 30) 
        self.t4_rdb2.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')  
        self.t4_rdb2.setHidden(True)
        self.t4_rdb1.setChecked(True) 
        self.t4_bgr1 = QButtonGroup(self)  
        self.t4_bgr1.addButton(self.t4_rdb1) 
        self.t4_bgr1.addButton(self.t4_rdb2)
        
        # variable label
        self.t4_txt4 = QLabel(self)
        self.t4_txt4.move(30, 340)
        self.t4_txt4.setFixedSize(250, 30)
        self.t4_txt4.setText(' Variable') 
        self.t4_txt4.setAlignment(Qt.AlignLeft)
        self.t4_txt4.setStyleSheet('font-size: 14pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t4_txt4.setFont(font)
        self.t4_txt4.setHidden(True)
        
        # frame around variable
        self.t4_frm3 = QFrame(self)   
        self.t4_frm3.setGeometry(20, 365, 280, 110)  
        self.t4_frm3.setFrameShape(QFrame.Panel)
        self.t4_frm3.setLineWidth(1)  
        self.t4_frm3.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t4_frm3.setHidden(True)
        
        # variable label
        self.t4_txt5 = QLabel(self)
        self.t4_txt5.move(30, 375)
        self.t4_txt5.setFixedSize(200, 30)
        self.t4_txt5.setText(' none') 
        self.t4_txt5.setAlignment(Qt.AlignLeft)
        self.t4_txt5.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t4_txt5.setHidden(True)
        
        # variable slider
        self.t4_sld2 = QScrollBar(Qt.Horizontal, self)
        self.t4_sld2.setGeometry(35, 400, 240, 20)
        self.t4_sld2.setStyleSheet(stylesheet)
        self.t4_sld2.setMinimum(1)
        self.t4_sld2.setSingleStep(1)   
        self.t4_sld2.setPageStep(3) 
        self.t4_sld2.setHidden(True)
        
        # variable menu
        self.t4_mnu2 = QComboBox(self)
        self.t4_mnu2.move(35, 435)                                             
        self.t4_mnu2.resize(240, 25)
        self.t4_mnu2.setStyleSheet('QListView{background-color: white}')
        self.t4_mnu2.view().setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.t4_mnu2.addItem('none', 1)
        self.t4_mnu2.setHidden(True)
        
        # response label
        self.t4_txt6 = QLabel(self)
        self.t4_txt6.move(30, 495)
        self.t4_txt6.setFixedSize(250, 30)
        self.t4_txt6.setText(' Responding to') 
        self.t4_txt6.setAlignment(Qt.AlignLeft)
        self.t4_txt6.setStyleSheet('font-size: 14pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t4_txt6.setFont(font)
        self.t4_txt6.setHidden(True)
        
        # frame around response
        self.t4_frm4 = QFrame(self)   
        self.t4_frm4.setGeometry(20, 520, 280, 110)  
        self.t4_frm4.setFrameShape(QFrame.Panel)
        self.t4_frm4.setLineWidth(1)  
        self.t4_frm4.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t4_frm4.setHidden(True)
        
        # response selection label
        self.t4_txt7 = QLabel(self)
        self.t4_txt7.move(30, 530)
        self.t4_txt7.setFixedSize(200, 30)
        self.t4_txt7.setText(' none') 
        self.t4_txt7.setAlignment(Qt.AlignLeft)
        self.t4_txt7.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t4_txt7.setHidden(True)
        
        # response slider
        self.t4_sld3 = QScrollBar(Qt.Horizontal, self)
        self.t4_sld3.setGeometry(35, 555, 240, 20)
        self.t4_sld3.setStyleSheet(stylesheet)
        self.t4_sld3.setMinimum(1)
        self.t4_sld3.setSingleStep(1)   
        self.t4_sld3.setPageStep(3) 
        self.t4_sld3.setHidden(True)
        
        # response menu
        self.t4_mnu3 = QComboBox(self)
        self.t4_mnu3.move(35, 590)                                             
        self.t4_mnu3.resize(240, 25)
        self.t4_mnu3.setStyleSheet('QListView{background-color: white}')
        self.t4_mnu3.view().setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.t4_mnu3.addItem('none', 1)
        self.t4_mnu3.setHidden(True)
        
        # default no-graphic image
        self.t4_img1 = QLabel(self) 
        self.t4_img1.setFixedSize(660, 570)
        self.t4_img1.setPixmap(QPixmap(join(self.interface_path, 'no_graphic.png')).scaled(660, 570))
        self.t4_img1.move(320, 60) 
        self.t4_img1.setStyleSheet('border: 1px solid black;background-color: rgb' + str(self.background_color))
        self.t4_img1.setHidden(True)
        
        # if graphic view is activated, update tab 4 with graphics to turn it into a graphics navigator
        self.update_tab_4()
        

    def hide_tab_4(self):
    
        # hide all controls
        self.t4_txt1.setHidden(True)
        self.t4_txt2.setHidden(True)
        self.t4_txt3.setHidden(True)
        self.t4_txt4.setHidden(True)
        self.t4_txt5.setHidden(True)
        self.t4_txt6.setHidden(True)
        self.t4_txt7.setHidden(True)
        self.t4_frm1.setHidden(True)
        self.t4_frm2.setHidden(True)
        self.t4_frm3.setHidden(True)
        self.t4_frm4.setHidden(True)
        self.t4_sld1.setHidden(True)
        self.t4_sld2.setHidden(True)
        self.t4_sld3.setHidden(True)
        self.t4_rdb1.setHidden(True)
        self.t4_rdb2.setHidden(True)
        self.t4_mnu1.setHidden(True)
        self.t4_mnu2.setHidden(True)
        self.t4_mnu3.setHidden(True)
        self.t4_img1.setHidden(True)
        # update tab color
        self.tab_pbt4.setStyleSheet("background:rgb" + str(self.backtabs_color) + ";") 


    def show_tab_4(self): 
        
        # show all controls
        self.t4_txt1.setVisible(True)
        self.t4_txt2.setVisible(True)
        self.t4_txt3.setVisible(True)
        self.t4_txt4.setVisible(True)
        self.t4_txt5.setVisible(True)
        self.t4_txt6.setVisible(True)
        self.t4_txt7.setVisible(True)
        self.t4_frm1.setVisible(True)
        self.t4_frm2.setVisible(True)
        self.t4_frm3.setVisible(True)
        self.t4_frm4.setVisible(True)
        self.t4_sld1.setVisible(True)
        self.t4_sld2.setVisible(True)
        self.t4_sld3.setVisible(True)
        self.t4_rdb1.setVisible(True)
        self.t4_rdb2.setVisible(True)
        self.t4_mnu1.setVisible(True)
        self.t4_mnu2.setVisible(True)
        self.t4_mnu3.setVisible(True)
        self.t4_img1.setVisible(True)


    def update_tab_4(self):
        
        if self.view_graphics:
            # recover path to graphics folder
            self.__get_graphics_folder_path()
            # recover the names of all images in graphics folder
            self.__get_image_names()
            # generate rollmenus from the list of splitted images
            self.__get_rollmenus()
            # initiate application, variable, and response values
            self.__initiate_values()
            # update all controls
            self.__update_application_slider()
            self.__update_application_rollmenu()
            self.__update_variable_slider()
            self.__update_variable_rollmenu()
            self.__update_response_slider()
            self.__update_response_rollmenu()              
            # set all controls to initial positions
            self.__set_application_label()
            self.__set_application_slider()         
            self.__set_application_rollmenu() 
            self.__set_variable_label()
            self.__set_variable_slider()        
            self.__set_variable_rollmenu()
            self.__set_response_label()
            self.__set_response_slider()
            self.__set_response_rollmenu()              
            # initiate callbacks
            self.__initiate_callbacks()
            # initiate image
            self.__update_image()
            
     
    def cb_t4_sld1(self):
        # update current application
        self.current_application = self.t4_mnu1.itemText(self.t4_sld1.value())   
        # set application controls
        self.__set_application_label()
        self.__set_application_rollmenu()  
        # update variables 
        self.__update_current_variables()
        self.__update_current_variable()
        # update variable controls
        self.__update_variable_slider()
        self.__update_variable_rollmenu()
        # set variable controls
        self.__set_variable_label()
        self.__set_variable_slider()        
        self.__set_variable_rollmenu()
        # update responses
        self.__update_current_responses()
        self.__update_current_response()
        # update response controls
        self.__update_response_slider()
        self.__update_response_rollmenu()
        # set response controls
        self.__set_response_label()
        self.__set_response_slider()
        self.__set_response_rollmenu()        
        # update image
        self.__update_image()
            
        
    def cb_t4_mnu1(self, index):
        # update current application
        if self.t4_mnu1.itemData(index) != 0:
            self.current_application = self.t4_mnu1.currentText()
        # set application controls
        self.__set_application_label()
        self.__set_application_slider()         
        self.__set_application_rollmenu() 
        # update variables 
        self.__update_current_variables()
        self.__update_current_variable()
        # update variable controls
        self.__update_variable_slider()
        self.__update_variable_rollmenu()
        # set variable controls
        self.__set_variable_label()
        self.__set_variable_slider()        
        self.__set_variable_rollmenu()
        # update responses
        self.__update_current_responses()
        self.__update_current_response()
        # update response controls
        self.__update_response_slider()
        self.__update_response_rollmenu()
        # set response controls
        self.__set_response_label()
        self.__set_response_slider()
        self.__set_response_rollmenu()   
        # update image
        self.__update_image()
        

    def cb_t4_bgr1(self):
        # if joint plot is True, update and set current variable to all
        if self.t4_rdb1.isChecked() == True:
           self.joint_plot = True
        # else, update and set current variable to first variable that is not all
        elif self.t4_rdb2.isChecked() == True:
            self.joint_plot = False
        # update variables
        self.__update_current_variable()
        # update variable controls
        self.__update_variable_slider()
        self.__update_variable_rollmenu()        
        # set variable controls
        self.__set_variable_label()
        self.__set_variable_slider()
        self.__set_variable_rollmenu()      
        self.__switch_variable_controls()        
        # update responses
        self.__update_current_responses()
        self.__update_current_response()
        # update response controls
        self.__update_response_slider()
        self.__update_response_rollmenu()
        # set response controls
        self.__switch_response_controls()
        self.__set_response_label()
        self.__set_response_slider()
        self.__set_response_rollmenu()        
        # update image
        self.__update_image()
        

    def cb_t4_sld2(self):
        # update current variable
        self.current_variable =  self.t4_mnu2.itemText(self.t4_sld2.value())
        # set variable controls
        self.__set_variable_label()
        self.__set_variable_rollmenu()
        # update responses
        self.__update_current_responses()
        self.__update_current_response()
        # update response controls
        self.__update_response_slider()
        self.__update_response_rollmenu()
        # set response controls
        self.__set_response_label()
        self.__set_response_slider()
        self.__set_response_rollmenu()
        # update image
        self.__update_image()
        

    def cb_t4_mnu2(self, index):
        # update current variable
        if self.t4_mnu2.itemData(index) != 0:
            self.current_variable = self.t4_mnu2.currentText()        
        # set variable controls
        self.__set_variable_label()
        self.__set_variable_slider()
        self.__set_variable_rollmenu()
        # update responses
        self.__update_current_responses()
        self.__update_current_response()
        # update response controls
        self.__update_response_slider()
        self.__update_response_rollmenu()
        # set response controls
        self.__set_response_label()
        self.__set_response_slider()
        self.__set_response_rollmenu()
        # update image
        self.__update_image()
        
        
    def cb_t4_sld3(self):
        # update current response
        self.current_response =  self.t4_mnu3.itemText(self.t4_sld3.value())
        # set response controls
        self.__set_response_label()
        self.__set_response_rollmenu()
        # update image
        self.__update_image()
        

    def cb_t4_mnu3(self, index):
        # update current response
        if self.t4_mnu3.itemData(index) != 0:
            self.current_response = self.t4_mnu3.currentText()
        # set response controls
        self.__set_response_label()
        self.__set_response_slider()
        self.__set_response_rollmenu()
        # update image
        self.__update_image()
        

    def __get_graphics_folder_path(self):
        project_path = self.user_inputs['tab_1']['project_path']
        graphics_folder_path = join(project_path, 'graphics')        
        self.graphics_folder_path = graphics_folder_path


    def __get_image_names(self):
        # get list of all image files in graphics folder
        image_list = listdir(self.graphics_folder_path)
        # initiate list of split names
        split_image_list = []
        # loop over files and separate each name into application, variable and response
        for image in image_list:
            # application is first part, before underscore
            application = image.split('_')[0]
            # variables and response are parts after underscore, removing png extentions
            variable_and_response = image.split('_', maxsplit = 1)[1].split('.png')[0]
            # variable is part before @, response part after @ (if any)
            split_variable_and_response = variable_and_response.split('@')
            variable = split_variable_and_response[0]
            if len(split_variable_and_response) == 2:
                response = split_variable_and_response[1]
            else:
                response = 'none'
            split_image_list.append([application, variable, response])
        self.split_image_list = split_image_list
        # also, get list of all applications (obtained in arbitrary order for now)
        applications = list(dict.fromkeys([element[0] for element in split_image_list]))
        # reorganize applications so that they are in the right order
        sorted_applications = []        
        possible_applications = ['fit', 'residuals', 'forecasts', 'conditional_forecasts', \
                                 'irf', 'fevd', 'hd']
        for application in possible_applications:
            if application in applications:
                sorted_applications.append(application)
        self.applications = sorted_applications
        # check whether there are no images to display
        if not self.applications:
            raise TypeError('Image error: graphics and figures are selected, but there are no images to display. Select applications producing figures.')


    def __get_rollmenus(self):
        # initiate dictionary of rollmenus
        rollmenus = {}
        split_image_list = self.split_image_list
        applications = self.applications
        rollmenus['applications'] = applications
        # loop over applications
        for application in applications:
            # add applications to dictionary of rollmenus
            rollmenus[application] = {}
            # list of all files matching the application
            application_files = [element for element in split_image_list if element[0] == application]
            # list of all variables 
            variables = sorted(list(dict.fromkeys([element[1] for element in application_files])))
            if 'all' in variables:
                variables.remove('all')
                variables.insert(0, 'all')
            # add variables to application dictionary
            rollmenus[application]['variables'] = variables
            # loop over variables
            for variable in variables:
                # list of all files matching the variable
                variable_files = [element for element in application_files if element[1] == variable]    
                # list of all corresponding responses
                responses = sorted(list(dict.fromkeys([element[2] for element in variable_files])))
                if 'all' in responses:
                    responses.remove('all')
                    responses.insert(0, 'all')
                # add responses to variable dictionary
                rollmenus[application][variable] = responses
        self.rollmenus = rollmenus
        
        
    def __initiate_values(self):
        self.current_application = self.applications[0]
        self.joint_plot = True
        self.current_variables = self.rollmenus[self.current_application]['variables']
        self.current_variable = self.current_variables[0]
        self.current_responses = self.rollmenus[self.current_application][self.current_variable]
        self.current_response = self.current_responses[0]
 

    def __initiate_callbacks(self):
        self.t4_sld1.valueChanged.connect(self.cb_t4_sld1)
        self.t4_mnu1.activated.connect(self.cb_t4_mnu1)
        self.t4_bgr1.buttonClicked.connect(self.cb_t4_bgr1)        
        self.t4_sld2.valueChanged.connect(self.cb_t4_sld2)
        self.t4_mnu2.activated.connect(self.cb_t4_mnu2)
        self.t4_sld3.valueChanged.connect(self.cb_t4_sld3)
        self.t4_mnu3.activated.connect(self.cb_t4_mnu3)
      
        
    def __switch_variable_controls(self):
        if self.joint_plot:
            self.t4_sld2.setEnabled(False)
            self.t4_mnu2.setEnabled(False)  
        else:
            self.t4_sld2.setEnabled(True)
            self.t4_mnu2.setEnabled(True) 


    def __switch_response_controls(self):
        if self.joint_plot:
            self.t4_sld3.setEnabled(False)
            self.t4_mnu3.setEnabled(False) 
        else:
            self.t4_sld3.setEnabled(True)
            self.t4_mnu3.setEnabled(True) 


    def __update_current_application(self):
        self.current_application = self.rollmenus['applications'][0]


    def __update_current_variables(self):
        self.current_variables = self.rollmenus[self.current_application]['variables']
            
            
    def __update_current_variable(self):
        # recover list of all variables of current application
        current_variables = self.current_variables.copy()
        # if joint plot, current variable must be all by default
        if self.joint_plot:
            self.current_variable = 'all'
        # if not joint plot, make sure all is not in the list of current variables
        else:
            if 'all' in current_variables:
                current_variables.remove('all')
        # now, if current variable is not in current variables, use first variable by default
        if self.current_variable not in current_variables:
            self.current_variable = current_variables[0] 


    def __update_current_responses(self):
        self.current_responses = self.rollmenus[self.current_application][self.current_variable]        
        
        
    def __update_current_response(self):
        if self.current_response not in self.current_responses:
            self.current_response = self.current_responses[0]    


    def __update_application_slider(self):
        self.t4_sld1.setMaximum(len(self.applications))


    def __update_application_rollmenu(self):
        self.t4_mnu1.clear()
        self.t4_mnu1.addItem('select', 0)  
        self.t4_mnu1.setCurrentIndex(0)
        for i, application in enumerate(self.applications):
            self.t4_mnu1.addItem(application, i+1)


    def __update_variable_slider(self):
        self.t4_sld2.blockSignals(True)        
        if self.joint_plot:
            self.t4_sld2.setEnabled(False)
            self.t4_sld2.setMaximum(1)
        else:
            self.t4_sld2.setEnabled(True)
            current_variables = self.current_variables.copy()
            if 'all' in current_variables:
                current_variables.remove('all')
            self.t4_sld2.setMaximum(len(current_variables))
        self.t4_sld2.blockSignals(False)            


    def __update_variable_rollmenu(self):
        self.t4_mnu2.clear()
        if self.joint_plot:
            self.t4_mnu2.setEnabled(False)
            self.t4_mnu2.addItem('none', 0)  
        else:
            self.t4_mnu2.setEnabled(True)
            self.t4_mnu2.addItem('select', 0)
            current_variables = self.current_variables.copy()
            if 'all' in current_variables:
                current_variables.remove('all')
            for i, variable in enumerate(current_variables):
                self.t4_mnu2.addItem(variable, i+1)  


    def __update_response_slider(self):
        self.t4_sld3.blockSignals(True)
        if self.joint_plot:
            self.t4_sld3.setEnabled(False)
            self.t4_sld3.setMaximum(1)
        else:
            self.t4_sld3.setEnabled(True)
            self.t4_sld3.setMaximum(len(self.current_responses))
        self.t4_sld3.blockSignals(False)
        

    def __update_response_rollmenu(self):
        self.t4_mnu3.clear()
        if self.joint_plot:
            self.t4_mnu3.setEnabled(False)
            self.t4_mnu3.addItem('none', 0)  
        else:
            self.t4_mnu3.setEnabled(True)
            self.t4_mnu3.addItem('select', 0)  
            for i, response in enumerate(self.current_responses):
                self.t4_mnu3.addItem(response, i+1)


    def __set_application_label(self):
        self.t4_txt2.setText(' ' + self.current_application) 
        self.t4_txt2.repaint()        


    def __set_application_slider(self):
        self.t4_sld1.blockSignals(True)
        self.t4_sld1.setSliderPosition(self.t4_mnu1.findText(self.current_application))
        self.t4_sld1.blockSignals(False)
        
        
    def __set_application_rollmenu(self):
        self.t4_mnu1.setCurrentIndex(0)


    def __set_variable_label(self):
        if self.current_variable == 'all':
            self.t4_txt5.setText(' none')
        else:
            self.t4_txt5.setText(' ' + self.current_variable)
        self.t4_txt5.repaint()
        
        
    def __set_variable_slider(self):
        self.t4_sld2.blockSignals(True)
        self.t4_sld2.setSliderPosition(self.t4_mnu2.findText(self.current_variable))
        self.t4_sld2.blockSignals(False)    

      
    def __set_variable_rollmenu(self):
        self.t4_mnu2.setCurrentIndex(0)


    def __set_response_label(self):
        if self.current_response == 'none':
            self.t4_txt7.setText(' none')
        else:
            self.t4_txt7.setText(' ' + self.current_response)
        self.t4_txt7.repaint()
        
        
    def __set_response_slider(self):
        self.t4_sld3.blockSignals(True)
        self.t4_sld3.setSliderPosition(self.t4_mnu3.findText(self.current_response))  
        self.t4_sld3.blockSignals(False)    
             
        
    def __set_response_rollmenu(self):
        self.t4_mnu3.setCurrentIndex(0)
        
    
    def __update_image(self):
        image_name = self.current_application + '_' + self.current_variable
        if self.current_response != 'none':
            image_name += '@' + self.current_response
        image_name += '.png'
        self.image_name = image_name
        self.t4_img1.setPixmap(QPixmap(join(self.graphics_folder_path, self.image_name)).scaled(660, 570, transformMode = Qt.SmoothTransformation))


    
        
