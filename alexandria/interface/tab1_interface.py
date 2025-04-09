# imports
from os import remove
from os.path import isfile, join
from PyQt5.QtWidgets import QLabel, QFrame, QComboBox, QTextEdit, QLineEdit, QRadioButton, QButtonGroup, QPushButton
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QPixmap, QIcon



class Tab1Interface(object):

    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass


    def create_tab_1(self):
        
        # main title
        self.t1_txt1 = QLabel(self)
        self.t1_txt1.move(10, 45)
        self.t1_txt1.setFixedSize(600, 70)
        self.t1_txt1.setText(' Alexandria') 
        self.t1_txt1.setAlignment(Qt.AlignLeft)
        self.t1_txt1.setStyleSheet('font-size: 34pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t1_txt1.setFont(font)
        
        # subtitle
        self.t1_txt2 = QLabel(self)
        self.t1_txt2.move(15, 100)
        self.t1_txt2.setFixedSize(600, 30)
        self.t1_txt2.setText(' The library of Bayesian time-series models') 
        self.t1_txt2.setAlignment(Qt.AlignLeft)
        self.t1_txt2.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t1_txt2.setFont(font)
        
        # version
        self.t1_txt3 = QLabel(self)
        self.t1_txt3.move(15, 130)
        self.t1_txt3.setFixedSize(600, 30)
        self.t1_txt3.setText(' V 1.0 - Python edition') 
        self.t1_txt3.setAlignment(Qt.AlignLeft)
        self.t1_txt3.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t1_txt3.setFont(font)
        
        # Python logo
        self.t1_img1 = QLabel(self) 
        self.t1_img1.setFixedSize(120, 120)
        self.t1_img1.setPixmap(QPixmap(join(self.interface_path, 'python.png')).scaled(110, 110))
        self.t1_img1.move(855, 55) 
        self.t1_img1.setStyleSheet('background-color: rgb' + str(self.background_color))
        
        # model label
        self.t1_txt4 = QLabel(self)
        self.t1_txt4.move(30, 190)
        self.t1_txt4.setFixedSize(600, 30)
        self.t1_txt4.setText(' Model') 
        self.t1_txt4.setAlignment(Qt.AlignLeft)
        self.t1_txt4.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t1_txt4.setFont(font)
        
        # frame around model
        self.t1_frm1 = QFrame(self)   
        self.t1_frm1.setGeometry(20, 220, 380, 410)  
        self.t1_frm1.setFrameShape(QFrame.Panel)
        self.t1_frm1.setLineWidth(1)  
        self.t1_frm1.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        
        # model selection label
        self.t1_txt5 = QLabel(self)
        self.t1_txt5.move(30, 230)
        self.t1_txt5.setFixedSize(300, 30)
        self.t1_txt5.setText(' model selection') 
        self.t1_txt5.setAlignment(Qt.AlignLeft)
        self.t1_txt5.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        
        # model selection menu
        self.t1_mnu1 = QComboBox(self)
        self.t1_mnu1.move(35,255)                                             
        self.t1_mnu1.resize(250,25)
        self.t1_mnu1.setStyleSheet('QListView{background-color: white}')
        self.t1_mnu1.addItem('1. linear regression', 1)
        self.t1_mnu1.addItem('2. vector autoregression', 2)
        self.t1_mnu1.setCurrentIndex(self.user_inputs['tab_1']['model'] - 1)
        self.t1_mnu1.activated.connect(self.cb_t1_mnu1)
        
        # endogenous label
        self.t1_txt6 = QLabel(self)
        self.t1_txt6.move(30, 300)
        self.t1_txt6.setFixedSize(300, 30)
        self.t1_txt6.setText(' endogenous variables') 
        self.t1_txt6.setAlignment(Qt.AlignLeft)
        self.t1_txt6.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        
        # endogenous edit
        self.t1_edt1 = QTextEdit(self) 
        self.t1_edt1.move(35, 320)       
        self.t1_edt1.resize(340, 60)                                           
        self.t1_edt1.setAlignment(Qt.AlignLeft)     
        self.t1_edt1.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t1_edt1.setText(self.user_inputs['tab_1']['endogenous_variables'])                              
        self.t1_edt1.textChanged.connect(self.cb_t1_edt1)
        
        # exogenous label
        self.t1_txt7 = QLabel(self)
        self.t1_txt7.move(30, 400)
        self.t1_txt7.setFixedSize(300, 30)
        self.t1_txt7.setText(' exogenous variables') 
        self.t1_txt7.setAlignment(Qt.AlignLeft)
        self.t1_txt7.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        
        # exogenous edit
        self.t1_edt2 = QTextEdit(self) 
        self.t1_edt2.move(35, 420)       
        self.t1_edt2.resize(340, 60)                                           
        self.t1_edt2.setAlignment(Qt.AlignLeft)     
        self.t1_edt2.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t1_edt2.setText(self.user_inputs['tab_1']['exogenous_variables'])  
        self.t1_edt2.textChanged.connect(self.cb_t1_edt2)
        
        # frequency label
        self.t1_txt8 = QLabel(self)
        self.t1_txt8.move(30, 495)
        self.t1_txt8.setFixedSize(300, 30)
        self.t1_txt8.setText(' data frequency') 
        self.t1_txt8.setAlignment(Qt.AlignLeft)
        self.t1_txt8.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        
        # data frequency menu
        self.t1_mnu2 = QComboBox(self)
        self.t1_mnu2.move(35,520)                                             
        self.t1_mnu2.resize(250,25)
        self.t1_mnu2.setStyleSheet('QListView{background-color: white}')
        # self.t1_mnu2.addItem('1. cross-sectional/undated', 1)
        self.t1_mnu2.addItem('1. cross-sectional/undated', 1)
        self.t1_mnu2.addItem('2. annual', 2)
        self.t1_mnu2.addItem('3. quarterly', 3)
        self.t1_mnu2.addItem('4. monthly', 4)
        self.t1_mnu2.addItem('5. weekly', 5)
        self.t1_mnu2.addItem('6. daily', 6)
        self.t1_mnu2.setCurrentIndex(self.user_inputs['tab_1']['frequency'] - 1)
        self.t1_mnu2.activated.connect(self.cb_t1_mnu2)
        
        # sample label
        self.t1_txt9 = QLabel(self)
        self.t1_txt9.move(30, 565)
        self.t1_txt9.setFixedSize(300, 30)
        self.t1_txt9.setText(' estimation sample') 
        self.t1_txt9.setAlignment(Qt.AlignLeft)
        self.t1_txt9.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        
        # sample edit
        self.t1_edt3 = QLineEdit(self)
        self.t1_edt3.move(35, 590)       
        self.t1_edt3.resize(340, 25)                                           
        self.t1_edt3.setAlignment(Qt.AlignLeft)     
        self.t1_edt3.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t1_edt3.setText(self.user_inputs['tab_1']['sample']) 
        self.t1_edt3.textChanged.connect(self.cb_t1_edt3)
        
        # settings label
        self.t1_txt10 = QLabel(self)
        self.t1_txt10.move(430, 190)
        self.t1_txt10.setFixedSize(600, 30)
        self.t1_txt10.setText('Settings') 
        self.t1_txt10.setAlignment(Qt.AlignLeft)
        self.t1_txt10.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t1_txt10.setFont(font)
        
        # frame around settings
        self.t1_frm2 = QFrame(self)   
        self.t1_frm2.setGeometry(420, 220, 380, 410)  
        self.t1_frm2.setFrameShape(QFrame.Panel)
        self.t1_frm2.setLineWidth(1)  
        self.t1_frm2.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        
        # project path label
        self.t1_txt11 = QLabel(self)
        self.t1_txt11.move(430, 230)
        self.t1_txt11.setFixedSize(300, 30)
        self.t1_txt11.setText(' path to project folder') 
        self.t1_txt11.setAlignment(Qt.AlignLeft)
        self.t1_txt11.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        
        # project path edit
        self.t1_edt4 = QLineEdit(self)
        self.t1_edt4.move(435, 255)       
        self.t1_edt4.resize(340, 25)                                           
        self.t1_edt4.setAlignment(Qt.AlignLeft)     
        self.t1_edt4.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t1_edt4.setText(self.user_inputs['tab_1']['project_path']) 
        self.t1_edt4.textChanged.connect(self.cb_t1_edt4)
        
        # data file label
        self.t1_txt12 = QLabel(self)
        self.t1_txt12.move(430, 300)
        self.t1_txt12.setFixedSize(300, 30)
        self.t1_txt12.setText(' data file') 
        self.t1_txt12.setAlignment(Qt.AlignLeft)
        self.t1_txt12.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        
        # data file edit
        self.t1_edt5 = QLineEdit(self)
        self.t1_edt5.move(435, 320)       
        self.t1_edt5.resize(340, 25)                                           
        self.t1_edt5.setAlignment(Qt.AlignLeft)     
        self.t1_edt5.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t1_edt5.setText(self.user_inputs['tab_1']['data_file']) 
        self.t1_edt5.textChanged.connect(self.cb_t1_edt5)
        
        # progress bar label
        self.t1_txt13 = QLabel(self)
        self.t1_txt13.move(430, 365)
        self.t1_txt13.setFixedSize(300, 30)
        self.t1_txt13.setText(' progress bar') 
        self.t1_txt13.setAlignment(Qt.AlignLeft)
        self.t1_txt13.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        
        # progress bar radiobuttons
        self.t1_rdb1 = QRadioButton(' yes', self)  
        self.t1_rdb1.move(435, 385)           
        self.t1_rdb1.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}') 
        self.t1_rdb1.setChecked(self.user_inputs['tab_1']['progress_bar']) 
        self.t1_rdb1.toggled.connect(self.cb_t1_bgr1)           
        self.t1_rdb2 = QRadioButton(' no', self)   
        self.t1_rdb2.move(515,385)        
        self.t1_rdb2.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')
        self.t1_rdb2.setChecked(not self.user_inputs['tab_1']['progress_bar']) 
        self.t1_rdb2.toggled.connect(self.cb_t1_bgr1) 
        self.t1_bgr1 = QButtonGroup(self)  
        self.t1_bgr1.addButton(self.t1_rdb1) 
        self.t1_bgr1.addButton(self.t1_rdb2) 
        
        # graphics label
        self.t1_txt14 = QLabel(self)
        self.t1_txt14.move(430, 430)
        self.t1_txt14.setFixedSize(300, 30)
        self.t1_txt14.setText(' graphics and figures') 
        self.t1_txt14.setAlignment(Qt.AlignLeft)
        self.t1_txt14.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        
        # graphics radiobuttons
        self.t1_rdb3 = QRadioButton(' yes', self)  
        self.t1_rdb3.move(435, 450)          
        self.t1_rdb3.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')
        self.t1_rdb3.setChecked(self.user_inputs['tab_1']['create_graphics'])                            
        self.t1_rdb3.toggled.connect(self.cb_t1_bgr2)           
        self.t1_rdb4 = QRadioButton(' no', self)   
        self.t1_rdb4.move(515,450)        
        self.t1_rdb4.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')
        self.t1_rdb4.setChecked(not self.user_inputs['tab_1']['create_graphics']) 
        self.t1_rdb4.toggled.connect(self.cb_t1_bgr2) 
        self.t1_bgr2 = QButtonGroup(self)  
        self.t1_bgr2.addButton(self.t1_rdb3) 
        self.t1_bgr2.addButton(self.t1_rdb4) 
        
        # save label
        self.t1_txt15 = QLabel(self)
        self.t1_txt15.move(430, 495)
        self.t1_txt15.setFixedSize(300, 30)
        self.t1_txt15.setText(' save results in project folder') 
        self.t1_txt15.setAlignment(Qt.AlignLeft)
        self.t1_txt15.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        
        # save radiobuttons
        self.t1_rdb5 = QRadioButton(' yes', self)  
        self.t1_rdb5.move(435, 515)           
        self.t1_rdb5.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')
        self.t1_rdb5.setChecked(self.user_inputs['tab_1']['save_results'])                           
        self.t1_rdb5.toggled.connect(self.cb_t1_bgr3)           
        self.t1_rdb6 = QRadioButton(' no', self)   
        self.t1_rdb6.move(515, 515)        
        self.t1_rdb6.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')
        self.t1_rdb6.setChecked(not self.user_inputs['tab_1']['save_results'])                           
        self.t1_rdb6.toggled.connect(self.cb_t1_bgr3) 
        self.t1_bgr3 = QButtonGroup(self)  
        self.t1_bgr3.addButton(self.t1_rdb5) 
        self.t1_bgr3.addButton(self.t1_rdb6)
        
        # reset pushbutton
        self.t1_pbt1 = QPushButton(self)
        self.t1_pbt1.move(510, 570)  
        self.t1_pbt1.resize(200, 30)
        self.t1_pbt1.setText('Reset all') 
        self.t1_pbt1.setStyleSheet('font-size: 17px; font-family: Serif') 
        self.t1_pbt1.clicked.connect(self.cb_t1_pbt1)
        
        # run pushbutton
        self.t1_pbt2 = QPushButton(self)
        self.t1_pbt2.move(820, 340)  
        self.t1_pbt2.resize(160, 260)
        self.t1_pbt2.setStyleSheet('font-size: 17px; font-family: Serif')
        self.t1_pbt2.setIcon(QIcon(join(self.interface_path, 'run_button.png')))  
        self.t1_pbt2.setIconSize(QSize(160, 260))
        self.t1_pbt2.clicked.connect(self.cb_t1_pbt2)
        
        # initiate current tab
        self.current_tab = 'tab_1';


    def hide_tab_1(self):
    
        # hide all controls
        self.t1_txt1.setHidden(True)
        self.t1_txt2.setHidden(True)
        self.t1_txt3.setHidden(True)
        self.t1_txt4.setHidden(True)
        self.t1_txt5.setHidden(True)
        self.t1_txt6.setHidden(True)
        self.t1_txt7.setHidden(True)
        self.t1_txt8.setHidden(True)
        self.t1_txt9.setHidden(True)
        self.t1_txt10.setHidden(True)
        self.t1_txt11.setHidden(True)
        self.t1_txt12.setHidden(True)
        self.t1_txt13.setHidden(True)
        self.t1_txt14.setHidden(True)
        self.t1_txt15.setHidden(True)
        self.t1_img1.setHidden(True)
        self.t1_frm1.setHidden(True)
        self.t1_frm2.setHidden(True)
        self.t1_mnu1.setHidden(True)
        self.t1_mnu2.setHidden(True)
        self.t1_edt1.setHidden(True)
        self.t1_edt2.setHidden(True)
        self.t1_edt3.setHidden(True)
        self.t1_edt4.setHidden(True)
        self.t1_edt5.setHidden(True)
        self.t1_rdb1.setHidden(True)
        self.t1_rdb2.setHidden(True)
        self.t1_rdb3.setHidden(True)
        self.t1_rdb4.setHidden(True)
        self.t1_rdb5.setHidden(True)
        self.t1_rdb6.setHidden(True)
        self.t1_pbt1.setHidden(True)
        self.t1_pbt2.setHidden(True)
        
        # update tab color
        self.tab_pbt1.setStyleSheet("background:rgb" + str(self.backtabs_color) + ";")  
    
    
    def show_tab_1(self):    
        
        # show all controls
        self.t1_txt1.setHidden(False)
        self.t1_txt2.setHidden(False)
        self.t1_txt3.setHidden(False)
        self.t1_txt4.setHidden(False)
        self.t1_txt5.setHidden(False)
        self.t1_txt6.setHidden(False)
        self.t1_txt7.setHidden(False)
        self.t1_txt8.setHidden(False)
        self.t1_txt9.setHidden(False)
        self.t1_txt10.setHidden(False)
        self.t1_txt11.setHidden(False)
        self.t1_txt12.setHidden(False)
        self.t1_txt13.setHidden(False)
        self.t1_txt14.setHidden(False)
        self.t1_txt15.setHidden(False)
        self.t1_img1.setHidden(False)
        self.t1_frm1.setHidden(False)
        self.t1_frm2.setHidden(False)
        self.t1_mnu1.setHidden(False)
        self.t1_mnu2.setHidden(False)
        self.t1_edt1.setHidden(False)
        self.t1_edt2.setHidden(False)
        self.t1_edt3.setHidden(False)
        self.t1_edt4.setHidden(False)
        self.t1_edt5.setHidden(False)
        self.t1_rdb1.setHidden(False)
        self.t1_rdb2.setHidden(False)
        self.t1_rdb3.setHidden(False)
        self.t1_rdb4.setHidden(False)
        self.t1_rdb5.setHidden(False)
        self.t1_rdb6.setHidden(False)
        self.t1_pbt1.setHidden(False)
        self.t1_pbt2.setHidden(False)    
        

    def cb_t1_mnu1(self, index):        
        self.user_inputs['tab_1']['model'] = self.t1_mnu1.itemData(index)
        

    def cb_t1_edt1(self):
        self.user_inputs['tab_1']['endogenous_variables'] = self.t1_edt1.toPlainText()
        
        
    def cb_t1_edt2(self):
        self.user_inputs['tab_1']['exogenous_variables'] = self.t1_edt2.toPlainText()        
        

    def cb_t1_mnu2(self, index):        
        self.user_inputs['tab_1']['frequency'] = self.t1_mnu2.itemData(index)


    def cb_t1_edt3(self):
        self.user_inputs['tab_1']['sample'] = self.t1_edt3.text() 


    def cb_t1_edt4(self):
        self.user_inputs['tab_1']['project_path'] = self.t1_edt4.text() 


    def cb_t1_edt5(self):
        self.user_inputs['tab_1']['data_file'] = self.t1_edt5.text() 
        
    
    def cb_t1_bgr1(self):
        if self.t1_rdb1.isChecked() == True:
            self.user_inputs['tab_1']['progress_bar'] = True
        elif self.t1_rdb2.isChecked() == True:
            self.user_inputs['tab_1']['progress_bar'] = False


    def cb_t1_bgr2(self):
        if self.t1_rdb3.isChecked() == True:
            self.user_inputs['tab_1']['create_graphics'] = True
        elif self.t1_rdb4.isChecked() == True:
            self.user_inputs['tab_1']['create_graphics'] = False


    def cb_t1_bgr3(self):
        if self.t1_rdb5.isChecked() == True:
            self.user_inputs['tab_1']['save_results'] = True
        elif self.t1_rdb6.isChecked() == True:
            self.user_inputs['tab_1']['save_results'] = False


    def cb_t1_pbt1(self):
        if isfile(join(self.interface_path, "user_inputs.json")):
            remove(join(self.interface_path, "user_inputs.json"));
        self.create_default_inputs()
        self.create_tab_2_lr()
        self.reset_default_inputs()
        self.created_tab_2_var = False
        
        
    def cb_t1_pbt2(self):
        self.validate_interface()
