# imports
from os.path import join
from PyQt5.QtWidgets import QLabel, QFrame, QRadioButton, QButtonGroup, QLineEdit, QComboBox, QCheckBox, QPushButton
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon



class Tab3Interface(object):

    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass    


    def create_tab_3(self):
        
        # applications label
        self.t3_txt1 = QLabel(self)
        self.t3_txt1.move(30, 60)
        self.t3_txt1.setFixedSize(450, 30)
        self.t3_txt1.setText(' Applications and credibility levels') 
        self.t3_txt1.setAlignment(Qt.AlignLeft)
        self.t3_txt1.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t3_txt1.setFont(font)
        self.t3_txt1.setHidden(True)
        
        # frame around applications
        self.t3_frm1 = QFrame(self)   
        self.t3_frm1.setGeometry(20, 90, 780, 190)  
        self.t3_frm1.setFrameShape(QFrame.Panel)
        self.t3_frm1.setLineWidth(1)  
        self.t3_frm1.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t3_frm1.setHidden(True)
        
        # forecast activation label
        self.t3_txt2 = QLabel(self)
        self.t3_txt2.move(30, 100)
        self.t3_txt2.setFixedSize(200, 20)
        self.t3_txt2.setText(' forecasts') 
        self.t3_txt2.setAlignment(Qt.AlignLeft)
        self.t3_txt2.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t3_txt2.setHidden(True)
        
        # forecast radiobuttons
        self.t3_rdb1 = QRadioButton(' yes', self)  
        self.t3_rdb1.setGeometry(510, 100, 80, 25)
        self.t3_rdb1.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')  
        self.t3_rdb1.toggled.connect(self.cb_t3_bgr1)
        self.t3_rdb1.setHidden(True)  
        self.t3_rdb2 = QRadioButton(' no', self)   
        self.t3_rdb2.setGeometry(600, 100, 80, 25) 
        self.t3_rdb2.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')  
        self.t3_rdb2.toggled.connect(self.cb_t3_bgr1) 
        self.t3_rdb2.setHidden(True)
        if self.user_inputs['tab_3']['forecast']:
            self.t3_rdb1.setChecked(True) 
        else:
            self.t3_rdb2.setChecked(True)
        self.t3_bgr1 = QButtonGroup(self)  
        self.t3_bgr1.addButton(self.t3_rdb1) 
        self.t3_bgr1.addButton(self.t3_rdb2)
        
        # forecast credibility edit
        self.t3_edt1 = QLineEdit(self)
        self.t3_edt1.move(690, 100)       
        self.t3_edt1.resize(70, 25)                                           
        self.t3_edt1.setAlignment(Qt.AlignCenter)     
        self.t3_edt1.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t3_edt1.setText(self.user_inputs['tab_3']['forecast_credibility'])
        self.t3_edt1.textChanged.connect(self.cb_t3_edt1)
        self.t3_edt1.setHidden(True)
        
        # conditional forecast activation label
        self.t3_txt3 = QLabel(self)
        self.t3_txt3.move(30, 135)
        self.t3_txt3.setFixedSize(200, 20)
        self.t3_txt3.setText(' conditional forecasts') 
        self.t3_txt3.setAlignment(Qt.AlignLeft)
        self.t3_txt3.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t3_txt3.setHidden(True)
        
        # conditional forecast radiobuttons
        self.t3_rdb3 = QRadioButton(' yes', self)  
        self.t3_rdb3.setGeometry(510, 135, 80, 25)
        self.t3_rdb3.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')  
        self.t3_rdb3.toggled.connect(self.cb_t3_bgr2)
        self.t3_rdb3.setHidden(True)  
        self.t3_rdb4 = QRadioButton(' no', self)   
        self.t3_rdb4.setGeometry(600, 135, 80, 25) 
        self.t3_rdb4.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')  
        self.t3_rdb4.toggled.connect(self.cb_t3_bgr2) 
        self.t3_rdb4.setHidden(True)
        if self.user_inputs['tab_3']['conditional_forecast']:
            self.t3_rdb3.setChecked(True) 
        else:
            self.t3_rdb4.setChecked(True)
        self.t3_bgr2 = QButtonGroup(self)  
        self.t3_bgr2.addButton(self.t3_rdb3) 
        self.t3_bgr2.addButton(self.t3_rdb4) 
        
        # conditional forecast credibility edit
        self.t3_edt2 = QLineEdit(self)
        self.t3_edt2.move(690, 135)
        self.t3_edt2.resize(70, 25)                                           
        self.t3_edt2.setAlignment(Qt.AlignCenter)     
        self.t3_edt2.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t3_edt2.setText(self.user_inputs['tab_3']['conditional_forecast_credibility'])
        self.t3_edt2.textChanged.connect(self.cb_t3_edt2)
        self.t3_edt2.setHidden(True)
        
        # irf activation label
        self.t3_txt4 = QLabel(self)
        self.t3_txt4.move(30, 170)
        self.t3_txt4.setFixedSize(450, 20)
        self.t3_txt4.setText(' impulse response functions') 
        self.t3_txt4.setAlignment(Qt.AlignLeft)
        self.t3_txt4.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t3_txt4.setHidden(True)
        
        # irf radiobuttons
        self.t3_rdb5 = QRadioButton(' yes', self)  
        self.t3_rdb5.setGeometry(510, 170, 80, 25)
        self.t3_rdb5.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')  
        self.t3_rdb5.toggled.connect(self.cb_t3_bgr3)
        self.t3_rdb5.setHidden(True)     
        self.t3_rdb6 = QRadioButton(' no', self)   
        self.t3_rdb6.setGeometry(600, 170, 80, 25) 
        self.t3_rdb6.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')  
        self.t3_rdb6.toggled.connect(self.cb_t3_bgr3) 
        self.t3_rdb6.setHidden(True)
        if self.user_inputs['tab_3']['irf']:
            self.t3_rdb5.setChecked(True) 
        else:
            self.t3_rdb6.setChecked(True)
        self.t3_bgr3 = QButtonGroup(self)  
        self.t3_bgr3.addButton(self.t3_rdb5) 
        self.t3_bgr3.addButton(self.t3_rdb6) 
        
        # irf credibility edit
        self.t3_edt3 = QLineEdit(self)
        self.t3_edt3.move(690, 170)       
        self.t3_edt3.resize(70, 25)                                           
        self.t3_edt3.setAlignment(Qt.AlignCenter)     
        self.t3_edt3.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t3_edt3.setText(self.user_inputs['tab_3']['irf_credibility'])
        self.t3_edt3.textChanged.connect(self.cb_t3_edt3)
        self.t3_edt3.setHidden(True)
        
        # fevd activation label
        self.t3_txt5 = QLabel(self)
        self.t3_txt5.move(30, 205)
        self.t3_txt5.setFixedSize(450, 20)
        self.t3_txt5.setText(' forecast error variance decomposition') 
        self.t3_txt5.setAlignment(Qt.AlignLeft)
        self.t3_txt5.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t3_txt5.setHidden(True)
        
        # fevd radiobuttons
        self.t3_rdb7 = QRadioButton(' yes', self)  
        self.t3_rdb7.setGeometry(510, 205, 80, 25)
        self.t3_rdb7.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')  
        self.t3_rdb7.toggled.connect(self.cb_t3_bgr4)
        self.t3_rdb7.setHidden(True)   
        self.t3_rdb8 = QRadioButton(' no', self)   
        self.t3_rdb8.setGeometry(600, 205, 80, 25) 
        self.t3_rdb8.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')  
        self.t3_rdb8.toggled.connect(self.cb_t3_bgr4) 
        self.t3_rdb8.setHidden(True)
        if self.user_inputs['tab_3']['fevd']:
            self.t3_rdb7.setChecked(True) 
        else:
            self.t3_rdb8.setChecked(True)
        self.t3_bgr4 = QButtonGroup(self)  
        self.t3_bgr4.addButton(self.t3_rdb7) 
        self.t3_bgr4.addButton(self.t3_rdb8) 
        
        # fevd credibility edit
        self.t3_edt4 = QLineEdit(self)
        self.t3_edt4.move(690, 205)       
        self.t3_edt4.resize(70, 25)                                           
        self.t3_edt4.setAlignment(Qt.AlignCenter)     
        self.t3_edt4.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t3_edt4.setText(self.user_inputs['tab_3']['fevd_credibility'])
        self.t3_edt4.textChanged.connect(self.cb_t3_edt4)
        self.t3_edt4.setHidden(True)
        
        # historical decomposition activation label
        self.t3_txt6 = QLabel(self)
        self.t3_txt6.move(30, 240)
        self.t3_txt6.setFixedSize(450, 20)
        self.t3_txt6.setText(' historical decomposition') 
        self.t3_txt6.setAlignment(Qt.AlignLeft)
        self.t3_txt6.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t3_txt6.setHidden(True)
        
        # historical decomposition radiobuttons
        self.t3_rdb9 = QRadioButton(' yes', self)  
        self.t3_rdb9.setGeometry(510, 240, 80, 25)
        self.t3_rdb9.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')  
        self.t3_rdb9.toggled.connect(self.cb_t3_bgr5)
        self.t3_rdb9.setHidden(True)   
        self.t3_rdb10 = QRadioButton(' no', self)   
        self.t3_rdb10.setGeometry(600, 240, 80, 25) 
        self.t3_rdb10.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')  
        self.t3_rdb10.toggled.connect(self.cb_t3_bgr5) 
        self.t3_rdb10.setHidden(True)
        if self.user_inputs['tab_3']['hd']:
            self.t3_rdb9.setChecked(True) 
        else:
            self.t3_rdb10.setChecked(True)
        self.t3_bgr5 = QButtonGroup(self)  
        self.t3_bgr5.addButton(self.t3_rdb9) 
        self.t3_bgr5.addButton(self.t3_rdb10) 
        
        # historical decomposition credibility edit
        self.t3_edt5 = QLineEdit(self)
        self.t3_edt5.move(690, 240)       
        self.t3_edt5.resize(70, 25)                                           
        self.t3_edt5.setAlignment(Qt.AlignCenter)     
        self.t3_edt5.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t3_edt5.setText(self.user_inputs['tab_3']['hd_credibility'])
        self.t3_edt5.textChanged.connect(self.cb_t3_edt5)
        self.t3_edt5.setHidden(True)
        
        # forecasts label
        self.t3_txt7 = QLabel(self)
        self.t3_txt7.move(30, 300)
        self.t3_txt7.setFixedSize(300, 30)
        self.t3_txt7.setText(' Forecasts') 
        self.t3_txt7.setAlignment(Qt.AlignLeft)
        self.t3_txt7.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t3_txt7.setFont(font)
        self.t3_txt7.setHidden(True)
        
        # frame around forecast settings
        self.t3_frm2 = QFrame(self)   
        self.t3_frm2.setGeometry(20, 330, 380, 300)  
        self.t3_frm2.setFrameShape(QFrame.Panel)
        self.t3_frm2.setLineWidth(1)  
        self.t3_frm2.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t3_frm2.setHidden(True)
        
        # forecast periods label
        self.t3_txt8 = QLabel(self)
        self.t3_txt8.move(30, 345)
        self.t3_txt8.setFixedSize(200, 30)
        self.t3_txt8.setText(' forecast periods') 
        self.t3_txt8.setAlignment(Qt.AlignLeft)
        self.t3_txt8.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t3_txt8.setHidden(True)
        
        # forecast periods edit
        self.t3_edt6 = QLineEdit(self)
        self.t3_edt6.move(290, 340)
        self.t3_edt6.resize(70, 25)                                           
        self.t3_edt6.setAlignment(Qt.AlignCenter)     
        self.t3_edt6.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t3_edt6.setText(self.user_inputs['tab_3']['forecast_periods'])
        self.t3_edt6.textChanged.connect(self.cb_t3_edt6)
        self.t3_edt6.setHidden(True)        
        
        # conditional forecast type label
        self.t3_txt9 = QLabel(self)
        self.t3_txt9.move(30, 385)
        self.t3_txt9.setFixedSize(250, 30)
        self.t3_txt9.setText(' conditional forecast type') 
        self.t3_txt9.setAlignment(Qt.AlignLeft)
        self.t3_txt9.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t3_txt9.setHidden(True)

        # conditional forecast type menu
        self.t3_mnu1 = QComboBox(self)
        self.t3_mnu1.move(35, 415)                                        
        self.t3_mnu1.resize(200, 25)
        self.t3_mnu1.setStyleSheet('QListView{background-color: white}')
        self.t3_mnu1.addItem('1. agnostic', 1)
        self.t3_mnu1.addItem('2. structural shocks', 2)        
        self.t3_mnu1.setCurrentIndex(self.user_inputs['tab_3']['conditional_forecast_type'] - 1)
        self.t3_mnu1.activated.connect(self.cb_t3_mnu1)
        self.t3_mnu1.setHidden(True)        
        
        # forecast input file label
        self.t3_txt10 = QLabel(self)
        self.t3_txt10.move(30, 460)
        self.t3_txt10.setFixedSize(250, 30)
        self.t3_txt10.setText(' forecast input file') 
        self.t3_txt10.setAlignment(Qt.AlignLeft)
        self.t3_txt10.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t3_txt10.setHidden(True)
        
        # forecast file edit
        self.t3_edt7 = QLineEdit(self)
        self.t3_edt7.move(35, 485)       
        self.t3_edt7.resize(325, 25)                                           
        self.t3_edt7.setAlignment(Qt.AlignLeft)     
        self.t3_edt7.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t3_edt7.setText(self.user_inputs['tab_3']['forecast_file'])
        self.t3_edt7.textChanged.connect(self.cb_t3_edt7)
        self.t3_edt7.setHidden(True)   
        
        # conditional forecast input file label
        self.t3_txt11 = QLabel(self)
        self.t3_txt11.move(30, 530)
        self.t3_txt11.setFixedSize(250, 30)
        self.t3_txt11.setText(' conditional forecast input file') 
        self.t3_txt11.setAlignment(Qt.AlignLeft)
        self.t3_txt11.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t3_txt11.setHidden(True)
        
        # conditional forecast file edit
        self.t3_edt8 = QLineEdit(self)
        self.t3_edt8.move(35, 555)       
        self.t3_edt8.resize(325, 25)                                           
        self.t3_edt8.setAlignment(Qt.AlignLeft)     
        self.t3_edt8.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t3_edt8.setText(self.user_inputs['tab_3']['conditional_forecast_file'])
        self.t3_edt8.textChanged.connect(self.cb_t3_edt8)
        self.t3_edt8.setHidden(True) 

        # forecast evaluation label
        self.t3_txt12 = QLabel(self)
        self.t3_txt12.move(30, 600)
        self.t3_txt12.setFixedSize(200, 25)
        self.t3_txt12.setText(' forecast evaluation') 
        self.t3_txt12.setAlignment(Qt.AlignLeft)
        self.t3_txt12.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t3_txt12.setHidden(True)
        
        # forecast evaluation checkbox
        self.t3_cbx1 = QCheckBox(self)
        self.t3_cbx1.setGeometry(345, 600, 20, 20) 
        self.t3_cbx1.setStyleSheet("QCheckBox::indicator:unchecked \
        {background-color : rgb(255, 255, 255); border: 0.5px solid rgb(0, 0, 0)}; \
        QCheckBox::indicator {width: 5px; height: 12px}") 
        self.t3_cbx1.setChecked(self.user_inputs['tab_3']['forecast_evaluation'])
        self.t3_cbx1.stateChanged.connect(self.cb_t3_cbx1) 
        self.t3_cbx1.setHidden(True)
        
        # IRF label
        self.t3_txt13 = QLabel(self)
        self.t3_txt13.move(430, 300)
        self.t3_txt13.setFixedSize(350, 30)
        self.t3_txt13.setText(' Impulse response function') 
        self.t3_txt13.setAlignment(Qt.AlignLeft)
        self.t3_txt13.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t3_txt13.setFont(font)
        self.t3_txt13.setHidden(True)
        
        # frame around IRF settings
        self.t3_frm3 = QFrame(self)   
        self.t3_frm3.setGeometry(420, 330, 380, 300)  
        self.t3_frm3.setFrameShape(QFrame.Panel)
        self.t3_frm3.setLineWidth(1)  
        self.t3_frm3.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t3_frm3.setHidden(True)
        
        # irf periods label
        self.t3_txt14 = QLabel(self)
        self.t3_txt14.move(430, 345)
        self.t3_txt14.setFixedSize(200, 30)
        self.t3_txt14.setText(' IRF periods')
        self.t3_txt14.setAlignment(Qt.AlignLeft)
        self.t3_txt14.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t3_txt14.setHidden(True)
        
        # irf periods edit
        self.t3_edt9 = QLineEdit(self)
        self.t3_edt9.move(690, 340)   
        self.t3_edt9.resize(70, 25)                                           
        self.t3_edt9.setAlignment(Qt.AlignCenter)     
        self.t3_edt9.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t3_edt9.setText(self.user_inputs['tab_3']['irf_periods'])
        self.t3_edt9.textChanged.connect(self.cb_t3_edt9)
        self.t3_edt9.setHidden(True)    

        # structural identification label
        self.t3_txt15 = QLabel(self)
        self.t3_txt15.move(430, 385)
        self.t3_txt15.setFixedSize(250, 30)
        self.t3_txt15.setText(' structural identification')
        self.t3_txt15.setAlignment(Qt.AlignLeft)
        self.t3_txt15.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t3_txt15.setHidden(True)
        
        # structural identification menu
        self.t3_mnu2 = QComboBox(self)
        self.t3_mnu2.move(435, 415)                                             
        self.t3_mnu2.resize(200, 25)
        self.t3_mnu2.setStyleSheet('QListView{background-color: white}')
        self.t3_mnu2.addItem('1. none', 1)
        self.t3_mnu2.addItem('2. Cholesky', 2)
        self.t3_mnu2.addItem('3. triangular', 3)   
        self.t3_mnu2.addItem('4. restrictions', 4)   
        self.t3_mnu2.setCurrentIndex(self.user_inputs['tab_3']['structural_identification'] - 1)
        self.t3_mnu2.activated.connect(self.cb_t3_mnu2)
        self.t3_mnu2.setHidden(True)   
        
        # structural identification file label
        self.t3_txt16 = QLabel(self)
        self.t3_txt16.move(430, 460)
        self.t3_txt16.setFixedSize(300, 30)
        self.t3_txt16.setText(' structural identification file')
        self.t3_txt16.setAlignment(Qt.AlignLeft)
        self.t3_txt16.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t3_txt16.setHidden(True)
        
        # structural identification file edit
        self.t3_edt10 = QLineEdit(self)
        self.t3_edt10.move(435, 485)       
        self.t3_edt10.resize(325, 25)                                           
        self.t3_edt10.setAlignment(Qt.AlignLeft)     
        self.t3_edt10.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t3_edt10.setText(self.user_inputs['tab_3']['structural_identification_file'])
        self.t3_edt10.textChanged.connect(self.cb_t3_edt10)
        self.t3_edt10.setHidden(True)
        
        # run pushbutton
        self.t3_pbt1 = QPushButton(self)
        self.t3_pbt1.move(820, 340)  
        self.t3_pbt1.resize(160, 260)
        self.t3_pbt1.setStyleSheet('font-size: 17px; font-family: Serif')
        self.t3_pbt1.setIcon(QIcon(join(self.interface_path, 'run_button.png')))  
        self.t3_pbt1.setIconSize(QSize(160, 260))
        self.t3_pbt1.clicked.connect(self.cb_t3_pbt1)
        self.t3_pbt1.setHidden(True)    
    
    
    def hide_tab_3(self):    
        
        # hide all controls
        self.t3_txt1.setHidden(True)
        self.t3_txt2.setHidden(True)
        self.t3_txt3.setHidden(True)        
        self.t3_txt4.setHidden(True)
        self.t3_txt5.setHidden(True)
        self.t3_txt6.setHidden(True)
        self.t3_txt7.setHidden(True)
        self.t3_txt8.setHidden(True)
        self.t3_txt9.setHidden(True)
        self.t3_txt10.setHidden(True)
        self.t3_txt11.setHidden(True)
        self.t3_txt12.setHidden(True)
        self.t3_txt13.setHidden(True)
        self.t3_txt14.setHidden(True)
        self.t3_txt15.setHidden(True)
        self.t3_txt16.setHidden(True)
        self.t3_frm1.setHidden(True)
        self.t3_frm2.setHidden(True)
        self.t3_frm3.setHidden(True)
        self.t3_rdb1.setHidden(True)
        self.t3_rdb2.setHidden(True)
        self.t3_rdb3.setHidden(True)
        self.t3_rdb4.setHidden(True)
        self.t3_rdb5.setHidden(True)
        self.t3_rdb6.setHidden(True)
        self.t3_rdb7.setHidden(True)
        self.t3_rdb8.setHidden(True)
        self.t3_rdb9.setHidden(True)
        self.t3_rdb10.setHidden(True)
        self.t3_edt1.setHidden(True)
        self.t3_edt2.setHidden(True)
        self.t3_edt3.setHidden(True)
        self.t3_edt4.setHidden(True)
        self.t3_edt5.setHidden(True)
        self.t3_edt6.setHidden(True)
        self.t3_edt7.setHidden(True)
        self.t3_edt8.setHidden(True)
        self.t3_edt9.setHidden(True)
        self.t3_edt10.setHidden(True)
        self.t3_pbt1.setHidden(True)
        self.t3_cbx1.setHidden(True)
        self.t3_mnu1.setHidden(True)
        self.t3_mnu2.setHidden(True)
    
        # update tab color
        self.tab_pbt3.setStyleSheet("background:rgb" + str(self.backtabs_color) + ";") 


    def show_tab_3(self):    
        
        # show all controls
        self.t3_txt1.setVisible(True)
        self.t3_txt2.setVisible(True)
        self.t3_txt3.setVisible(True)
        self.t3_txt4.setVisible(True)
        self.t3_txt5.setVisible(True)
        self.t3_txt6.setVisible(True)
        self.t3_txt7.setVisible(True)
        self.t3_txt8.setVisible(True)
        self.t3_txt9.setVisible(True)
        self.t3_txt10.setVisible(True)
        self.t3_txt11.setVisible(True)
        self.t3_txt12.setVisible(True)
        self.t3_txt13.setVisible(True)
        self.t3_txt14.setVisible(True)
        self.t3_txt15.setVisible(True)
        self.t3_txt16.setVisible(True)
        self.t3_frm1.setVisible(True)
        self.t3_frm2.setVisible(True)
        self.t3_frm3.setVisible(True)
        self.t3_rdb1.setVisible(True)
        self.t3_rdb2.setVisible(True)
        self.t3_rdb3.setVisible(True)
        self.t3_rdb4.setVisible(True)
        self.t3_rdb5.setVisible(True)
        self.t3_rdb6.setVisible(True)
        self.t3_rdb7.setVisible(True)
        self.t3_rdb8.setVisible(True)
        self.t3_rdb9.setVisible(True)
        self.t3_rdb10.setVisible(True)
        self.t3_edt1.setVisible(True)
        self.t3_edt2.setVisible(True)
        self.t3_edt3.setVisible(True)
        self.t3_edt4.setVisible(True)
        self.t3_edt5.setVisible(True)
        self.t3_edt6.setVisible(True)
        self.t3_edt7.setVisible(True)
        self.t3_edt8.setVisible(True)
        self.t3_edt9.setVisible(True)
        self.t3_edt10.setVisible(True)
        self.t3_pbt1.setVisible(True)
        self.t3_cbx1.setVisible(True)
        self.t3_mnu1.setVisible(True)
        self.t3_mnu2.setVisible(True)


    def cb_t3_bgr1(self):
        if self.t3_rdb1.isChecked() == True:
            self.user_inputs['tab_3']['forecast'] = True
        elif self.t3_rdb2.isChecked() == True:
            self.user_inputs['tab_3']['forecast'] = False
 
    
    def cb_t3_edt1(self):
        self.user_inputs['tab_3']['forecast_credibility'] = self.t3_edt1.text()      
   
    
    def cb_t3_bgr2(self):
        if self.t3_rdb3.isChecked() == True:
            self.user_inputs['tab_3']['conditional_forecast'] = True
        elif self.t3_rdb4.isChecked() == True:
            self.user_inputs['tab_3']['conditional_forecast'] = False
 
    
    def cb_t3_edt2(self):
        self.user_inputs['tab_3']['conditional_forecast_credibility'] = self.t3_edt2.text()     
 
    
    def cb_t3_bgr3(self):
        if self.t3_rdb5.isChecked() == True:
            self.user_inputs['tab_3']['irf'] = True
        elif self.t3_rdb6.isChecked() == True:
            self.user_inputs['tab_3']['irf'] = False
 
    
    def cb_t3_edt3(self):
        self.user_inputs['tab_3']['irf_credibility'] = self.t3_edt3.text()     
    
    
    def cb_t3_bgr4(self):
        if self.t3_rdb7.isChecked() == True:
            self.user_inputs['tab_3']['fevd'] = True
        elif self.t3_rdb8.isChecked() == True:
            self.user_inputs['tab_3']['fevd'] = False
 
    
    def cb_t3_edt4(self):
        self.user_inputs['tab_3']['fevd_credibility'] = self.t3_edt4.text()     
 
    
    def cb_t3_bgr5(self):
        if self.t3_rdb9.isChecked() == True:
            self.user_inputs['tab_3']['hd'] = True
        elif self.t3_rdb10.isChecked() == True:
            self.user_inputs['tab_3']['hd'] = False
 
    
    def cb_t3_edt5(self):
        self.user_inputs['tab_3']['hd_credibility'] = self.t3_edt5.text()     
 
    
    def cb_t3_edt6(self):
        self.user_inputs['tab_3']['forecast_periods'] = self.t3_edt6.text()      
 
    
    def cb_t3_mnu1(self, index):        
        self.user_inputs['tab_3']['conditional_forecast_type'] = self.t3_mnu1.itemData(index) 


    def cb_t3_edt7(self):
        self.user_inputs['tab_3']['forecast_file'] = self.t3_edt7.text()     
 
    
    def cb_t3_edt8(self):
        self.user_inputs['tab_3']['conditional_forecast_file'] = self.t3_edt8.text() 
         
    
    def cb_t3_cbx1(self, state):     
        if (state == Qt.Checked):
            self.user_inputs['tab_3']['forecast_evaluation'] = True 
        else:
            self.user_inputs['tab_3']['forecast_evaluation'] = False              
             
            
    def cb_t3_edt9(self):
        self.user_inputs['tab_3']['irf_periods'] = self.t3_edt9.text()     


    def cb_t3_mnu2(self, index):        
        self.user_inputs['tab_3']['structural_identification'] = self.t3_mnu2.itemData(index) 
    

    def cb_t3_edt10(self):
        self.user_inputs['tab_3']['structural_identification_file'] = self.t3_edt10.text()  


    def cb_t3_pbt1(self):
        self.validate_interface()

        