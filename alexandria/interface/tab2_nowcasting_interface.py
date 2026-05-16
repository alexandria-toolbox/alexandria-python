# imports
from PyQt5.QtWidgets import QLabel, QFrame, QComboBox, QRadioButton, QButtonGroup, QLineEdit, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont



class Tab2NowcastingInterface(object):


    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass


    def create_tab_2_now(self):
    
        # model label
        self.t2_now_txt1 = QLabel(self)
        self.t2_now_txt1.move(30, 60)
        self.t2_now_txt1.setFixedSize(300, 30)
        self.t2_now_txt1.setText(' Model') 
        self.t2_now_txt1.setAlignment(Qt.AlignLeft)
        self.t2_now_txt1.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_now_txt1.setFont(font)
        self.t2_now_txt1.setHidden(True)
     
        # frame around model
        self.t2_now_frm1 = QFrame(self)   
        self.t2_now_frm1.setGeometry(20, 90, 470, 145)  
        self.t2_now_frm1.setFrameShape(QFrame.Panel)
        self.t2_now_frm1.setLineWidth(1)  
        self.t2_now_frm1.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t2_now_frm1.setHidden(True)   
        
        # Selection label
        self.t2_now_txt2 = QLabel(self)
        self.t2_now_txt2.move(30, 95)
        self.t2_now_txt2.setFixedSize(200, 30)
        self.t2_now_txt2.setText(' Selection') 
        self.t2_now_txt2.setAlignment(Qt.AlignLeft)
        self.t2_now_txt2.setStyleSheet('font-size: 14pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_now_txt2.setFont(font)
        self.t2_now_txt2.setHidden(True)
    
        # model radiobuttons
        self.t2_now_rdb1 = QRadioButton(' mixed frequency BVAR', self)
        self.t2_now_rdb1.setGeometry(30, 120, 200, 30)
        self.t2_now_rdb1.setStyleSheet("font-size: 12pt; font-family: Serif;")  
        self.t2_now_rdb1.toggled.connect(self.cb_t2_now_bgr1)
        self.t2_now_rdb1.setHidden(True)
        self.t2_now_rdb2 = QRadioButton(' dynamic factor model', self)
        self.t2_now_rdb2.setGeometry(30, 147, 200, 30)
        self.t2_now_rdb2.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_now_rdb2.toggled.connect(self.cb_t2_now_bgr1)
        self.t2_now_rdb2.setHidden(True)
        self.t2_now_rdb3 = QRadioButton(' Midas regression', self)
        self.t2_now_rdb3.setGeometry(30, 174, 200, 30)
        self.t2_now_rdb3.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_now_rdb3.toggled.connect(self.cb_t2_now_bgr1)
        self.t2_now_rdb3.setHidden(True)
        if self.user_inputs['tab_2_now']['model'] == 1:
            self.t2_now_rdb1.setChecked(True) 
        elif self.user_inputs['tab_2_now']['model'] == 2:
            self.t2_now_rdb2.setChecked(True) 
        elif self.user_inputs['tab_2_now']['model'] == 3:
            self.t2_now_rdb3.setChecked(True) 
        self.t2_now_bgr1 = QButtonGroup(self)  
        self.t2_now_bgr1.addButton(self.t2_now_rdb1) 
        self.t2_now_bgr1.addButton(self.t2_now_rdb2)     
        self.t2_now_bgr1.addButton(self.t2_now_rdb3) 

        # Gibbs sampling label
        self.t2_now_txt3 = QLabel(self)
        self.t2_now_txt3.move(260, 95)
        self.t2_now_txt3.setFixedSize(200, 30)
        self.t2_now_txt3.setText(' Gibbs sampling') 
        self.t2_now_txt3.setAlignment(Qt.AlignLeft)
        self.t2_now_txt3.setStyleSheet('font-size: 14pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_now_txt3.setFont(font)
        self.t2_now_txt3.setHidden(True)        
    
        # iteration label
        self.t2_now_txt4 = QLabel(self)
        self.t2_now_txt4.move(260, 124)
        self.t2_now_txt4.setFixedSize(200, 25)
        self.t2_now_txt4.setText(' iterations') 
        self.t2_now_txt4.setAlignment(Qt.AlignLeft)
        self.t2_now_txt4.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt4.setHidden(True)

        # iteration edit
        self.t2_now_edt1 = QLineEdit(self)
        self.t2_now_edt1.move(410, 121)       
        self.t2_now_edt1.resize(70, 23)                                           
        self.t2_now_edt1.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt1.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt1.setText(self.user_inputs['tab_2_now']['iterations'])
        self.t2_now_edt1.textChanged.connect(self.cb_t2_now_edt1)
        self.t2_now_edt1.setHidden(True)    
    
        # burn-in label
        self.t2_now_txt5 = QLabel(self)
        self.t2_now_txt5.move(260, 151)
        self.t2_now_txt5.setFixedSize(200, 25)
        self.t2_now_txt5.setText(' burn-in') 
        self.t2_now_txt5.setAlignment(Qt.AlignLeft)
        self.t2_now_txt5.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt5.setHidden(True)

        # burn-in edit
        self.t2_now_edt2 = QLineEdit(self)
        self.t2_now_edt2.move(410, 148)       
        self.t2_now_edt2.resize(70, 23)                                           
        self.t2_now_edt2.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt2.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt2.setText(self.user_inputs['tab_2_now']['burnin'])
        self.t2_now_edt2.textChanged.connect(self.cb_t2_now_edt2)
        self.t2_now_edt2.setHidden(True)  
    
        # credibility label
        self.t2_now_txt6 = QLabel(self)
        self.t2_now_txt6.move(260, 178)
        self.t2_now_txt6.setFixedSize(200, 20)
        self.t2_now_txt6.setText(' credibility level') 
        self.t2_now_txt6.setAlignment(Qt.AlignLeft)
        self.t2_now_txt6.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt6.setHidden(True)        

        # credibility edit
        self.t2_now_edt3 = QLineEdit(self)
        self.t2_now_edt3.move(410, 175)       
        self.t2_now_edt3.resize(70, 23)                                           
        self.t2_now_edt3.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt3.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt3.setText(self.user_inputs['tab_2_now']['model_credibility'])
        self.t2_now_edt3.textChanged.connect(self.cb_t2_now_edt3)
        self.t2_now_edt3.setHidden(True)    
    
        # midas label
        self.t2_now_txt7 = QLabel(self)
        self.t2_now_txt7.move(520, 60)
        self.t2_now_txt7.setFixedSize(300, 30)
        self.t2_now_txt7.setText(' Bayesian MIDAS regression') 
        self.t2_now_txt7.setAlignment(Qt.AlignLeft)
        self.t2_now_txt7.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_now_txt7.setFont(font)
        self.t2_now_txt7.setHidden(True)

        # frame around midas regression
        self.t2_now_frm2 = QFrame(self)   
        self.t2_now_frm2.setGeometry(510, 90, 470, 145)  
        self.t2_now_frm2.setFrameShape(QFrame.Panel)
        self.t2_now_frm2.setLineWidth(1)  
        self.t2_now_frm2.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t2_now_frm2.setHidden(True)       
    
        # prior type label
        self.t2_now_txt11 = QLabel(self)
        self.t2_now_txt11.move(520, 97)
        self.t2_now_txt11.setFixedSize(200, 25)
        self.t2_now_txt11.setText(' model') 
        self.t2_now_txt11.setAlignment(Qt.AlignLeft)
        self.t2_now_txt11.setStyleSheet('font-size: 11pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt11.setHidden(True)  
    
        # model menu
        self.t2_now_mnu1 = QComboBox(self)
        self.t2_now_mnu1.move(680,94)                                             
        self.t2_now_mnu1.resize(290,23)
        self.t2_now_mnu1.setStyleSheet('QListView{background-color: white}')
        self.t2_now_mnu1.addItem('unrestricted - minnesota', 1)
        self.t2_now_mnu1.addItem('unrestricted - horseshoe', 2)
        self.t2_now_mnu1.addItem('unrestricted - lasso', 3)
        self.t2_now_mnu1.addItem('almon - minnesota', 4)
        self.t2_now_mnu1.addItem('almon - horseshoe', 5) 
        self.t2_now_mnu1.addItem('almon - lasso', 6)         
        self.t2_now_mnu1.addItem('fourier - minnesota', 7)         
        self.t2_now_mnu1.addItem('fourier - horseshoe', 8) 
        self.t2_now_mnu1.addItem('fourier - lasso', 9)         
        # self.t2_now_mnu1.setCurrentIndex(self.user_inputs['tab_2_now']['midas_prior_type'] - 1)
        self.t2_now_mnu1.setCurrentIndex(self.user_inputs['tab_2_now']['midas_model'] - 1)
        self.t2_now_mnu1.activated.connect(self.cb_t2_now_mnu1)
        self.t2_now_mnu1.setHidden(True)      
    
        # endogenous lags label
        self.t2_now_txt8 = QLabel(self)
        self.t2_now_txt8.move(520, 124)
        self.t2_now_txt8.setFixedSize(200, 25)
        self.t2_now_txt8.setText(' endogenous lags') 
        self.t2_now_txt8.setAlignment(Qt.AlignLeft)
        self.t2_now_txt8.setStyleSheet('font-size: 11pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt8.setHidden(True)

        # endogenous lags edit
        self.t2_now_edt4 = QLineEdit(self)
        self.t2_now_edt4.move(680, 121)       
        self.t2_now_edt4.resize(70, 23)                                           
        self.t2_now_edt4.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt4.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt4.setText(self.user_inputs['tab_2_now']['midas_endogenous_lags'])
        self.t2_now_edt4.textChanged.connect(self.cb_t2_now_edt4)
        self.t2_now_edt4.setHidden(True)  
    
        # exogenous lags label
        self.t2_now_txt9 = QLabel(self)
        self.t2_now_txt9.move(760, 124)
        self.t2_now_txt9.setFixedSize(200, 25)
        self.t2_now_txt9.setText(' exogenous lags') 
        self.t2_now_txt9.setAlignment(Qt.AlignLeft)
        self.t2_now_txt9.setStyleSheet('font-size: 11pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt9.setHidden(True)

        # exogenous lags edit
        self.t2_now_edt5 = QLineEdit(self)
        self.t2_now_edt5.move(900, 121)    
        self.t2_now_edt5.resize(70, 23)                                           
        self.t2_now_edt5.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt5.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt5.setText(self.user_inputs['tab_2_now']['midas_exogenous_lags'])
        self.t2_now_edt5.textChanged.connect(self.cb_t2_now_edt5)
        self.t2_now_edt5.setHidden(True)   
    
        # endogenous tightness label
        self.t2_now_txt12 = QLabel(self)
        self.t2_now_txt12.move(520, 151)
        self.t2_now_txt12.setFixedSize(200, 25)
        self.t2_now_txt12.setText(' ω₁: endo tightness') 
        self.t2_now_txt12.setAlignment(Qt.AlignLeft)
        self.t2_now_txt12.setStyleSheet('font-size: 11pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt12.setHidden(True)  
        
        # endogenous tightness edit
        self.t2_now_edt7 = QLineEdit(self)
        self.t2_now_edt7.move(680, 148) 
        self.t2_now_edt7.resize(70, 23)                                           
        self.t2_now_edt7.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt7.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt7.setText(self.user_inputs['tab_2_now']['midas_omega1'])
        self.t2_now_edt7.textChanged.connect(self.cb_t2_now_edt7)
        self.t2_now_edt7.setHidden(True)     
    
        # endogenous lag decay label
        self.t2_now_txt13 = QLabel(self)
        self.t2_now_txt13.move(520, 178)
        self.t2_now_txt13.setFixedSize(200, 25)
        self.t2_now_txt13.setText(' ω₂: endo lag decay') 
        self.t2_now_txt13.setAlignment(Qt.AlignLeft)
        self.t2_now_txt13.setStyleSheet('font-size: 11pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt13.setHidden(True)  
        
        # endogenous lag decay edit
        self.t2_now_edt8 = QLineEdit(self)
        self.t2_now_edt8.move(680, 175)       
        self.t2_now_edt8.resize(70, 23)                                           
        self.t2_now_edt8.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt8.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt8.setText(self.user_inputs['tab_2_now']['midas_omega2'])
        self.t2_now_edt8.textChanged.connect(self.cb_t2_now_edt8)
        self.t2_now_edt8.setHidden(True)    
    
        # exogenous tightness label
        self.t2_now_txt14 = QLabel(self)
        self.t2_now_txt14.move(760, 151)
        self.t2_now_txt14.setFixedSize(200, 25)
        self.t2_now_txt14.setText(' υ₁: exo tightness') 
        self.t2_now_txt14.setAlignment(Qt.AlignLeft)
        self.t2_now_txt14.setStyleSheet('font-size: 11pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt14.setHidden(True)  
        
        # exogenous tightness edit
        self.t2_now_edt9 = QLineEdit(self)
        self.t2_now_edt9.move(900, 148)
        self.t2_now_edt9.resize(70, 23)                                           
        self.t2_now_edt9.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt9.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt9.setText(self.user_inputs['tab_2_now']['midas_upsilon1'])
        self.t2_now_edt9.textChanged.connect(self.cb_t2_now_edt9)
        self.t2_now_edt9.setHidden(True)
 
        # exogenous lag decay label
        self.t2_now_txt15 = QLabel(self)
        self.t2_now_txt15.move(760, 178)
        self.t2_now_txt15.setFixedSize(200, 25)
        self.t2_now_txt15.setText(' υ₂: exo lag decay') 
        self.t2_now_txt15.setAlignment(Qt.AlignLeft)
        self.t2_now_txt15.setStyleSheet('font-size: 11pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt15.setHidden(True)  
        
        # exogenous lag decay edit
        self.t2_now_edt10 = QLineEdit(self)
        self.t2_now_edt10.move(900, 175)       
        self.t2_now_edt10.resize(70, 23)                                           
        self.t2_now_edt10.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt10.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt10.setText(self.user_inputs['tab_2_now']['midas_upsilon2'])
        self.t2_now_edt10.textChanged.connect(self.cb_t2_now_edt10)
        self.t2_now_edt10.setHidden(True)     
    
        # polynomial order label
        self.t2_now_txt10 = QLabel(self)
        self.t2_now_txt10.move(520, 205)
        self.t2_now_txt10.setFixedSize(200, 25)
        self.t2_now_txt10.setText(' polynomial order') 
        self.t2_now_txt10.setAlignment(Qt.AlignLeft)
        self.t2_now_txt10.setStyleSheet('font-size: 11pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt10.setHidden(True)

        # polynomial order edit
        self.t2_now_edt6 = QLineEdit(self)
        self.t2_now_edt6.move(680, 202)       
        self.t2_now_edt6.resize(70, 23)                                           
        self.t2_now_edt6.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt6.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt6.setText(self.user_inputs['tab_2_now']['midas_polynomial_order'])
        self.t2_now_edt6.textChanged.connect(self.cb_t2_now_edt6)
        self.t2_now_edt6.setHidden(True)         
    
        # mfbvar label
        self.t2_now_txt16 = QLabel(self)
        self.t2_now_txt16.move(30, 253)
        self.t2_now_txt16.setFixedSize(400, 30)
        self.t2_now_txt16.setText(' Mixed frequency Bayesian VAR') 
        self.t2_now_txt16.setAlignment(Qt.AlignLeft)
        self.t2_now_txt16.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_now_txt16.setFont(font)
        self.t2_now_txt16.setHidden(True)
        
        # frame around mfbvar
        self.t2_now_frm3 = QFrame(self)   
        self.t2_now_frm3.setGeometry(20, 283, 470, 347)  
        self.t2_now_frm3.setFrameShape(QFrame.Panel)
        self.t2_now_frm3.setLineWidth(1)  
        self.t2_now_frm3.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t2_now_frm3.setHidden(True)

        # constant label
        self.t2_now_txt17 = QLabel(self)
        self.t2_now_txt17.move(30, 293)
        self.t2_now_txt17.setFixedSize(300, 30)
        self.t2_now_txt17.setText(' constant') 
        self.t2_now_txt17.setAlignment(Qt.AlignLeft)
        self.t2_now_txt17.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt17.setHidden(True)   
        
        # constant checkbox
        self.t2_now_cbx1 = QCheckBox(self)
        self.t2_now_cbx1.setGeometry(460, 293, 20, 20) 
        self.t2_now_cbx1.setStyleSheet("QCheckBox::indicator:unchecked \
        {background-color : rgb(255, 255, 255); border: 0.5px solid rgb(0, 0, 0)}; \
        QCheckBox::indicator {width: 12px; height: 12px}") 
        self.t2_now_cbx1.setChecked(self.user_inputs['tab_2_now']['mfbvar_constant'])
        self.t2_now_cbx1.stateChanged.connect(self.cb_t2_now_cbx1) 
        self.t2_now_cbx1.setHidden(True)        
        
        # linear trend label
        self.t2_now_txt18 = QLabel(self)
        self.t2_now_txt18.move(30, 320)
        self.t2_now_txt18.setFixedSize(300, 30)
        self.t2_now_txt18.setText(' linear trend') 
        self.t2_now_txt18.setAlignment(Qt.AlignLeft)
        self.t2_now_txt18.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt18.setHidden(True)     
    
        # linear trend checkbox
        self.t2_now_cbx2 = QCheckBox(self)
        self.t2_now_cbx2.setGeometry(460, 320, 20, 20) 
        self.t2_now_cbx2.setStyleSheet("QCheckBox::indicator:unchecked \
        {background-color : rgb(255, 255, 255); border: 0.5px solid rgb(0, 0, 0)}; \
        QCheckBox::indicator {width: 12px; height: 12px}") 
        self.t2_now_cbx2.setChecked(self.user_inputs['tab_2_now']['mfbvar_trend'])
        self.t2_now_cbx2.stateChanged.connect(self.cb_t2_now_cbx2) 
        self.t2_now_cbx2.setHidden(True)     
    
        # quadratic trend label
        self.t2_now_txt19 = QLabel(self)
        self.t2_now_txt19.move(30, 347)
        self.t2_now_txt19.setFixedSize(300, 30)
        self.t2_now_txt19.setText(' quadratic trend') 
        self.t2_now_txt19.setAlignment(Qt.AlignLeft)
        self.t2_now_txt19.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt19.setHidden(True)      
    
        # quadratic trend checkbox
        self.t2_now_cbx3 = QCheckBox(self)
        self.t2_now_cbx3.setGeometry(460, 347, 20, 20) 
        self.t2_now_cbx3.setStyleSheet("QCheckBox::indicator:unchecked \
        {background-color : rgb(255, 255, 255); border: 0.5px solid rgb(0, 0, 0)}; \
        QCheckBox::indicator {width: 12px; height: 12px}") 
        self.t2_now_cbx3.setChecked(self.user_inputs['tab_2_now']['mfbvar_quadratic_trend'])
        self.t2_now_cbx3.stateChanged.connect(self.cb_t2_now_cbx3) 
        self.t2_now_cbx3.setHidden(True)      
    
        # decomposition label
        self.t2_now_txt20 = QLabel(self)
        self.t2_now_txt20.move(30, 374)
        self.t2_now_txt20.setFixedSize(300, 30)
        self.t2_now_txt20.setText(' decomposition') 
        self.t2_now_txt20.setAlignment(Qt.AlignLeft)
        self.t2_now_txt20.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt20.setHidden(True)      
    
        # decomposition checkbox
        self.t2_now_cbx4 = QCheckBox(self)
        self.t2_now_cbx4.setGeometry(460, 374, 20, 20) 
        self.t2_now_cbx4.setStyleSheet("QCheckBox::indicator:unchecked \
        {background-color : rgb(255, 255, 255); border: 0.5px solid rgb(0, 0, 0)}; \
        QCheckBox::indicator {width: 12px; height: 12px}") 
        self.t2_now_cbx4.setChecked(self.user_inputs['tab_2_now']['mfbvar_decomposition'])
        self.t2_now_cbx4.stateChanged.connect(self.cb_t2_now_cbx4) 
        self.t2_now_cbx4.setHidden(True)     
    
        # lags label
        self.t2_now_txt21 = QLabel(self)
        self.t2_now_txt21.move(30, 401)
        self.t2_now_txt21.setFixedSize(300, 30)
        self.t2_now_txt21.setText(' p:    lags') 
        self.t2_now_txt21.setAlignment(Qt.AlignLeft)
        self.t2_now_txt21.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt21.setHidden(True)    
    
        # lags edit
        self.t2_now_edt11 = QLineEdit(self)
        self.t2_now_edt11.move(335, 404)       
        self.t2_now_edt11.resize(140, 22)                                           
        self.t2_now_edt11.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt11.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt11.setText(self.user_inputs['tab_2_now']['mfbvar_lags'])
        self.t2_now_edt11.textChanged.connect(self.cb_t2_now_edt11)
        self.t2_now_edt11.setHidden(True)    
    
        # ar coefficients label
        self.t2_now_txt22 = QLabel(self)
        self.t2_now_txt22.move(30, 428)
        self.t2_now_txt22.setFixedSize(300, 30)
        self.t2_now_txt22.setText(' δ:    AR coefficients') 
        self.t2_now_txt22.setAlignment(Qt.AlignLeft)
        self.t2_now_txt22.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt22.setHidden(True)      
    
        # ar coefficients edit
        self.t2_now_edt12 = QLineEdit(self)
        self.t2_now_edt12.move(335, 431)       
        self.t2_now_edt12.resize(140, 22)                                           
        self.t2_now_edt12.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt12.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt12.setText(self.user_inputs['tab_2_now']['mfbvar_ar_coefficients'])
        self.t2_now_edt12.textChanged.connect(self.cb_t2_now_edt12)
        self.t2_now_edt12.setHidden(True)     
    
        # pi1 label
        self.t2_now_txt23 = QLabel(self)
        self.t2_now_txt23.move(30, 455)
        self.t2_now_txt23.setFixedSize(300, 30)
        self.t2_now_txt23.setText(' π₁:  overall tightness') 
        self.t2_now_txt23.setAlignment(Qt.AlignLeft)
        self.t2_now_txt23.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt23.setHidden(True)     
    
        # pi1 edit
        self.t2_now_edt13 = QLineEdit(self)
        self.t2_now_edt13.move(335, 458)       
        self.t2_now_edt13.resize(140, 22)                                           
        self.t2_now_edt13.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt13.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt13.setText(self.user_inputs['tab_2_now']['mfbvar_pi1'])
        self.t2_now_edt13.textChanged.connect(self.cb_t2_now_edt13)
        self.t2_now_edt13.setHidden(True)     
    
        # pi2 label
        self.t2_now_txt24 = QLabel(self)
        self.t2_now_txt24.move(30, 482)
        self.t2_now_txt24.setFixedSize(300, 30)
        self.t2_now_txt24.setText(' π₂:  cross-variable shrinkage') 
        self.t2_now_txt24.setAlignment(Qt.AlignLeft)
        self.t2_now_txt24.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt24.setHidden(True)      
    
        # pi2 edit
        self.t2_now_edt14 = QLineEdit(self)
        self.t2_now_edt14.move(335, 485)       
        self.t2_now_edt14.resize(140, 22)                                           
        self.t2_now_edt14.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt14.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt14.setText(self.user_inputs['tab_2_now']['mfbvar_pi2'])
        self.t2_now_edt14.textChanged.connect(self.cb_t2_now_edt14)
        self.t2_now_edt14.setHidden(True)     
    
        # pi3 label
        self.t2_now_txt25 = QLabel(self)
        self.t2_now_txt25.move(30, 509)
        self.t2_now_txt25.setFixedSize(300, 30)
        self.t2_now_txt25.setText(' π₃:  lag decay') 
        self.t2_now_txt25.setAlignment(Qt.AlignLeft)
        self.t2_now_txt25.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt25.setHidden(True)      
    
        # pi3 edit
        self.t2_now_edt15 = QLineEdit(self)
        self.t2_now_edt15.move(335, 512)       
        self.t2_now_edt15.resize(140, 22)                                           
        self.t2_now_edt15.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt15.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt15.setText(self.user_inputs['tab_2_now']['mfbvar_pi3'])
        self.t2_now_edt15.textChanged.connect(self.cb_t2_now_edt15)
        self.t2_now_edt15.setHidden(True)  
    
        # pi4 label
        self.t2_now_txt26 = QLabel(self)
        self.t2_now_txt26.move(30, 536)
        self.t2_now_txt26.setFixedSize(300, 30)
        self.t2_now_txt26.setText(' π₄:  exogenous slackness') 
        self.t2_now_txt26.setAlignment(Qt.AlignLeft)
        self.t2_now_txt26.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt26.setHidden(True)      
    
        # pi4 edit
        self.t2_now_edt16 = QLineEdit(self)
        self.t2_now_edt16.move(335, 539)       
        self.t2_now_edt16.resize(140, 22)                                           
        self.t2_now_edt16.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt16.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt16.setText(self.user_inputs['tab_2_now']['mfbvar_pi4'])
        self.t2_now_edt16.textChanged.connect(self.cb_t2_now_edt16)
        self.t2_now_edt16.setHidden(True)     
    
        # decomposition file label
        self.t2_now_txt27 = QLabel(self)
        self.t2_now_txt27.move(30, 563)
        self.t2_now_txt27.setFixedSize(300, 30)
        self.t2_now_txt27.setText(' file: decomposition table') 
        self.t2_now_txt27.setAlignment(Qt.AlignLeft)
        self.t2_now_txt27.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt27.setHidden(True)      

        # decomposition file edit
        self.t2_now_edt17 = QLineEdit(self)
        self.t2_now_edt17.move(30, 593)       
        self.t2_now_edt17.resize(445, 22)                                            
        self.t2_now_edt17.setAlignment(Qt.AlignLeft)   
        self.t2_now_edt17.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt17.setText(self.user_inputs['tab_2_now']['mfbvar_decomposition_file'])
        self.t2_now_edt17.textChanged.connect(self.cb_t2_now_edt17)
        self.t2_now_edt17.setHidden(True)     
    
        # dfm label
        self.t2_now_txt28 = QLabel(self)
        self.t2_now_txt28.move(520, 253)
        self.t2_now_txt28.setFixedSize(400, 30)
        self.t2_now_txt28.setText(' Bayesian dynamic factor model') 
        self.t2_now_txt28.setAlignment(Qt.AlignLeft)
        self.t2_now_txt28.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_now_txt28.setFont(font)
        self.t2_now_txt28.setHidden(True)
    
        # frame around dfm
        self.t2_now_frm4 = QFrame(self)   
        self.t2_now_frm4.setGeometry(510, 283, 470, 347)  
        self.t2_now_frm4.setFrameShape(QFrame.Panel)
        self.t2_now_frm4.setLineWidth(1)  
        self.t2_now_frm4.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t2_now_frm4.setHidden(True)    
    
        # factor label
        self.t2_now_txt29 = QLabel(self)
        self.t2_now_txt29.move(520, 293)
        self.t2_now_txt29.setFixedSize(300, 30)
        self.t2_now_txt29.setText(' m:   factors') 
        self.t2_now_txt29.setAlignment(Qt.AlignLeft)
        self.t2_now_txt29.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt29.setHidden(True)  
    
        # factor edit
        self.t2_now_edt18 = QLineEdit(self)
        self.t2_now_edt18.move(825, 296)       
        self.t2_now_edt18.resize(140, 22)                                           
        self.t2_now_edt18.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt18.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt18.setText(self.user_inputs['tab_2_now']['dfm_factors'])
        self.t2_now_edt18.textChanged.connect(self.cb_t2_now_edt18)
        self.t2_now_edt18.setHidden(True)      

        # loadings lag label
        self.t2_now_txt30 = QLabel(self)
        self.t2_now_txt30.move(520, 320)
        self.t2_now_txt30.setFixedSize(300, 30)
        self.t2_now_txt30.setText(' q:    loadings lags') 
        self.t2_now_txt30.setAlignment(Qt.AlignLeft)
        self.t2_now_txt30.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt30.setHidden(True)      
    
        # loadings lag edit
        self.t2_now_edt19 = QLineEdit(self)
        self.t2_now_edt19.move(825, 323)       
        self.t2_now_edt19.resize(140, 22)                                           
        self.t2_now_edt19.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt19.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt19.setText(self.user_inputs['tab_2_now']['dfm_loadings_lags'])
        self.t2_now_edt19.textChanged.connect(self.cb_t2_now_edt19)
        self.t2_now_edt19.setHidden(True)  
    
        # factor lag label
        self.t2_now_txt31 = QLabel(self)
        self.t2_now_txt31.move(520, 347)
        self.t2_now_txt31.setFixedSize(300, 30)
        self.t2_now_txt31.setText(' p:    factor lags') 
        self.t2_now_txt31.setAlignment(Qt.AlignLeft)
        self.t2_now_txt31.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt31.setHidden(True)       

        # factor lag edit
        self.t2_now_edt20 = QLineEdit(self)
        self.t2_now_edt20.move(825, 350)       
        self.t2_now_edt20.resize(140, 22)                                           
        self.t2_now_edt20.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt20.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt20.setText(self.user_inputs['tab_2_now']['dfm_factor_lags'])
        self.t2_now_edt20.textChanged.connect(self.cb_t2_now_edt20)
        self.t2_now_edt20.setHidden(True)  

        # residual lag label
        self.t2_now_txt32 = QLabel(self)
        self.t2_now_txt32.move(520, 374)
        self.t2_now_txt32.setFixedSize(300, 30)
        self.t2_now_txt32.setText(' r:    residual lags') 
        self.t2_now_txt32.setAlignment(Qt.AlignLeft)
        self.t2_now_txt32.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt32.setHidden(True)   

        # residual lag edit
        self.t2_now_edt21 = QLineEdit(self)
        self.t2_now_edt21.move(825, 377)       
        self.t2_now_edt21.resize(140, 22)                                           
        self.t2_now_edt21.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt21.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt21.setText(self.user_inputs['tab_2_now']['dfm_residual_lags'])
        self.t2_now_edt21.textChanged.connect(self.cb_t2_now_edt21)
        self.t2_now_edt21.setHidden(True)  

        # residual variance label
        self.t2_now_txt33 = QLabel(self)
        self.t2_now_txt33.move(520, 401)
        self.t2_now_txt33.setFixedSize(300, 30)
        self.t2_now_txt33.setText(' σ:    residual variance') 
        self.t2_now_txt33.setAlignment(Qt.AlignLeft)
        self.t2_now_txt33.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt33.setHidden(True)   

        # residual variance edit
        self.t2_now_edt22 = QLineEdit(self)
        self.t2_now_edt22.move(825, 404)       
        self.t2_now_edt22.resize(140, 22)                                           
        self.t2_now_edt22.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt22.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt22.setText(self.user_inputs['tab_2_now']['dfm_sigma'])
        self.t2_now_edt22.textChanged.connect(self.cb_t2_now_edt22)
        self.t2_now_edt22.setHidden(True)  

        # factor variance label
        self.t2_now_txt34 = QLabel(self)
        self.t2_now_txt34.move(520, 428)
        self.t2_now_txt34.setFixedSize(300, 30)
        self.t2_now_txt34.setText(' ω:    factor variance') 
        self.t2_now_txt34.setAlignment(Qt.AlignLeft)
        self.t2_now_txt34.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt34.setHidden(True)  

        # factor variance edit
        self.t2_now_edt23 = QLineEdit(self)
        self.t2_now_edt23.move(825, 431)       
        self.t2_now_edt23.resize(140, 22)                                           
        self.t2_now_edt23.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt23.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt23.setText(self.user_inputs['tab_2_now']['dfm_omega'])
        self.t2_now_edt23.textChanged.connect(self.cb_t2_now_edt23)
        self.t2_now_edt23.setHidden(True)  

        # loadings tightness label
        self.t2_now_txt35 = QLabel(self)
        self.t2_now_txt35.move(520, 455)
        self.t2_now_txt35.setFixedSize(300, 30)
        self.t2_now_txt35.setText(' δ₁:   loadings tightness') 
        self.t2_now_txt35.setAlignment(Qt.AlignLeft)
        self.t2_now_txt35.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt35.setHidden(True) 

        # loadings tightness edit
        self.t2_now_edt24 = QLineEdit(self)
        self.t2_now_edt24.move(825, 458)       
        self.t2_now_edt24.resize(140, 22)                                           
        self.t2_now_edt24.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt24.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt24.setText(self.user_inputs['tab_2_now']['dfm_delta1'])
        self.t2_now_edt24.textChanged.connect(self.cb_t2_now_edt24)
        self.t2_now_edt24.setHidden(True)  

        # pi1 label
        self.t2_now_txt36 = QLabel(self)
        self.t2_now_txt36.move(520, 482)
        self.t2_now_txt36.setFixedSize(300, 30)
        self.t2_now_txt36.setText(' π₁:   factor tightness') 
        self.t2_now_txt36.setAlignment(Qt.AlignLeft)
        self.t2_now_txt36.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt36.setHidden(True) 

        # pi1 edit
        self.t2_now_edt25 = QLineEdit(self)
        self.t2_now_edt25.move(825, 485)       
        self.t2_now_edt25.resize(140, 22)                                           
        self.t2_now_edt25.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt25.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt25.setText(self.user_inputs['tab_2_now']['dfm_pi1'])
        self.t2_now_edt25.textChanged.connect(self.cb_t2_now_edt25)
        self.t2_now_edt25.setHidden(True)  

        # pi2 label
        self.t2_now_txt37 = QLabel(self)
        self.t2_now_txt37.move(520, 509)
        self.t2_now_txt37.setFixedSize(300, 30)
        self.t2_now_txt37.setText(' π₂:   cross-variable shrinkage') 
        self.t2_now_txt37.setAlignment(Qt.AlignLeft)
        self.t2_now_txt37.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt37.setHidden(True) 

        # pi2 edit
        self.t2_now_edt26 = QLineEdit(self)
        self.t2_now_edt26.move(825, 512)       
        self.t2_now_edt26.resize(140, 22)                                           
        self.t2_now_edt26.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt26.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt26.setText(self.user_inputs['tab_2_now']['dfm_pi2'])
        self.t2_now_edt26.textChanged.connect(self.cb_t2_now_edt26)
        self.t2_now_edt26.setHidden(True)  

        # pi3 label
        self.t2_now_txt38 = QLabel(self)
        self.t2_now_txt38.move(520, 536)
        self.t2_now_txt38.setFixedSize(300, 30)
        self.t2_now_txt38.setText(' π₃:   lag decay') 
        self.t2_now_txt38.setAlignment(Qt.AlignLeft)
        self.t2_now_txt38.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt38.setHidden(True) 

        # pi3 edit
        self.t2_now_edt27 = QLineEdit(self)
        self.t2_now_edt27.move(825, 539)       
        self.t2_now_edt27.resize(140, 22)                                           
        self.t2_now_edt27.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt27.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt27.setText(self.user_inputs['tab_2_now']['dfm_pi3'])
        self.t2_now_edt27.textChanged.connect(self.cb_t2_now_edt27)
        self.t2_now_edt27.setHidden(True) 

        # omega1 label
        self.t2_now_txt39 = QLabel(self)
        self.t2_now_txt39.move(520, 563)
        self.t2_now_txt39.setFixedSize(300, 30)
        self.t2_now_txt39.setText(' ω₁:   residual tightness') 
        self.t2_now_txt39.setAlignment(Qt.AlignLeft)
        self.t2_now_txt39.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_now_txt39.setHidden(True) 

        # omega1 edit
        self.t2_now_edt28 = QLineEdit(self)
        self.t2_now_edt28.move(825, 566)       
        self.t2_now_edt28.resize(140, 22)                                           
        self.t2_now_edt28.setAlignment(Qt.AlignCenter)     
        self.t2_now_edt28.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_now_edt28.setText(self.user_inputs['tab_2_now']['dfm_omega1'])
        self.t2_now_edt28.textChanged.connect(self.cb_t2_now_edt28)
        self.t2_now_edt28.setHidden(True) 

    
    def hide_tab_2_now(self):
        
        # hide all controls
        self.t2_now_txt1.setHidden(True)    
        self.t2_now_txt2.setHidden(True)   
        self.t2_now_txt3.setHidden(True)   
        self.t2_now_txt4.setHidden(True)
        self.t2_now_txt5.setHidden(True)
        self.t2_now_txt6.setHidden(True)
        self.t2_now_txt7.setHidden(True)
        self.t2_now_txt8.setHidden(True)
        self.t2_now_txt9.setHidden(True)
        self.t2_now_txt10.setHidden(True)
        self.t2_now_txt11.setHidden(True)
        self.t2_now_txt12.setHidden(True)
        self.t2_now_txt13.setHidden(True)
        self.t2_now_txt14.setHidden(True)
        self.t2_now_txt15.setHidden(True)
        self.t2_now_txt16.setHidden(True)
        self.t2_now_txt17.setHidden(True)
        self.t2_now_txt18.setHidden(True)
        self.t2_now_txt19.setHidden(True)
        self.t2_now_txt20.setHidden(True)
        self.t2_now_txt21.setHidden(True)
        self.t2_now_txt22.setHidden(True)
        self.t2_now_txt23.setHidden(True)
        self.t2_now_txt24.setHidden(True)
        self.t2_now_txt25.setHidden(True)
        self.t2_now_txt26.setHidden(True)
        self.t2_now_txt27.setHidden(True)
        self.t2_now_txt28.setHidden(True)
        self.t2_now_txt29.setHidden(True)
        self.t2_now_txt30.setHidden(True)
        self.t2_now_txt31.setHidden(True)
        self.t2_now_txt32.setHidden(True)
        self.t2_now_txt33.setHidden(True)
        self.t2_now_txt34.setHidden(True)
        self.t2_now_txt35.setHidden(True)
        self.t2_now_txt36.setHidden(True)
        self.t2_now_txt37.setHidden(True)
        self.t2_now_txt38.setHidden(True)
        self.t2_now_txt39.setHidden(True)
        self.t2_now_frm1.setHidden(True)   
        self.t2_now_frm2.setHidden(True)    
        self.t2_now_frm3.setHidden(True)  
        self.t2_now_frm4.setHidden(True)  
        self.t2_now_rdb1.setHidden(True)
        self.t2_now_rdb2.setHidden(True)
        self.t2_now_rdb3.setHidden(True)
        self.t2_now_edt1.setHidden(True)
        self.t2_now_edt2.setHidden(True)
        self.t2_now_edt3.setHidden(True)
        self.t2_now_edt4.setHidden(True)
        self.t2_now_edt5.setHidden(True)
        self.t2_now_edt6.setHidden(True)
        self.t2_now_edt7.setHidden(True)
        self.t2_now_edt8.setHidden(True)
        self.t2_now_edt9.setHidden(True)
        self.t2_now_edt10.setHidden(True)
        self.t2_now_edt11.setHidden(True)
        self.t2_now_edt12.setHidden(True)
        self.t2_now_edt13.setHidden(True)
        self.t2_now_edt14.setHidden(True)
        self.t2_now_edt15.setHidden(True)
        self.t2_now_edt16.setHidden(True)
        self.t2_now_edt17.setHidden(True)
        self.t2_now_edt18.setHidden(True)
        self.t2_now_edt19.setHidden(True)
        self.t2_now_edt20.setHidden(True)
        self.t2_now_edt21.setHidden(True)
        self.t2_now_edt22.setHidden(True)
        self.t2_now_edt23.setHidden(True)
        self.t2_now_edt24.setHidden(True)
        self.t2_now_edt25.setHidden(True)
        self.t2_now_edt26.setHidden(True)
        self.t2_now_edt27.setHidden(True)
        self.t2_now_edt28.setHidden(True)
        self.t2_now_mnu1.setHidden(True)
        self.t2_now_cbx1.setHidden(True)
        self.t2_now_cbx2.setHidden(True)
        self.t2_now_cbx3.setHidden(True)
        self.t2_now_cbx4.setHidden(True)
        
        # update tab color
        self.tab_pbt2.setStyleSheet("background:rgb" + str(self.backtabs_color) + ";")      

    
    def show_tab_2_now(self): 
    
        # show all controls
        self.t2_now_txt1.setVisible(True)
        self.t2_now_txt2.setVisible(True)
        self.t2_now_txt3.setVisible(True)
        self.t2_now_txt4.setVisible(True)
        self.t2_now_txt5.setVisible(True)
        self.t2_now_txt6.setVisible(True)
        self.t2_now_txt7.setVisible(True)
        self.t2_now_txt8.setVisible(True)
        self.t2_now_txt9.setVisible(True)
        self.t2_now_txt10.setVisible(True)
        self.t2_now_txt11.setVisible(True)
        self.t2_now_txt12.setVisible(True)
        self.t2_now_txt13.setVisible(True)
        self.t2_now_txt14.setVisible(True)
        self.t2_now_txt15.setVisible(True)
        self.t2_now_txt16.setVisible(True)
        self.t2_now_txt17.setVisible(True)
        self.t2_now_txt18.setVisible(True)
        self.t2_now_txt19.setVisible(True)
        self.t2_now_txt20.setVisible(True)
        self.t2_now_txt21.setVisible(True)
        self.t2_now_txt22.setVisible(True)
        self.t2_now_txt23.setVisible(True)
        self.t2_now_txt24.setVisible(True)
        self.t2_now_txt25.setVisible(True)
        self.t2_now_txt26.setVisible(True)
        self.t2_now_txt27.setVisible(True)
        self.t2_now_txt28.setVisible(True)
        self.t2_now_txt29.setVisible(True)
        self.t2_now_txt30.setVisible(True)
        self.t2_now_txt31.setVisible(True)
        self.t2_now_txt32.setVisible(True)
        self.t2_now_txt33.setVisible(True)
        self.t2_now_txt34.setVisible(True)
        self.t2_now_txt35.setVisible(True)
        self.t2_now_txt36.setVisible(True)
        self.t2_now_txt37.setVisible(True)
        self.t2_now_txt38.setVisible(True)
        self.t2_now_txt39.setVisible(True)
        self.t2_now_frm1.setVisible(True)    
        self.t2_now_frm2.setVisible(True)   
        self.t2_now_frm3.setVisible(True) 
        self.t2_now_frm4.setVisible(True) 
        self.t2_now_rdb1.setVisible(True)
        self.t2_now_rdb2.setVisible(True)
        self.t2_now_rdb3.setVisible(True)    
        self.t2_now_edt1.setVisible(True)
        self.t2_now_edt2.setVisible(True)
        self.t2_now_edt3.setVisible(True)
        self.t2_now_edt4.setVisible(True)
        self.t2_now_edt5.setVisible(True)
        self.t2_now_edt6.setVisible(True) 
        self.t2_now_edt7.setVisible(True) 
        self.t2_now_edt8.setVisible(True) 
        self.t2_now_edt9.setVisible(True) 
        self.t2_now_edt10.setVisible(True)
        self.t2_now_edt11.setVisible(True)
        self.t2_now_edt12.setVisible(True)
        self.t2_now_edt13.setVisible(True)
        self.t2_now_edt14.setVisible(True)
        self.t2_now_edt15.setVisible(True)
        self.t2_now_edt16.setVisible(True)
        self.t2_now_edt17.setVisible(True)
        self.t2_now_edt18.setVisible(True)
        self.t2_now_edt19.setVisible(True)
        self.t2_now_edt20.setVisible(True)
        self.t2_now_edt21.setVisible(True)
        self.t2_now_edt22.setVisible(True)
        self.t2_now_edt23.setVisible(True)
        self.t2_now_edt24.setVisible(True)
        self.t2_now_edt25.setVisible(True)
        self.t2_now_edt26.setVisible(True)
        self.t2_now_edt27.setVisible(True)
        self.t2_now_edt28.setVisible(True)
        self.t2_now_mnu1.setVisible(True)
        self.t2_now_cbx1.setVisible(True)
        self.t2_now_cbx2.setVisible(True)
        self.t2_now_cbx3.setVisible(True)
        self.t2_now_cbx4.setVisible(True)
    

    def cb_t2_now_bgr1(self):
        if self.t2_now_rdb1.isChecked() == True:
            self.user_inputs['tab_2_now']['model'] = 1
        elif self.t2_now_rdb2.isChecked() == True:
            self.user_inputs['tab_2_now']['model'] = 2
        elif self.t2_now_rdb3.isChecked() == True:
            self.user_inputs['tab_2_now']['model'] = 3          
    
    def cb_t2_now_edt1(self):
        self.user_inputs['tab_2_now']['iterations'] = self.t2_now_edt1.text()         

    def cb_t2_now_edt2(self):
        self.user_inputs['tab_2_now']['burnin'] = self.t2_now_edt2.text()         

    def cb_t2_now_edt3(self):
        self.user_inputs['tab_2_now']['model_credibility'] = self.t2_now_edt3.text()  

    def cb_t2_now_edt4(self):
        self.user_inputs['tab_2_now']['midas_endogenous_lags'] = self.t2_now_edt4.text()  

    def cb_t2_now_edt5(self):
        self.user_inputs['tab_2_now']['midas_exogenous_lags'] = self.t2_now_edt5.text() 

    def cb_t2_now_edt6(self):
        self.user_inputs['tab_2_now']['midas_polynomial_order'] = self.t2_now_edt6.text() 

    def cb_t2_now_mnu1(self, index):     
        self.user_inputs['tab_2_now']['midas_model'] = self.t2_now_mnu1.itemData(index)
        # self.user_inputs['tab_2_now']['midas_prior_type'] = self.t2_now_mnu1.itemData(index)

    def cb_t2_now_edt7(self):
        self.user_inputs['tab_2_now']['midas_omega1'] = self.t2_now_edt7.text() 

    def cb_t2_now_edt8(self):
        self.user_inputs['tab_2_now']['midas_omega2'] = self.t2_now_edt8.text() 

    def cb_t2_now_edt9(self):
        self.user_inputs['tab_2_now']['midas_upsilon1'] = self.t2_now_edt9.text() 
        
    def cb_t2_now_edt10(self):
        self.user_inputs['tab_2_now']['midas_upsilon2'] = self.t2_now_edt10.text() 

    def cb_t2_now_cbx1(self, state):     
        if (state == Qt.Checked):
            self.user_inputs['tab_2_now']['mfbvar_constant'] = True 
        else:
            self.user_inputs['tab_2_now']['mfbvar_constant'] = False  

    def cb_t2_now_cbx2(self, state):     
        if (state == Qt.Checked):
            self.user_inputs['tab_2_now']['mfbvar_trend'] = True 
        else:
            self.user_inputs['tab_2_now']['mfbvar_trend'] = False  

    def cb_t2_now_cbx3(self, state):     
        if (state == Qt.Checked):
            self.user_inputs['tab_2_now']['mfbvar_quadratic_trend'] = True 
        else:
            self.user_inputs['tab_2_now']['mfbvar_quadratic_trend'] = False  

    def cb_t2_now_cbx4(self, state):     
        if (state == Qt.Checked):
            self.user_inputs['tab_2_now']['mfbvar_decomposition'] = True 
        else:
            self.user_inputs['tab_2_now']['mfbvar_decomposition'] = False  

    def cb_t2_now_edt11(self):
        self.user_inputs['tab_2_now']['mfbvar_lags'] = self.t2_now_edt11.text() 
        
    def cb_t2_now_edt12(self):
        self.user_inputs['tab_2_now']['mfbvar_ar_coefficients'] = self.t2_now_edt12.text()         
        
    def cb_t2_now_edt13(self):
        self.user_inputs['tab_2_now']['mfbvar_pi1'] = self.t2_now_edt13.text()  
        
    def cb_t2_now_edt14(self):
        self.user_inputs['tab_2_now']['mfbvar_pi2'] = self.t2_now_edt14.text()  
        
    def cb_t2_now_edt15(self):
        self.user_inputs['tab_2_now']['mfbvar_pi3'] = self.t2_now_edt15.text()  
        
    def cb_t2_now_edt16(self):
        self.user_inputs['tab_2_now']['mfbvar_pi4'] = self.t2_now_edt16.text()           
        
    def cb_t2_now_edt17(self):
        self.user_inputs['tab_2_now']['mfbvar_decomposition_file'] = self.t2_now_edt17.text()         
        
    def cb_t2_now_edt18(self):
        self.user_inputs['tab_2_now']['dfm_factors'] = self.t2_now_edt18.text()          

    def cb_t2_now_edt19(self):
        self.user_inputs['tab_2_now']['dfm_loadings_lags'] = self.t2_now_edt19.text()  

    def cb_t2_now_edt20(self):
        self.user_inputs['tab_2_now']['dfm_factor_lags'] = self.t2_now_edt20.text()  

    def cb_t2_now_edt21(self):
        self.user_inputs['tab_2_now']['dfm_residual_lags'] = self.t2_now_edt21.text()  
        
    def cb_t2_now_edt22(self):
        self.user_inputs['tab_2_now']['dfm_sigma'] = self.t2_now_edt22.text()          

    def cb_t2_now_edt23(self):
        self.user_inputs['tab_2_now']['dfm_omega'] = self.t2_now_edt23.text()  
        
    def cb_t2_now_edt24(self):
        self.user_inputs['tab_2_now']['dfm_delta1'] = self.t2_now_edt24.text()  

    def cb_t2_now_edt25(self):
        self.user_inputs['tab_2_now']['dfm_pi1'] = self.t2_now_edt25.text()  

    def cb_t2_now_edt26(self):
        self.user_inputs['tab_2_now']['dfm_pi2'] = self.t2_now_edt26.text()  

    def cb_t2_now_edt27(self):
        self.user_inputs['tab_2_now']['dfm_pi3'] = self.t2_now_edt27.text()  

    def cb_t2_now_edt28(self):
        self.user_inputs['tab_2_now']['dfm_omega1'] = self.t2_now_edt28.text()          
        

            