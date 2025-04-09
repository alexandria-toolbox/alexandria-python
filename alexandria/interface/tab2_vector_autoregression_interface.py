# imports
from PyQt5.QtWidgets import QLabel, QFrame, QRadioButton, QButtonGroup, QLineEdit, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont



class Tab2VectorAutoregressionInterface(object):


    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass


    def create_tab_2_var(self):
    
        # vector autoregression label
        self.t2_var_txt1 = QLabel(self)
        self.t2_var_txt1.move(30, 60)
        self.t2_var_txt1.setFixedSize(300, 30)
        self.t2_var_txt1.setText(' VAR type') 
        self.t2_var_txt1.setAlignment(Qt.AlignLeft)
        self.t2_var_txt1.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_var_txt1.setFont(font)
        self.t2_var_txt1.setHidden(True)
        
        # frame around VAR type
        self.t2_var_frm1 = QFrame(self)   
        self.t2_var_frm1.setGeometry(20, 90, 470, 110)  
        self.t2_var_frm1.setFrameShape(QFrame.Panel)
        self.t2_var_frm1.setLineWidth(1)  
        self.t2_var_frm1.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t2_var_frm1.setHidden(True)
        
        # VAR type radiobuttons
        self.t2_var_rdb1 = QRadioButton(' maximum likelihood', self)
        self.t2_var_rdb1.setGeometry(30, 93, 200, 30)
        self.t2_var_rdb1.setStyleSheet("font-size: 12pt; font-family: Serif;")  
        self.t2_var_rdb1.toggled.connect(self.cb_t2_var_bgr1)
        self.t2_var_rdb1.setHidden(True)
        self.t2_var_rdb2 = QRadioButton(' Minnesota', self)
        self.t2_var_rdb2.setGeometry(260, 93, 200, 30)
        self.t2_var_rdb2.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_var_rdb2.toggled.connect(self.cb_t2_var_bgr1)
        self.t2_var_rdb2.setHidden(True)
        self.t2_var_rdb3 = QRadioButton(' normal-Wishart', self)
        self.t2_var_rdb3.setGeometry(30, 118, 200, 30)
        self.t2_var_rdb3.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_var_rdb3.toggled.connect(self.cb_t2_var_bgr1)
        self.t2_var_rdb3.setHidden(True)
        self.t2_var_rdb4 = QRadioButton(' independent', self)
        self.t2_var_rdb4.setGeometry(260, 118, 200, 30)
        self.t2_var_rdb4.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_var_rdb4.toggled.connect(self.cb_t2_var_bgr1)
        self.t2_var_rdb4.setHidden(True)
        self.t2_var_rdb5 = QRadioButton(' dummy observations', self)
        self.t2_var_rdb5.setGeometry(30, 143, 200, 30)
        self.t2_var_rdb5.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_var_rdb5.toggled.connect(self.cb_t2_var_bgr1)
        self.t2_var_rdb5.setHidden(True)
        self.t2_var_rdb6 = QRadioButton(' large Bayesian VAR', self)
        self.t2_var_rdb6.setGeometry(260, 143, 200, 30)
        self.t2_var_rdb6.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_var_rdb6.toggled.connect(self.cb_t2_var_bgr1)
        self.t2_var_rdb6.setHidden(True)
        self.t2_var_rdb7 = QRadioButton(' proxy-SVAR', self)
        self.t2_var_rdb7.setGeometry(30, 173, 200, 25)
        self.t2_var_rdb7.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_var_rdb7.toggled.connect(self.cb_t2_var_bgr1)
        self.t2_var_rdb7.setHidden(True)
        if self.user_inputs['tab_2_var']['var_type'] == 1:
            self.t2_var_rdb1.setChecked(True) 
        elif self.user_inputs['tab_2_var']['var_type'] == 2:
            self.t2_var_rdb2.setChecked(True) 
        elif self.user_inputs['tab_2_var']['var_type'] == 3:
            self.t2_var_rdb3.setChecked(True) 
        elif self.user_inputs['tab_2_var']['var_type'] == 4:
            self.t2_var_rdb4.setChecked(True) 
        elif self.user_inputs['tab_2_var']['var_type'] == 5:
            self.t2_var_rdb5.setChecked(True) 
        elif self.user_inputs['tab_2_var']['var_type'] == 6:
            self.t2_var_rdb6.setChecked(True)
        elif self.user_inputs['tab_2_var']['var_type'] == 7:
            self.t2_var_rdb7.setChecked(True) 
        self.t2_var_bgr1 = QButtonGroup(self)  
        self.t2_var_bgr1.addButton(self.t2_var_rdb1) 
        self.t2_var_bgr1.addButton(self.t2_var_rdb2)     
        self.t2_var_bgr1.addButton(self.t2_var_rdb3) 
        self.t2_var_bgr1.addButton(self.t2_var_rdb4) 
        self.t2_var_bgr1.addButton(self.t2_var_rdb5) 
        self.t2_var_bgr1.addButton(self.t2_var_rdb6) 
        self.t2_var_bgr1.addButton(self.t2_var_rdb7) 
        
        # estimation label
        self.t2_var_txt2 = QLabel(self)
        self.t2_var_txt2.move(520, 60)
        self.t2_var_txt2.setFixedSize(300, 30)
        self.t2_var_txt2.setText(' Estimation') 
        self.t2_var_txt2.setAlignment(Qt.AlignLeft)
        self.t2_var_txt2.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_var_txt2.setFont(font)
        self.t2_var_txt2.setHidden(True)

        # frame around estimation
        self.t2_var_frm2 = QFrame(self)   
        self.t2_var_frm2.setGeometry(510, 90, 470, 110)  
        self.t2_var_frm2.setFrameShape(QFrame.Panel)
        self.t2_var_frm2.setLineWidth(1)  
        self.t2_var_frm2.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t2_var_frm2.setHidden(True)   

        # Gibbs sampling label
        self.t2_var_txt3 = QLabel(self)
        self.t2_var_txt3.move(520, 95)
        self.t2_var_txt3.setFixedSize(200, 30)
        self.t2_var_txt3.setText(' Gibbs sampling') 
        self.t2_var_txt3.setAlignment(Qt.AlignLeft)
        self.t2_var_txt3.setStyleSheet('font-size: 14pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_var_txt3.setFont(font)
        self.t2_var_txt3.setHidden(True)

        # iteration label
        self.t2_var_txt4 = QLabel(self)
        self.t2_var_txt4.move(520, 125)
        self.t2_var_txt4.setFixedSize(200, 25)
        self.t2_var_txt4.setText(' iterations') 
        self.t2_var_txt4.setAlignment(Qt.AlignLeft)
        self.t2_var_txt4.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt4.setHidden(True)

        # iteration edit
        self.t2_var_edt1 = QLineEdit(self)
        self.t2_var_edt1.move(670, 122)       
        self.t2_var_edt1.resize(70, 23)                                           
        self.t2_var_edt1.setAlignment(Qt.AlignCenter)     
        self.t2_var_edt1.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_var_edt1.setText(self.user_inputs['tab_2_var']['iterations'])
        self.t2_var_edt1.textChanged.connect(self.cb_t2_var_edt1)
        self.t2_var_edt1.setHidden(True)

        # burn-in label
        self.t2_var_txt5 = QLabel(self)
        self.t2_var_txt5.move(520, 150)
        self.t2_var_txt5.setFixedSize(200, 25)
        self.t2_var_txt5.setText(' burn-in') 
        self.t2_var_txt5.setAlignment(Qt.AlignLeft)
        self.t2_var_txt5.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt5.setHidden(True)

        # burn-in edit
        self.t2_var_edt2 = QLineEdit(self)
        self.t2_var_edt2.move(670, 147)       
        self.t2_var_edt2.resize(70, 23)                                           
        self.t2_var_edt2.setAlignment(Qt.AlignCenter)     
        self.t2_var_edt2.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_var_edt2.setText(self.user_inputs['tab_2_var']['burnin'])
        self.t2_var_edt2.textChanged.connect(self.cb_t2_var_edt2)
        self.t2_var_edt2.setHidden(True)
        
        # credibility label
        self.t2_var_txt6 = QLabel(self)
        self.t2_var_txt6.move(520, 175)
        self.t2_var_txt6.setFixedSize(200, 20)
        self.t2_var_txt6.setText(' credibility level') 
        self.t2_var_txt6.setAlignment(Qt.AlignLeft)
        self.t2_var_txt6.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt6.setHidden(True)        

        # credibility edit
        self.t2_var_edt3 = QLineEdit(self)
        self.t2_var_edt3.move(670, 172)       
        self.t2_var_edt3.resize(70, 23)                                           
        self.t2_var_edt3.setAlignment(Qt.AlignCenter)     
        self.t2_var_edt3.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_var_edt3.setText(self.user_inputs['tab_2_var']['model_credibility'])
        self.t2_var_edt3.textChanged.connect(self.cb_t2_var_edt3)
        self.t2_var_edt3.setHidden(True)

        # Exogenous label
        self.t2_var_txt7 = QLabel(self)
        self.t2_var_txt7.move(770, 95)
        self.t2_var_txt7.setFixedSize(200, 30)
        self.t2_var_txt7.setText(' Exogenous') 
        self.t2_var_txt7.setAlignment(Qt.AlignLeft)
        self.t2_var_txt7.setStyleSheet('font-size: 14pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_var_txt7.setFont(font)
        self.t2_var_txt7.setHidden(True)

        # constant label
        self.t2_var_txt8 = QLabel(self)
        self.t2_var_txt8.move(770, 125)
        self.t2_var_txt8.setFixedSize(200, 25)
        self.t2_var_txt8.setText(' constant') 
        self.t2_var_txt8.setAlignment(Qt.AlignLeft)
        self.t2_var_txt8.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt8.setHidden(True)

        # constant checkbox
        self.t2_var_cbx1 = QCheckBox(self)
        self.t2_var_cbx1.setGeometry(950, 125, 20, 20)  
        self.t2_var_cbx1.setStyleSheet("QCheckBox::indicator:unchecked \
        {background-color : rgb(255, 255, 255); border: 0.5px solid rgb(0, 0, 0)}; \
        QCheckBox::indicator {width: 12px; height: 12px}") 
        self.t2_var_cbx1.setChecked(self.user_inputs['tab_2_var']['constant'])
        self.t2_var_cbx1.stateChanged.connect(self.cb_t2_var_cbx1) 
        self.t2_var_cbx1.setHidden(True)

        # linear trend label
        self.t2_var_txt9 = QLabel(self)
        self.t2_var_txt9.move(770, 150)
        self.t2_var_txt9.setFixedSize(200, 25)
        self.t2_var_txt9.setText(' linear trend') 
        self.t2_var_txt9.setAlignment(Qt.AlignLeft)
        self.t2_var_txt9.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt9.setHidden(True)

        # linear trend checkbox
        self.t2_var_cbx2 = QCheckBox(self)
        self.t2_var_cbx2.setGeometry(950, 150, 20, 20) 
        self.t2_var_cbx2.setStyleSheet("QCheckBox::indicator:unchecked \
        {background-color : rgb(255, 255, 255); border: 0.5px solid rgb(0, 0, 0)}; \
        QCheckBox::indicator {width: 12px; height: 12px}") 
        self.t2_var_cbx2.setChecked(self.user_inputs['tab_2_var']['trend'])
        self.t2_var_cbx2.stateChanged.connect(self.cb_t2_var_cbx2) 
        self.t2_var_cbx2.setHidden(True)

        # quadratic trend
        self.t2_var_txt10 = QLabel(self)
        self.t2_var_txt10.move(770, 175)
        self.t2_var_txt10.setFixedSize(200, 20)
        self.t2_var_txt10.setText(' quadratic trend') 
        self.t2_var_txt10.setAlignment(Qt.AlignLeft)
        self.t2_var_txt10.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt10.setHidden(True)
        
        # quadratic trend checkbox
        self.t2_var_cbx3 = QCheckBox(self)
        self.t2_var_cbx3.setGeometry(950, 175, 20, 20) 
        self.t2_var_cbx3.setStyleSheet("QCheckBox::indicator:unchecked \
        {background-color : rgb(255, 255, 255); border: 0.5px solid rgb(0, 0, 0)}; \
        QCheckBox::indicator {width: 12px; height: 12px}") 
        self.t2_var_cbx3.setChecked(self.user_inputs['tab_2_var']['quadratic_trend'])
        self.t2_var_cbx3.stateChanged.connect(self.cb_t2_var_cbx3) 
        self.t2_var_cbx3.setHidden(True)
    
        # hyperparameter label
        self.t2_var_txt11 = QLabel(self)
        self.t2_var_txt11.move(30, 220)
        self.t2_var_txt11.setFixedSize(300, 30)
        self.t2_var_txt11.setText(' Hyperparameters') 
        self.t2_var_txt11.setAlignment(Qt.AlignLeft)
        self.t2_var_txt11.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_var_txt11.setFont(font)
        self.t2_var_txt11.setHidden(True)
    
        # frame around hyperparameters
        self.t2_var_frm3 = QFrame(self)   
        self.t2_var_frm3.setGeometry(20, 250, 470, 380)  
        self.t2_var_frm3.setFrameShape(QFrame.Panel)
        self.t2_var_frm3.setLineWidth(1)  
        self.t2_var_frm3.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t2_var_frm3.setHidden(True)

        # specification label
        self.t2_var_txt12 = QLabel(self)
        self.t2_var_txt12.move(30, 255)
        self.t2_var_txt12.setFixedSize(300, 25)
        self.t2_var_txt12.setText(' VAR specification') 
        self.t2_var_txt12.setAlignment(Qt.AlignLeft)
        self.t2_var_txt12.setStyleSheet('font-size: 14pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_var_txt12.setFont(font)
        self.t2_var_txt12.setHidden(True)    
        
        # lag label
        self.t2_var_txt13 = QLabel(self)
        self.t2_var_txt13.move(30, 283)
        self.t2_var_txt13.setFixedSize(300, 25)
        self.t2_var_txt13.setText(' p:    lags') 
        self.t2_var_txt13.setAlignment(Qt.AlignLeft)
        self.t2_var_txt13.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt13.setHidden(True)   

        # lag edit
        self.t2_var_edt4 = QLineEdit(self)
        self.t2_var_edt4.move(330, 281)       
        self.t2_var_edt4.resize(140, 22)                                           
        self.t2_var_edt4.setAlignment(Qt.AlignCenter)     
        self.t2_var_edt4.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_var_edt4.setText(self.user_inputs['tab_2_var']['lags'])
        self.t2_var_edt4.textChanged.connect(self.cb_t2_var_edt4)
        self.t2_var_edt4.setHidden(True)

        # AR coefficients label
        self.t2_var_txt14 = QLabel(self)
        self.t2_var_txt14.move(30, 307)
        self.t2_var_txt14.setFixedSize(300, 25)
        self.t2_var_txt14.setText(' δ:    AR coefficients') 
        self.t2_var_txt14.setAlignment(Qt.AlignLeft)
        self.t2_var_txt14.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt14.setHidden(True)      
    
        # AR coefficients edit
        self.t2_var_edt5 = QLineEdit(self)
        self.t2_var_edt5.move(330, 305)       
        self.t2_var_edt5.resize(140, 22)                                           
        self.t2_var_edt5.setAlignment(Qt.AlignCenter)     
        self.t2_var_edt5.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_var_edt5.setText(self.user_inputs['tab_2_var']['ar_coefficients'])
        self.t2_var_edt5.textChanged.connect(self.cb_t2_var_edt5)
        self.t2_var_edt5.setHidden(True)    

        # pi1 label
        self.t2_var_txt15 = QLabel(self)
        self.t2_var_txt15.move(30, 331)
        self.t2_var_txt15.setFixedSize(300, 25)
        self.t2_var_txt15.setText(' π₁:  overall tightness') 
        self.t2_var_txt15.setAlignment(Qt.AlignLeft)
        self.t2_var_txt15.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt15.setHidden(True) 
        
        # pi1 edit
        self.t2_var_edt6 = QLineEdit(self)
        self.t2_var_edt6.move(330, 329)       
        self.t2_var_edt6.resize(140, 22)                                           
        self.t2_var_edt6.setAlignment(Qt.AlignCenter)     
        self.t2_var_edt6.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_var_edt6.setText(self.user_inputs['tab_2_var']['pi1'])
        self.t2_var_edt6.textChanged.connect(self.cb_t2_var_edt6)
        self.t2_var_edt6.setHidden(True)           
        
        # pi2 label
        self.t2_var_txt16 = QLabel(self)
        self.t2_var_txt16.move(30, 355)
        self.t2_var_txt16.setFixedSize(300, 25)
        self.t2_var_txt16.setText(' π₂:  cross-variable shrinkage') 
        self.t2_var_txt16.setAlignment(Qt.AlignLeft)
        self.t2_var_txt16.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt16.setHidden(True) 
        
        # pi2 edit
        self.t2_var_edt7 = QLineEdit(self)
        self.t2_var_edt7.move(330, 353)       
        self.t2_var_edt7.resize(140, 22)                                           
        self.t2_var_edt7.setAlignment(Qt.AlignCenter)     
        self.t2_var_edt7.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_var_edt7.setText(self.user_inputs['tab_2_var']['pi2'])
        self.t2_var_edt7.textChanged.connect(self.cb_t2_var_edt7)
        self.t2_var_edt7.setHidden(True) 

        # pi3 label
        self.t2_var_txt17 = QLabel(self)
        self.t2_var_txt17.move(30, 379)
        self.t2_var_txt17.setFixedSize(300, 25)
        self.t2_var_txt17.setText(' π₃:  lag decay') 
        self.t2_var_txt17.setAlignment(Qt.AlignLeft)
        self.t2_var_txt17.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt17.setHidden(True) 

        # pi3 edit
        self.t2_var_edt8 = QLineEdit(self)
        self.t2_var_edt8.move(330, 377)       
        self.t2_var_edt8.resize(140, 22)                                           
        self.t2_var_edt8.setAlignment(Qt.AlignCenter)     
        self.t2_var_edt8.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_var_edt8.setText(self.user_inputs['tab_2_var']['pi3'])
        self.t2_var_edt8.textChanged.connect(self.cb_t2_var_edt8)
        self.t2_var_edt8.setHidden(True) 

        # pi4 label
        self.t2_var_txt18 = QLabel(self)
        self.t2_var_txt18.move(30, 403)
        self.t2_var_txt18.setFixedSize(300, 25)
        self.t2_var_txt18.setText(' π₄:  exogenous slackness') 
        self.t2_var_txt18.setAlignment(Qt.AlignLeft)
        self.t2_var_txt18.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt18.setHidden(True) 

        # pi4 edit
        self.t2_var_edt9 = QLineEdit(self)
        self.t2_var_edt9.move(330, 401)       
        self.t2_var_edt9.resize(140, 22)                                           
        self.t2_var_edt9.setAlignment(Qt.AlignCenter)     
        self.t2_var_edt9.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_var_edt9.setText(self.user_inputs['tab_2_var']['pi4'])
        self.t2_var_edt9.textChanged.connect(self.cb_t2_var_edt9)
        self.t2_var_edt9.setHidden(True) 

        # pi5 label
        self.t2_var_txt19 = QLabel(self)
        self.t2_var_txt19.move(30, 427)
        self.t2_var_txt19.setFixedSize(300, 25)
        self.t2_var_txt19.setText(' π₅:  sums-of-coefficients tightness') 
        self.t2_var_txt19.setAlignment(Qt.AlignLeft)
        self.t2_var_txt19.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt19.setHidden(True) 

        # pi5 edit
        self.t2_var_edt10 = QLineEdit(self)
        self.t2_var_edt10.move(330, 425)       
        self.t2_var_edt10.resize(140, 22)                                           
        self.t2_var_edt10.setAlignment(Qt.AlignCenter)     
        self.t2_var_edt10.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_var_edt10.setText(self.user_inputs['tab_2_var']['pi5'])
        self.t2_var_edt10.textChanged.connect(self.cb_t2_var_edt10)
        self.t2_var_edt10.setHidden(True) 

        # pi6 label
        self.t2_var_txt20 = QLabel(self)
        self.t2_var_txt20.move(30, 451)
        self.t2_var_txt20.setFixedSize(300, 25)
        self.t2_var_txt20.setText(' π₆:  initial observation tightness') 
        self.t2_var_txt20.setAlignment(Qt.AlignLeft)
        self.t2_var_txt20.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt20.setHidden(True) 
        
        # pi6 edit
        self.t2_var_edt11 = QLineEdit(self)
        self.t2_var_edt11.move(330, 449)       
        self.t2_var_edt11.resize(140, 22)                                           
        self.t2_var_edt11.setAlignment(Qt.AlignCenter)     
        self.t2_var_edt11.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_var_edt11.setText(self.user_inputs['tab_2_var']['pi6'])
        self.t2_var_edt11.textChanged.connect(self.cb_t2_var_edt11)
        self.t2_var_edt11.setHidden(True) 

        # pi7 label
        self.t2_var_txt21 = QLabel(self)
        self.t2_var_txt21.move(30, 475)
        self.t2_var_txt21.setFixedSize(300, 25)
        self.t2_var_txt21.setText(' π₇:  long-run tightness') 
        self.t2_var_txt21.setAlignment(Qt.AlignLeft)
        self.t2_var_txt21.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt21.setHidden(True) 

        # pi7 edit
        self.t2_var_edt12 = QLineEdit(self)
        self.t2_var_edt12.move(330, 473)       
        self.t2_var_edt12.resize(140, 22)                                           
        self.t2_var_edt12.setAlignment(Qt.AlignCenter)     
        self.t2_var_edt12.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_var_edt12.setText(self.user_inputs['tab_2_var']['pi7'])
        self.t2_var_edt12.textChanged.connect(self.cb_t2_var_edt12)
        self.t2_var_edt12.setHidden(True) 

        # proxy SVAR label
        self.t2_var_txt22 = QLabel(self)
        self.t2_var_txt22.move(30, 505)
        self.t2_var_txt22.setFixedSize(300, 25)
        self.t2_var_txt22.setText(' Proxy-SVAR') 
        self.t2_var_txt22.setAlignment(Qt.AlignLeft)
        self.t2_var_txt22.setStyleSheet('font-size: 14pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_var_txt22.setFont(font)
        self.t2_var_txt22.setHidden(True) 

        # proxy label
        self.t2_var_txt23 = QLabel(self)
        self.t2_var_txt23.move(30, 533)
        self.t2_var_txt23.setFixedSize(300, 25)
        self.t2_var_txt23.setText(' proxys') 
        self.t2_var_txt23.setAlignment(Qt.AlignLeft)
        self.t2_var_txt23.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt23.setHidden(True) 

        # proxy edit
        self.t2_var_edt13 = QLineEdit(self)
        self.t2_var_edt13.move(330, 531)       
        self.t2_var_edt13.resize(140, 22)                                           
        self.t2_var_edt13.setAlignment(Qt.AlignCenter)     
        self.t2_var_edt13.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_var_edt13.setText(self.user_inputs['tab_2_var']['proxy_variables'])
        self.t2_var_edt13.textChanged.connect(self.cb_t2_var_edt13)
        self.t2_var_edt13.setHidden(True) 

        # relevance label
        self.t2_var_txt24 = QLabel(self)
        self.t2_var_txt24.move(30, 557)
        self.t2_var_txt24.setFixedSize(300, 25)
        self.t2_var_txt24.setText(' λ:  relevance') 
        self.t2_var_txt24.setAlignment(Qt.AlignLeft)
        self.t2_var_txt24.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt24.setHidden(True)  

        # relevance edit
        self.t2_var_edt14 = QLineEdit(self)
        self.t2_var_edt14.move(330, 555)       
        self.t2_var_edt14.resize(140, 22)                                           
        self.t2_var_edt14.setAlignment(Qt.AlignCenter)     
        self.t2_var_edt14.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_var_edt14.setText(self.user_inputs['tab_2_var']['lamda'])
        self.t2_var_edt14.textChanged.connect(self.cb_t2_var_edt14)
        self.t2_var_edt14.setHidden(True) 

        # prior scheme label
        self.t2_var_txt25 = QLabel(self)
        self.t2_var_txt25.move(30, 582)
        self.t2_var_txt25.setFixedSize(300, 25)
        self.t2_var_txt25.setText(' prior scheme') 
        self.t2_var_txt25.setAlignment(Qt.AlignLeft)
        self.t2_var_txt25.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt25.setHidden(True)  

        # prior radiobuttons
        self.t2_var_rdb8 = QRadioButton(' uninformative', self)
        self.t2_var_rdb8.setGeometry(330, 578, 140, 25)
        self.t2_var_rdb8.setStyleSheet("font-size: 12pt; font-family: Serif;")  
        self.t2_var_rdb8.toggled.connect(self.cb_t2_var_bgr2)
        self.t2_var_rdb8.setHidden(True)
        self.t2_var_rdb9 = QRadioButton(' Minnesota', self)
        self.t2_var_rdb9.setGeometry(330, 603, 120, 25)
        self.t2_var_rdb9.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_var_rdb9.toggled.connect(self.cb_t2_var_bgr2)
        self.t2_var_rdb9.setHidden(True)
        if self.user_inputs['tab_2_var']['proxy_prior'] == 1:
            self.t2_var_rdb8.setChecked(True) 
        elif self.user_inputs['tab_2_var']['proxy_prior'] == 2:
            self.t2_var_rdb9.setChecked(True) 
        self.t2_var_bgr2 = QButtonGroup(self)  
        self.t2_var_bgr2.addButton(self.t2_var_rdb8) 
        self.t2_var_bgr2.addButton(self.t2_var_rdb9)     

        # options label
        self.t2_var_txt26 = QLabel(self)
        self.t2_var_txt26.move(520, 220)
        self.t2_var_txt26.setFixedSize(300, 30)
        self.t2_var_txt26.setText(' Options') 
        self.t2_var_txt26.setAlignment(Qt.AlignLeft)
        self.t2_var_txt26.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_var_txt26.setFont(font)
        self.t2_var_txt26.setHidden(True)
    
        # frame around options
        self.t2_var_frm4 = QFrame(self)   
        self.t2_var_frm4.setGeometry(510, 250, 470, 380)  
        self.t2_var_frm4.setFrameShape(QFrame.Panel)
        self.t2_var_frm4.setLineWidth(1)  
        self.t2_var_frm4.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t2_var_frm4.setHidden(True)

        # applications label
        self.t2_var_txt27 = QLabel(self)
        self.t2_var_txt27.move(520, 255)
        self.t2_var_txt27.setFixedSize(300, 25)
        self.t2_var_txt27.setText(' Applications') 
        self.t2_var_txt27.setAlignment(Qt.AlignLeft)
        self.t2_var_txt27.setStyleSheet('font-size: 14pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_var_txt27.setFont(font)
        self.t2_var_txt27.setHidden(True)   

        # in-sample fit label
        self.t2_var_txt38 = QLabel(self)
        self.t2_var_txt38.move(520, 283)
        self.t2_var_txt38.setFixedSize(300, 25)
        self.t2_var_txt38.setText(' in-sample fit') 
        self.t2_var_txt38.setAlignment(Qt.AlignLeft)
        self.t2_var_txt38.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt38.setHidden(True)  

        # in-sample fit radiobuttons
        self.t2_var_rdb24 = QRadioButton(' yes', self)
        self.t2_var_rdb24.setGeometry(830, 280, 50, 25)
        self.t2_var_rdb24.setStyleSheet("font-size: 12pt; font-family: Serif;")  
        self.t2_var_rdb24.toggled.connect(self.cb_t2_var_bgr10)
        self.t2_var_rdb24.setHidden(True)
        self.t2_var_rdb25 = QRadioButton(' no', self)
        self.t2_var_rdb25.setGeometry(910, 280, 50, 25)
        self.t2_var_rdb25.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_var_rdb25.toggled.connect(self.cb_t2_var_bgr10)
        self.t2_var_rdb25.setHidden(True)
        if self.user_inputs['tab_2_var']['insample_fit']:
            self.t2_var_rdb24.setChecked(True) 
        else:
            self.t2_var_rdb25.setChecked(True) 
        self.t2_var_bgr10 = QButtonGroup(self)  
        self.t2_var_bgr10.addButton(self.t2_var_rdb24) 
        self.t2_var_bgr10.addButton(self.t2_var_rdb25)  

        # constrained coefficients label
        self.t2_var_txt28 = QLabel(self)
        self.t2_var_txt28.move(520, 307)
        self.t2_var_txt28.setFixedSize(300, 25)
        self.t2_var_txt28.setText(' constrained coefficients') 
        self.t2_var_txt28.setAlignment(Qt.AlignLeft)
        self.t2_var_txt28.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt28.setHidden(True)  

        # constrained coefficients radiobuttons
        self.t2_var_rdb10 = QRadioButton(' yes', self)
        self.t2_var_rdb10.setGeometry(830, 304, 50, 25)
        self.t2_var_rdb10.setStyleSheet("font-size: 12pt; font-family: Serif;")  
        self.t2_var_rdb10.toggled.connect(self.cb_t2_var_bgr3)
        self.t2_var_rdb10.setHidden(True)
        self.t2_var_rdb11 = QRadioButton(' no', self)
        self.t2_var_rdb11.setGeometry(910, 304, 50, 25)
        self.t2_var_rdb11.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_var_rdb11.toggled.connect(self.cb_t2_var_bgr3)
        self.t2_var_rdb11.setHidden(True)
        if self.user_inputs['tab_2_var']['constrained_coefficients']:
            self.t2_var_rdb10.setChecked(True) 
        else:
            self.t2_var_rdb11.setChecked(True) 
        self.t2_var_bgr3 = QButtonGroup(self)  
        self.t2_var_bgr3.addButton(self.t2_var_rdb10) 
        self.t2_var_bgr3.addButton(self.t2_var_rdb11)  

        # sums-of-coefficients label
        self.t2_var_txt29 = QLabel(self)
        self.t2_var_txt29.move(520, 331)
        self.t2_var_txt29.setFixedSize(300, 25)
        self.t2_var_txt29.setText(' sums-of-coefficients') 
        self.t2_var_txt29.setAlignment(Qt.AlignLeft)
        self.t2_var_txt29.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt29.setHidden(True)   

        # sums-of-coefficients radiobuttons
        self.t2_var_rdb12 = QRadioButton(' yes', self)
        self.t2_var_rdb12.setGeometry(830, 328, 50, 25)
        self.t2_var_rdb12.setStyleSheet("font-size: 12pt; font-family: Serif;")  
        self.t2_var_rdb12.toggled.connect(self.cb_t2_var_bgr4)
        self.t2_var_rdb12.setHidden(True)
        self.t2_var_rdb13 = QRadioButton(' no', self)
        self.t2_var_rdb13.setGeometry(910, 328, 50, 25)
        self.t2_var_rdb13.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_var_rdb13.toggled.connect(self.cb_t2_var_bgr4)
        self.t2_var_rdb13.setHidden(True)
        if self.user_inputs['tab_2_var']['sums_of_coefficients']:
            self.t2_var_rdb12.setChecked(True) 
        else:
            self.t2_var_rdb13.setChecked(True) 
        self.t2_var_bgr4 = QButtonGroup(self)  
        self.t2_var_bgr4.addButton(self.t2_var_rdb12) 
        self.t2_var_bgr4.addButton(self.t2_var_rdb13)  

        # dummy initial observation label
        self.t2_var_txt30 = QLabel(self)
        self.t2_var_txt30.move(520, 355)
        self.t2_var_txt30.setFixedSize(300, 25)
        self.t2_var_txt30.setText(' dummy initial observation') 
        self.t2_var_txt30.setAlignment(Qt.AlignLeft)
        self.t2_var_txt30.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt30.setHidden(True) 

        # dummy initial observation radiobuttons
        self.t2_var_rdb14 = QRadioButton(' yes', self)
        self.t2_var_rdb14.setGeometry(830, 352, 50, 25)
        self.t2_var_rdb14.setStyleSheet("font-size: 12pt; font-family: Serif;")  
        self.t2_var_rdb14.toggled.connect(self.cb_t2_var_bgr5)
        self.t2_var_rdb14.setHidden(True)
        self.t2_var_rdb15 = QRadioButton(' no', self)
        self.t2_var_rdb15.setGeometry(910, 352, 50, 25)
        self.t2_var_rdb15.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_var_rdb15.toggled.connect(self.cb_t2_var_bgr5)
        self.t2_var_rdb15.setHidden(True)
        if self.user_inputs['tab_2_var']['initial_observation']:
            self.t2_var_rdb14.setChecked(True) 
        else:
            self.t2_var_rdb15.setChecked(True) 
        self.t2_var_bgr5 = QButtonGroup(self)  
        self.t2_var_bgr5.addButton(self.t2_var_rdb14) 
        self.t2_var_bgr5.addButton(self.t2_var_rdb15) 

        # long-run label
        self.t2_var_txt31 = QLabel(self)
        self.t2_var_txt31.move(520, 379)
        self.t2_var_txt31.setFixedSize(300, 25)
        self.t2_var_txt31.setText(' long-run prior') 
        self.t2_var_txt31.setAlignment(Qt.AlignLeft)
        self.t2_var_txt31.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt31.setHidden(True) 
        
        # long-run radiobuttons
        self.t2_var_rdb16 = QRadioButton(' yes', self)
        self.t2_var_rdb16.setGeometry(830, 376, 50, 25)
        self.t2_var_rdb16.setStyleSheet("font-size: 12pt; font-family: Serif;")  
        self.t2_var_rdb16.toggled.connect(self.cb_t2_var_bgr6)
        self.t2_var_rdb16.setHidden(True)
        self.t2_var_rdb17 = QRadioButton(' no', self)
        self.t2_var_rdb17.setGeometry(910, 376, 50, 25)
        self.t2_var_rdb17.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_var_rdb17.toggled.connect(self.cb_t2_var_bgr6)
        self.t2_var_rdb17.setHidden(True)
        if self.user_inputs['tab_2_var']['long_run']:
            self.t2_var_rdb16.setChecked(True) 
        else:
            self.t2_var_rdb17.setChecked(True) 
        self.t2_var_bgr6 = QButtonGroup(self)  
        self.t2_var_bgr6.addButton(self.t2_var_rdb16) 
        self.t2_var_bgr6.addButton(self.t2_var_rdb17)         
        
        # stationary prior label
        self.t2_var_txt32 = QLabel(self)
        self.t2_var_txt32.move(520, 403)
        self.t2_var_txt32.setFixedSize(300, 25)
        self.t2_var_txt32.setText(' stationary prior') 
        self.t2_var_txt32.setAlignment(Qt.AlignLeft)
        self.t2_var_txt32.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt32.setHidden(True) 

        # stationary prior radiobuttons
        self.t2_var_rdb18 = QRadioButton(' yes', self)
        self.t2_var_rdb18.setGeometry(830, 400, 50, 25)
        self.t2_var_rdb18.setStyleSheet("font-size: 12pt; font-family: Serif;")  
        self.t2_var_rdb18.toggled.connect(self.cb_t2_var_bgr7)
        self.t2_var_rdb18.setHidden(True)
        self.t2_var_rdb19 = QRadioButton(' no', self)
        self.t2_var_rdb19.setGeometry(910, 400, 50, 25)
        self.t2_var_rdb19.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_var_rdb19.toggled.connect(self.cb_t2_var_bgr7)
        self.t2_var_rdb19.setHidden(True)
        if self.user_inputs['tab_2_var']['stationary']:
            self.t2_var_rdb18.setChecked(True) 
        else:
            self.t2_var_rdb19.setChecked(True) 
        self.t2_var_bgr7 = QButtonGroup(self)  
        self.t2_var_bgr7.addButton(self.t2_var_rdb18) 
        self.t2_var_bgr7.addButton(self.t2_var_rdb19)    

        # marginal likelihood label
        self.t2_var_txt33 = QLabel(self)
        self.t2_var_txt33.move(520, 427)
        self.t2_var_txt33.setFixedSize(300, 25)
        self.t2_var_txt33.setText(' marginal likelihood') 
        self.t2_var_txt33.setAlignment(Qt.AlignLeft)
        self.t2_var_txt33.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt33.setHidden(True) 

        # marginal likelihood radiobuttons
        self.t2_var_rdb20 = QRadioButton(' yes', self)
        self.t2_var_rdb20.setGeometry(830, 424, 50, 25)
        self.t2_var_rdb20.setStyleSheet("font-size: 12pt; font-family: Serif;")  
        self.t2_var_rdb20.toggled.connect(self.cb_t2_var_bgr8)
        self.t2_var_rdb20.setHidden(True)
        self.t2_var_rdb21 = QRadioButton(' no', self)
        self.t2_var_rdb21.setGeometry(910, 424, 50, 25)
        self.t2_var_rdb21.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_var_rdb21.toggled.connect(self.cb_t2_var_bgr8)
        self.t2_var_rdb21.setHidden(True)
        if self.user_inputs['tab_2_var']['marginal_likelihood']:
            self.t2_var_rdb20.setChecked(True) 
        else:
            self.t2_var_rdb21.setChecked(True) 
        self.t2_var_bgr8 = QButtonGroup(self)  
        self.t2_var_bgr8.addButton(self.t2_var_rdb20) 
        self.t2_var_bgr8.addButton(self.t2_var_rdb21)  

        # optimization label
        self.t2_var_txt34 = QLabel(self)
        self.t2_var_txt34.move(520, 451)
        self.t2_var_txt34.setFixedSize(300, 25)
        self.t2_var_txt34.setText(' hyperparameter optimization') 
        self.t2_var_txt34.setAlignment(Qt.AlignLeft)
        self.t2_var_txt34.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt34.setHidden(True)    
    
        # optimization radiobuttons
        self.t2_var_rdb22 = QRadioButton(' yes', self)
        self.t2_var_rdb22.setGeometry(830, 448, 50, 25)
        self.t2_var_rdb22.setStyleSheet("font-size: 12pt; font-family: Serif;")  
        self.t2_var_rdb22.toggled.connect(self.cb_t2_var_bgr9)
        self.t2_var_rdb22.setHidden(True)
        self.t2_var_rdb23 = QRadioButton(' no', self)
        self.t2_var_rdb23.setGeometry(910, 448, 50, 25)
        self.t2_var_rdb23.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_var_rdb23.toggled.connect(self.cb_t2_var_bgr9)
        self.t2_var_rdb23.setHidden(True)
        if self.user_inputs['tab_2_var']['hyperparameter_optimization']:
            self.t2_var_rdb22.setChecked(True) 
        else:
            self.t2_var_rdb23.setChecked(True) 
        self.t2_var_bgr9 = QButtonGroup(self)  
        self.t2_var_bgr9.addButton(self.t2_var_rdb22) 
        self.t2_var_bgr9.addButton(self.t2_var_rdb23)      
    
        # files label
        self.t2_var_txt35 = QLabel(self)
        self.t2_var_txt35.move(520, 505)
        self.t2_var_txt35.setFixedSize(300, 25)
        self.t2_var_txt35.setText(' Files') 
        self.t2_var_txt35.setAlignment(Qt.AlignLeft)
        self.t2_var_txt35.setStyleSheet('font-size: 14pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_var_txt35.setFont(font)
        self.t2_var_txt35.setHidden(True)     
   
        # constrained coefficients label
        self.t2_var_txt36 = QLabel(self)
        self.t2_var_txt36.move(520, 533)
        self.t2_var_txt36.setFixedSize(300, 25)
        self.t2_var_txt36.setText(' constrained coefficients') 
        self.t2_var_txt36.setAlignment(Qt.AlignLeft)
        self.t2_var_txt36.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt36.setHidden(True)
        
        # constrained coefficients edit
        self.t2_var_edt15 = QLineEdit(self)
        self.t2_var_edt15.move(525, 555)       
        self.t2_var_edt15.resize(430, 22)         
        self.t2_var_edt15.setAlignment(Qt.AlignLeft)                                          
        self.t2_var_edt15.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_var_edt15.setText(self.user_inputs['tab_2_var']['coefficients_file'])
        self.t2_var_edt15.textChanged.connect(self.cb_t2_var_edt15)
        self.t2_var_edt15.setHidden(True) 
   
        # long run prior label
        self.t2_var_txt37 = QLabel(self)
        self.t2_var_txt37.move(520, 582)
        self.t2_var_txt37.setFixedSize(300, 25)
        self.t2_var_txt37.setText(' long-run prior') 
        self.t2_var_txt37.setAlignment(Qt.AlignLeft)
        self.t2_var_txt37.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_var_txt37.setHidden(True)     
   
        # long run prior edit
        self.t2_var_edt16 = QLineEdit(self)
        self.t2_var_edt16.move(525, 604)       
        self.t2_var_edt16.resize(430, 22)                                           
        self.t2_var_edt16.setAlignment(Qt.AlignLeft)     
        self.t2_var_edt16.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_var_edt16.setText(self.user_inputs['tab_2_var']['long_run_file'])
        self.t2_var_edt16.textChanged.connect(self.cb_t2_var_edt16)
        self.t2_var_edt16.setHidden(True)     
   
    
        # indicate that tab 2 for vector autoregression is now created
        self.created_tab_2_var = True    
       
    
    def hide_tab_2_var(self):
        
        # hide all controls
        self.t2_var_txt1.setHidden(True)
        self.t2_var_txt2.setHidden(True)
        self.t2_var_txt3.setHidden(True)
        self.t2_var_txt4.setHidden(True)
        self.t2_var_txt5.setHidden(True)
        self.t2_var_txt6.setHidden(True)
        self.t2_var_txt7.setHidden(True)
        self.t2_var_txt8.setHidden(True)
        self.t2_var_txt9.setHidden(True)
        self.t2_var_txt10.setHidden(True)
        self.t2_var_txt11.setHidden(True)
        self.t2_var_txt12.setHidden(True)
        self.t2_var_txt13.setHidden(True)
        self.t2_var_txt14.setHidden(True)
        self.t2_var_txt15.setHidden(True)
        self.t2_var_txt16.setHidden(True)
        self.t2_var_txt17.setHidden(True)
        self.t2_var_txt18.setHidden(True)
        self.t2_var_txt19.setHidden(True)
        self.t2_var_txt20.setHidden(True)
        self.t2_var_txt21.setHidden(True)
        self.t2_var_txt22.setHidden(True)
        self.t2_var_txt23.setHidden(True)
        self.t2_var_txt24.setHidden(True)
        self.t2_var_txt25.setHidden(True)
        self.t2_var_txt26.setHidden(True)
        self.t2_var_txt27.setHidden(True)
        self.t2_var_txt28.setHidden(True)
        self.t2_var_txt29.setHidden(True)
        self.t2_var_txt30.setHidden(True)
        self.t2_var_txt31.setHidden(True)
        self.t2_var_txt32.setHidden(True)
        self.t2_var_txt33.setHidden(True)
        self.t2_var_txt34.setHidden(True)
        self.t2_var_txt35.setHidden(True)
        self.t2_var_txt36.setHidden(True)
        self.t2_var_txt37.setHidden(True)
        self.t2_var_txt38.setHidden(True)
        self.t2_var_frm1.setHidden(True)
        self.t2_var_frm2.setHidden(True)
        self.t2_var_frm3.setHidden(True)
        self.t2_var_frm4.setHidden(True)
        self.t2_var_rdb1.setHidden(True)
        self.t2_var_rdb2.setHidden(True)
        self.t2_var_rdb3.setHidden(True)
        self.t2_var_rdb4.setHidden(True)
        self.t2_var_rdb5.setHidden(True)
        self.t2_var_rdb6.setHidden(True)
        self.t2_var_rdb7.setHidden(True)
        self.t2_var_rdb8.setHidden(True)
        self.t2_var_rdb9.setHidden(True)
        self.t2_var_rdb10.setHidden(True)
        self.t2_var_rdb11.setHidden(True)
        self.t2_var_rdb12.setHidden(True)
        self.t2_var_rdb13.setHidden(True)
        self.t2_var_rdb14.setHidden(True)
        self.t2_var_rdb15.setHidden(True)
        self.t2_var_rdb16.setHidden(True)
        self.t2_var_rdb17.setHidden(True)
        self.t2_var_rdb18.setHidden(True)
        self.t2_var_rdb19.setHidden(True)
        self.t2_var_rdb20.setHidden(True)
        self.t2_var_rdb21.setHidden(True)
        self.t2_var_rdb22.setHidden(True)
        self.t2_var_rdb23.setHidden(True)
        self.t2_var_rdb24.setHidden(True)
        self.t2_var_rdb25.setHidden(True)
        self.t2_var_edt1.setHidden(True)
        self.t2_var_edt2.setHidden(True)
        self.t2_var_edt3.setHidden(True)
        self.t2_var_edt4.setHidden(True)
        self.t2_var_edt5.setHidden(True)       
        self.t2_var_edt6.setHidden(True)       
        self.t2_var_edt7.setHidden(True)       
        self.t2_var_edt8.setHidden(True)       
        self.t2_var_edt9.setHidden(True)       
        self.t2_var_edt10.setHidden(True)       
        self.t2_var_edt11.setHidden(True)       
        self.t2_var_edt12.setHidden(True)     
        self.t2_var_edt13.setHidden(True)
        self.t2_var_edt14.setHidden(True)     
        self.t2_var_edt15.setHidden(True)  
        self.t2_var_edt16.setHidden(True)  
        self.t2_var_cbx1.setHidden(True)
        self.t2_var_cbx2.setHidden(True)
        self.t2_var_cbx3.setHidden(True)
        
        # update tab color
        self.tab_pbt2.setStyleSheet("background:rgb" + str(self.backtabs_color) + ";")       
        
        
    def show_tab_2_var(self): 
    
        # show all controls
        self.t2_var_txt1.setVisible(True)
        self.t2_var_txt2.setVisible(True)
        self.t2_var_txt3.setVisible(True)
        self.t2_var_txt4.setVisible(True)
        self.t2_var_txt5.setVisible(True)
        self.t2_var_txt6.setVisible(True)
        self.t2_var_txt7.setVisible(True)
        self.t2_var_txt8.setVisible(True)
        self.t2_var_txt9.setVisible(True)
        self.t2_var_txt10.setVisible(True)
        self.t2_var_txt11.setVisible(True)
        self.t2_var_txt12.setVisible(True)
        self.t2_var_txt13.setVisible(True)
        self.t2_var_txt14.setVisible(True)
        self.t2_var_txt15.setVisible(True)
        self.t2_var_txt16.setVisible(True)
        self.t2_var_txt17.setVisible(True)
        self.t2_var_txt18.setVisible(True)
        self.t2_var_txt19.setVisible(True)
        self.t2_var_txt20.setVisible(True)
        self.t2_var_txt21.setVisible(True)
        self.t2_var_txt22.setVisible(True)
        self.t2_var_txt23.setVisible(True)
        self.t2_var_txt24.setVisible(True)
        self.t2_var_txt25.setVisible(True)
        self.t2_var_txt26.setVisible(True)
        self.t2_var_txt27.setVisible(True)
        self.t2_var_txt28.setVisible(True)
        self.t2_var_txt29.setVisible(True)
        self.t2_var_txt30.setVisible(True)
        self.t2_var_txt31.setVisible(True)
        self.t2_var_txt32.setVisible(True)
        self.t2_var_txt33.setVisible(True)
        self.t2_var_txt34.setVisible(True)
        self.t2_var_txt35.setVisible(True)
        self.t2_var_txt36.setVisible(True)
        self.t2_var_txt37.setVisible(True)
        self.t2_var_txt38.setVisible(True)
        self.t2_var_frm1.setVisible(True)
        self.t2_var_frm2.setVisible(True)
        self.t2_var_frm3.setVisible(True)
        self.t2_var_frm4.setVisible(True)
        self.t2_var_rdb1.setVisible(True)
        self.t2_var_rdb2.setVisible(True)
        self.t2_var_rdb3.setVisible(True)
        self.t2_var_rdb4.setVisible(True)
        self.t2_var_rdb5.setVisible(True)
        self.t2_var_rdb6.setVisible(True)
        self.t2_var_rdb7.setVisible(True)
        self.t2_var_rdb8.setVisible(True)
        self.t2_var_rdb9.setVisible(True)
        self.t2_var_rdb10.setVisible(True)
        self.t2_var_rdb11.setVisible(True)
        self.t2_var_rdb12.setVisible(True)
        self.t2_var_rdb13.setVisible(True)
        self.t2_var_rdb14.setVisible(True)
        self.t2_var_rdb15.setVisible(True)
        self.t2_var_rdb16.setVisible(True)
        self.t2_var_rdb17.setVisible(True)
        self.t2_var_rdb18.setVisible(True)
        self.t2_var_rdb19.setVisible(True)
        self.t2_var_rdb20.setVisible(True)
        self.t2_var_rdb21.setVisible(True)
        self.t2_var_rdb22.setVisible(True)
        self.t2_var_rdb23.setVisible(True)
        self.t2_var_rdb24.setVisible(True)
        self.t2_var_rdb25.setVisible(True)
        self.t2_var_edt1.setVisible(True)
        self.t2_var_edt2.setVisible(True)
        self.t2_var_edt3.setVisible(True)
        self.t2_var_edt4.setVisible(True)
        self.t2_var_edt5.setVisible(True)  
        self.t2_var_edt6.setVisible(True)  
        self.t2_var_edt7.setVisible(True)  
        self.t2_var_edt8.setVisible(True)  
        self.t2_var_edt9.setVisible(True)  
        self.t2_var_edt10.setVisible(True)  
        self.t2_var_edt11.setVisible(True)  
        self.t2_var_edt12.setVisible(True) 
        self.t2_var_edt13.setVisible(True)
        self.t2_var_edt14.setVisible(True)
        self.t2_var_edt15.setVisible(True)
        self.t2_var_edt16.setVisible(True)
        self.t2_var_cbx1.setVisible(True)
        self.t2_var_cbx2.setVisible(True)
        self.t2_var_cbx3.setVisible(True)
            
        
    def cb_t2_var_bgr1(self):
        if self.t2_var_rdb1.isChecked() == True:
            self.user_inputs['tab_2_var']['var_type'] = 1
        elif self.t2_var_rdb2.isChecked() == True:
            self.user_inputs['tab_2_var']['var_type'] = 2
        elif self.t2_var_rdb3.isChecked() == True:
            self.user_inputs['tab_2_var']['var_type'] = 3       
        elif self.t2_var_rdb4.isChecked() == True:
            self.user_inputs['tab_2_var']['var_type'] = 4        
        elif self.t2_var_rdb5.isChecked() == True:
            self.user_inputs['tab_2_var']['var_type'] = 5        
        elif self.t2_var_rdb6.isChecked() == True:
            self.user_inputs['tab_2_var']['var_type'] = 6        
        elif self.t2_var_rdb7.isChecked() == True:
            self.user_inputs['tab_2_var']['var_type'] = 7          
    
    def cb_t2_var_edt1(self):
        self.user_inputs['tab_2_var']['iterations'] = self.t2_var_edt1.text()         
   
    def cb_t2_var_edt2(self):
        self.user_inputs['tab_2_var']['burnin'] = self.t2_var_edt2.text()   
   
    def cb_t2_var_edt3(self):
        self.user_inputs['tab_2_var']['model_credibility'] = self.t2_var_edt3.text()      
   
    def cb_t2_var_cbx1(self, state):     
        if (state == Qt.Checked):
            self.user_inputs['tab_2_var']['constant'] = True 
        else:
            self.user_inputs['tab_2_var']['constant'] = False     
   
    def cb_t2_var_cbx2(self, state):     
        if (state == Qt.Checked):
            self.user_inputs['tab_2_var']['trend'] = True 
        else:
            self.user_inputs['tab_2_var']['trend'] = False  
            
    def cb_t2_var_cbx3(self, state):     
        if (state == Qt.Checked):
            self.user_inputs['tab_2_var']['quadratic_trend'] = True 
        else:
            self.user_inputs['tab_2_var']['quadratic_trend'] = False             
            
    def cb_t2_var_bgr2(self):
        if self.t2_var_rdb8.isChecked() == True:
            self.user_inputs['tab_2_var']['proxy_prior'] = 1
        elif self.t2_var_rdb9.isChecked() == True:
            self.user_inputs['tab_2_var']['proxy_prior'] = 2

    def cb_t2_var_edt4(self):
        self.user_inputs['tab_2_var']['lags'] = self.t2_var_edt4.text() 
            
    def cb_t2_var_edt5(self):
        self.user_inputs['tab_2_var']['ar_coefficients'] = self.t2_var_edt5.text()             
            
    def cb_t2_var_edt6(self):
        self.user_inputs['tab_2_var']['pi1'] = self.t2_var_edt6.text()
        
    def cb_t2_var_edt7(self):
        self.user_inputs['tab_2_var']['pi2'] = self.t2_var_edt7.text() 
        
    def cb_t2_var_edt8(self):
        self.user_inputs['tab_2_var']['pi3'] = self.t2_var_edt8.text() 
        
    def cb_t2_var_edt9(self):
        self.user_inputs['tab_2_var']['pi4'] = self.t2_var_edt9.text() 
        
    def cb_t2_var_edt10(self):
        self.user_inputs['tab_2_var']['pi5'] = self.t2_var_edt10.text() 
        
    def cb_t2_var_edt11(self):
        self.user_inputs['tab_2_var']['pi6'] = self.t2_var_edt11.text() 
        
    def cb_t2_var_edt12(self):
        self.user_inputs['tab_2_var']['pi7'] = self.t2_var_edt12.text()             
        
    def cb_t2_var_edt13(self):
        self.user_inputs['tab_2_var']['proxy_variables'] = self.t2_var_edt13.text()            
            
    def cb_t2_var_edt14(self):
        self.user_inputs['tab_2_var']['lamda'] = self.t2_var_edt14.text()

    def cb_t2_var_bgr10(self):
        if self.t2_var_rdb24.isChecked() == True:
            self.user_inputs['tab_2_var']['insample_fit'] = True
        elif self.t2_var_rdb25.isChecked() == True:
            self.user_inputs['tab_2_var']['insample_fit'] = False

    def cb_t2_var_bgr3(self):
        if self.t2_var_rdb10.isChecked() == True:
            self.user_inputs['tab_2_var']['constrained_coefficients'] = True
        elif self.t2_var_rdb11.isChecked() == True:
            self.user_inputs['tab_2_var']['constrained_coefficients'] = False

    def cb_t2_var_bgr4(self):
        if self.t2_var_rdb12.isChecked() == True:
            self.user_inputs['tab_2_var']['sums_of_coefficients'] = True
        elif self.t2_var_rdb13.isChecked() == True:
            self.user_inputs['tab_2_var']['sums_of_coefficients'] = False

    def cb_t2_var_bgr5(self):
        if self.t2_var_rdb14.isChecked() == True:
            self.user_inputs['tab_2_var']['initial_observation'] = True
        elif self.t2_var_rdb15.isChecked() == True:
            self.user_inputs['tab_2_var']['initial_observation'] = False

    def cb_t2_var_bgr6(self):
        if self.t2_var_rdb16.isChecked() == True:
            self.user_inputs['tab_2_var']['long_run'] = True
        elif self.t2_var_rdb17.isChecked() == True:
            self.user_inputs['tab_2_var']['long_run'] = False

    def cb_t2_var_bgr7(self):
        if self.t2_var_rdb18.isChecked() == True:
            self.user_inputs['tab_2_var']['stationary'] = True
        elif self.t2_var_rdb19.isChecked() == True:
            self.user_inputs['tab_2_var']['stationary'] = False

    def cb_t2_var_bgr8(self):
        if self.t2_var_rdb20.isChecked() == True:
            self.user_inputs['tab_2_var']['marginal_likelihood'] = True
        elif self.t2_var_rdb21.isChecked() == True:
            self.user_inputs['tab_2_var']['marginal_likelihood'] = False           
            
    def cb_t2_var_bgr9(self):
        if self.t2_var_rdb22.isChecked() == True:
            self.user_inputs['tab_2_var']['hyperparameter_optimization'] = True
        elif self.t2_var_rdb23.isChecked() == True:
            self.user_inputs['tab_2_var']['hyperparameter_optimization'] = False             
            
    def cb_t2_var_edt15(self):
        self.user_inputs['tab_2_var']['coefficients_file'] = self.t2_var_edt15.text()             
            
    def cb_t2_var_edt16(self):
        self.user_inputs['tab_2_var']['long_run_file'] = self.t2_var_edt16.text()             
            
            
            
            