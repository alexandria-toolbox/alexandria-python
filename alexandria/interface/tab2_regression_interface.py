# imports
from PyQt5.QtWidgets import QLabel, QFrame, QRadioButton, QButtonGroup, QLineEdit, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont



class Tab2RegressionInterface(object):


    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass


    def create_tab_2_lr(self):
    
        # regression label
        self.t2_lr_txt1 = QLabel(self)
        self.t2_lr_txt1.move(30, 60)
        self.t2_lr_txt1.setFixedSize(300, 30)
        self.t2_lr_txt1.setText(' Regression type') 
        self.t2_lr_txt1.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt1.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_lr_txt1.setFont(font)
        self.t2_lr_txt1.setHidden(True)
        
        # frame around regression
        self.t2_lr_frm1 = QFrame(self)   
        self.t2_lr_frm1.setGeometry(20, 90, 470, 110)  
        self.t2_lr_frm1.setFrameShape(QFrame.Panel)
        self.t2_lr_frm1.setLineWidth(1)  
        self.t2_lr_frm1.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t2_lr_frm1.setHidden(True)
        
        # regression radiobuttons
        self.t2_lr_rdb1 = QRadioButton(' maximum likelihood', self)
        self.t2_lr_rdb1.setGeometry(30, 95, 200, 30)
        self.t2_lr_rdb1.setStyleSheet("font-size: 12pt; font-family: Serif;")  
        self.t2_lr_rdb1.toggled.connect(self.cb_t2_lr_bgr1)
        self.t2_lr_rdb1.setHidden(True)
        self.t2_lr_rdb2 = QRadioButton(' simple Bayesian', self)
        self.t2_lr_rdb2.setGeometry(260, 95, 200, 30)
        self.t2_lr_rdb2.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_lr_rdb2.toggled.connect(self.cb_t2_lr_bgr1)
        self.t2_lr_rdb2.setHidden(True)
        self.t2_lr_rdb3 = QRadioButton(' hierarchical', self)
        self.t2_lr_rdb3.setGeometry(30, 130, 200, 30)
        self.t2_lr_rdb3.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_lr_rdb3.toggled.connect(self.cb_t2_lr_bgr1)
        self.t2_lr_rdb3.setHidden(True)
        self.t2_lr_rdb4 = QRadioButton(' independent', self)
        self.t2_lr_rdb4.setGeometry(260, 130, 200, 30)
        self.t2_lr_rdb4.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_lr_rdb4.toggled.connect(self.cb_t2_lr_bgr1)
        self.t2_lr_rdb4.setHidden(True)
        self.t2_lr_rdb5 = QRadioButton(' heteroscedastic', self)
        self.t2_lr_rdb5.setGeometry(30, 165, 200, 30)
        self.t2_lr_rdb5.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_lr_rdb5.toggled.connect(self.cb_t2_lr_bgr1)
        self.t2_lr_rdb5.setHidden(True)
        self.t2_lr_rdb6 = QRadioButton(' autocorrelated', self)
        self.t2_lr_rdb6.setGeometry(260, 165, 200, 30)
        self.t2_lr_rdb6.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_lr_rdb6.toggled.connect(self.cb_t2_lr_bgr1)
        self.t2_lr_rdb6.setHidden(True)
        if self.user_inputs['tab_2_lr']['regression_type'] == 1:
            self.t2_lr_rdb1.setChecked(True) 
        elif self.user_inputs['tab_2_lr']['regression_type'] == 2:
            self.t2_lr_rdb2.setChecked(True) 
        elif self.user_inputs['tab_2_lr']['regression_type'] == 3:
            self.t2_lr_rdb3.setChecked(True) 
        elif self.user_inputs['tab_2_lr']['regression_type'] == 4:
            self.t2_lr_rdb4.setChecked(True) 
        elif self.user_inputs['tab_2_lr']['regression_type'] == 5:
            self.t2_lr_rdb5.setChecked(True) 
        elif self.user_inputs['tab_2_lr']['regression_type'] == 6:
            self.t2_lr_rdb6.setChecked(True)   
        self.t2_lr_bgr1 = QButtonGroup(self)  
        self.t2_lr_bgr1.addButton(self.t2_lr_rdb1) 
        self.t2_lr_bgr1.addButton(self.t2_lr_rdb2)     
        self.t2_lr_bgr1.addButton(self.t2_lr_rdb3) 
        self.t2_lr_bgr1.addButton(self.t2_lr_rdb4) 
        self.t2_lr_bgr1.addButton(self.t2_lr_rdb5) 
        self.t2_lr_bgr1.addButton(self.t2_lr_rdb6) 
            
        # estimation label
        self.t2_lr_txt2 = QLabel(self)
        self.t2_lr_txt2.move(520, 60)
        self.t2_lr_txt2.setFixedSize(300, 30)
        self.t2_lr_txt2.setText(' Estimation') 
        self.t2_lr_txt2.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt2.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_lr_txt2.setFont(font)
        self.t2_lr_txt2.setHidden(True)
        
        # frame around estimation
        self.t2_lr_frm2 = QFrame(self)   
        self.t2_lr_frm2.setGeometry(510, 90, 470, 110)  
        self.t2_lr_frm2.setFrameShape(QFrame.Panel)
        self.t2_lr_frm2.setLineWidth(1)  
        self.t2_lr_frm2.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t2_lr_frm2.setHidden(True)
        
        # iteration label
        self.t2_lr_txt3 = QLabel(self)
        self.t2_lr_txt3.move(520, 100)
        self.t2_lr_txt3.setFixedSize(200, 20)
        self.t2_lr_txt3.setText(' iterations') 
        self.t2_lr_txt3.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt3.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt3.setHidden(True)
        
        # iteration edit
        self.t2_lr_edt1 = QLineEdit(self)
        self.t2_lr_edt1.move(770, 100)       
        self.t2_lr_edt1.resize(70, 25)                                           
        self.t2_lr_edt1.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt1.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt1.setText(self.user_inputs['tab_2_lr']['iterations'])
        self.t2_lr_edt1.textChanged.connect(self.cb_t2_lr_edt1)
        self.t2_lr_edt1.setHidden(True)
        
        # burn-in label
        self.t2_lr_txt4 = QLabel(self)
        self.t2_lr_txt4.move(520, 135)
        self.t2_lr_txt4.setFixedSize(200, 20)
        self.t2_lr_txt4.setText(' burn-in') 
        self.t2_lr_txt4.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt4.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt4.setHidden(True)
        
        # burn-in edit
        self.t2_lr_edt2 = QLineEdit(self)
        self.t2_lr_edt2.move(770, 135)       
        self.t2_lr_edt2.resize(70, 25)                                           
        self.t2_lr_edt2.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt2.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt2.setText(self.user_inputs['tab_2_lr']['burnin'])
        self.t2_lr_edt2.textChanged.connect(self.cb_t2_lr_edt2)
        self.t2_lr_edt2.setHidden(True)
        
        # credibility label
        self.t2_lr_txt5 = QLabel(self)
        self.t2_lr_txt5.move(520, 170)
        self.t2_lr_txt5.setFixedSize(200, 20)
        self.t2_lr_txt5.setText(' credibility level') 
        self.t2_lr_txt5.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt5.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt5.setHidden(True)
        
        # credibility edit
        self.t2_lr_edt3 = QLineEdit(self)
        self.t2_lr_edt3.move(770, 170)       
        self.t2_lr_edt3.resize(70, 25)                                           
        self.t2_lr_edt3.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt3.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt3.setText(self.user_inputs['tab_2_lr']['model_credibility'])
        self.t2_lr_edt3.textChanged.connect(self.cb_t2_lr_edt3)
        self.t2_lr_edt3.setHidden(True)
        
        # hyperparameter label
        self.t2_lr_txt6 = QLabel(self)
        self.t2_lr_txt6.move(30, 220)
        self.t2_lr_txt6.setFixedSize(300, 30)
        self.t2_lr_txt6.setText(' Hyperparameters') 
        self.t2_lr_txt6.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt6.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_lr_txt6.setFont(font)
        self.t2_lr_txt6.setHidden(True)
        
        # frame around hyperparameters
        self.t2_lr_frm3 = QFrame(self)   
        self.t2_lr_frm3.setGeometry(20, 250, 470, 380)  
        self.t2_lr_frm3.setFrameShape(QFrame.Panel)
        self.t2_lr_frm3.setLineWidth(1)  
        self.t2_lr_frm3.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t2_lr_frm3.setHidden(True)
        
        # Bayesian label
        self.t2_lr_txt7 = QLabel(self)
        self.t2_lr_txt7.move(30, 260)
        self.t2_lr_txt7.setFixedSize(300, 25)
        self.t2_lr_txt7.setText(' All Bayesian') 
        self.t2_lr_txt7.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt7.setStyleSheet('font-size: 14pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_lr_txt7.setFont(font)
        self.t2_lr_txt7.setHidden(True)
        
        # prior mean b label
        self.t2_lr_txt8 = QLabel(self)
        self.t2_lr_txt8.move(30, 295)
        self.t2_lr_txt8.setFixedSize(20, 20)
        self.t2_lr_txt8.setText(' b') 
        self.t2_lr_txt8.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt8.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt8.setHidden(True)
        
        # prior mean b edit
        self.t2_lr_edt4 = QLineEdit(self)
        self.t2_lr_edt4.move(155, 290)       
        self.t2_lr_edt4.resize(70, 25)                                           
        self.t2_lr_edt4.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt4.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt4.setText(self.user_inputs['tab_2_lr']['b'])
        self.t2_lr_edt4.textChanged.connect(self.cb_t2_lr_edt4)
        self.t2_lr_edt4.setHidden(True)
        
        # prior mean V label
        self.t2_lr_txt9 = QLabel(self)
        self.t2_lr_txt9.move(30, 328)
        self.t2_lr_txt9.setFixedSize(20, 20)
        self.t2_lr_txt9.setText(' V') 
        self.t2_lr_txt9.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt9.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt9.setHidden(True)
        
        # prior mean V edit
        self.t2_lr_edt5 = QLineEdit(self)
        self.t2_lr_edt5.move(155, 323)
        self.t2_lr_edt5.resize(70, 25)                                           
        self.t2_lr_edt5.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt5.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt5.setText(self.user_inputs['tab_2_lr']['V'])
        self.t2_lr_edt5.textChanged.connect(self.cb_t2_lr_edt5)
        self.t2_lr_edt5.setHidden(True)
        
        # hierarchical label
        self.t2_lr_txt10 = QLabel(self)
        self.t2_lr_txt10.move(275, 260)
        self.t2_lr_txt10.setFixedSize(180, 22)
        self.t2_lr_txt10.setText(' Hierarchical') 
        self.t2_lr_txt10.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt10.setStyleSheet('font-size: 14pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_lr_txt10.setFont(font)
        self.t2_lr_txt10.setHidden(True)
        
        # shape alpha label
        self.t2_lr_txt11 = QLabel(self)
        self.t2_lr_txt11.move(275, 290)
        self.t2_lr_txt11.setFixedSize(200, 25)
        self.t2_lr_txt11.setText(' α') 
        self.t2_lr_txt11.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt11.setStyleSheet('font-size: 16pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt11.setHidden(True)
        
        # shape alpha edit
        self.t2_lr_edt6 = QLineEdit(self)
        self.t2_lr_edt6.move(400, 290)       
        self.t2_lr_edt6.resize(70, 25)                                           
        self.t2_lr_edt6.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt6.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt6.setText(self.user_inputs['tab_2_lr']['alpha'])
        self.t2_lr_edt6.textChanged.connect(self.cb_t2_lr_edt6)
        self.t2_lr_edt6.setHidden(True)
        
        # scale delta label
        self.t2_lr_txt12 = QLabel(self)
        self.t2_lr_txt12.move(275, 323)
        self.t2_lr_txt12.setFixedSize(200, 25)
        self.t2_lr_txt12.setText(' δ') 
        self.t2_lr_txt12.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt12.setStyleSheet('font-size: 16pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt12.setHidden(True)
        
        # scale delta edit
        self.t2_lr_edt7 = QLineEdit(self)
        self.t2_lr_edt7.move(400, 323)
        self.t2_lr_edt7.resize(70, 25)                                           
        self.t2_lr_edt7.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt7.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt7.setText(self.user_inputs['tab_2_lr']['delta'])
        self.t2_lr_edt7.textChanged.connect(self.cb_t2_lr_edt7)
        self.t2_lr_edt7.setHidden(True)
        
        # heteroscedastic label
        self.t2_lr_txt13 = QLabel(self)
        self.t2_lr_txt13.move(30, 368)
        self.t2_lr_txt13.setFixedSize(200, 22)
        self.t2_lr_txt13.setText(' Heteroscedastic') 
        self.t2_lr_txt13.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt13.setStyleSheet('font-size: 14pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_lr_txt13.setFont(font)
        self.t2_lr_txt13.setHidden(True)
        
        # heteroscedastic mean g label
        self.t2_lr_txt14 = QLabel(self)
        self.t2_lr_txt14.move(30, 402)
        self.t2_lr_txt14.setFixedSize(200, 22)
        self.t2_lr_txt14.setText(' g') 
        self.t2_lr_txt14.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt14.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt14.setHidden(True)
        
        # heteroscedastic mean g edit
        self.t2_lr_edt8 = QLineEdit(self)
        self.t2_lr_edt8.move(155, 398) 
        self.t2_lr_edt8.resize(70, 25)                                           
        self.t2_lr_edt8.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt8.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt8.setText(self.user_inputs['tab_2_lr']['g'])
        self.t2_lr_edt8.textChanged.connect(self.cb_t2_lr_edt8)
        self.t2_lr_edt8.setHidden(True)
        
        # heteroscedastic variance Q label
        self.t2_lr_txt15 = QLabel(self)
        self.t2_lr_txt15.move(30, 435)
        self.t2_lr_txt15.setFixedSize(200, 22)
        self.t2_lr_txt15.setText(' Q') 
        self.t2_lr_txt15.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt15.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt15.setHidden(True)
        
        # heteroscedastic variance Q edit
        self.t2_lr_edt9 = QLineEdit(self)
        self.t2_lr_edt9.move(155, 430)       
        self.t2_lr_edt9.resize(70, 25)                                           
        self.t2_lr_edt9.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt9.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt9.setText(self.user_inputs['tab_2_lr']['Q'])
        self.t2_lr_edt9.textChanged.connect(self.cb_t2_lr_edt9)
        self.t2_lr_edt9.setHidden(True)
        
        # kernel variance tau label
        self.t2_lr_txt16 = QLabel(self)
        self.t2_lr_txt16.move(30, 463)
        self.t2_lr_txt16.setFixedSize(200, 25)
        self.t2_lr_txt16.setText(' τ') 
        self.t2_lr_txt16.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt16.setStyleSheet('font-size: 16pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt16.setHidden(True)
        
        # kernel variance tau edit
        self.t2_lr_edt10 = QLineEdit(self) 
        self.t2_lr_edt10.move(155, 463)
        self.t2_lr_edt10.resize(70, 25)                                           
        self.t2_lr_edt10.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt10.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt10.setText(self.user_inputs['tab_2_lr']['tau'])
        self.t2_lr_edt10.textChanged.connect(self.cb_t2_lr_edt10)
        self.t2_lr_edt10.setHidden(True)
        
        # thinning label
        self.t2_lr_txt17 = QLabel(self)
        self.t2_lr_txt17.move(30, 500)
        self.t2_lr_txt17.setFixedSize(250, 25)
        self.t2_lr_txt17.setText(' thinning') 
        self.t2_lr_txt17.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt17.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt17.setHidden(True)
        
        # thinning checkbox
        self.t2_lr_cbx1 = QCheckBox(self)
        self.t2_lr_cbx1.move(211, 491)  
        self.t2_lr_cbx1.setStyleSheet("QCheckBox::indicator:unchecked \
        {background-color : rgb(255, 255, 255); border: 0.5px solid rgb(0, 0, 0)}; \
        QCheckBox::indicator {width: 12px; height: 12px}") 
        self.t2_lr_cbx1.setChecked(self.user_inputs['tab_2_lr']['thinning'])
        self.t2_lr_cbx1.stateChanged.connect(self.cb_t2_lr_cbx1) 
        self.t2_lr_cbx1.setHidden(True)
        
        # thinning frequency label
        self.t2_lr_txt18 = QLabel(self)
        self.t2_lr_txt18.move(30, 533)
        self.t2_lr_txt18.setFixedSize(250, 25)
        self.t2_lr_txt18.setText(' frequency') 
        self.t2_lr_txt18.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt18.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt18.setHidden(True)
        
        # thinning frequency edit
        self.t2_lr_edt11 = QLineEdit(self) 
        self.t2_lr_edt11.move(155, 528)   
        self.t2_lr_edt11.resize(70, 25)                                           
        self.t2_lr_edt11.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt11.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt11.setText(self.user_inputs['tab_2_lr']['thinning_frequency'])
        self.t2_lr_edt11.textChanged.connect(self.cb_t2_lr_edt11)
        self.t2_lr_edt11.setHidden(True)
        
        # heteroscedasticity regressors Z label
        self.t2_lr_txt19 = QLabel(self)
        self.t2_lr_txt19.move(30, 565)
        self.t2_lr_txt19.setFixedSize(300, 25)
        self.t2_lr_txt19.setText(' Z variables') 
        self.t2_lr_txt19.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt19.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt19.setHidden(True)
        
        # heteroscedasticity regressors Z edit
        self.t2_lr_edt12 = QLineEdit(self)
        self.t2_lr_edt12.move(35, 590)       
        self.t2_lr_edt12.resize(190, 25)                                           
        self.t2_lr_edt12.setAlignment(Qt.AlignLeft)     
        self.t2_lr_edt12.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt12.setText(self.user_inputs['tab_2_lr']['Z_variables'])
        self.t2_lr_edt12.textChanged.connect(self.cb_t2_lr_edt12)
        self.t2_lr_edt12.setHidden(True)
        
        # autocorrelation label
        self.t2_lr_txt20 = QLabel(self)
        self.t2_lr_txt20.move(275, 368)
        self.t2_lr_txt20.setFixedSize(200, 22)
        self.t2_lr_txt20.setText(' Autocorrelated') 
        self.t2_lr_txt20.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt20.setStyleSheet('font-size: 14pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_lr_txt20.setFont(font)
        self.t2_lr_txt20.setHidden(True)
        
        # autocorrelation order q label
        self.t2_lr_txt21 = QLabel(self)
        self.t2_lr_txt21.move(275, 403)
        self.t2_lr_txt21.setFixedSize(200, 22)
        self.t2_lr_txt21.setText(' q') 
        self.t2_lr_txt21.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt21.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt21.setHidden(True)
        
        # autocorrelation order q edit
        self.t2_lr_edt13 = QLineEdit(self)     
        self.t2_lr_edt13.move(400, 398)  
        self.t2_lr_edt13.resize(70, 25)                                           
        self.t2_lr_edt13.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt13.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt13.setText(self.user_inputs['tab_2_lr']['q'])
        self.t2_lr_edt13.textChanged.connect(self.cb_t2_lr_edt13)
        self.t2_lr_edt13.setHidden(True)
        
        # autocorrelation mean p label
        self.t2_lr_txt22 = QLabel(self)
        self.t2_lr_txt22.move(275, 435)
        self.t2_lr_txt22.setFixedSize(200, 22)
        self.t2_lr_txt22.setText(' p') 
        self.t2_lr_txt22.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt22.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt22.setHidden(True)
        
        # autocorrelation mean p edit
        self.t2_lr_edt14 = QLineEdit(self)
        self.t2_lr_edt14.move(400, 430)       
        self.t2_lr_edt14.resize(70, 25)                                           
        self.t2_lr_edt14.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt14.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt14.setText(self.user_inputs['tab_2_lr']['p'])
        self.t2_lr_edt14.textChanged.connect(self.cb_t2_lr_edt14)
        self.t2_lr_edt14.setHidden(True)
        
        # autocorrelation variance H label
        self.t2_lr_txt23 = QLabel(self)
        self.t2_lr_txt23.move(275, 468)
        self.t2_lr_txt23.setFixedSize(200, 20)
        self.t2_lr_txt23.setText(' H') 
        self.t2_lr_txt23.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt23.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt23.setHidden(True)
        
        # autocorrelation variance H edit
        self.t2_lr_edt15 = QLineEdit(self)   
        self.t2_lr_edt15.move(400, 463)   
        self.t2_lr_edt15.resize(70, 25)                                           
        self.t2_lr_edt15.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt15.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt15.setText(self.user_inputs['tab_2_lr']['H'])
        self.t2_lr_edt15.textChanged.connect(self.cb_t2_lr_edt15)
        self.t2_lr_edt15.setHidden(True)
        
        # exogenous label
        self.t2_lr_txt24 = QLabel(self)
        self.t2_lr_txt24.move(520, 220)
        self.t2_lr_txt24.setFixedSize(300, 30)
        self.t2_lr_txt24.setText(' Exogenous') 
        self.t2_lr_txt24.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt24.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_lr_txt24.setFont(font)
        self.t2_lr_txt24.setHidden(True)
        
        # frame around exogenous
        self.t2_lr_frm4 = QFrame(self)   
        self.t2_lr_frm4.setGeometry(510, 250, 470, 110)  
        self.t2_lr_frm4.setFrameShape(QFrame.Panel)
        self.t2_lr_frm4.setLineWidth(1)  
        self.t2_lr_frm4.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t2_lr_frm4.setHidden(True)
        
        # constant label
        self.t2_lr_txt25 = QLabel(self)
        self.t2_lr_txt25.move(520, 260)
        self.t2_lr_txt25.setFixedSize(200, 20)
        self.t2_lr_txt25.setText(' constant') 
        self.t2_lr_txt25.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt25.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt25.setHidden(True)
        
        # constant checkbox
        self.t2_lr_cbx2 = QCheckBox(self)
        self.t2_lr_cbx2.move(690, 255)  
        self.t2_lr_cbx2.setStyleSheet("QCheckBox::indicator:unchecked \
        {background-color : rgb(255, 255, 255); border: 0.5px solid rgb(0, 0, 0)}; \
        QCheckBox::indicator {width: 12px; height: 12px}") 
        self.t2_lr_cbx2.setChecked(self.user_inputs['tab_2_lr']['constant'])
        self.t2_lr_cbx2.stateChanged.connect(self.cb_t2_lr_cbx2) 
        self.t2_lr_cbx2.setHidden(True)
        
        # constant mean label
        self.t2_lr_txt26 = QLabel(self)
        self.t2_lr_txt26.move(740, 260)
        self.t2_lr_txt26.setFixedSize(20, 20)
        self.t2_lr_txt26.setText(' b') 
        self.t2_lr_txt26.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt26.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt26.setHidden(True)
        
        # constant mean edit
        self.t2_lr_edt16 = QLineEdit(self)
        self.t2_lr_edt16.move(770, 255)       
        self.t2_lr_edt16.resize(70, 25)                                           
        self.t2_lr_edt16.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt16.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt16.setText(self.user_inputs['tab_2_lr']['b_constant'])
        self.t2_lr_edt16.textChanged.connect(self.cb_t2_lr_edt16)
        self.t2_lr_edt16.setHidden(True)
        
        # constant variance label
        self.t2_lr_txt27 = QLabel(self)
        self.t2_lr_txt27.move(870, 260)
        self.t2_lr_txt27.setFixedSize(20, 20)
        self.t2_lr_txt27.setText(' V') 
        self.t2_lr_txt27.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt27.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt27.setHidden(True)
        
        # constant variance edit
        self.t2_lr_edt17 = QLineEdit(self)
        self.t2_lr_edt17.move(900, 255)       
        self.t2_lr_edt17.resize(70, 25)                                           
        self.t2_lr_edt17.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt17.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt17.setText(self.user_inputs['tab_2_lr']['V_constant'])
        self.t2_lr_edt17.textChanged.connect(self.cb_t2_lr_edt17)
        self.t2_lr_edt17.setHidden(True)
        
        # trend label
        self.t2_lr_txt28 = QLabel(self)
        self.t2_lr_txt28.move(520, 295)
        self.t2_lr_txt28.setFixedSize(200, 20)
        self.t2_lr_txt28.setText(' linear trend') 
        self.t2_lr_txt28.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt28.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt28.setHidden(True)
        
        # trend checkbox
        self.t2_lr_cbx3 = QCheckBox(self)
        self.t2_lr_cbx3.move(690, 290)  
        self.t2_lr_cbx3.setStyleSheet("QCheckBox::indicator:unchecked \
        {background-color : rgb(255, 255, 255); border: 0.5px solid rgb(0, 0, 0)}; \
        QCheckBox::indicator {width: 12px; height: 12px}") 
        self.t2_lr_cbx3.setChecked(self.user_inputs['tab_2_lr']['trend'])
        self.t2_lr_cbx3.stateChanged.connect(self.cb_t2_lr_cbx3) 
        self.t2_lr_cbx3.setHidden(True)
        
        # trend mean label
        self.t2_lr_txt29 = QLabel(self)
        self.t2_lr_txt29.move(740, 295)
        self.t2_lr_txt29.setFixedSize(20, 20)
        self.t2_lr_txt29.setText(' b') 
        self.t2_lr_txt29.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt29.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt29.setHidden(True)
        
        # trend mean edit
        self.t2_lr_edt18 = QLineEdit(self)
        self.t2_lr_edt18.move(770, 290)       
        self.t2_lr_edt18.resize(70, 25)                                           
        self.t2_lr_edt18.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt18.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt18.setText(self.user_inputs['tab_2_lr']['b_trend'])
        self.t2_lr_edt18.textChanged.connect(self.cb_t2_lr_edt18)
        self.t2_lr_edt18.setHidden(True)
        
        # trend variance label
        self.t2_lr_txt30 = QLabel(self)
        self.t2_lr_txt30.move(870, 295)
        self.t2_lr_txt30.setFixedSize(20, 20)
        self.t2_lr_txt30.setText(' V') 
        self.t2_lr_txt30.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt30.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt30.setHidden(True)
        
        # trend variance edit
        self.t2_lr_edt19 = QLineEdit(self)
        self.t2_lr_edt19.move(900, 290)       
        self.t2_lr_edt19.resize(70, 25)                                           
        self.t2_lr_edt19.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt19.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt19.setText(self.user_inputs['tab_2_lr']['V_trend'])
        self.t2_lr_edt19.textChanged.connect(self.cb_t2_lr_edt19)
        self.t2_lr_edt19.setHidden(True)
        
        # quadratic trend label
        self.t2_lr_txt31 = QLabel(self)
        self.t2_lr_txt31.move(520, 330)
        self.t2_lr_txt31.setFixedSize(200, 20)
        self.t2_lr_txt31.setText(' quadratic trend') 
        self.t2_lr_txt31.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt31.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt31.setHidden(True)
        
        # quadratic trend checkbox
        self.t2_lr_cbx4 = QCheckBox(self)
        self.t2_lr_cbx4.move(690, 325)  
        self.t2_lr_cbx4.setStyleSheet("QCheckBox::indicator:unchecked \
        {background-color : rgb(255, 255, 255); border: 0.5px solid rgb(0, 0, 0)}; \
        QCheckBox::indicator {width: 12px; height: 12px}") 
        self.t2_lr_cbx4.setChecked(self.user_inputs['tab_2_lr']['quadratic_trend'])
        self.t2_lr_cbx4.stateChanged.connect(self.cb_t2_lr_cbx4) 
        self.t2_lr_cbx4.setHidden(True)
        
        # quadratic trend mean label
        self.t2_lr_txt32 = QLabel(self)
        self.t2_lr_txt32.move(740, 330)
        self.t2_lr_txt32.setFixedSize(20, 20)
        self.t2_lr_txt32.setText(' b') 
        self.t2_lr_txt32.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt32.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt32.setHidden(True)
        
        # quadratic trend mean edit
        self.t2_lr_edt20 = QLineEdit(self)
        self.t2_lr_edt20.move(770, 325)       
        self.t2_lr_edt20.resize(70, 25)                                           
        self.t2_lr_edt20.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt20.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt20.setText(self.user_inputs['tab_2_lr']['b_quadratic_trend'])
        self.t2_lr_edt20.textChanged.connect(self.cb_t2_lr_edt20)
        self.t2_lr_edt20.setHidden(True)
        
        # quadratic trend variance label
        self.t2_lr_txt33 = QLabel(self)
        self.t2_lr_txt33.move(870, 330)
        self.t2_lr_txt33.setFixedSize(20, 20)
        self.t2_lr_txt33.setText(' V') 
        self.t2_lr_txt33.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt33.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt33.setHidden(True)
        
        # quadratic trend variance edit
        self.t2_lr_edt21 = QLineEdit(self)
        self.t2_lr_edt21.move(900, 325)       
        self.t2_lr_edt21.resize(70, 25)                                           
        self.t2_lr_edt21.setAlignment(Qt.AlignCenter)     
        self.t2_lr_edt21.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_lr_edt21.setText(self.user_inputs['tab_2_lr']['V_quadratic_trend'])
        self.t2_lr_edt21.textChanged.connect(self.cb_t2_lr_edt21)
        self.t2_lr_edt21.setHidden(True)
        
        # option label
        self.t2_lr_txt34 = QLabel(self)
        self.t2_lr_txt34.move(520, 380)
        self.t2_lr_txt34.setFixedSize(300, 30)
        self.t2_lr_txt34.setText(' Options') 
        self.t2_lr_txt34.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt34.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_lr_txt34.setFont(font)
        self.t2_lr_txt34.setHidden(True)
        
        # frame around option
        self.t2_lr_frm5 = QFrame(self)   
        self.t2_lr_frm5.setGeometry(510, 410, 470, 220)  
        self.t2_lr_frm5.setFrameShape(QFrame.Panel)
        self.t2_lr_frm5.setLineWidth(1)  
        self.t2_lr_frm5.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t2_lr_frm5.setHidden(True)
        
        # fit label
        self.t2_lr_txt35 = QLabel(self)
        self.t2_lr_txt35.move(520, 430)
        self.t2_lr_txt35.setFixedSize(300, 20)
        self.t2_lr_txt35.setText(' in-sample fit') 
        self.t2_lr_txt35.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt35.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt35.setHidden(True)
        
        # fit checkbox
        self.t2_lr_cbx5 = QCheckBox(self)
        self.t2_lr_cbx5.setGeometry(925, 425, 20, 20) 
        self.t2_lr_cbx5.setStyleSheet("QCheckBox::indicator:unchecked \
        {background-color : rgb(255, 255, 255); border: 0.5px solid rgb(0, 0, 0)}; \
        QCheckBox::indicator {width: 5px; height: 12px}") 
        self.t2_lr_cbx5.setChecked(self.user_inputs['tab_2_lr']['insample_fit'])
        self.t2_lr_cbx5.stateChanged.connect(self.cb_t2_lr_cbx5) 
        self.t2_lr_cbx5.setHidden(True)
        
        # marginal likelihood label
        self.t2_lr_txt36 = QLabel(self)
        self.t2_lr_txt36.move(520, 470)
        self.t2_lr_txt36.setFixedSize(300, 22)
        self.t2_lr_txt36.setText(' marginal likelihood') 
        self.t2_lr_txt36.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt36.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt36.setHidden(True)
        
        # marginal likelihood checkbox
        self.t2_lr_cbx6 = QCheckBox(self)
        self.t2_lr_cbx6.setGeometry(925, 465, 20, 20) 
        self.t2_lr_cbx6.setStyleSheet("QCheckBox::indicator:unchecked \
        {background-color : rgb(255, 255, 255); border: 0.5px solid rgb(0, 0, 0)}; \
        QCheckBox::indicator {width: 5px; height: 12px}") 
        self.t2_lr_cbx6.setChecked(self.user_inputs['tab_2_lr']['marginal_likelihood'])
        self.t2_lr_cbx6.stateChanged.connect(self.cb_t2_lr_cbx6) 
        self.t2_lr_cbx6.setHidden(True)
        
        # optimization label
        self.t2_lr_txt37 = QLabel(self)
        self.t2_lr_txt37.move(520, 510)
        self.t2_lr_txt37.setFixedSize(300, 22)
        self.t2_lr_txt37.setText(' hyperparameter optimization') 
        self.t2_lr_txt37.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt37.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt37.setHidden(True)
        
        # optimization checkbox
        self.t2_lr_cbx7 = QCheckBox(self)
        self.t2_lr_cbx7.setGeometry(925, 505, 20, 20) 
        self.t2_lr_cbx7.setStyleSheet("QCheckBox::indicator:unchecked \
        {background-color : rgb(255, 255, 255); border: 0.5px solid rgb(0, 0, 0)}; \
        QCheckBox::indicator {width: 5px; height: 12px}") 
        self.t2_lr_cbx7.setChecked(self.user_inputs['tab_2_lr']['hyperparameter_optimization'])
        self.t2_lr_cbx7.stateChanged.connect(self.cb_t2_lr_cbx7) 
        self.t2_lr_cbx7.setHidden(True)
        
        # optimization type label
        self.t2_lr_txt38 = QLabel(self)
        self.t2_lr_txt38.move(520, 550)
        self.t2_lr_txt38.setFixedSize(300, 22)
        self.t2_lr_txt38.setText(' optimization type') 
        self.t2_lr_txt38.setAlignment(Qt.AlignLeft)
        self.t2_lr_txt38.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_lr_txt38.setHidden(True)
        
        # optimization radiobuttons
        self.t2_lr_rdb7 = QRadioButton(' simple', self)  
        self.t2_lr_rdb7.setGeometry(790, 550, 80, 20)
        self.t2_lr_rdb7.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')  
        self.t2_lr_rdb7.toggled.connect(self.cb_t2_lr_bgr2)
        self.t2_lr_rdb7.setHidden(True)       
        self.t2_lr_rdb8 = QRadioButton(' full', self)   
        self.t2_lr_rdb8.setGeometry(890, 550, 80, 20) 
        self.t2_lr_rdb8.setStyleSheet('font-size: 12pt; font-family: Serif; \
                                  QRadioButton::indicator {width: 15px; height: 15px}')  
        self.t2_lr_rdb8.toggled.connect(self.cb_t2_lr_bgr2) 
        self.t2_lr_rdb8.setHidden(True)
        if self.user_inputs['tab_2_lr']['optimization_type'] == 1:
            self.t2_lr_rdb7.setChecked(True) 
        elif self.user_inputs['tab_2_lr']['optimization_type'] == 2:
            self.t2_lr_rdb8.setChecked(True) 
        self.t2_lr_bgr2 = QButtonGroup(self)  
        self.t2_lr_bgr2.addButton(self.t2_lr_rdb7) 
        self.t2_lr_bgr2.addButton(self.t2_lr_rdb8) 
        
        # indicate that tab 2 for linear regression is now created
        self.created_tab_2_lr = True
    
    
    def hide_tab_2_lr(self):
        
        # hide all controls
        self.t2_lr_txt1.setHidden(True)
        self.t2_lr_txt2.setHidden(True)
        self.t2_lr_txt3.setHidden(True)
        self.t2_lr_txt4.setHidden(True)
        self.t2_lr_txt5.setHidden(True)
        self.t2_lr_txt6.setHidden(True)
        self.t2_lr_txt7.setHidden(True)
        self.t2_lr_txt8.setHidden(True)
        self.t2_lr_txt9.setHidden(True)
        self.t2_lr_txt10.setHidden(True)
        self.t2_lr_txt11.setHidden(True)
        self.t2_lr_txt12.setHidden(True)
        self.t2_lr_txt13.setHidden(True)
        self.t2_lr_txt14.setHidden(True)
        self.t2_lr_txt15.setHidden(True)
        self.t2_lr_txt16.setHidden(True)
        self.t2_lr_txt17.setHidden(True)
        self.t2_lr_txt18.setHidden(True)
        self.t2_lr_txt19.setHidden(True)
        self.t2_lr_txt20.setHidden(True)
        self.t2_lr_txt21.setHidden(True)
        self.t2_lr_txt22.setHidden(True)
        self.t2_lr_txt23.setHidden(True)
        self.t2_lr_txt24.setHidden(True)
        self.t2_lr_txt25.setHidden(True)
        self.t2_lr_txt26.setHidden(True)
        self.t2_lr_txt27.setHidden(True)
        self.t2_lr_txt28.setHidden(True)
        self.t2_lr_txt29.setHidden(True)
        self.t2_lr_txt30.setHidden(True)
        self.t2_lr_txt31.setHidden(True)
        self.t2_lr_txt32.setHidden(True)
        self.t2_lr_txt33.setHidden(True)
        self.t2_lr_txt34.setHidden(True)
        self.t2_lr_txt35.setHidden(True)
        self.t2_lr_txt36.setHidden(True)
        self.t2_lr_txt37.setHidden(True)
        self.t2_lr_txt38.setHidden(True)
        self.t2_lr_frm1.setHidden(True)
        self.t2_lr_frm2.setHidden(True)
        self.t2_lr_frm3.setHidden(True)
        self.t2_lr_frm4.setHidden(True)
        self.t2_lr_frm5.setHidden(True)
        self.t2_lr_rdb1.setHidden(True)
        self.t2_lr_rdb2.setHidden(True)
        self.t2_lr_rdb3.setHidden(True)
        self.t2_lr_rdb4.setHidden(True)
        self.t2_lr_rdb5.setHidden(True)
        self.t2_lr_rdb6.setHidden(True)
        self.t2_lr_rdb7.setHidden(True)
        self.t2_lr_rdb8.setHidden(True)
        self.t2_lr_edt1.setHidden(True)
        self.t2_lr_edt2.setHidden(True)
        self.t2_lr_edt3.setHidden(True)
        self.t2_lr_edt4.setHidden(True)
        self.t2_lr_edt5.setHidden(True)
        self.t2_lr_edt6.setHidden(True)
        self.t2_lr_edt7.setHidden(True)
        self.t2_lr_edt8.setHidden(True)
        self.t2_lr_edt9.setHidden(True)
        self.t2_lr_edt10.setHidden(True)
        self.t2_lr_edt11.setHidden(True)
        self.t2_lr_edt12.setHidden(True)
        self.t2_lr_edt13.setHidden(True)
        self.t2_lr_edt14.setHidden(True)
        self.t2_lr_edt15.setHidden(True)
        self.t2_lr_edt16.setHidden(True)
        self.t2_lr_edt17.setHidden(True)
        self.t2_lr_edt18.setHidden(True)
        self.t2_lr_edt19.setHidden(True)
        self.t2_lr_edt20.setHidden(True)
        self.t2_lr_edt21.setHidden(True)
        self.t2_lr_cbx1.setHidden(True)
        self.t2_lr_cbx2.setHidden(True)
        self.t2_lr_cbx3.setHidden(True)
        self.t2_lr_cbx4.setHidden(True)
        self.t2_lr_cbx5.setHidden(True)
        self.t2_lr_cbx6.setHidden(True)
        self.t2_lr_cbx7.setHidden(True)
        
        # update tab color
        self.tab_pbt2.setStyleSheet("background:rgb" + str(self.backtabs_color) + ";")       
        
        
    def show_tab_2_lr(self): 
    
        # show all controls
        self.t2_lr_txt1.setVisible(True)
        self.t2_lr_txt2.setVisible(True)
        self.t2_lr_txt3.setVisible(True)
        self.t2_lr_txt4.setVisible(True)
        self.t2_lr_txt5.setVisible(True)
        self.t2_lr_txt6.setVisible(True)
        self.t2_lr_txt7.setVisible(True)
        self.t2_lr_txt8.setVisible(True)
        self.t2_lr_txt9.setVisible(True)
        self.t2_lr_txt10.setVisible(True)
        self.t2_lr_txt11.setVisible(True)
        self.t2_lr_txt12.setVisible(True)
        self.t2_lr_txt13.setVisible(True)
        self.t2_lr_txt14.setVisible(True)
        self.t2_lr_txt15.setVisible(True)
        self.t2_lr_txt16.setVisible(True)
        self.t2_lr_txt17.setVisible(True)
        self.t2_lr_txt18.setVisible(True)
        self.t2_lr_txt19.setVisible(True)
        self.t2_lr_txt20.setVisible(True)
        self.t2_lr_txt21.setVisible(True)
        self.t2_lr_txt22.setVisible(True)
        self.t2_lr_txt23.setVisible(True)
        self.t2_lr_txt24.setVisible(True)
        self.t2_lr_txt25.setVisible(True)
        self.t2_lr_txt26.setVisible(True)
        self.t2_lr_txt27.setVisible(True)
        self.t2_lr_txt28.setVisible(True)
        self.t2_lr_txt29.setVisible(True)
        self.t2_lr_txt30.setVisible(True)
        self.t2_lr_txt31.setVisible(True)
        self.t2_lr_txt32.setVisible(True)
        self.t2_lr_txt33.setVisible(True)
        self.t2_lr_txt34.setVisible(True)
        self.t2_lr_txt35.setVisible(True)
        self.t2_lr_txt36.setVisible(True)
        self.t2_lr_txt37.setVisible(True)
        self.t2_lr_txt38.setVisible(True)
        self.t2_lr_frm1.setVisible(True)
        self.t2_lr_frm2.setVisible(True)
        self.t2_lr_frm3.setVisible(True)
        self.t2_lr_frm4.setVisible(True)
        self.t2_lr_frm5.setVisible(True)
        self.t2_lr_rdb1.setVisible(True)
        self.t2_lr_rdb2.setVisible(True)
        self.t2_lr_rdb3.setVisible(True)
        self.t2_lr_rdb4.setVisible(True)
        self.t2_lr_rdb5.setVisible(True)
        self.t2_lr_rdb6.setVisible(True)
        self.t2_lr_rdb7.setVisible(True)
        self.t2_lr_rdb8.setVisible(True)
        self.t2_lr_edt1.setVisible(True)
        self.t2_lr_edt2.setVisible(True)
        self.t2_lr_edt3.setVisible(True)
        self.t2_lr_edt4.setVisible(True)
        self.t2_lr_edt5.setVisible(True)
        self.t2_lr_edt6.setVisible(True)
        self.t2_lr_edt7.setVisible(True)
        self.t2_lr_edt8.setVisible(True)
        self.t2_lr_edt9.setVisible(True)
        self.t2_lr_edt10.setVisible(True)
        self.t2_lr_edt11.setVisible(True)
        self.t2_lr_edt12.setVisible(True)
        self.t2_lr_edt13.setVisible(True)
        self.t2_lr_edt14.setVisible(True)
        self.t2_lr_edt15.setVisible(True)
        self.t2_lr_edt16.setVisible(True)
        self.t2_lr_edt17.setVisible(True)
        self.t2_lr_edt18.setVisible(True)
        self.t2_lr_edt19.setVisible(True)
        self.t2_lr_edt20.setVisible(True)
        self.t2_lr_edt21.setVisible(True)
        self.t2_lr_cbx1.setVisible(True)
        self.t2_lr_cbx2.setVisible(True)
        self.t2_lr_cbx3.setVisible(True)
        self.t2_lr_cbx4.setVisible(True)
        self.t2_lr_cbx5.setVisible(True)
        self.t2_lr_cbx6.setVisible(True)
        self.t2_lr_cbx7.setVisible(True)    
            
        
    def cb_t2_lr_bgr1(self):
        if self.t2_lr_rdb1.isChecked() == True:
            self.user_inputs['tab_2_lr']['regression_type'] = 1
        elif self.t2_lr_rdb2.isChecked() == True:
            self.user_inputs['tab_2_lr']['regression_type'] = 2
        elif self.t2_lr_rdb3.isChecked() == True:
            self.user_inputs['tab_2_lr']['regression_type'] = 3       
        elif self.t2_lr_rdb4.isChecked() == True:
            self.user_inputs['tab_2_lr']['regression_type'] = 4        
        elif self.t2_lr_rdb5.isChecked() == True:
            self.user_inputs['tab_2_lr']['regression_type'] = 5        
        elif self.t2_lr_rdb6.isChecked() == True:
            self.user_inputs['tab_2_lr']['regression_type'] = 6        
        
        
    def cb_t2_lr_edt1(self):
        self.user_inputs['tab_2_lr']['iterations'] = self.t2_lr_edt1.text()         
        
      
    def cb_t2_lr_edt2(self):
        self.user_inputs['tab_2_lr']['burnin'] = self.t2_lr_edt2.text() 


    def cb_t2_lr_edt3(self):
        self.user_inputs['tab_2_lr']['model_credibility'] = self.t2_lr_edt3.text() 


    def cb_t2_lr_edt4(self):
        self.user_inputs['tab_2_lr']['b'] = self.t2_lr_edt4.text() 


    def cb_t2_lr_edt5(self):
        self.user_inputs['tab_2_lr']['V'] = self.t2_lr_edt5.text() 


    def cb_t2_lr_edt6(self):
        self.user_inputs['tab_2_lr']['alpha'] = self.t2_lr_edt6.text()


    def cb_t2_lr_edt7(self):
        self.user_inputs['tab_2_lr']['delta'] = self.t2_lr_edt7.text()


    def cb_t2_lr_edt8(self):
        self.user_inputs['tab_2_lr']['g'] = self.t2_lr_edt8.text()


    def cb_t2_lr_edt9(self):
        self.user_inputs['tab_2_lr']['Q'] = self.t2_lr_edt9.text()        
        
        
    def cb_t2_lr_edt10(self):
        self.user_inputs['tab_2_lr']['tau'] = self.t2_lr_edt10.text()             
        
        
    def cb_t2_lr_cbx1(self, state):     
        if (state == Qt.Checked):
            self.user_inputs['tab_2_lr']['thinning'] = True 
        else:
            self.user_inputs['tab_2_lr']['thinning'] = False        
        
        
    def cb_t2_lr_edt11(self):
        self.user_inputs['tab_2_lr']['thinning_frequency'] = self.t2_lr_edt11.text()              
        
        
    def cb_t2_lr_edt12(self):
        self.user_inputs['tab_2_lr']['Z_variables'] = self.t2_lr_edt12.text()                   
   
        
    def cb_t2_lr_edt13(self):
        self.user_inputs['tab_2_lr']['q'] = self.t2_lr_edt13.text()     
    
   
    def cb_t2_lr_edt14(self):
        self.user_inputs['tab_2_lr']['p'] = self.t2_lr_edt14.text()     
   
    
    def cb_t2_lr_edt15(self):
        self.user_inputs['tab_2_lr']['H'] = self.t2_lr_edt15.text()    
    
   
    def cb_t2_lr_cbx2(self, state):     
        if (state == Qt.Checked):
            self.user_inputs['tab_2_lr']['constant'] = True 
        else:
            self.user_inputs['tab_2_lr']['constant'] = False      
   
    
    def cb_t2_lr_edt16(self):
        self.user_inputs['tab_2_lr']['b_constant'] = self.t2_lr_edt16.text()    
    
   
    def cb_t2_lr_edt17(self):
        self.user_inputs['tab_2_lr']['V_constant'] = self.t2_lr_edt17.text()     
   
    
    def cb_t2_lr_cbx3(self, state):     
        if (state == Qt.Checked):
            self.user_inputs['tab_2_lr']['trend'] = True 
        else:
            self.user_inputs['tab_2_lr']['trend'] = False    
    
    
    def cb_t2_lr_edt18(self):
        self.user_inputs['tab_2_lr']['b_trend'] = self.t2_lr_edt18.text()    
    
   
    def cb_t2_lr_edt19(self):
        self.user_inputs['tab_2_lr']['V_trend'] = self.t2_lr_edt19.text()    
    
   
    def cb_t2_lr_cbx4(self, state):     
        if (state == Qt.Checked):
            self.user_inputs['tab_2_lr']['quadratic_trend'] = True 
        else:
            self.user_inputs['tab_2_lr']['quadratic_trend'] = False    
    
    
    def cb_t2_lr_edt20(self):
        self.user_inputs['tab_2_lr']['b_quadratic_trend'] = self.t2_lr_edt20.text()    
    
   
    def cb_t2_lr_edt21(self):
        self.user_inputs['tab_2_lr']['V_quadratic_trend'] = self.t2_lr_edt21.text()      
   
    
    def cb_t2_lr_cbx5(self, state):     
        if (state == Qt.Checked):
            self.user_inputs['tab_2_lr']['insample_fit'] = True 
        else:
            self.user_inputs['tab_2_lr']['insample_fit'] = False
            
            
    def cb_t2_lr_cbx6(self, state):     
        if (state == Qt.Checked):
            self.user_inputs['tab_2_lr']['marginal_likelihood'] = True 
        else:
            self.user_inputs['tab_2_lr']['marginal_likelihood'] = False            
            
            
    def cb_t2_lr_cbx7(self, state):     
        if (state == Qt.Checked):
            self.user_inputs['tab_2_lr']['hyperparameter_optimization'] = True 
        else:
            self.user_inputs['tab_2_lr']['hyperparameter_optimization'] = False              
            
            
    def cb_t2_lr_bgr2(self):
        if self.t2_lr_rdb7.isChecked() == True:
            self.user_inputs['tab_2_lr']['optimization_type'] = 1
        elif self.t2_lr_rdb8.isChecked() == True:
            self.user_inputs['tab_2_lr']['optimization_type'] = 2         
    
    
    