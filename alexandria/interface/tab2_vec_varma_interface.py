# imports
from PyQt5.QtWidgets import QLabel, QFrame, QRadioButton, QButtonGroup, QLineEdit, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont



class Tab2VecVarmaInterface(object):


    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass


    def create_tab_2_ext(self):
    
        # model label
        self.t2_ext_txt1 = QLabel(self)
        self.t2_ext_txt1.move(30, 60)
        self.t2_ext_txt1.setFixedSize(300, 30)
        self.t2_ext_txt1.setText(' Model') 
        self.t2_ext_txt1.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt1.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_ext_txt1.setFont(font)
        self.t2_ext_txt1.setHidden(True)
        
        # frame around model
        self.t2_ext_frm1 = QFrame(self)   
        self.t2_ext_frm1.setGeometry(20, 90, 470, 110)  
        self.t2_ext_frm1.setFrameShape(QFrame.Panel)
        self.t2_ext_frm1.setLineWidth(1)  
        self.t2_ext_frm1.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t2_ext_frm1.setHidden(True)
        
        # model radiobuttons
        self.t2_ext_rdb1 = QRadioButton(' VEC', self)
        self.t2_ext_rdb1.setGeometry(30, 93, 400, 30)
        self.t2_ext_rdb1.setStyleSheet("font-size: 12pt; font-family: Serif;")  
        self.t2_ext_rdb1.toggled.connect(self.cb_t2_ext_bgr1)
        self.t2_ext_rdb1.setHidden(True)
        self.t2_ext_rdb2 = QRadioButton(' VARMA', self)
        self.t2_ext_rdb2.setGeometry(30, 143, 400, 30)
        self.t2_ext_rdb2.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_ext_rdb2.toggled.connect(self.cb_t2_ext_bgr1)
        self.t2_ext_rdb2.setHidden(True)
        if self.user_inputs['tab_2_ext']['model'] == 1:
            self.t2_ext_rdb1.setChecked(True) 
        elif self.user_inputs['tab_2_ext']['model'] == 2:
            self.t2_ext_rdb2.setChecked(True) 
        self.t2_ext_bgr1 = QButtonGroup(self)  
        self.t2_ext_bgr1.addButton(self.t2_ext_rdb1) 
        self.t2_ext_bgr1.addButton(self.t2_ext_rdb2)     
 
        # VEC label
        self.t2_ext_txt2 = QLabel(self)
        self.t2_ext_txt2.move(50, 120)
        self.t2_ext_txt2.setFixedSize(400, 25)
        self.t2_ext_txt2.setText(' (Bayesian Vector Error Correction)') 
        self.t2_ext_txt2.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt2.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt2.setHidden(True)

        # VARMA label
        self.t2_ext_txt3 = QLabel(self)
        self.t2_ext_txt3.move(50, 170)
        self.t2_ext_txt3.setFixedSize(400, 25)
        self.t2_ext_txt3.setText(' (Bayesian Vector Autoregressive Moving Average)') 
        self.t2_ext_txt3.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt3.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt3.setHidden(True)


        # estimation label
        self.t2_ext_txt4 = QLabel(self)
        self.t2_ext_txt4.move(520, 60)
        self.t2_ext_txt4.setFixedSize(300, 30)
        self.t2_ext_txt4.setText(' Estimation') 
        self.t2_ext_txt4.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt4.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_ext_txt4.setFont(font)
        self.t2_ext_txt4.setHidden(True)

        # frame around estimation
        self.t2_ext_frm2 = QFrame(self)   
        self.t2_ext_frm2.setGeometry(510, 90, 470, 110)  
        self.t2_ext_frm2.setFrameShape(QFrame.Panel)
        self.t2_ext_frm2.setLineWidth(1)  
        self.t2_ext_frm2.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t2_ext_frm2.setHidden(True) 

        # Gibbs sampling label
        self.t2_ext_txt5 = QLabel(self)
        self.t2_ext_txt5.move(520, 95)
        self.t2_ext_txt5.setFixedSize(200, 30)
        self.t2_ext_txt5.setText(' Gibbs sampling') 
        self.t2_ext_txt5.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt5.setStyleSheet('font-size: 14pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_ext_txt5.setFont(font)
        self.t2_ext_txt5.setHidden(True)

        # iteration label
        self.t2_ext_txt6 = QLabel(self)
        self.t2_ext_txt6.move(520, 125)
        self.t2_ext_txt6.setFixedSize(200, 25)
        self.t2_ext_txt6.setText(' iterations') 
        self.t2_ext_txt6.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt6.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt6.setHidden(True)

        # iteration edit
        self.t2_ext_edt1 = QLineEdit(self)
        self.t2_ext_edt1.move(670, 122)       
        self.t2_ext_edt1.resize(70, 23)                                           
        self.t2_ext_edt1.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt1.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt1.setText(self.user_inputs['tab_2_ext']['iterations'])
        self.t2_ext_edt1.textChanged.connect(self.cb_t2_ext_edt1)
        self.t2_ext_edt1.setHidden(True)

        # burn-in label
        self.t2_ext_txt7 = QLabel(self)
        self.t2_ext_txt7.move(520, 150)
        self.t2_ext_txt7.setFixedSize(200, 25)
        self.t2_ext_txt7.setText(' burn-in') 
        self.t2_ext_txt7.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt7.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt7.setHidden(True)

        # burn-in edit
        self.t2_ext_edt2 = QLineEdit(self)
        self.t2_ext_edt2.move(670, 147)       
        self.t2_ext_edt2.resize(70, 23)                                           
        self.t2_ext_edt2.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt2.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt2.setText(self.user_inputs['tab_2_ext']['burnin'])
        self.t2_ext_edt2.textChanged.connect(self.cb_t2_ext_edt2)
        self.t2_ext_edt2.setHidden(True)
        
        # credibility label
        self.t2_ext_txt8 = QLabel(self)
        self.t2_ext_txt8.move(520, 175)
        self.t2_ext_txt8.setFixedSize(200, 20)
        self.t2_ext_txt8.setText(' credibility level') 
        self.t2_ext_txt8.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt8.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt8.setHidden(True)        

        # credibility edit
        self.t2_ext_edt3 = QLineEdit(self)
        self.t2_ext_edt3.move(670, 172)       
        self.t2_ext_edt3.resize(70, 23)                                           
        self.t2_ext_edt3.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt3.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt3.setText(self.user_inputs['tab_2_ext']['model_credibility'])
        self.t2_ext_edt3.textChanged.connect(self.cb_t2_ext_edt3)
        self.t2_ext_edt3.setHidden(True)

        # Exogenous label
        self.t2_ext_txt9 = QLabel(self)
        self.t2_ext_txt9.move(770, 95)
        self.t2_ext_txt9.setFixedSize(200, 30)
        self.t2_ext_txt9.setText(' Exogenous') 
        self.t2_ext_txt9.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt9.setStyleSheet('font-size: 14pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_ext_txt9.setFont(font)
        self.t2_ext_txt9.setHidden(True)

        # constant label
        self.t2_ext_txt10 = QLabel(self)
        self.t2_ext_txt10.move(770, 125)
        self.t2_ext_txt10.setFixedSize(200, 25)
        self.t2_ext_txt10.setText(' constant') 
        self.t2_ext_txt10.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt10.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt10.setHidden(True)

        # constant checkbox
        self.t2_ext_cbx1 = QCheckBox(self)
        self.t2_ext_cbx1.setGeometry(950, 125, 20, 20)  
        self.t2_ext_cbx1.setStyleSheet("QCheckBox::indicator:unchecked \
        {background-color : rgb(255, 255, 255); border: 0.5px solid rgb(0, 0, 0)}; \
        QCheckBox::indicator {width: 12px; height: 12px}") 
        self.t2_ext_cbx1.setChecked(self.user_inputs['tab_2_ext']['constant'])
        self.t2_ext_cbx1.stateChanged.connect(self.cb_t2_ext_cbx1) 
        self.t2_ext_cbx1.setHidden(True)

        # linear trend label
        self.t2_ext_txt11 = QLabel(self)
        self.t2_ext_txt11.move(770, 150)
        self.t2_ext_txt11.setFixedSize(200, 25)
        self.t2_ext_txt11.setText(' linear trend') 
        self.t2_ext_txt11.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt11.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt11.setHidden(True)

        # linear trend checkbox
        self.t2_ext_cbx2 = QCheckBox(self)
        self.t2_ext_cbx2.setGeometry(950, 150, 20, 20) 
        self.t2_ext_cbx2.setStyleSheet("QCheckBox::indicator:unchecked \
        {background-color : rgb(255, 255, 255); border: 0.5px solid rgb(0, 0, 0)}; \
        QCheckBox::indicator {width: 12px; height: 12px}") 
        self.t2_ext_cbx2.setChecked(self.user_inputs['tab_2_ext']['trend'])
        self.t2_ext_cbx2.stateChanged.connect(self.cb_t2_ext_cbx2) 
        self.t2_ext_cbx2.setHidden(True)

        # quadratic trend
        self.t2_ext_txt12 = QLabel(self)
        self.t2_ext_txt12.move(770, 175)
        self.t2_ext_txt12.setFixedSize(200, 20)
        self.t2_ext_txt12.setText(' quadratic trend') 
        self.t2_ext_txt12.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt12.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt12.setHidden(True)
        
        # quadratic trend checkbox
        self.t2_ext_cbx3 = QCheckBox(self)
        self.t2_ext_cbx3.setGeometry(950, 175, 20, 20) 
        self.t2_ext_cbx3.setStyleSheet("QCheckBox::indicator:unchecked \
        {background-color : rgb(255, 255, 255); border: 0.5px solid rgb(0, 0, 0)}; \
        QCheckBox::indicator {width: 12px; height: 12px}") 
        self.t2_ext_cbx3.setChecked(self.user_inputs['tab_2_ext']['quadratic_trend'])
        self.t2_ext_cbx3.stateChanged.connect(self.cb_t2_ext_cbx3) 
        self.t2_ext_cbx3.setHidden(True)

        # vec label
        self.t2_ext_txt13 = QLabel(self)
        self.t2_ext_txt13.move(30, 220)
        self.t2_ext_txt13.setFixedSize(300, 30)
        self.t2_ext_txt13.setText(' VEC') 
        self.t2_ext_txt13.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt13.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_ext_txt13.setFont(font)
        self.t2_ext_txt13.setHidden(True)
    
        # frame around vec
        self.t2_ext_frm3 = QFrame(self)   
        self.t2_ext_frm3.setGeometry(20, 250, 470, 380)  
        self.t2_ext_frm3.setFrameShape(QFrame.Panel)
        self.t2_ext_frm3.setLineWidth(1)  
        self.t2_ext_frm3.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t2_ext_frm3.setHidden(True)

        # specification label
        self.t2_ext_txt14 = QLabel(self)
        self.t2_ext_txt14.move(30, 255)
        self.t2_ext_txt14.setFixedSize(300, 25)
        self.t2_ext_txt14.setText(' Autoregressive specification') 
        self.t2_ext_txt14.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt14.setStyleSheet('font-size: 14pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_ext_txt14.setFont(font)
        self.t2_ext_txt14.setHidden(True)    
        
        # lag label
        self.t2_ext_txt15 = QLabel(self)
        self.t2_ext_txt15.move(30, 285)
        self.t2_ext_txt15.setFixedSize(300, 25)
        self.t2_ext_txt15.setText(' p:    lags') 
        self.t2_ext_txt15.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt15.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt15.setHidden(True)   

        # lag edit
        self.t2_ext_edt4 = QLineEdit(self)
        self.t2_ext_edt4.move(330, 283)       
        self.t2_ext_edt4.resize(140, 22)                                           
        self.t2_ext_edt4.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt4.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt4.setText(self.user_inputs['tab_2_ext']['vec_lags'])
        self.t2_ext_edt4.textChanged.connect(self.cb_t2_ext_edt4)
        self.t2_ext_edt4.setHidden(True)

        # pi1 label
        self.t2_ext_txt16 = QLabel(self)
        self.t2_ext_txt16.move(30, 335)
        self.t2_ext_txt16.setFixedSize(300, 25)
        self.t2_ext_txt16.setText(' π₁:  overall tightness') 
        self.t2_ext_txt16.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt16.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt16.setHidden(True) 
        
        # pi1 edit
        self.t2_ext_edt5 = QLineEdit(self)
        self.t2_ext_edt5.move(330, 333)       
        self.t2_ext_edt5.resize(140, 22)                                           
        self.t2_ext_edt5.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt5.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt5.setText(self.user_inputs['tab_2_ext']['vec_pi1'])
        self.t2_ext_edt5.textChanged.connect(self.cb_t2_ext_edt5)
        self.t2_ext_edt5.setHidden(True)           
        
        # pi2 label
        self.t2_ext_txt17 = QLabel(self)
        self.t2_ext_txt17.move(30, 360)
        self.t2_ext_txt17.setFixedSize(300, 25)
        self.t2_ext_txt17.setText(' π₂:  cross-variable shrinkage') 
        self.t2_ext_txt17.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt17.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt17.setHidden(True) 
        
        # pi2 edit
        self.t2_ext_edt6 = QLineEdit(self)
        self.t2_ext_edt6.move(330, 358)       
        self.t2_ext_edt6.resize(140, 22)                                           
        self.t2_ext_edt6.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt6.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt6.setText(self.user_inputs['tab_2_ext']['vec_pi2'])
        self.t2_ext_edt6.textChanged.connect(self.cb_t2_ext_edt6)
        self.t2_ext_edt6.setHidden(True) 

        # pi3 label
        self.t2_ext_txt18 = QLabel(self)
        self.t2_ext_txt18.move(30, 385)
        self.t2_ext_txt18.setFixedSize(300, 25)
        self.t2_ext_txt18.setText(' π₃:  lag decay') 
        self.t2_ext_txt18.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt18.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt18.setHidden(True) 

        # pi3 edit
        self.t2_ext_edt7 = QLineEdit(self)
        self.t2_ext_edt7.move(330, 383)       
        self.t2_ext_edt7.resize(140, 22)                                           
        self.t2_ext_edt7.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt7.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt7.setText(self.user_inputs['tab_2_ext']['vec_pi3'])
        self.t2_ext_edt7.textChanged.connect(self.cb_t2_ext_edt7)
        self.t2_ext_edt7.setHidden(True) 

        # pi4 label
        self.t2_ext_txt19 = QLabel(self)
        self.t2_ext_txt19.move(30, 410)
        self.t2_ext_txt19.setFixedSize(300, 25)
        self.t2_ext_txt19.setText(' π₄:  exogenous slackness') 
        self.t2_ext_txt19.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt19.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt19.setHidden(True) 

        # pi4 edit
        self.t2_ext_edt8 = QLineEdit(self)
        self.t2_ext_edt8.move(330, 408)       
        self.t2_ext_edt8.resize(140, 22)                                           
        self.t2_ext_edt8.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt8.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt8.setText(self.user_inputs['tab_2_ext']['vec_pi4'])
        self.t2_ext_edt8.textChanged.connect(self.cb_t2_ext_edt8)
        self.t2_ext_edt8.setHidden(True) 

        # error correction label
        self.t2_ext_txt20 = QLabel(self)
        self.t2_ext_txt20.move(30, 440)
        self.t2_ext_txt20.setFixedSize(300, 25)
        self.t2_ext_txt20.setText(' Error correction specification') 
        self.t2_ext_txt20.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt20.setStyleSheet('font-size: 14pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_ext_txt20.setFont(font)
        self.t2_ext_txt20.setHidden(True) 

        # prior label
        self.t2_ext_txt21 = QLabel(self)
        self.t2_ext_txt21.move(30, 470)
        self.t2_ext_txt21.setFixedSize(300, 25)
        self.t2_ext_txt21.setText(' prior:') 
        self.t2_ext_txt21.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt21.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt21.setHidden(True) 

        # prior radiobuttons
        self.t2_ext_rdb3 = QRadioButton(' uninformative', self)
        self.t2_ext_rdb3.setGeometry(330, 470, 150, 30)
        self.t2_ext_rdb3.setStyleSheet("font-size: 12pt; font-family: Serif;")  
        self.t2_ext_rdb3.toggled.connect(self.cb_t2_ext_bgr2)
        self.t2_ext_rdb3.setHidden(True)
        self.t2_ext_rdb4 = QRadioButton(' horseshoe', self)
        self.t2_ext_rdb4.setGeometry(330, 495, 150, 30)
        self.t2_ext_rdb4.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_ext_rdb4.toggled.connect(self.cb_t2_ext_bgr2)
        self.t2_ext_rdb4.setHidden(True)
        self.t2_ext_rdb5 = QRadioButton(' selection', self)
        self.t2_ext_rdb5.setGeometry(330, 520, 150, 30)
        self.t2_ext_rdb5.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_ext_rdb5.toggled.connect(self.cb_t2_ext_bgr2)
        self.t2_ext_rdb5.setHidden(True)
        if self.user_inputs['tab_2_ext']['prior_type'] == 1:
            self.t2_ext_rdb3.setChecked(True) 
        elif self.user_inputs['tab_2_ext']['prior_type'] == 2:
            self.t2_ext_rdb4.setChecked(True) 
        elif self.user_inputs['tab_2_ext']['prior_type'] == 3:
            self.t2_ext_rdb5.setChecked(True)             
        self.t2_ext_bgr2 = QButtonGroup(self)  
        self.t2_ext_bgr2.addButton(self.t2_ext_rdb3) 
        self.t2_ext_bgr2.addButton(self.t2_ext_rdb4)    
        self.t2_ext_bgr2.addButton(self.t2_ext_rdb5)  

        # correction type label
        self.t2_ext_txt22 = QLabel(self)
        self.t2_ext_txt22.move(30, 545)
        self.t2_ext_txt22.setFixedSize(300, 25)
        self.t2_ext_txt22.setText(' correction type:') 
        self.t2_ext_txt22.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt22.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt22.setHidden(True) 

        # error correction radiobuttons
        self.t2_ext_rdb6 = QRadioButton(' general', self)
        self.t2_ext_rdb6.setGeometry(330, 545, 150, 30)
        self.t2_ext_rdb6.setStyleSheet("font-size: 12pt; font-family: Serif;")  
        self.t2_ext_rdb6.toggled.connect(self.cb_t2_ext_bgr3)
        self.t2_ext_rdb6.setHidden(True)
        self.t2_ext_rdb7 = QRadioButton(' reduced-rank', self)
        self.t2_ext_rdb7.setGeometry(330, 570, 150, 30)
        self.t2_ext_rdb7.setStyleSheet("font-size: 12pt; font-family: Serif;") 
        self.t2_ext_rdb7.toggled.connect(self.cb_t2_ext_bgr3)
        self.t2_ext_rdb7.setHidden(True)
        if self.user_inputs['tab_2_ext']['error_correction_type'] == 1:
            self.t2_ext_rdb6.setChecked(True) 
        elif self.user_inputs['tab_2_ext']['error_correction_type'] == 2:
            self.t2_ext_rdb7.setChecked(True) 
        self.t2_ext_bgr3 = QButtonGroup(self)  
        self.t2_ext_bgr3.addButton(self.t2_ext_rdb6) 
        self.t2_ext_bgr3.addButton(self.t2_ext_rdb7)    

        # cointegration rank label
        self.t2_ext_txt23 = QLabel(self)
        self.t2_ext_txt23.move(30, 600)
        self.t2_ext_txt23.setFixedSize(200, 25)
        self.t2_ext_txt23.setText(' r: max cointegration rank') 
        self.t2_ext_txt23.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt23.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt23.setHidden(True)

        # cointegration rank edit
        self.t2_ext_edt9 = QLineEdit(self)
        self.t2_ext_edt9.move(330, 598)      
        self.t2_ext_edt9.resize(140, 22)                                           
        self.t2_ext_edt9.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt9.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt9.setText(self.user_inputs['tab_2_ext']['max_cointegration_rank'])
        self.t2_ext_edt9.textChanged.connect(self.cb_t2_ext_edt9)
        self.t2_ext_edt9.setHidden(True)

        # varma label
        self.t2_ext_txt24 = QLabel(self)
        self.t2_ext_txt24.move(520, 220)
        self.t2_ext_txt24.setFixedSize(300, 30)
        self.t2_ext_txt24.setText(' VARMA') 
        self.t2_ext_txt24.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt24.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_ext_txt24.setFont(font)
        self.t2_ext_txt24.setHidden(True)
    
        # frame around varma
        self.t2_ext_frm4 = QFrame(self)   
        self.t2_ext_frm4.setGeometry(510, 250, 470, 380)  
        self.t2_ext_frm4.setFrameShape(QFrame.Panel)
        self.t2_ext_frm4.setLineWidth(1)  
        self.t2_ext_frm4.setStyleSheet('border: 1px solid rgb(150, 150, 150)')
        self.t2_ext_frm4.setHidden(True)

        # specification label
        self.t2_ext_txt25 = QLabel(self)
        self.t2_ext_txt25.move(520, 255)
        self.t2_ext_txt25.setFixedSize(300, 25)
        self.t2_ext_txt25.setText(' Autoregressive specification') 
        self.t2_ext_txt25.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt25.setStyleSheet('font-size: 14pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_ext_txt25.setFont(font)
        self.t2_ext_txt25.setHidden(True)  

        # lag label
        self.t2_ext_txt26 = QLabel(self)
        self.t2_ext_txt26.move(520, 285)
        self.t2_ext_txt26.setFixedSize(300, 25)
        self.t2_ext_txt26.setText(' p:    lags') 
        self.t2_ext_txt26.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt26.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt26.setHidden(True)   

        # lag edit
        self.t2_ext_edt10 = QLineEdit(self)
        self.t2_ext_edt10.move(820, 283)       
        self.t2_ext_edt10.resize(140, 22)                                           
        self.t2_ext_edt10.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt10.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt10.setText(self.user_inputs['tab_2_ext']['varma_lags'])
        self.t2_ext_edt10.textChanged.connect(self.cb_t2_ext_edt10)
        self.t2_ext_edt10.setHidden(True)

        # AR coefficients label
        self.t2_ext_txt27 = QLabel(self)
        self.t2_ext_txt27.move(520, 310)
        self.t2_ext_txt27.setFixedSize(300, 25)
        self.t2_ext_txt27.setText(' δ:    AR coefficients') 
        self.t2_ext_txt27.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt27.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt27.setHidden(True)      
    
        # AR coefficients edit
        self.t2_ext_edt11 = QLineEdit(self)
        self.t2_ext_edt11.move(820, 308)       
        self.t2_ext_edt11.resize(140, 22)                                           
        self.t2_ext_edt11.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt11.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt11.setText(self.user_inputs['tab_2_ext']['ar_coefficients'])
        self.t2_ext_edt11.textChanged.connect(self.cb_t2_ext_edt11)
        self.t2_ext_edt11.setHidden(True)

        # pi1 label
        self.t2_ext_txt28 = QLabel(self)
        self.t2_ext_txt28.move(520, 335)
        self.t2_ext_txt28.setFixedSize(300, 25)
        self.t2_ext_txt28.setText(' π₁:  overall tightness') 
        self.t2_ext_txt28.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt28.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt28.setHidden(True) 
        
        # pi1 edit
        self.t2_ext_edt12 = QLineEdit(self)
        self.t2_ext_edt12.move(820, 333)       
        self.t2_ext_edt12.resize(140, 22)                                           
        self.t2_ext_edt12.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt12.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt12.setText(self.user_inputs['tab_2_ext']['varma_pi1'])
        self.t2_ext_edt12.textChanged.connect(self.cb_t2_ext_edt12)
        self.t2_ext_edt12.setHidden(True)           
        
        # pi2 label
        self.t2_ext_txt29 = QLabel(self)
        self.t2_ext_txt29.move(520, 360)
        self.t2_ext_txt29.setFixedSize(300, 25)
        self.t2_ext_txt29.setText(' π₂:  cross-variable shrinkage') 
        self.t2_ext_txt29.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt29.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt29.setHidden(True) 
        
        # pi2 edit
        self.t2_ext_edt13 = QLineEdit(self)
        self.t2_ext_edt13.move(820, 358)       
        self.t2_ext_edt13.resize(140, 22)                                           
        self.t2_ext_edt13.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt13.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt13.setText(self.user_inputs['tab_2_ext']['varma_pi2'])
        self.t2_ext_edt13.textChanged.connect(self.cb_t2_ext_edt13)
        self.t2_ext_edt13.setHidden(True) 

        # pi3 label
        self.t2_ext_txt30 = QLabel(self)
        self.t2_ext_txt30.move(520, 385)
        self.t2_ext_txt30.setFixedSize(300, 25)
        self.t2_ext_txt30.setText(' π₃:  lag decay') 
        self.t2_ext_txt30.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt30.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt30.setHidden(True) 

        # pi3 edit
        self.t2_ext_edt14 = QLineEdit(self)
        self.t2_ext_edt14.move(820, 383)       
        self.t2_ext_edt14.resize(140, 22)                                           
        self.t2_ext_edt14.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt14.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt14.setText(self.user_inputs['tab_2_ext']['varma_pi3'])
        self.t2_ext_edt14.textChanged.connect(self.cb_t2_ext_edt14)
        self.t2_ext_edt14.setHidden(True) 

        # pi4 label
        self.t2_ext_txt31 = QLabel(self)
        self.t2_ext_txt31.move(520, 410)
        self.t2_ext_txt31.setFixedSize(300, 25)
        self.t2_ext_txt31.setText(' π₄:  exogenous slackness') 
        self.t2_ext_txt31.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt31.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt31.setHidden(True) 

        # pi4 edit
        self.t2_ext_edt15 = QLineEdit(self)
        self.t2_ext_edt15.move(820, 408)       
        self.t2_ext_edt15.resize(140, 22)                                           
        self.t2_ext_edt15.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt15.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt15.setText(self.user_inputs['tab_2_ext']['varma_pi4'])
        self.t2_ext_edt15.textChanged.connect(self.cb_t2_ext_edt15)
        self.t2_ext_edt15.setHidden(True) 

        # moving average label
        self.t2_ext_txt32 = QLabel(self)
        self.t2_ext_txt32.move(520, 440)
        self.t2_ext_txt32.setFixedSize(300, 25)
        self.t2_ext_txt32.setText(' Moving average specification') 
        self.t2_ext_txt32.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt32.setStyleSheet('font-size: 14pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t2_ext_txt32.setFont(font)
        self.t2_ext_txt32.setHidden(True) 

        # residual lag label
        self.t2_ext_txt33 = QLabel(self)
        self.t2_ext_txt33.move(520, 470)
        self.t2_ext_txt33.setFixedSize(300, 25)
        self.t2_ext_txt33.setText(' q:    residual lags') 
        self.t2_ext_txt33.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt33.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt33.setHidden(True) 

        # residual lag edit
        self.t2_ext_edt16 = QLineEdit(self)
        self.t2_ext_edt16.move(820, 468)       
        self.t2_ext_edt16.resize(140, 22)                                           
        self.t2_ext_edt16.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt16.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt16.setText(self.user_inputs['tab_2_ext']['residual_lags'])
        self.t2_ext_edt16.textChanged.connect(self.cb_t2_ext_edt16)
        self.t2_ext_edt16.setHidden(True) 

        # lambda1 label
        self.t2_ext_txt34 = QLabel(self)
        self.t2_ext_txt34.move(520, 495)
        self.t2_ext_txt34.setFixedSize(300, 25)
        self.t2_ext_txt34.setText(' λ₁:  overall tightness') 
        self.t2_ext_txt34.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt34.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt34.setHidden(True) 

        # lambda1 edit
        self.t2_ext_edt17 = QLineEdit(self)
        self.t2_ext_edt17.move(820, 493)       
        self.t2_ext_edt17.resize(140, 22)                                           
        self.t2_ext_edt17.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt17.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt17.setText(self.user_inputs['tab_2_ext']['lambda1'])
        self.t2_ext_edt17.textChanged.connect(self.cb_t2_ext_edt17)
        self.t2_ext_edt17.setHidden(True) 

        # lambda2 label
        self.t2_ext_txt35 = QLabel(self)
        self.t2_ext_txt35.move(520, 520)
        self.t2_ext_txt35.setFixedSize(300, 25)
        self.t2_ext_txt35.setText(' λ₂:  cross-variable shrinkage') 
        self.t2_ext_txt35.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt35.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt35.setHidden(True) 

        # lambda2 edit
        self.t2_ext_edt18 = QLineEdit(self)
        self.t2_ext_edt18.move(820, 518)       
        self.t2_ext_edt18.resize(140, 22)                                           
        self.t2_ext_edt18.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt18.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt18.setText(self.user_inputs['tab_2_ext']['lambda2'])
        self.t2_ext_edt18.textChanged.connect(self.cb_t2_ext_edt18)
        self.t2_ext_edt18.setHidden(True) 

        # lambda3 label
        self.t2_ext_txt36 = QLabel(self)
        self.t2_ext_txt36.move(520, 545)
        self.t2_ext_txt36.setFixedSize(300, 25)
        self.t2_ext_txt36.setText(' λ₃:  lag decay') 
        self.t2_ext_txt36.setAlignment(Qt.AlignLeft)
        self.t2_ext_txt36.setStyleSheet('font-size: 12pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        self.t2_ext_txt36.setHidden(True) 

        # lambda3 edit
        self.t2_ext_edt19 = QLineEdit(self)
        self.t2_ext_edt19.move(820, 543)       
        self.t2_ext_edt19.resize(140, 22)                                           
        self.t2_ext_edt19.setAlignment(Qt.AlignCenter)     
        self.t2_ext_edt19.setStyleSheet('background-color: rgb(255, 255, 255); \
                                    border: 0.5px solid rgb(130, 130, 130)')
        self.t2_ext_edt19.setText(self.user_inputs['tab_2_ext']['lambda3'])
        self.t2_ext_edt19.textChanged.connect(self.cb_t2_ext_edt19)
        self.t2_ext_edt19.setHidden(True) 
    
        # indicate that tab 2 for vector autoregression is now created
        self.created_tab_2_ext = True    
       
    
    def hide_tab_2_ext(self):
        
        # hide all controls
        self.t2_ext_txt1.setHidden(True)
        self.t2_ext_txt2.setHidden(True)
        self.t2_ext_txt3.setHidden(True)
        self.t2_ext_txt4.setHidden(True)
        self.t2_ext_txt5.setHidden(True)
        self.t2_ext_txt6.setHidden(True)
        self.t2_ext_txt7.setHidden(True)
        self.t2_ext_txt8.setHidden(True)
        self.t2_ext_txt9.setHidden(True)
        self.t2_ext_txt10.setHidden(True)
        self.t2_ext_txt11.setHidden(True)
        self.t2_ext_txt12.setHidden(True)
        self.t2_ext_txt13.setHidden(True)
        self.t2_ext_txt14.setHidden(True)
        self.t2_ext_txt15.setHidden(True)
        self.t2_ext_txt16.setHidden(True)
        self.t2_ext_txt17.setHidden(True)
        self.t2_ext_txt18.setHidden(True)
        self.t2_ext_txt19.setHidden(True)
        self.t2_ext_txt20.setHidden(True)
        self.t2_ext_txt21.setHidden(True)
        self.t2_ext_txt22.setHidden(True)
        self.t2_ext_txt23.setHidden(True)
        self.t2_ext_txt24.setHidden(True)
        self.t2_ext_txt25.setHidden(True)
        self.t2_ext_txt26.setHidden(True)
        self.t2_ext_txt27.setHidden(True)
        self.t2_ext_txt28.setHidden(True)
        self.t2_ext_txt29.setHidden(True)
        self.t2_ext_txt30.setHidden(True)
        self.t2_ext_txt31.setHidden(True)
        self.t2_ext_txt32.setHidden(True)
        self.t2_ext_txt33.setHidden(True)
        self.t2_ext_txt34.setHidden(True)
        self.t2_ext_txt35.setHidden(True)
        self.t2_ext_txt36.setHidden(True)
        self.t2_ext_frm1.setHidden(True)
        self.t2_ext_frm2.setHidden(True)
        self.t2_ext_frm3.setHidden(True)
        self.t2_ext_frm4.setHidden(True)
        self.t2_ext_rdb1.setHidden(True)
        self.t2_ext_rdb2.setHidden(True)
        self.t2_ext_rdb3.setHidden(True)
        self.t2_ext_rdb4.setHidden(True)
        self.t2_ext_rdb5.setHidden(True)
        self.t2_ext_rdb6.setHidden(True)
        self.t2_ext_rdb7.setHidden(True)
        self.t2_ext_edt1.setHidden(True)
        self.t2_ext_edt2.setHidden(True)
        self.t2_ext_edt3.setHidden(True)
        self.t2_ext_edt4.setHidden(True)
        self.t2_ext_edt5.setHidden(True)       
        self.t2_ext_edt6.setHidden(True)       
        self.t2_ext_edt7.setHidden(True)       
        self.t2_ext_edt8.setHidden(True)       
        self.t2_ext_edt9.setHidden(True)       
        self.t2_ext_edt10.setHidden(True)       
        self.t2_ext_edt11.setHidden(True)       
        self.t2_ext_edt12.setHidden(True)     
        self.t2_ext_edt13.setHidden(True)
        self.t2_ext_edt14.setHidden(True)     
        self.t2_ext_edt15.setHidden(True)  
        self.t2_ext_edt16.setHidden(True)  
        self.t2_ext_edt17.setHidden(True)  
        self.t2_ext_edt18.setHidden(True)  
        self.t2_ext_edt19.setHidden(True)  
        self.t2_ext_cbx1.setHidden(True)
        self.t2_ext_cbx2.setHidden(True)
        self.t2_ext_cbx3.setHidden(True)
        
        # update tab color
        self.tab_pbt2.setStyleSheet("background:rgb" + str(self.backtabs_color) + ";")       
        
        
    def show_tab_2_ext(self): 
    
        # show all controls
        self.t2_ext_txt1.setVisible(True)
        self.t2_ext_txt2.setVisible(True)
        self.t2_ext_txt3.setVisible(True)
        self.t2_ext_txt4.setVisible(True)
        self.t2_ext_txt5.setVisible(True)
        self.t2_ext_txt6.setVisible(True)
        self.t2_ext_txt7.setVisible(True)
        self.t2_ext_txt8.setVisible(True)
        self.t2_ext_txt9.setVisible(True)
        self.t2_ext_txt10.setVisible(True)
        self.t2_ext_txt11.setVisible(True)
        self.t2_ext_txt12.setVisible(True)
        self.t2_ext_txt13.setVisible(True)
        self.t2_ext_txt14.setVisible(True)
        self.t2_ext_txt15.setVisible(True)
        self.t2_ext_txt16.setVisible(True)
        self.t2_ext_txt17.setVisible(True)
        self.t2_ext_txt18.setVisible(True)
        self.t2_ext_txt19.setVisible(True)
        self.t2_ext_txt20.setVisible(True)
        self.t2_ext_txt21.setVisible(True)
        self.t2_ext_txt22.setVisible(True)
        self.t2_ext_txt23.setVisible(True)
        self.t2_ext_txt24.setVisible(True)
        self.t2_ext_txt25.setVisible(True)
        self.t2_ext_txt26.setVisible(True)
        self.t2_ext_txt27.setVisible(True)
        self.t2_ext_txt28.setVisible(True)
        self.t2_ext_txt29.setVisible(True)
        self.t2_ext_txt30.setVisible(True)
        self.t2_ext_txt31.setVisible(True)
        self.t2_ext_txt32.setVisible(True)
        self.t2_ext_txt33.setVisible(True)
        self.t2_ext_txt34.setVisible(True)
        self.t2_ext_txt35.setVisible(True)
        self.t2_ext_txt36.setVisible(True)
        self.t2_ext_frm1.setVisible(True)
        self.t2_ext_frm2.setVisible(True)
        self.t2_ext_frm3.setVisible(True)
        self.t2_ext_frm4.setVisible(True)
        self.t2_ext_rdb1.setVisible(True)
        self.t2_ext_rdb2.setVisible(True)
        self.t2_ext_rdb3.setVisible(True)
        self.t2_ext_rdb4.setVisible(True)
        self.t2_ext_rdb5.setVisible(True)
        self.t2_ext_rdb6.setVisible(True)
        self.t2_ext_rdb7.setVisible(True)
        self.t2_ext_edt1.setVisible(True)
        self.t2_ext_edt2.setVisible(True)
        self.t2_ext_edt3.setVisible(True)
        self.t2_ext_edt4.setVisible(True)
        self.t2_ext_edt5.setVisible(True)  
        self.t2_ext_edt6.setVisible(True)  
        self.t2_ext_edt7.setVisible(True)  
        self.t2_ext_edt8.setVisible(True)  
        self.t2_ext_edt9.setVisible(True)  
        self.t2_ext_edt10.setVisible(True)  
        self.t2_ext_edt11.setVisible(True)  
        self.t2_ext_edt12.setVisible(True) 
        self.t2_ext_edt13.setVisible(True)
        self.t2_ext_edt14.setVisible(True)
        self.t2_ext_edt15.setVisible(True)
        self.t2_ext_edt16.setVisible(True)
        self.t2_ext_edt17.setVisible(True)
        self.t2_ext_edt18.setVisible(True)
        self.t2_ext_edt19.setVisible(True)
        self.t2_ext_cbx1.setVisible(True)
        self.t2_ext_cbx2.setVisible(True)
        self.t2_ext_cbx3.setVisible(True)
        
        
    def cb_t2_ext_bgr1(self):
        if self.t2_ext_rdb1.isChecked() == True:
            self.user_inputs['tab_2_ext']['model'] = 1
        elif self.t2_ext_rdb2.isChecked() == True:
            self.user_inputs['tab_2_ext']['model'] = 2

    def cb_t2_ext_edt1(self):
        self.user_inputs['tab_2_ext']['iterations'] = self.t2_ext_edt1.text()         
   
    def cb_t2_ext_edt2(self):
        self.user_inputs['tab_2_ext']['burnin'] = self.t2_ext_edt2.text()   
   
    def cb_t2_ext_edt3(self):
        self.user_inputs['tab_2_ext']['model_credibility'] = self.t2_ext_edt3.text()      
   
    def cb_t2_ext_cbx1(self, state):     
        if (state == Qt.Checked):
            self.user_inputs['tab_2_ext']['constant'] = True 
        else:
            self.user_inputs['tab_2_ext']['constant'] = False     
   
    def cb_t2_ext_cbx2(self, state):     
        if (state == Qt.Checked):
            self.user_inputs['tab_2_ext']['trend'] = True 
        else:
            self.user_inputs['tab_2_ext']['trend'] = False  
            
    def cb_t2_ext_cbx3(self, state):     
        if (state == Qt.Checked):
            self.user_inputs['tab_2_ext']['quadratic_trend'] = True 
        else:
            self.user_inputs['tab_2_ext']['quadratic_trend'] = False             
            
    def cb_t2_ext_edt4(self):
        self.user_inputs['tab_2_ext']['vec_lags'] = self.t2_ext_edt4.text() 
            
    def cb_t2_ext_edt5(self):
        self.user_inputs['tab_2_ext']['vec_pi1'] = self.t2_ext_edt5.text()
        
    def cb_t2_ext_edt6(self):
        self.user_inputs['tab_2_ext']['vec_pi2'] = self.t2_ext_edt6.text() 
        
    def cb_t2_ext_edt7(self):
        self.user_inputs['tab_2_ext']['vec_pi3'] = self.t2_ext_edt7.text() 
        
    def cb_t2_ext_edt8(self):
        self.user_inputs['tab_2_ext']['vec_pi4'] = self.t2_ext_edt8.text()             
            
    def cb_t2_ext_bgr2(self):
        if self.t2_ext_rdb3.isChecked() == True:
            self.user_inputs['tab_2_ext']['prior_type'] = 1
        elif self.t2_ext_rdb4.isChecked() == True:
            self.user_inputs['tab_2_ext']['prior_type'] = 2            
        elif self.t2_ext_rdb5.isChecked() == True:
            self.user_inputs['tab_2_ext']['prior_type'] = 3           
            
    def cb_t2_ext_bgr3(self):
        if self.t2_ext_rdb6.isChecked() == True:
            self.user_inputs['tab_2_ext']['error_correction_type'] = 1
        elif self.t2_ext_rdb7.isChecked() == True:
            self.user_inputs['tab_2_ext']['error_correction_type'] = 2              
            
    def cb_t2_ext_edt9(self):
        self.user_inputs['tab_2_ext']['max_cointegration_rank'] = self.t2_ext_edt9.text() 

    def cb_t2_ext_edt10(self):
        self.user_inputs['tab_2_ext']['varma_lags'] = self.t2_ext_edt10.text() 

    def cb_t2_ext_edt11(self):
        self.user_inputs['tab_2_ext']['ar_coefficients'] = self.t2_ext_edt11.text() 

    def cb_t2_ext_edt12(self):
        self.user_inputs['tab_2_ext']['varma_pi1'] = self.t2_ext_edt12.text() 
            
    def cb_t2_ext_edt13(self):
        self.user_inputs['tab_2_ext']['varma_pi2'] = self.t2_ext_edt13.text()         
        
    def cb_t2_ext_edt14(self):
        self.user_inputs['tab_2_ext']['varma_pi3'] = self.t2_ext_edt14.text()         
        
    def cb_t2_ext_edt15(self):
        self.user_inputs['tab_2_ext']['varma_pi4'] = self.t2_ext_edt15.text()         

    def cb_t2_ext_edt16(self):
        self.user_inputs['tab_2_ext']['residual_lags'] = self.t2_ext_edt16.text()         
        
    def cb_t2_ext_edt17(self):
        self.user_inputs['tab_2_ext']['lambda1'] = self.t2_ext_edt17.text()    
     
    def cb_t2_ext_edt18(self):
        self.user_inputs['tab_2_ext']['lambda2'] = self.t2_ext_edt18.text()        
     
    def cb_t2_ext_edt19(self):
        self.user_inputs['tab_2_ext']['lambda3'] = self.t2_ext_edt19.text() 


