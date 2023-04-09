# imports
from os.path import join
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap



class Tab5Interface(object):

    
    #---------------------------------------------------
    # Methods (Access = public)
    #---------------------------------------------------  


    def __init__(self):
        pass    


    def create_tab_5(self):
    
        # main title
        self.t5_txt1 = QLabel(self)
        self.t5_txt1.move(10, 60)
        self.t5_txt1.setFixedSize(300, 70)
        self.t5_txt1.setText(' Alexandria') 
        self.t5_txt1.setAlignment(Qt.AlignLeft)
        self.t5_txt1.setStyleSheet('font-size: 34pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t5_txt1.setFont(font)
        self.t5_txt1.setHidden(True)
        
        # subtitle
        self.t5_txt2 = QLabel(self)
        self.t5_txt2.move(15, 130)
        self.t5_txt2.setFixedSize(350, 30)
        self.t5_txt2.setText(' By Romain Legrand') 
        self.t5_txt2.setAlignment(Qt.AlignLeft)
        self.t5_txt2.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t5_txt2.setFont(font)
        self.t5_txt2.setHidden(True)
        
        # copyright label
        self.t5_txt3 = QLabel(self)
        self.t5_txt3.move(15, 170)
        self.t5_txt3.setFixedSize(460, 30)
        self.t5_txt3.setText(' Copyright © Romain Legrand 2021') 
        self.t5_txt3.setAlignment(Qt.AlignLeft)
        self.t5_txt3.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t5_txt3.setFont(font)
        self.t5_txt3.setHidden(True)
        
        # information label
        self.t5_txt4 = QLabel(self)
        self.t5_txt4.move(15, 270)
        self.t5_txt4.setFixedSize(460, 30)
        self.t5_txt4.setText(' for more information:') 
        self.t5_txt4.setAlignment(Qt.AlignLeft)
        self.t5_txt4.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t5_txt4.setFont(font)
        self.t5_txt4.setHidden(True)
        
        # mail label
        self.t5_txt5 = QLabel(self)
        self.t5_txt5.move(15, 310)
        self.t5_txt5.setFixedSize(460, 30)
        self.t5_txt5.setText(' alexandria.toolbox@gmail.com') 
        self.t5_txt5.setAlignment(Qt.AlignLeft)
        self.t5_txt5.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t5_txt5.setFont(font)
        self.t5_txt5.setHidden(True)
        
        # website label
        self.t5_txt6 = QLabel(self)
        self.t5_txt6.move(15, 350)
        self.t5_txt6.setFixedSize(460, 30)
        self.t5_txt6.setText(' alexandria-toolbox.github.io') 
        self.t5_txt6.setAlignment(Qt.AlignLeft)
        self.t5_txt6.setStyleSheet('font-size: 16pt; font-family: Serif; \
                font-weight: bold; background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t5_txt6.setFont(font)
        self.t5_txt6.setHidden(True)
        
        # disclaimer line 1
        self.t5_txt7 = QLabel(self)
        self.t5_txt7.move(15, 450)
        self.t5_txt7.setFixedSize(460, 30)
        self.t5_txt7.setText(' Use of this software implies acceptance of the ') 
        self.t5_txt7.setAlignment(Qt.AlignLeft)
        self.t5_txt7.setStyleSheet('font-size: 11pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t5_txt7.setFont(font)
        self.t5_txt7.setHidden(True)
        
        # disclaimer line 2
        self.t5_txt8 = QLabel(self)
        self.t5_txt8.move(15, 480)
        self.t5_txt8.setFixedSize(460, 30)
        self.t5_txt8.setText(' End User Licence Agreement (EULA) for Alexandria.') 
        self.t5_txt8.setAlignment(Qt.AlignLeft)
        self.t5_txt8.setStyleSheet('font-size: 11pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t5_txt8.setFont(font)
        self.t5_txt8.setHidden(True)
        
        # disclaimer line 3
        self.t5_txt9 = QLabel(self)
        self.t5_txt9.move(15, 510)
        self.t5_txt9.setFixedSize(460, 30)
        self.t5_txt9.setText(' Please read and accept the End-User License ') 
        self.t5_txt9.setAlignment(Qt.AlignLeft)
        self.t5_txt9.setStyleSheet('font-size: 11pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t5_txt9.setFont(font)
        self.t5_txt9.setHidden(True)
        
        # disclaimer line 4
        self.t5_txt10 = QLabel(self)
        self.t5_txt10.move(15, 540)
        self.t5_txt10.setFixedSize(460, 30)
        self.t5_txt10.setText(' Agreement carefully before downloading, installing') 
        self.t5_txt10.setAlignment(Qt.AlignLeft)
        self.t5_txt10.setStyleSheet('font-size: 11pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t5_txt10.setFont(font)
        self.t5_txt10.setHidden(True)
        
        # disclaimer line 5
        self.t5_txt11 = QLabel(self)
        self.t5_txt11.move(15, 570)
        self.t5_txt11.setFixedSize(460, 30)
        self.t5_txt11.setText(' or using Alexandria.') 
        self.t5_txt11.setAlignment(Qt.AlignLeft)
        self.t5_txt11.setStyleSheet('font-size: 11pt; font-family: Serif; \
                background-color: rgb' + str(self.background_color))
        font = QFont(); font.setItalic(True); self.t5_txt11.setFont(font)
        self.t5_txt11.setHidden(True)
        
        # credit image
        self.t5_img1 = QLabel(self) 
        self.t5_img1.setFixedSize(480, 360)
        self.t5_img1.setPixmap(QPixmap(join(self.interface_path, 'credits.png')).scaled(480, 360))
        self.t5_img1.move(480, 170) 
        self.t5_img1.setStyleSheet('border: 1px solid black;background-color: rgb' + str(self.background_color))
        self.t5_img1.setHidden(True)
        
    
    def hide_tab_5(self):
    
        # hide all controls
        self.t5_txt1.setHidden(True)
        self.t5_txt2.setHidden(True)
        self.t5_txt3.setHidden(True)
        self.t5_txt4.setHidden(True)
        self.t5_txt5.setHidden(True)
        self.t5_txt6.setHidden(True)
        self.t5_txt7.setHidden(True)
        self.t5_txt8.setHidden(True)
        self.t5_txt9.setHidden(True)
        self.t5_txt10.setHidden(True)
        self.t5_txt11.setHidden(True)
        self.t5_img1.setHidden(True)
        
        # update tab color
        self.tab_pbt5.setStyleSheet("background:rgb" + str(self.backtabs_color) + ";")    
    
    
    def show_tab_5(self):
    
        # show all controls
        self.t5_txt1.setVisible(True)
        self.t5_txt2.setVisible(True)
        self.t5_txt3.setVisible(True)
        self.t5_txt4.setVisible(True)
        self.t5_txt5.setVisible(True)
        self.t5_txt6.setVisible(True)
        self.t5_txt7.setVisible(True)
        self.t5_txt8.setVisible(True)
        self.t5_txt9.setVisible(True)
        self.t5_txt10.setVisible(True)
        self.t5_txt11.setVisible(True)
        self.t5_img1.setVisible(True)
    

