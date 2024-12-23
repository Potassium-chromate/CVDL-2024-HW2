# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Q1.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from Q1_lib import Q1



class Ui_Dialog(object):
    def __init__(self):
        self.Q1 = Q1()
    def Load_image_func(self):
        self.Q1.load_image()
    def Show_augmented_func(self):
        self.Q1.show_augmented_image()
    def Show_Structure_func(self):
        self.Q1.show_structure()
    def Show_AccAndLoss_func(self):
        self.Q1.show_acc_and_loss()
    def Inference_func(self):
        self.Q1.inference()
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1070, 822)
        self.verticalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(140, 140, 160, 201))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        ####### Load Image #######
        
        self.pushButton.clicked.connect(self.Load_image_func)
        
        ##########################
        self.verticalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout.addWidget(self.pushButton_2)
        ####### Show argumented image #######
        
        self.pushButton_2.clicked.connect(self.Show_augmented_func)
        
        ##########################
        self.pushButton_3 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout.addWidget(self.pushButton_3)
        ####### Show structure #######
        
        self.pushButton_3.clicked.connect(self.Show_Structure_func)
        
        ##########################
        self.pushButton_4 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout.addWidget(self.pushButton_4)
        ####### Show acc and loss #######
        
        self.pushButton_4.clicked.connect(self.Show_AccAndLoss_func)
        
        ##########################
        self.pushButton_5 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_5.setObjectName("pushButton_5")
        self.verticalLayout.addWidget(self.pushButton_5)
        ####### Show Inference #######
        
        self.pushButton_5.clicked.connect(self.Inference_func)
        
        ##########################

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "Load Image"))
        self.pushButton_2.setText(_translate("Dialog", "1. Show Augmented Images"))
        self.pushButton_3.setText(_translate("Dialog", "2. Show Model Stucture"))
        self.pushButton_4.setText(_translate("Dialog", "3. Show Accuracy and Loss"))
        self.pushButton_5.setText(_translate("Dialog", "4. Inference"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
