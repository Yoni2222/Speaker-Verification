# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'addRecToSpeaker.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_addRecToSpeaker(object):
    def setupUi(self, addRecToSpeaker):
        addRecToSpeaker.setObjectName("addRecToSpeaker")
        addRecToSpeaker.resize(800, 207)
        self.centralwidget = QtWidgets.QWidget(addRecToSpeaker)
        self.centralwidget.setObjectName("centralwidget")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(130, 40, 601, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.name = QtWidgets.QTextEdit(self.centralwidget)
        self.name.setGeometry(QtCore.QRect(300, 90, 251, 31))
        self.name.setObjectName("name")
        self.submit = QtWidgets.QPushButton(self.centralwidget)
        self.submit.setGeometry(QtCore.QRect(300, 130, 251, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.submit.setFont(font)
        self.submit.setObjectName("submit")
        ######################################
        #self.submit.setStyleSheet("background-color: rgb(255, 209, 128);")
        ########################################
        addRecToSpeaker.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(addRecToSpeaker)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        addRecToSpeaker.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(addRecToSpeaker)
        self.statusbar.setObjectName("statusbar")
        addRecToSpeaker.setStatusBar(self.statusbar)

        self.retranslateUi(addRecToSpeaker)
        QtCore.QMetaObject.connectSlotsByName(addRecToSpeaker)

    def retranslateUi(self, addRecToSpeaker):
        _translate = QtCore.QCoreApplication.translate
        addRecToSpeaker.setWindowTitle(_translate("addRecToSpeaker", "MainWindow"))
        self.label_2.setText(_translate("addRecToSpeaker", "Enter the name of the speaker you want to add a recording of:"))
        self.submit.setText(_translate("addRecToSpeaker", "Submit"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    addRecToSpeaker = QtWidgets.QMainWindow()
    ui = Ui_addRecToSpeaker()
    ui.setupUi(addRecToSpeaker)
    addRecToSpeaker.show()
    sys.exit(app.exec_())