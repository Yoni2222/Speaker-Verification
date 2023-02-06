# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_speakerVerificationMain(object):
    def setupUi(self, speakerVerificationMain):
        speakerVerificationMain.setObjectName("speakerVerificationMain")
        speakerVerificationMain.resize(873, 574)
        self.centralwidget = QtWidgets.QWidget(speakerVerificationMain)
        self.centralwidget.setObjectName("centralwidget")
        self.speakerVerifi = QtWidgets.QLabel(self.centralwidget)
        self.speakerVerifi.setGeometry(QtCore.QRect(160, 100, 561, 51))
        font = QtGui.QFont()
        #font.setFamily("Cambria")
        font.setFamily("Lucida Calligraphy")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.speakerVerifi.setFont(font)
        self.speakerVerifi.setAlignment(QtCore.Qt.AlignCenter)
        self.speakerVerifi.setObjectName("speakerVerifi")
        self.verifyButton = QtWidgets.QPushButton(self.centralwidget)
        self.verifyButton.setGeometry(QtCore.QRect(340, 350, 191, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.verifyButton.setFont(font)
        self.verifyButton.setStyleSheet("background-color: rgb(255, 209, 128);")
        self.verifyButton.setObjectName("verifyButton")
        self.enrollButton = QtWidgets.QPushButton(self.centralwidget)
        self.enrollButton.setGeometry(QtCore.QRect(340, 290, 191, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.enrollButton.setFont(font)
        self.enrollButton.setStyleSheet("\n"
"background-color: rgb(255, 209, 128);")
        self.enrollButton.setObjectName("enrollButton")
        self.Login = QtWidgets.QPushButton(self.centralwidget)
        self.Login.setGeometry(QtCore.QRect(340, 410, 191, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.Login.setFont(font)
        self.Login.setStyleSheet("background-color: rgb(255, 209, 128);")
        self.Login.setObjectName("Login")
        speakerVerificationMain.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(speakerVerificationMain)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 873, 21))
        self.menubar.setObjectName("menubar")
        speakerVerificationMain.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(speakerVerificationMain)
        self.statusbar.setObjectName("statusbar")
        speakerVerificationMain.setStatusBar(self.statusbar)

        self.retranslateUi(speakerVerificationMain)
        QtCore.QMetaObject.connectSlotsByName(speakerVerificationMain)

    def retranslateUi(self, speakerVerificationMain):
        _translate = QtCore.QCoreApplication.translate
        speakerVerificationMain.setWindowTitle(_translate("speakerVerificationMain", "MainWindow"))
        self.speakerVerifi.setText(_translate("speakerVerificationMain", "Speaker Recognition"))
        self.verifyButton.setText(_translate("speakerVerificationMain", "Verify"))
        self.enrollButton.setText(_translate("speakerVerificationMain", "Enroll"))
        self.Login.setText(_translate("speakerVerificationMain", "Log In"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    speakerVerificationMain = QtWidgets.QMainWindow()
    ui = Ui_speakerVerificationMain()
    ui.setupUi(speakerVerificationMain)
    speakerVerificationMain.show()
    sys.exit(app.exec_())
