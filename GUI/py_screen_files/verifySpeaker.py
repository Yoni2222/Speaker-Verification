# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'verifySpeaker.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_verifySpeaker(object):
    def setupUi(self, verifySpeaker):
        verifySpeaker.setObjectName("verifySpeaker")
        verifySpeaker.resize(714, 290)
        self.centralwidget = QtWidgets.QWidget(verifySpeaker)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(310, 39, 551, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.name = QtWidgets.QTextEdit(self.centralwidget)
        self.name.setGeometry(QtCore.QRect(270, 130, 231, 31))
        self.name.setObjectName("name")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(220, 90, 361, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(290, 182, 181, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("submit")
        verifySpeaker.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(verifySpeaker)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 714, 21))
        self.menubar.setObjectName("menubar")
        verifySpeaker.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(verifySpeaker)
        self.statusbar.setObjectName("statusbar")
        verifySpeaker.setStatusBar(self.statusbar)

        self.retranslateUi(verifySpeaker)
        QtCore.QMetaObject.connectSlotsByName(verifySpeaker)

    def retranslateUi(self, verifySpeaker):
        _translate = QtCore.QCoreApplication.translate
        verifySpeaker.setWindowTitle(_translate("verifySpeaker", "MainWindow"))
        self.label.setText(_translate("verifySpeaker", "Verify Speaker"))
        self.label_2.setText(_translate("verifySpeaker", "Enter the name of the speaker you would like to verify: "))
        self.pushButton.setText(_translate("verifySpeaker", "Submit"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    verifySpeaker = QtWidgets.QMainWindow()
    ui = Ui_verifySpeaker()
    ui.setupUi(verifySpeaker)
    verifySpeaker.show()
    sys.exit(app.exec_())