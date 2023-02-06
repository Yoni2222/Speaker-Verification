# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'enrollment.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Enrollment(object):
    def setupUi(self, Enrollment):
        Enrollment.setObjectName("Enrollment")
        Enrollment.resize(800, 599)
        self.centralwidget = QtWidgets.QWidget(Enrollment)
        self.centralwidget.setObjectName("centralwidget")
        self.enrollment_title = QtWidgets.QLabel(self.centralwidget)
        self.enrollment_title.setGeometry(QtCore.QRect(310, 80, 201, 51))
        font = QtGui.QFont()
        font.setFamily("Hadassah Friedlaender")
        font.setPointSize(24)
        font.setBold(True)
        font.setWeight(75)
        self.enrollment_title.setFont(font)
        self.enrollment_title.setObjectName("enrollment_title")
        self.submit = QtWidgets.QPushButton(self.centralwidget)
        self.submit.setGeometry(QtCore.QRect(280, 220, 221, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.submit.setFont(font)
        self.submit.setObjectName("submit")
        self.uploadRec = QtWidgets.QPushButton(self.centralwidget)
        self.uploadRec.setGeometry(QtCore.QRect(280, 360, 221, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.uploadRec.setFont(font)
        self.uploadRec.setObjectName("uploadRec")
        self.delete_rec = QtWidgets.QPushButton(self.centralwidget)
        self.delete_rec.setGeometry(QtCore.QRect(280, 290, 221, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.delete_rec.setFont(font)
        self.delete_rec.setObjectName("delete_rec")
        self.back = QtWidgets.QPushButton(self.centralwidget)
        self.back.setGeometry(QtCore.QRect(0, 480, 221, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.back.setFont(font)
        self.back.setObjectName("back")
        self.chosen_path = QtWidgets.QTextEdit(self.centralwidget)
        self.chosen_path.setGeometry(QtCore.QRect(283, 430, 221, 31))
        self.chosen_path.setObjectName("chosen_path")
        Enrollment.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Enrollment)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        Enrollment.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Enrollment)
        self.statusbar.setObjectName("statusbar")
        Enrollment.setStatusBar(self.statusbar)

        self.retranslateUi(Enrollment)
        QtCore.QMetaObject.connectSlotsByName(Enrollment)

    def retranslateUi(self, Enrollment):
        _translate = QtCore.QCoreApplication.translate
        Enrollment.setWindowTitle(_translate("Enrollment", "MainWindow"))
        self.enrollment_title.setText(_translate("Enrollment", "Enrollment"))
        self.submit.setText(_translate("Enrollment", "Submit"))
        self.uploadRec.setText(_translate("Enrollment", "Upload Record"))
        self.delete_rec.setText(_translate("Enrollment", "Delete Recording"))
        self.back.setText(_translate("Enrollment", "Back"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Enrollment = QtWidgets.QMainWindow()
    ui = Ui_Enrollment()
    ui.setupUi(Enrollment)
    Enrollment.show()
    sys.exit(app.exec_())
