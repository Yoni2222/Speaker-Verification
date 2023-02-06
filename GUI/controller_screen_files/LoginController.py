import GUI.general_screen_functions as gf
from GUI.py_screen_files.id_enroll import Ui_id_enroll
from GUI.py_screen_files.enrollment import Ui_Enrollment
from GUI.py_screen_files.already_enrolled import  Ui_already_enrolled
from GUI.py_screen_files.administrator import  Ui_adminWindow
from PyQt5 import QtCore, QtGui, QtWidgets

from functools import partial

admins = {"yonatan" : "123456", "deniss" : "123456"}
username, password = None, None

def install_Login(installer, main_window):
    gf.set_background(main_window)
    submitButton = gf.get_object_by_name(main_window, "submit")
    submitButton.setStyleSheet("background-color: rgb(255, 209, 128);")
    func = partial(install_submit_button, installer, main_window)
    submitButton.clicked.connect(func)



def install_submit_button(installer, main_window):

    print("line 21")
    username_obj = gf.get_object_by_name(main_window, "username")
    print("line 23")
    global username
    print("line 25")
    username = username_obj.toPlainText()                           # save username of speaker
    print("line 27")
    global password
    print("line 29")
    password_obj = gf.get_object_by_name(main_window, "password")
    print("line 31")
    #password = password_obj.toPlainText()  # save password of speaker
    password = password_obj.text()
    print("line 33")
    #password.setEchoMode(QtWidgets.QLineEdit.Password)
    if username is None and password is not None:
        gf.error_message("Error", "Please enter username")
    elif username is not None and password is None:
        gf.error_message("Error", "Please enter password")
    elif username is None and password is None:
        gf.error_message("Error", "Please enter username and password")
    else:
        print("line 41")
        if username in admins and admins[username] == password:
            print("line 43")
            print("username is ", username, "password is ", password)
            installer.open_window(Ui_adminWindow) # open next screen
        else:
            print("line 46")
            print("username is ", username, "password is ", password)
            gf.error_message("Error", "Password or username is incorrect!")
