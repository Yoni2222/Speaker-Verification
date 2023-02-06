import os.path

import GUI.general_screen_functions as gf
from GUI.py_screen_files.id_verify import  Ui_id_verify
from GUI.py_screen_files.didntEnroll import  Ui_didntEnroll
from GUI.py_screen_files.alreadyVerified import  Ui_alreadyVerified
from GUI.py_screen_files.verification import  Ui_verificationWin
from GUI.controller_screen_files.id_enrollController import enrolled
curr_path = "C:\\finalProject\\datasets\\timit\\GUI\\Users"

from functools import partial

verified = []
curr_id = []

def install_idVerifyWin(installer, main_window):
    #installer.open_window(Ui_MainWindow)
    gf.set_background(main_window)
    ok_button = gf.get_object_by_name(main_window, "submitID")
    ok_button.setStyleSheet("background-color: rgb(255, 209, 128);")
    func = partial(install_ok_button, installer, main_window)
    ok_button.clicked.connect(func)

def install_ok_button(installer, main_window):

    id_num_obj = gf.get_object_by_name(main_window, "id_text")
    id_num = id_num_obj.toPlainText()  # save id inserted by user to 'id_num'
    speaker_dir = os.path.join(curr_path, id_num)
    if id_num == '':
        gf.error_message("Error", "Please enter a valid id number!")
    else:
        if id_num in verified or os.path.exists(os.path.join(speaker_dir, "verified")):
            #installer.open_window(Ui_alreadyVerified)
            gf.error_message("Verification has already been done", "You have already verified yourself! Good job!")

        elif not os.path.exists(os.path.join(speaker_dir, "enrolled")):
            #installer.open_window(Ui_didntEnroll)
            gf.error_message("Error", "You have not enrolled yourself yet.")

        else:
            curr_id.clear()
            curr_id.append(id_num)
            verified.append(id_num)
            os.makedirs(os.path.join(speaker_dir, "verified"), exist_ok = True)
            installer.open_window(Ui_verificationWin)
    #verified.clear()