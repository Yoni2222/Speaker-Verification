import os

import GUI.general_screen_functions as gf
from GUI.py_screen_files.id_enroll import Ui_id_enroll
from GUI.py_screen_files.enrollment import Ui_Enrollment
from GUI.py_screen_files.already_enrolled import  Ui_already_enrolled
from functools import partial
curr_path = "C:\\finalProject\\datasets\\timit\\GUI\\Users"

enrolled = []
curr_id = []

def install_idEnrollWin(installer, main_window):
    gf.set_background(main_window)
    ok_button = gf.get_object_by_name(main_window, "submitID")
    ok_button.setStyleSheet("background-color: rgb(255, 209, 128);")
    func = partial(install_ok_button,installer, main_window)
    ok_button.clicked.connect(func)

def install_ok_button(installer, main_window):

    id_num_obj = gf.get_object_by_name(main_window, "id_box")
    id_num = id_num_obj.toPlainText()  # save id inserted by user to 'id_num'
    #id_num = 'a'
    speaker_dir = os.path.join(curr_path, id_num)
    if id_num == '':
        gf.error_message("Error", "Please enter a valid id number!")
    else:
        if id_num in enrolled or os.path.exists(speaker_dir):
            #installer.open_window(Ui_already_enrolled)
            gf.error_message("Error", "You have already enrolled yourself.\n You may go back to menu and perform verification.")
        else:
            enrolled.append(id_num)

            os.makedirs(speaker_dir, exist_ok = True)
            os.makedirs(os.path.join(speaker_dir, "enrolled"), exist_ok = True)
            #enrolled.clear()
            curr_id.clear()
            curr_id.append(id_num)
            print(curr_id[0])
            print("line 41")
            installer.open_window(Ui_Enrollment)
    #enrolled.clear()

