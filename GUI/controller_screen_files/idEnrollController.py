import GUI.general_screen_functions as gf
from GUI.py_screen_files.id_enroll import Ui_id_enroll
from GUI.py_screen_files.enrollment import Ui_Enrollment
from GUI.py_screen_files.already_enrolled import  Ui_already_enrolled
from functools import partial

enrolled = []

def install_idEnrollWin1(installer, main_window):

    ok_button = gf.get_object_by_name(main_window, "submitID")
    ok_button.setStyleSheet("background-color: rgb(255, 209, 128);")
    func = partial(install_ok_button,installer, main_window)
    ok_button.clicked.connect(func)
    ui = Ui_Enrollment()

def install_ok_button(installer, main_window):
    gf.set_background(main_window)
    id_num_obj = gf.get_object_by_name(main_window, "id_box")
    id_num = id_num_obj.toPlainText()  # save id inserted by user to 'id_num'
    if id_num == '':
        gf.error_message("Error", "Please enter a valid id number!")
    else:
        if id_num in enrolled:
            #installer.open_window(Ui_already_enrolled)
            gf.error_message("Error", "You have already enrolled yourself.\n You may go back to menu and perform verification.")
        else:
            enrolled.append(id_num)
            ui1 = Ui_Enrollment()
            #enrolled.clear()
            installer.open_window(Ui_Enrollment)
    enrolled.clear()