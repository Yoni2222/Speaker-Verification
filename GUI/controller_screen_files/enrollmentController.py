import os

import numpy as np

import GUI.general_screen_functions as gf
from GUI.py_screen_files.enrollment import Ui_Enrollment
from GUI.controller_screen_files.id_enrollController import curr_id


from functools import partial

path_to_enroll = None
ids_and_rec_paths = {}
ids_and_npy_paths = {}
curr_path = "C:\\finalProject\\datasets\\timit\\GUI\\Users"
from GUI.controller_screen_files.verificationController import pre_process


def install_enrollmentWin(installer, main_window):
    #installer.open_window(Ui_MainWindow)
    gf.set_background(main_window)
    print("line 22")
    upload_button = gf.get_object_by_name(main_window, "uploadRec")
    print("line 24")
    upload_button.setStyleSheet("background-color: rgb(255, 209, 128);")
    print("line 26")
    path_box = gf.get_object_by_name(main_window, "chosen_path")
    print("line 28")
    func = partial(install_upload_btn_func, path_box)
    print("line 30")
    upload_button.clicked.connect(func)

    submit_button = gf.get_object_by_name(main_window, "submit")
    submit_button.setStyleSheet("background-color: rgb(255, 209, 128);")
    func = partial(install_submit_btn_func, installer)
    submit_button.clicked.connect(func)

    #print(installer.get_saved_data()["data"])

def install_upload_btn_func(path):
    file_path_string = gf.get_path_by_dialog()
    #while not file_path_string.endswith('.wav'):
    #    gf.error_message("Invalid file format", "Please choose only files having .wav extension.")
    #    file_path_string = gf.get_path_by_dialog()
    if not file_path_string.endswith('.wav'):
        gf.error_message("Invalid file format", "Please choose only files having .wav extension.")
    else:
        global path_to_enroll
        path.setText(file_path_string)
        path_to_enroll = file_path_string
        ids_and_rec_paths[curr_id[0]] = file_path_string
        print(path_to_enroll)


def install_submit_btn_func(installer):

    print(curr_id[0])
    #ids_and_rec_paths.clear()
    if path_to_enroll is None:
        return None
    if not path_to_enroll.endswith('.wav'):
        gf.error_message("Invalid file format", "Please choose only files having .wav extension.")
    else:
        feature_path = os.path.join(curr_path, curr_id[0], "enrolled", "a_enrollment.npy")
        enroll_record_features  = pre_process(path_to_enroll)
        print("im back")
        np.save(feature_path, enroll_record_features)
        gf.info_message("Enrollment Completed", "You have enrolled successfully!")
        #gf.error_message("Enrollment Completed", "You have enrolled successfully!")
    """
    show a prompt
    """
