import os

import GUI.general_screen_functions as gf
from GUI.py_screen_files.mainWindow import Ui_speakerVerificationMain
from GUI.py_screen_files.id_enroll import Ui_id_enroll
from GUI.py_screen_files.id_verify import Ui_id_verify
from GUI.py_screen_files.administrator import Ui_adminWindow
from GUI.py_screen_files.login import Ui_Login
from GUI.py_screen_files.verifySpeaker import Ui_verifySpeaker
from GUI.py_screen_files.recordingSelector import Ui_recordingSelector
name = []


from functools import partial

def install_verifySpeaker(installer, main_window):
    gf.set_background(main_window)
    print("line 16")
    submit_button_obj = gf.get_object_by_name(main_window, "submit")
    submit_button_obj.setStyleSheet("background-color: rgb(255, 209, 128);")
    print("line 18")
    func = partial(install_submit_btn, installer, main_window)
    print("line 20")
    submit_button_obj.clicked.connect(func)
    print("line 23")

def install_submit_btn(installer, main_window):
    speaker_in_db = []

    speaker_in_db = os.listdir(r"C:\finalProject\datasets\timit\data\TEST\DR1")

    name_button_obj = gf.get_object_by_name(main_window, "name")

    speaker_name = name_button_obj.toPlainText()

    if speaker_name not in speaker_in_db:
        gf.error_message("Error", "The name of this speaker is not in the database.")
    else:

        name.append(speaker_name)
        installer.open_window(Ui_recordingSelector)
