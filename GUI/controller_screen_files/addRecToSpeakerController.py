import os

import GUI.general_screen_functions as gf
from pytorch_speaker_verification.PyTorch_Speaker_Verification_master.OUR_TEST_NEW import Verification
from functools import partial
from GUI.py_screen_files.recordingSelector2 import Ui_recordingSelector2
name = []
def install_addRecToSpeaker(installer, main_window):
    gf.set_background(main_window)

    submit_btn_obj = gf.get_object_by_name(main_window, 'submit')
    submit_btn_obj.setStyleSheet("background-color: rgb(255, 209, 128);")
    func = partial(install_submit_btn, installer, main_window)
    submit_btn_obj.clicked.connect(func)

def install_submit_btn(installer, main_window):
    print("line 14")
    name_obj = gf.get_object_by_name(main_window, 'name')
    print("line 16")
    speaker_name = name_obj.toPlainText()
    print("line 18")
    if speaker_name is None:
       return None
    else:
        print("line 22")
        list_of_speakers = []
        list_of_speakers = os.listdir(r"C:\finalProject\datasets\timit\data\TEST\DR1")
        if speaker_name not in list_of_speakers:
            gf.error_message("Error", "This speaker is not in the database!")
        else:
            name.append(speaker_name)
            installer.open_window(Ui_recordingSelector2)