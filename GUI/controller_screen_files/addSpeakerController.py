import os

import GUI.general_screen_functions as gf
from pytorch_speaker_verification.PyTorch_Speaker_Verification_master.OUR_TEST_NEW import Verification
from functools import partial


def install_addSpeaker(installer, main_window):
    print("line 7")
    gf.set_background(main_window)
    print("line 9")
    submit_button_obj = gf.get_object_by_name(main_window, "submit")
    submit_button_obj.setStyleSheet("background-color: rgb(255, 209, 128);")
    print("line 11")
    func = partial(install_submit_btn, installer, main_window)
    print("line 15")
    submit_button_obj.clicked.connect(func)
    print("line 17")

def install_submit_btn(installer, main_window):
    print("line 19")
    name_obj = gf.get_object_by_name(main_window, 'name')
    new_name = name_obj.toPlainText()
    curr_dir = r"C:\finalProject\datasets\timit\data\TEST\DR1"
    if new_name in os.listdir(curr_dir):
        gf.error_message("Speaker Exists", "This speaker name already exists in the database")
    else:
        os.makedirs(os.path.join(curr_dir, new_name), exist_ok= True)
        gf.info_message("Done", "New Speaker was added successfully!")