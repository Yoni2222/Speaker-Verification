import os

import numpy as np
import shutil
import GUI.general_screen_functions as gf
from pytorch_speaker_verification.PyTorch_Speaker_Verification_master.OUR_TEST_NEW import Verification
from functools import partial
from GUI.controller_screen_files.addRecToSpeakerController import name
from pytorch_speaker_verification.PyTorch_Speaker_Verification_master.db_manager import DataBaseVendor

path_st = None

def install_recordingSelector2(installer, main_window):
    gf.set_background(main_window)

    upload_button = gf.get_object_by_name(main_window, "uploadRec")
    upload_button.setStyleSheet("background-color: rgb(255, 209, 128);")
    path_box = gf.get_object_by_name(main_window, "chosen_path")
    func = partial(install_upload_btn, path_box)
    upload_button.clicked.connect(func)

    submit_button = gf.get_object_by_name(main_window, "submit")
    submit_button.setStyleSheet("background-color: rgb(255, 209, 128);")
    func = partial(install_submit_btn_func, installer, main_window)
    submit_button.clicked.connect(func)

def install_upload_btn(path_box):
    file_path_string = gf.get_path_by_dialog()
    # while not file_path_string.endswith('.wav'):
    #    gf.error_message("Invalid file format", "Please choose only files having .wav extension.")
    #    file_path_string = gf.get_path_by_dialog()
    if not file_path_string.endswith('.wav'):
        gf.error_message("Invalid file format", "Please choose only files having .wav extension.")
    else:
        global path_st
        path_box.setText(file_path_string)
        path_st = file_path_string

def install_submit_btn_func(installer, main_window):
    all_accuracies = []
    print(path_st)
    if path_st is None:
        print("path_st is None")
        return None
    if not path_st.endswith('.wav'):
        gf.error_message("Invalid file format", "Please choose only files having .wav extension.")
        print("line 47")
    else:
        # global path_of_dir
        original = path_st
        target = r"C:\finalProject\datasets\timit\data\TEST\DR1\\" + str(name[0])
        #shutil.copyfile(original, target)
        print(target)
        print("line 51")
        shutil.copy2(path_st, target)
        gf.info_message("Done", "Recording was uploaded successfully!")