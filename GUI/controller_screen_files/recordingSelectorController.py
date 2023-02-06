import os

import GUI.general_screen_functions as gf
from GUI.py_screen_files.mainWindow import Ui_speakerVerificationMain
from GUI.py_screen_files.id_enroll import Ui_id_enroll
from GUI.py_screen_files.id_verify import Ui_id_verify
from GUI.py_screen_files.administrator import Ui_adminWindow
from GUI.py_screen_files.login import Ui_Login
from GUI.py_screen_files.verifySpeaker import Ui_verifySpeaker
from GUI.py_screen_files.recordingSelector import Ui_recordingSelector
from GUI.controller_screen_files.verifySpeakerController import name
from pytorch_speaker_verification.PyTorch_Speaker_Verification_master.OUR_TEST_NEW import Verification
from pytorch_speaker_verification.PyTorch_Speaker_Verification_master import db_manager
from pytorch_speaker_verification.PyTorch_Speaker_Verification_master.db_manager import DataBaseVendor
from functools import partial
path_st = None
database_path = r"C:\finalProject\datasets\timit\data"
model_path = r"C:\finalProject\datasets\timit\pytorch_speaker_verification\PyTorch_Speaker_Verification_master\models\stft\final_epoch_800_batch_id_141.model"

def install_recordingSelector(installer, main_window):
    print("line 20")
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

def install_upload_btn(path):
    file_path_string = gf.get_path_by_dialog()
    # while not file_path_string.endswith('.wav'):
    #    gf.error_message("Invalid file format", "Please choose only files having .wav extension.")
    #    file_path_string = gf.get_path_by_dialog()
    if not file_path_string.endswith('.wav'):
        gf.error_message("Invalid file format", "Please choose only files having .wav extension.")
    else:
        global path_st
        path.setText(file_path_string)
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
        #global path_of_dir

        dv = DataBaseVendor(r"C:\finalProject\datasets\timit\data")

        model_path = r"C:\finalProject\datasets\timit\pytorch_speaker_verification\PyTorch_Speaker_Verification_master\speech_id_checkpoint\final_epoch_800_batch_id_141.model"
        ver = Verification(model_path, dv)
        print("path st is ", path_st)
        ver.fit(path_st, 3)
        print("line 60")
        all_accuracies = ver.TLD.get_time_accuracy(name)
        ver.get_plot_bars(all_accuracies)

        """
            model_path = r"C:\finalProject\datasets\timit\pytorch_speaker_verification\models\stft\final_epoch_800_batch_id_141.model"
            dv = DataBaseVendor(r"C:\finalProject\datasets\timit\data")
            enroll_path = r"C:\finalProject\datasets\timit\pytorch_speaker_verification\PyTorch_Speaker_Verification_master\bibi_benet.wav"

            ver = Verification(model_path, dv)
            ver.fit(enroll_path, 10)
        """