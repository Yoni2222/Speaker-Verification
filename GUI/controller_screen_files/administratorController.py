import GUI.general_screen_functions as gf
from GUI.py_screen_files.id_enroll import Ui_id_enroll
from GUI.py_screen_files.enrollment import Ui_Enrollment
from GUI.py_screen_files.already_enrolled import  Ui_already_enrolled
from GUI.py_screen_files.verifySpeaker import Ui_verifySpeaker
from GUI.py_screen_files.addRecToSpeaker import Ui_addRecToSpeaker
from GUI.py_screen_files.addSpeaker import Ui_addSpeaker
from pytorch_speaker_verification.PyTorch_Speaker_Verification_master.OUR_TEST_NEW import Verification
from functools import partial

from pytorch_speaker_verification.PyTorch_Speaker_Verification_master.db_manager import DataBaseVendor


def install_administrator(installer, main_window):
    gf.set_background(main_window)
    add_btn_obj = gf.get_object_by_name(main_window, "addSpeaker")
    add_btn_obj.setStyleSheet("background-color: rgb(255, 209, 128);")
    func = partial(install_add_speaker_button, main_window, installer)
    add_btn_obj.clicked.connect(func)


    submit_btn_obj = gf.get_object_by_name(main_window, "addRec")
    submit_btn_obj.setStyleSheet("background-color: rgb(255, 209, 128);")
    func = partial(install_addRecToSpeaker_button, main_window, installer)
    submit_btn_obj.clicked.connect(func)


    verify_speaker_btn_obj = gf.get_object_by_name(main_window, "verifySpeaker")
    verify_speaker_btn_obj.setStyleSheet("background-color: rgb(255, 209, 128);")
    func = partial(install_verifySpeaker_button, installer, main_window)
    verify_speaker_btn_obj.clicked.connect(func)


def install_add_speaker_button(main_window, installer):
    installer.open_window(Ui_addSpeaker)

def install_addRecToSpeaker_button(main_window, installer):
    installer.open_window(Ui_addRecToSpeaker)

def install_verifySpeaker_button(installer, main_window):
    installer.open_window(Ui_verifySpeaker)
    #pass


