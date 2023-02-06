import GUI.general_screen_functions as gf
from GUI.py_screen_files.already_enrolled import Ui_already_enrolled
from GUI.py_screen_files.mainWindow import Ui_speakerVerificationMain
from functools import partial

def install_alreadyEnrolledWin(installer, main_window):

    back_button = gf.get_object_by_name(main_window, "backToMenu")
    func = partial(install_back_to_menu_button, installer)
    back_button.clicked.connect(func)

def install_back_to_menu_button(installer):
    installer.open_window(Ui_MainWindow)
    #gf.error_message("","You have already enrolled yourself in the system")