import GUI.general_screen_functions as gf
from GUI.py_screen_files.mainWindow import Ui_speakerVerificationMain
from GUI.py_screen_files.id_enroll import Ui_id_enroll
from GUI.py_screen_files.id_verify import Ui_id_verify
from GUI.py_screen_files.administrator import Ui_adminWindow
from GUI.py_screen_files.login import Ui_Login


from functools import partial

def install_mainWindow(installer, main_window):

    #main_window.setStyleSheet("#" + str(main_window.objectName()) + " { border-image: url(../GUI/images/voice.jpeg) 0 0 0 0 stretch stretch; }")
    gf.set_background(main_window)
    enrollButton = gf.get_object_by_name(main_window, "enrollButton")
    func = partial(install_mainWindow_function_enroll, installer)
    enrollButton.clicked.connect(func)

    verifyButton = gf.get_object_by_name(main_window, "verifyButton")
    func = partial(install_mainWindow_function_verify, installer)
    verifyButton.clicked.connect(func)
    #installer.get_saved_data()["data"] = "yonatan"

    loginButton = gf.get_object_by_name(main_window, "Login")
    func = partial(install_mainWindow_Login, installer)
    loginButton.clicked.connect(func)


def install_mainWindow_function_enroll(installer):
    installer.open_window(Ui_id_enroll)

def install_mainWindow_function_verify(installer):
    installer.open_window(Ui_id_verify)

def install_mainWindow_Login(installer):
    installer.open_window(Ui_Login)
"""
# install = from it, we can create new windows
def install_MainMenu(installer, main_menu):
    openMessage = gf.get_object_by_name(main_menu, "openMessage")
    func = partial(install_MainMenu_function_openMessage, installer)
    openMessage.clicked.connect(func)

    ExitBut = gf.get_object_by_name(main_menu, "ExitBut")
    textEdit = gf.get_object_by_name(main_menu, "textEdit")
    func = partial(install_MainMenu_function_ExitBut, textEdit)
    ExitBut.clicked.connect(func)
def install_MainMenu_function_openMessage(installer):
    installer.open_window(Ui_AddNewProfileWindow)

def install_MainMenu_function_ExitBut(text_box):
    path = gf.get_path_by_dialog()
    text_box.setText(path)"""