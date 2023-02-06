from PyQt5 import QtWidgets

from GUI.controller_screen_files.alreadyVerifiedController import install_alreadyVerifiedWin
from GUI.controller_screen_files.already_enrolledController import install_alreadyEnrolledWin
from GUI.controller_screen_files.didntEnrollController import install_didntEnrollWin
from GUI.controller_screen_files.enrollSuccessController import install_enrollSuccessWin
from GUI.controller_screen_files.enrollmentController import install_enrollmentWin
from GUI.controller_screen_files.id_enrollController import install_idEnrollWin
#from GUI.controller_screen_files.idEnrollController import install_idEnrollWin1
from GUI.controller_screen_files.id_verifyController import install_idVerifyWin
from GUI.controller_screen_files.mainWindowController import install_mainWindow



# this object aim is to route the screen object to the correct install file.
# and save all the screen that open, for protecting the screens from the garbage collector
# - because of those reasons the screen has to open from special method that existing in the object.
# - and called "open_window"
from GUI.controller_screen_files.verificationController import install_verificationWin
from GUI.controller_screen_files.verificationFailedController import install_verificationFailedWin
from GUI.controller_screen_files.verificationSuccessController import install_verificationSuccessWin
from GUI.controller_screen_files.LoginController import install_Login
from GUI.controller_screen_files.administratorController import install_administrator
from GUI.controller_screen_files.verifySpeakerController import install_verifySpeaker
from GUI.controller_screen_files.recordingSelectorController import install_recordingSelector
from GUI.controller_screen_files.addRecToSpeakerController import install_addRecToSpeaker
from GUI.controller_screen_files.recordingSelector2Controller import install_recordingSelector2
from GUI.controller_screen_files.addSpeakerController import install_addSpeaker
class InstallerDefinition:
    def __init__(self):
        # "screens_to_init_function" - it's a dictionary that connect between the screen to specific init function
        self.__screens = []
        self.__screens_to_init_function = {"speakerVerificationMain": install_mainWindow, "already_enrolled" : install_alreadyEnrolledWin, "alreadyVerified": install_alreadyVerifiedWin,
                                           "didntEnroll": install_didntEnrollWin, "Enrollment": install_enrollmentWin, "enrollSuccess": install_enrollSuccessWin, "id_enroll": install_idEnrollWin,
                                          "id_verify": install_idVerifyWin, "verificationWin": install_verificationWin, "verificationFailed": install_verificationFailedWin,
                                           "verificationSuccess": install_verificationSuccessWin, "Login" : install_Login, "adminWindow" : install_administrator, "verifySpeaker" : install_verifySpeaker, "recordingSelector" : install_recordingSelector,
                                           "addRecToSpeaker" : install_addRecToSpeaker, "recordingSelector2" : install_recordingSelector2, "addSpeaker" : install_addSpeaker}
        self.saved_data_dict = dict([])

    def get_saved_data(self):
        return self.saved_data_dict

    def install_definition(self, main_window):
        """
        :type main_window: PyQt5.QtWidgets.QMainWindow.QMainWindow
        :param main_window:
        :return:
        """
        self.__screens.append(main_window)
        func = self.__screens_to_init_function[main_window.objectName()]
        func(self, main_window)

    def open_window(self, ui_object):
        Window = QtWidgets.QMainWindow()
        ui = ui_object()
        ui.setupUi(Window)
        self.install_definition(Window)
        Window.show()

    """def open_message_window(self, text):
        Window = QtWidgets.QMainWindow()
        ui = Ui_MessageWindow()
        ui.setupUi(Window)
        install_MessageWindow(self, Window, text)
        self.__screens.append(Window)
        Window.show()"""


if __name__ == '__main__':
    import sys
    ins = InstallerDefinition()
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    ins.install_definition(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())



#obj.clicked.connect(func)