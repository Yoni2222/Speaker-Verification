from GUI.py_screen_files.mainWindow import *

from GUI.screen_definition import InstallerDefinition
import sys
from GUI.py_screen_files.mainWindow import Ui_speakerVerificationMain
from GUI.py_screen_files.enrollment import Ui_Enrollment

# the main
if __name__ == '__main__':
    ins = InstallerDefinition()
    app = QtWidgets.QApplication(sys.argv)
    ins.open_window(Ui_speakerVerificationMain)
    sys.exit(app.exec_())

