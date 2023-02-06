import PyQt5
from tkinter import filedialog
from tkinter import Tk
from tkinter import messagebox


# showing an error message on the screen, get the title and the message as strings
def error_message(title="", message=""):
    tk = Tk()
    messagebox.showerror(title, message)
    tk.destroy()

def info_message(title="", message="", **options):
    tk = Tk()
    messagebox.showinfo(title=title, message=message, **options)
    tk.destroy()

# open folder dialog / folder browser for choosing the path of file
# return the path of the chosen file
def get_path_by_dialog():
    tk = Tk()
    file_path_string = filedialog.askopenfilename()
    tk.destroy()
    #print("the chosen path: ", file_path_string)
    return file_path_string


# get QMainWindow object, and name of object in the QMainWindow
# the function returns the object himself from the QMainWindow by the name
def get_object_by_name(main_window, obj_name):
    """
    :type main_window: PyQt5.QtWidgets.QMainWindow.QMainWindow
    :param main_window:
    :param obj_name:
    :return:
    """
    for child in main_window.centralWidget().children():
        if child.objectName() == obj_name:
            return child
    return None

def set_background(main_window):
    main_window.setStyleSheet("#" + str(main_window.objectName()) + " { border-image: url(../GUI/images/voice.jpeg) 0 0 0 0 stretch stretch; }")
