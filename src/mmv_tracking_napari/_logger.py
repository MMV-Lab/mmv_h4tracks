import sys
import time
from pathlib import Path

from qtpy.QtWidgets import QMessageBox, QInputDialog, QApplication
from napari.qt.threading import thread_worker


def setup_logging():
    """
    Sets up for print to write to the log file
    """
    plugin_directory = Path(__file__).parent.parent.parent.absolute()
    print(plugin_directory)
    path = plugin_directory / "hitl4trk.log"
    with open(path, "w", encoding="utf-8") as file:
        #file = open(path, "w")
        sys.stdout = file
        sys.stderr = file
    print("Logging initialized")


def notify(text):
    """
    Shows a notification dialog

    Parameters
    ----------
    text : str
        The text displayed as the notification
    """
    msg = QMessageBox()
    msg.setWindowTitle("napari")
    msg.setText(text)
    print(f"Notifying user: '{text}'")
    msg.exec()

@thread_worker
def notify_with_delay(text):
    """
    Shows a notification dialog after a brief delay

    This is used to ensure the mouse release event is sent to the viewer
    before the message is displayed

    Parameters
    ----------
    text : str
        The text displayed as the notification
    """
    time.sleep(0.2)
    notify(text)

def choice_dialog(text, choices):
    """
    Shows a dialog where the user has to make a decision

    Parameters
    ----------
    text : str
        The text displayed as the prompt for the decision
    choices : list of tuple or types of buttons
        Tuples of the potential choices, consisting of ("button text", "button type") or button types
    """
    msg = QMessageBox()
    msg.setWindowTitle("napari")
    msg.setText(text)
    for choice in choices:
        if type(choice) is tuple:
            msg.addButton(choice[0], choice[1])
        else:
            msg.addButton(choice)
    print(f"Prompting user: '{text}'")
    return msg.exec()


def handle_exception(exception):
    notify(str(exception))
    QApplication.restoreOverrideCursor()
