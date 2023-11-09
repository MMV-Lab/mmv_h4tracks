import sys
from pathlib import Path
import time

from qtpy.QtWidgets import QMessageBox, QInputDialog
from napari.qt.threading import thread_worker


def setup_logging():
    """
    Sets up for print to write to the log file
    """
    plugin_directory = Path(__file__).parent.parent.parent.absolute()
    print(plugin_directory)
    path = plugin_directory / "hitl4trk.log"
    file = open(path, "w")
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
    print("Notifying user: '{}'".format(text))
    msg.exec()


@thread_worker
def notify_with_delay(text):
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
    print("Prompting user: '{}'".format(text))
    return msg.exec()

def layer_select(parent, layertype):
    title = "Select Layer"
    text = f"Please select the layer that has the {layertype}"
    items = []
    for layer in parent.viewer.layers:
        items.append(layer.name)
    return QInputDialog.getItem(parent, title, text, items, editable = False)
