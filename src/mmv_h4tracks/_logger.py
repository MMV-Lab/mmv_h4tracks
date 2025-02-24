from qtpy.QtWidgets import QMessageBox, QApplication

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
    msg.exec()

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
    return msg.exec()


def handle_exception(exception):
    """
    Handles an exception by showing a notification dialog and restoring the cursor
    """
    notify(str(exception))
    QApplication.restoreOverrideCursor()
