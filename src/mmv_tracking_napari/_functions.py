from qtpy.QtWidgets import QMessageBox


def message(text,title="napari",informative_text="",buttons=[]):
    msg = QMessageBox()
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setInformativeText(informative_text)
    for button in buttons:
        if type(button) is tuple:
            msg.addButton(button[0],button[1])
        else:
            msg.addButton(button)
    return msg.exec()