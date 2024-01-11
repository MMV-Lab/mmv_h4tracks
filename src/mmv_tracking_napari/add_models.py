import json
from pathlib import Path
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)
from qtpy.QtCore import Qt
import napari

SHOW_OPTIONS_TEXT = "Show advanced options"
HIDE_OPTIONS_TEXT = "Hide advanced options"


class ModelWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.setLayout(QGridLayout())
        self.setWindowTitle("Custom Model")
        self.parent = parent
        self.mode_path: str
        self.advanced_options = []
        try:
            self.setStyleSheet(napari.qt.get_stylesheet(theme="dark"))
        except TypeError:
            self.setStyleSheet(napari.qt.get_stylesheet(theme_id="dark"))

        ## QObjects
        # Labels
        label_name = QLabel("Name")  # basic
        label_name.setToolTip("The name your model will be displayed as")
        label_file = QLabel("File")  # basic
        # self.label_selected_file = QLabel()  # basic
        label_diameter = QLabel("diameter")  # basic
        label_channels = QLabel("channels")  # basic
        label_batch_size = QLabel("batch_size")
        self.advanced_options.append(label_batch_size)
        label_channel_axis = QLabel("channel_axis")
        self.advanced_options.append(label_channel_axis)
        label_z_axis = QLabel("z_axis")
        self.advanced_options.append(label_z_axis)
        label_tile_overlap = QLabel("tile_overlap")
        self.advanced_options.append(label_tile_overlap)
        label_flow_threshold = QLabel("flow-threshold")
        self.advanced_options.append(label_flow_threshold)
        label_cellprob_threshold = QLabel("cellprob-threshold")
        self.advanced_options.append(label_cellprob_threshold)
        label_min_size = QLabel("min_size")
        self.advanced_options.append(label_min_size)
        label_stitch_threshold = QLabel("stitch_threshold")
        self.advanced_options.append(label_stitch_threshold)
        label_rescale = QLabel("rescale")
        self.advanced_options.append(label_rescale)

        # Buttons
        btn_file = QPushButton("Select")  # basic
        self.btn_advanced_options = QPushButton(SHOW_OPTIONS_TEXT)  # basic
        btn_add_model = QPushButton("Add Model")  # basic
        btn_cancel = QPushButton("Cancel")  # basic

        btn_file.clicked.connect(self.select_file)
        self.btn_advanced_options.clicked.connect(self.toggle_advanced_options)
        btn_add_model.clicked.connect(self.add_model)
        btn_cancel.clicked.connect(self.cancel)

        # Lineedits
        self.lineedit_name = QLineEdit()  # basic
        self.lineedit_file = QLineEdit()  # basic
        self.lineedit_diameter = QLineEdit()  # basic
        self.lineedit_channels = QLineEdit("0, 0")  # basic
        self.lineedit_batch_size = QLineEdit("8")
        self.advanced_options.append(self.lineedit_batch_size)
        self.lineedit_channel_axis = QLineEdit("")
        self.advanced_options.append(self.lineedit_channel_axis)
        self.lineedit_z_axis = QLineEdit("")
        self.advanced_options.append(self.lineedit_z_axis)
        self.lineedit_tile_overlap = QLineEdit("0.1")
        self.advanced_options.append(self.lineedit_tile_overlap)
        self.lineedit_flow_threshold = QLineEdit("0.4")
        self.advanced_options.append(self.lineedit_flow_threshold)
        self.lineedit_cellprob_threshold = QLineEdit("0.0")
        self.advanced_options.append(self.lineedit_cellprob_threshold)
        self.lineedit_min_size = QLineEdit("15")
        self.advanced_options.append(self.lineedit_min_size)
        self.lineedit_stitch_threshold = QLineEdit("0.0")
        self.advanced_options.append(self.lineedit_stitch_threshold)
        self.lineedit_rescale = QLineEdit("")
        self.advanced_options.append(self.lineedit_rescale)

        # Checkboxes
        self.checkbox_invert = QCheckBox("invert")
        self.advanced_options.append(self.checkbox_invert)
        self.checkbox_normalize = QCheckBox("normalize")
        self.checkbox_normalize.setChecked(True)
        self.advanced_options.append(self.checkbox_normalize)
        self.checkbox_net_avg = QCheckBox("net_avg")
        self.advanced_options.append(self.checkbox_net_avg)
        self.checkbox_augment = QCheckBox("augment")
        self.advanced_options.append(self.checkbox_augment)
        self.checkbox_tile = QCheckBox("tile")
        self.checkbox_tile.setChecked(True)
        self.advanced_options.append(self.checkbox_tile)
        self.checkbox_resample = QCheckBox("resample")
        self.checkbox_resample.setChecked(True)
        self.advanced_options.append(self.checkbox_resample)
        self.checkbox_interp = QCheckBox("interp")
        self.checkbox_interp.setChecked(True)
        self.advanced_options.append(self.checkbox_interp)

        # Widget
        # self.advanced_options = QWidget()
        # self.advanced_options.setLayout(QGridLayout())
        # self.advanced_options.layout().addWidget(label_batch_size, 0, 0)
        # self.advanced_options.layout().addWidget(self.lineedit_batch_size, 0, 1)
        # self.advanced_options.layout().addWidget(label_channel_axis, 1, 0)
        # self.advanced_options.layout().addWidget(self.lineedit_channel_axis, 1, 1)
        # self.advanced_options.layout().addWidget(label_z_axis, 2, 0)
        # self.advanced_options.layout().addWidget(self.lineedit_z_axis, 2, 1)
        # self.advanced_options.layout().addWidget(label_tile_overlap, 6, 0)
        # self.advanced_options.layout().addWidget(self.lineedit_tile_overlap, 6, 1)
        # self.advanced_options.layout().addWidget(label_flow_threshold, 8, 0)
        # self.advanced_options.layout().addWidget(self.lineedit_flow_threshold, 8, 1)
        # self.advanced_options.layout().addWidget(label_cellprob_threshold, 9, 0)
        # self.advanced_options.layout().addWidget(self.lineedit_cellprob_threshold, 9, 1)
        # self.advanced_options.layout().addWidget(label_min_size, 10, 0)
        # self.advanced_options.layout().addWidget(self.lineedit_min_size, 10, 1)
        # self.advanced_options.layout().addWidget(label_stitch_threshold, 11, 0)
        # self.advanced_options.layout().addWidget(self.lineedit_stitch_threshold, 11, 1)
        # self.advanced_options.layout().addWidget(label_rescale, 12, 0)
        # self.advanced_options.layout().addWidget(self.lineedit_rescale, 12, 1)
        # self.advanced_options.layout().addWidget(self.checkbox_invert, 3, 0)
        # self.advanced_options.layout().addWidget(self.checkbox_normalize, 3, 1)
        # self.advanced_options.layout().addWidget(self.checkbox_net_avg, 4, 0)
        # self.advanced_options.layout().addWidget(self.checkbox_augment, 4, 1)
        # self.advanced_options.layout().addWidget(self.checkbox_tile, 5, 0, 1, -1)
        # self.advanced_options.layout().addWidget(self.checkbox_resample, 7, 0)
        # self.advanced_options.layout().addWidget(self.checkbox_interp, 7, 1)

        # self.advanced_options.hide()

        # Add elements to layout
        self.layout().addWidget(label_name, 0, 0, 1, 2)
        self.layout().addWidget(self.lineedit_name, 0, 2, 1, -1)
        self.layout().addWidget(label_file, 1, 0, 1, 2)
        self.layout().addWidget(btn_file, 1, 2, 1, 2)
        self.layout().addWidget(self.lineedit_file, 1, 4, 1, -1)
        self.layout().addWidget(label_diameter, 2, 0, 1, 2)
        self.layout().addWidget(self.lineedit_diameter, 2, 2, 1, -1)
        self.layout().addWidget(label_channels, 3, 0, 1, 2)
        self.layout().addWidget(self.lineedit_channels, 3, 2, 1, -1)
        self.layout().addWidget(self.btn_advanced_options, 4, 0, 1, -1)

        self.layout().addWidget(label_batch_size, 5, 0, 1, 2,)
        self.layout().addWidget(self.lineedit_batch_size, 5, 2, 1, -1)
        self.layout().addWidget(label_channel_axis, 6, 0, 1, 2)
        self.layout().addWidget(self.lineedit_channel_axis, 6, 2, 1, -1)
        self.layout().addWidget(label_z_axis, 7, 0, 1, 2)
        self.layout().addWidget(self.lineedit_z_axis, 7, 2, 1, -1)
        self.layout().addWidget(label_tile_overlap, 8, 0, 1, 2)
        self.layout().addWidget(self.lineedit_tile_overlap, 8, 2, 1, -1)
        self.layout().addWidget(label_flow_threshold, 9, 0, 1, 2)
        self.layout().addWidget(self.lineedit_flow_threshold, 9, 2, 1, -1)
        self.layout().addWidget(label_cellprob_threshold, 10, 0, 1, 2)
        self.layout().addWidget(self.lineedit_cellprob_threshold, 10, 2, 1, -1)
        self.layout().addWidget(label_min_size, 11, 0, 1, 2)
        self.layout().addWidget(self.lineedit_min_size, 11, 2, 1, -1)
        self.layout().addWidget(label_stitch_threshold, 12, 0, 1, 2)
        self.layout().addWidget(self.lineedit_stitch_threshold, 12, 2, 1, -1)
        self.layout().addWidget(label_rescale, 13, 0, 1, 2)
        self.layout().addWidget(self.lineedit_rescale, 13, 2, 1, -1)
        self.layout().addWidget(self.checkbox_invert, 14, 0, 1, 2)
        self.layout().addWidget(self.checkbox_normalize, 14, 2, 1, -1)
        self.layout().addWidget(self.checkbox_net_avg, 15, 0, 1, 2)
        self.layout().addWidget(self.checkbox_augment, 15, 2, 1, -1)
        self.layout().addWidget(self.checkbox_tile, 16, 0, 1, 2)
        self.layout().addWidget(self.checkbox_resample, 16, 2, 1, -1)
        self.layout().addWidget(self.checkbox_interp, 17, 0, 1, 2)

        # self.layout().addWidget(self.advanced_options, 5, 0, 1, -1)
        self.layout().addWidget(btn_add_model, 18, 0, 1, 3)
        self.layout().addWidget(btn_cancel, 18, 3, 1, 3)

        [widget.hide() for widget in self.advanced_options]

    def select_file(self):
        retval = QFileDialog().getOpenFileName(self, "Select Cellpose Model")
        if retval[0] == "":
            return
        self.model_path = retval[0]
        self.label_selected_file.setText(self.model_path)

    def add_model(self):
        params = {
            "diameter": float(self.lineedit_diameter.text()),
            "channels": [int(i) for i in self.lineedit_channels.text().split(",")],
        }

        batch_size = int(self.lineedit_batch_size.text())
        if batch_size != 8:
            params["batch_size"] = batch_size

        channel_axis = self.lineedit_channel_axis.text()
        if channel_axis != "":
            params["channel_axis"] = int(channel_axis)

        z_axis = self.lineedit_z_axis.text()
        if z_axis != "":
            params["z_axis"] = int(z_axis)

        if self.checkbox_invert.isChecked():
            params["invert"] = True

        if not self.checkbox_normalize.isChecked():
            params["normalize"] = False

        if self.checkbox_net_avg.isChecked():
            params["net_avg"] = True

        if self.checkbox_augment.isChecked():
            params["augment"] = True

        if not self.checkbox_tile.isChecked():
            params["tile"] = False

        tile_overlap = float(self.lineedit_tile_overlap.text())
        if tile_overlap != 0.1:
            params["tile_overlap"] = tile_overlap

        if not self.checkbox_resample.isChecked():
            params["resample"] = False

        if not self.checkbox_interp.isChecked():
            params["interp"] = False

        flow_threshold = float(self.lineedit_flow_threshold.text())
        if flow_threshold != 0.4:
            params["flow_threshold"] = flow_threshold

        cellprob_threshold = float(self.lineedit_cellprob_threshold.text())
        if cellprob_threshold != 0:
            params["cellprob_threshold"] = cellprob_threshold

        min_size = int(self.lineedit_min_size.text())
        if min_size != 15:
            params["min_size"] = min_size

        stitch_threshold = float(self.lineedit_stitch_threshold.text())
        if stitch_threshold != 0:
            params["stitch_threshold"] = stitch_threshold

        rescale = self.lineedit_rescale.text()
        if rescale != "":
            params["rescale"] = float(rescale)

        model_entry = {"filename": self.model_path, "params": params}
        self.parent.custom_models[self.lineedit_name.text()] = model_entry
        with open(Path(__file__).parent / "custom_models.json", "w") as file:
            json.dump(self.parent.custom_models, file)
        self.parent.read_models()
        self.close()

    def toggle_advanced_options(self):
        if self.btn_advanced_options.text() == SHOW_OPTIONS_TEXT:
            self.btn_advanced_options.setText(HIDE_OPTIONS_TEXT)
            [widget.show() for widget in self.advanced_options]
            # self.advanced_options.show()
            # self.show_advanced_options()
        else:
            self.btn_advanced_options.setText(SHOW_OPTIONS_TEXT)
            [widget.hide() for widget in self.advanced_options]
            # self.advanced_options.hide()
            self.adjustSize()
            # self.hide_advanced_options()

    def cancel(self):
        self.close()

    # def show_advanced_options(self):
    #     self.advanced_options.show()
    #     pass

    # def hide_advanced_options(self):
    #     pass
