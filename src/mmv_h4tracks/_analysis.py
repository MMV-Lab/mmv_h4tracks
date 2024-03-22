from multiprocessing import Pool

import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QPushButton,
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QFileDialog,
    QSizePolicy,
    QMessageBox,
    QApplication,
)
from qtpy.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from napari.qt.threading import thread_worker
import napari
from skimage import measure

from ._grabber import grab_layer
from ._logger import notify
from mmv_h4tracks._logger import handle_exception
from ._selector import Selector
from ._writer import save_csv


class AnalysisWindow(QWidget):
    """
    A (QWidget) window to run analysis on the data.

    Attributes
    ----------

    Methods
    -------
    """

    def __init__(self, parent):
        """
        Parameters
        ----------
        """
        super().__init__()
        self.parent = parent
        self.viewer = parent.viewer
        self.setLayout(QVBoxLayout())
        try:
            self.setStyleSheet(napari.qt.get_stylesheet(theme="dark"))
        except TypeError:
            self.setStyleSheet(napari.qt.get_stylesheet(theme_id="dark"))

        ### QObjects

        # Labels
        label_min_movement = QLabel("Movement minmum:")
        label_min_movement.setToolTip(
            "Sort exported tracks by smaller/larger than movement minimum in pixels"
        )
        label_min_duration = QLabel("Minimum track length:")
        label_min_duration.setToolTip(
            "Sort exported tracks by shorter/longer than minimum track length in timesteps"
        )
        label_metric = QLabel("Metric:")

        # Buttons
        btn_plot = QPushButton("Plot")
        btn_export = QPushButton("Export")
        self.btn_select_all = QPushButton("Select all")

        btn_plot.setToolTip(
            "Plot selected metric\n" "Only displayed tracks are plotted"
        )
        btn_export.setToolTip(
            "Export selected metrics as csv\n" "All tracks are exported"
        )

        btn_plot.clicked.connect(self._start_plot_worker)
        btn_export.clicked.connect(self._start_export_worker)
        self.btn_select_all.clicked.connect(self._select_all_metrics)

        # Comboboxes
        self.combobox_plots = QComboBox()
        self.combobox_plots.addItems(
            [
                "Speed",
                "Size",
                "Track duration",
                "Velocity",
                "Direction",
                "Perimeter",
                "Eccentricity",
                "Accumulated distance",
                "Euclidean distance",
            ]
        )

        # Checkboxes
        checkbox_speed = QCheckBox("Speed")
        checkbox_size = QCheckBox("Size")
        checkbox_direction = QCheckBox("Direction")
        checkbox_euclidean_distance = QCheckBox("Euclidean distance")
        checkbox_accumulated_distance = QCheckBox("Accumulated distance")
        checkbox_velocity = QCheckBox("Velocity")
        checkbox_perimeter = QCheckBox("Perimeter")
        checkbox_eccentricity = QCheckBox("Eccentricity")

        self.checkboxes = [
            checkbox_speed,
            checkbox_size,
            checkbox_direction,
            checkbox_euclidean_distance,
            checkbox_accumulated_distance,
            checkbox_velocity,
            checkbox_perimeter,
            checkbox_eccentricity,
        ]

        # Line Edits
        self.lineedit_movement = QLineEdit("")
        self.lineedit_movement.setMaximumWidth(40)
        self.lineedit_track_duration = QLineEdit("")
        self.lineedit_track_duration.setMaximumWidth(40)

        # Spacer
        v_spacer = QWidget()
        v_spacer.setFixedWidth(4)
        v_spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        h_spacer_1 = QWidget()
        h_spacer_1.setFixedHeight(0)
        h_spacer_1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_spacer_2 = QWidget()
        h_spacer_2.setFixedHeight(0)
        h_spacer_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_spacer_3 = QWidget()
        h_spacer_3.setFixedHeight(0)
        h_spacer_3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_spacer_4 = QWidget()
        h_spacer_4.setFixedHeight(10)
        h_spacer_4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # QGroupBoxes
        plot = QGroupBox("Plot")
        plot.setLayout(QGridLayout())
        plot.layout().addWidget(h_spacer_1, 0, 0, 1, -1)
        plot.layout().addWidget(label_metric, 1, 0)
        plot.layout().addWidget(self.combobox_plots, 1, 1)
        plot.layout().addWidget(btn_plot, 1, 2)

        export = QGroupBox("Export")
        export.setLayout(QGridLayout())
        export.layout().addWidget(h_spacer_2, 0, 0, 1, -1)
        export.layout().addWidget(label_min_movement, 1, 0)
        export.layout().addWidget(self.lineedit_movement, 1, 1)
        export.layout().addWidget(label_min_duration, 2, 0)
        export.layout().addWidget(self.lineedit_track_duration, 2, 1)
        export.layout().addWidget(h_spacer_3, 3, 0, 1, -1)
        export.layout().addWidget(checkbox_speed, 4, 0)
        export.layout().addWidget(checkbox_accumulated_distance, 4, 1)
        export.layout().addWidget(checkbox_direction, 4, 2)
        export.layout().addWidget(checkbox_velocity, 5, 0)
        export.layout().addWidget(checkbox_euclidean_distance, 5, 1)
        export.layout().addWidget(checkbox_size, 5, 2)
        export.layout().addWidget(checkbox_perimeter, 6, 0)
        export.layout().addWidget(checkbox_eccentricity, 6, 1)
        export.layout().addWidget(self.btn_select_all, 6, 2)
        export.layout().addWidget(h_spacer_4, 7, 0, 1, -1)
        export.layout().addWidget(btn_export, 8, 0, 1, -1)

        ### Organize objects via widgets
        content = QWidget()
        content.setLayout(QVBoxLayout())
        content.layout().addWidget(plot)
        content.layout().addWidget(export)
        content.layout().addWidget(v_spacer)

        self.layout().addWidget(content)

    def _select_all_metrics(self):
        """
        Selects or unselects all metrics, depending on the current state of the button
        """
        if self.btn_select_all.text() == "Select all":
            for checkbox in self.checkboxes:
                checkbox.setCheckState(2)
            self.btn_select_all.setText("Unselect all")
        else:
            for checkbox in self.checkboxes:
                checkbox.setCheckState(0)
            self.btn_select_all.setText("Select all")

    def _calculate_speed(self, tracks):
        """
        Calculates the speed of each track

        Parameters
        ----------
        tracks : nd array
            (N,4) shape array, which follows napari's trackslayer format (ID, z, y, x)

        Returns
        -------
        nd array
            (N,3) shape array, which contains the ID, average speed and standard deviation of speed
        """
        for unique_id in np.unique(tracks[:, 0]):
            track = np.delete(tracks, np.where(tracks[:, 0] != unique_id), 0)
            accumulated_distance = []
            for i in range(0, len(track) - 1):
                accumulated_distance.append(
                    np.hypot(
                        track[i, 2] - track[i + 1, 2], track[i, 3] - track[i + 1, 3]
                    )
                )
            average = np.around(np.average(accumulated_distance), 3)
            std_deviation = np.around(np.std(accumulated_distance), 3)
            try:
                speeds = np.append(speeds, [[unique_id, average, std_deviation]], 0)
            except UnboundLocalError:
                speeds = np.array([[unique_id, average, std_deviation]])
        return speeds

    def _calculate_size(self, tracks, segmentation):
        """
        Calculates the size of each tracked cell

        Parameters
        ----------
        tracks : nd array
            (N,4) shape array, which follows napari's trackslayer format (ID, z, y, x)
        segmentation : nd array
            (z,y,x) shape array, which follows napari's labelslayer format

        Returns
        -------
        speeds: nd array
            (N,3) shape array, which contains the ID, average speed and standard deviation of speed
        """
        track_and_segmentation = []
        for unique_id in np.unique(tracks[:, 0]):
            track_and_segmentation.append(
                [tracks[np.where(tracks[:, 0] == unique_id)], segmentation]
            )
        AMOUNT_OF_PROCESSES = self.parent.get_process_limit()

        with Pool(AMOUNT_OF_PROCESSES) as p:
            sizes = p.starmap(calculate_size_single_track, track_and_segmentation)

        return np.array(sizes)

    def _calculate_direction(self, tracks):
        """
        Calculates the angle [°] of each trajectory

        Parameters
        ----------
        tracks : nd array
            (N,4) shape array, which follows napari's trackslayer format (ID, z, y, x)

        Returns
        -------
        sizes: nd array
            (N,3) shape array, which contains the ID, average speed and standard deviation of speed
        """
        for unique_id in np.unique(tracks[:, 0]):
            track = np.delete(tracks, np.where(tracks[:, 0] != unique_id), 0)
            x = track[-1, 3] - track[0, 3]
            y = track[0, 2] - track[-1, 2]
            if x == 0:
                if y < 0:
                    direction = 270
                elif y > 0:
                    direction = 90
                else:
                    direction = 0
            else:
                direction = 180 + np.arctan(y / x) / np.pi * 180
                if x > 0:
                    direction += 180
            if direction >= 360:
                direction -= 360
            direction = np.around(direction, 3)
            try:
                retval = np.append(retval, [[unique_id, x, y, direction]], 0)
            except UnboundLocalError:
                retval = np.array([[unique_id, x, y, direction]])
        return retval

    def _calculate_euclidean_distance(self, tracks):
        """
        Calculates the euclidean distance of each trajectory

        Parameters
        ----------
        tracks : nd array
            (N,4) shape array, which follows napari's trackslayer format (ID, z, y, x)

        Returns
        -------
        euclidean_distances: nd array
            (N,3) shape array, which contains the ID, average euclidean distance and track length
        """
        for id in np.unique(tracks[:, 0]):
            track = np.delete(tracks, np.where(tracks[:, 0] != id), 0)
            x = track[-1, 3] - track[0, 3]
            y = track[0, 2] - track[-1, 2]
            euclidean_distance = np.around(np.sqrt(np.square(x) + np.square(y)), 3)
            try:
                euclidean_distances = np.append(
                    euclidean_distances, [[id, euclidean_distance, len(track)]], 0
                )
            except UnboundLocalError:
                euclidean_distances = np.array([[id, euclidean_distance, len(track)]])
        return euclidean_distances

    def _calculate_velocity(self, tracks):
        """
        Calculates the velocity of each trajectory

        Parameters
        ----------
        tracks : nd array
            (N,4) shape array, which follows napari's trackslayer format (ID, z, y, x)

        Returns
        -------
        velocities: nd array
            (N,3) shape array, which contains the ID, velocity and ID again
        """
        euclidean_distances = self._calculate_euclidean_distance(tracks)
        for unique_id in np.unique(tracks[:, 0]):
            euclidean_distance = np.delete(
                euclidean_distances, np.where(euclidean_distances[:, 0] != unique_id), 0
            )
            track = np.delete(tracks, np.where(tracks[:, 0] != unique_id), 0)
            directed_speed = np.around(euclidean_distance[0, 1] / len(track), 3)
            try:
                directed_speeds = np.append(
                    directed_speeds, [[unique_id, directed_speed, unique_id]], 0
                )
            except UnboundLocalError:
                directed_speeds = np.array([[unique_id, directed_speed, unique_id]])
        return directed_speeds

    def _calculate_accumulated_distance(self, tracks):
        """
        Calculates the accumulated distance of each trajectory

        Parameters
        ----------
        tracks : nd array
            (N,4) shape array, which follows napari's trackslayer format (ID, z, y, x)

        Returns
        -------
        accumulated_distances: nd array
            (N,3) shape array, which contains the ID, average accumulated distance and track length
        """
        for unique_id in np.unique(tracks[:, 0]):
            track = np.delete(tracks, np.where(tracks[:, 0] != unique_id), 0)
            steps = []
            for i in range(0, len(track) - 1):
                steps.append(
                    np.hypot(
                        track[i, 2] - track[i + 1, 2], track[i, 3] - track[i + 1, 3]
                    )
                )
            accumulated_distance = np.around(np.sum(steps), 3)
            try:
                accumulated_distances = np.append(
                    accumulated_distances,
                    [[unique_id, accumulated_distance, len(track)]],
                    0,
                )
            except UnboundLocalError:
                accumulated_distances = np.array(
                    [[unique_id, accumulated_distance, len(track)]]
                )
        return accumulated_distances

    def calculate_cell_eccentricity(self, tracks, segmentation):
        """
        Calculates the eccentricity of each cell

        Parameters
        ----------
        tracks : nd array
            (N,4) shape array, which follows napari's trackslayer format (ID, z, y, x)
        segmentation : nd array
            (z,y,x) shape array, which follows napari's labelslayer format

        Returns
        -------
        eccentricities: nd array
            (N,3) shape array, which contains the ID, average eccentricity and standard deviation of eccentricity
        """
        for unique_id in np.unique(tracks[:, 0]):
            track = np.delete(tracks, np.where(tracks[:, 0] != unique_id), 0)
            eccentricity = []
            for i in range(len(track)):
                seg_id = segmentation[track[i, 1], track[i, 2], track[i, 3]]
                properties = measure.regionprops(
                    (segmentation[track[i, 1]] == seg_id).astype(int)
                )
                eccentricity.append(properties[0].eccentricity)
            try:
                eccentricities = np.append(
                    eccentricities,
                    [
                        [
                            unique_id,
                            np.around(np.average(eccentricity), 3),
                            np.around(np.std(eccentricity), 3),
                        ]
                    ],
                    0,
                )
            except UnboundLocalError:
                eccentricities = np.array(
                    [
                        [
                            unique_id,
                            np.around(np.average(eccentricity), 3),
                            np.around(np.std(eccentricity), 3),
                        ]
                    ]
                )
        return eccentricities

    def calculate_cell_perimeter(self, tracks, segmentation):
        """
        Calculates the perimeter of each cell

        Parameters
        ----------
        tracks : nd array
            (N,4) shape array, which follows napari's trackslayer format (ID, z, y, x)
        segmentation : nd array
            (z,y,x) shape array, which follows napari's labelslayer format

        Returns
        -------
        perimeters: nd array
            (N,3) shape array, which contains the ID, average perimeter and standard deviation of perimeter
        """
        for unique_id in np.unique(tracks[:, 0]):
            track = np.delete(tracks, np.where(tracks[:, 0] != unique_id), 0)
            perimeter = []
            for i in range(len(track)):
                seg_id = segmentation[track[i, 1], track[i, 2], track[i, 3]]
                properties = measure.regionprops(
                    (segmentation[track[i, 1]] == seg_id).astype(int)
                )
                perimeter.append(properties[0].perimeter)
            try:
                perimeters = np.append(
                    perimeters,
                    [
                        [
                            unique_id,
                            np.around(np.average(perimeter), 3),
                            np.around(np.std(perimeter), 3),
                        ]
                    ],
                    0,
                )
            except UnboundLocalError:
                perimeters = np.array(
                    [
                        [
                            unique_id,
                            np.around(np.average(perimeter), 3),
                            np.around(np.std(perimeter), 3),
                        ]
                    ]
                )
        return perimeters

    def _start_plot_worker(self):
        """
        Starts the worker to plot the selected metric
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        worker = self._sort_plot_data(self.combobox_plots.currentText())
        worker.returned.connect(self._plot)

    def _plot(self, plot_dict):
        """
        Plots the selected metric

        Parameters
        ----------
        plot_dict : dict
            dictionary containing the metric data and results

        """
        fig = Figure(figsize=(8, 8))
        fig.patch.set_facecolor("#262930")
        axes = fig.add_subplot(111)
        axes.set_facecolor("#262930")
        axes.spines["bottom"].set_color("white")
        axes.spines["top"].set_color("white")
        axes.spines["right"].set_color("white")
        axes.spines["left"].set_color("white")
        axes.xaxis.label.set_color("white")
        axes.yaxis.label.set_color("white")
        axes.tick_params(axis="x", colors="white", labelsize=15)
        axes.tick_params(axis="y", colors="white", labelsize=15)

        title = plot_dict["Name"]
        results = plot_dict["Results"]

        axes.scatter(
            results[:, 1], results[:, 2], c=np.array([[0, 0.240802676, 0.70703125, 1]])
        )

        if plot_dict["Name"] == "Direction [°]":
            xabs_max = abs(max(axes.get_xlim(), key=abs))
            yabs_max = abs(max(axes.get_ylim(), key=abs))
            abs_max = max(xabs_max, yabs_max)

            axes.set_xlim(xmin=-abs_max, xmax=abs_max)
            axes.set_ylim(ymin=-abs_max, ymax=abs_max)

            axes.axhline(0, color="gray", linewidth=0.5)
            axes.axvline(0, color="gray", linewidth=0.5)
        else:
            axes.margins(x=0.2, y=0.2)
            axes.set_xlim(
                left=min(0, axes.get_xlim()[0]), right=max(0, axes.get_xlim()[1])
            )
            axes.set_ylim(
                bottom=min(0, axes.get_ylim()[0]), top=max(0, axes.get_ylim()[1])
            )

        axes.set_title(title, {"fontsize": 22, "color": "white"})
        axes.set_xlabel(plot_dict["x_label"], fontsize=15)
        axes.set_ylabel(plot_dict["y_label"], fontsize=15)

        canvas = FigureCanvas(fig)
        self.parent.plot_window = QWidget()
        try:
            self.parent.plot_window.setStyleSheet(
                napari.qt.get_stylesheet(theme="dark")
            )
        except TypeError:
            self.parent.plot_window.setStyleSheet(
                napari.qt.get_stylesheet(theme_id="dark")
            )
        self.parent.plot_window.setLayout(QVBoxLayout())
        description = QLabel(plot_dict["Description"])
        description.setMaximumHeight(20)
        self.parent.plot_window.layout().addWidget(description)
        self.selector = Selector(self, axes, results)

        self.parent.plot_window.layout().addWidget(canvas)
        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self.selector.apply)
        self.parent.plot_window.layout().addWidget(btn_apply)
        self.parent.plot_window.show()
        QApplication.restoreOverrideCursor()

    @thread_worker(connect={"errored": handle_exception})
    def _sort_plot_data(self, metric):
        """
        Create dictionary holding metric data and results
        Parameters
        ----------
        metric : str
            metric to plot

        Returns
        -------
        retval: dict
            dictionary containing the metric data and results
        """
        tracks_layer = grab_layer(
            self.parent.viewer, self.parent.combobox_tracks.currentText()
        )
        retval = {}
        if metric == "Speed":
            retval.update({"Name": "Speed [px/frame]"})
            retval.update(
                {"Description": "Scatterplot Standard Deviation vs Average: Speed"}
            )
            retval.update({"x_label": "Average", "y_label": "Standard Deviation"})
            retval.update({"Results": self._calculate_speed(tracks_layer.data)})

        elif metric == "Size":
            retval.update({"Name": "Size [pixels]"})
            segmentation_layer = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            )
            retval.update(
                {"Description": "Scatterplot Standard Deviation vs Average: Size"}
            )
            retval.update({"x_label": "Average", "y_label": "Standard Deviation"})
            retval.update(
                {
                    "Results": self._calculate_size(
                        tracks_layer.data, segmentation_layer.data
                    )
                }
            )

        elif metric == "Direction":
            retval.update({"Name": "Direction [°]"})
            retval.update({"Description": "Scatterplot Travel direction & Distance"})
            retval.update({"Results": self._calculate_direction(tracks_layer.data)})
            retval.update({"x_label": "Δx", "y_label": "Δy"})

        elif metric == "Velocity":
            retval.update({"Name": "Velocity"})
            retval.update({"Description": "Scatterplot velocity vs track id"})
            retval.update({"Results": self._calculate_velocity(tracks_layer.data)})
            retval.update({"x_label": "Velocity [px/frame]", "y_label": "ID"})

        elif metric == "Euclidean distance":
            retval.update({"Name": "Euclidean distance"})
            retval.update(
                {"Description": "Scatterplot euclidean distance vs track duration"}
            )
            retval.update(
                {
                    "x_label": "Euclidean distance [pixels]",
                    "y_label": "Track duration [frames]",
                }
            )
            retval.update(
                {"Results": self._calculate_euclidean_distance(tracks_layer.data)}
            )

        elif metric == "Accumulated distance":
            retval.update({"Name": "Accumulated distance"})
            retval.update(
                {"Description": "Scatterplot accumulated distance vs track duration"}
            )
            retval.update(
                {
                    "x_label": "Accumulated distance [pixels]",
                    "y_label": "Track duration [frames]",
                }
            )
            retval.update(
                {"Results": self._calculate_accumulated_distance(tracks_layer.data)}
            )

        elif metric == "Eccentricity":
            retval.update({"Name": "Eccentricity [a.u.]"})
            segmentation_layer = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            )
            retval.update(
                {
                    "Description": "Scatterplot Standard Deviation vs Average: Eccentricity"
                }
            )
            retval.update({"x_label": "Average", "y_label": "Standard Deviation"})
            retval.update(
                {
                    "Results": self.calculate_cell_eccentricity(
                        tracks_layer.data, segmentation_layer.data
                    )
                }
            )

        elif metric == "Perimeter":
            retval.update({"Name": "Perimeter [pixels]"})
            segmentation_layer = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            )
            retval.update(
                {"Description": "Scatterplot Standard Deviation vs Average: Perimeter"}
            )
            retval.update({"x_label": "Average", "y_label": "Standard Deviation"})
            retval.update(
                {
                    "Results": self.calculate_cell_perimeter(
                        tracks_layer.data, segmentation_layer.data
                    )
                }
            )

        elif metric == "Track duration":
            retval.update({"Name": "Track duration"})
            retval.update({"Description": "Scatterplot track ID vs track duration"})
            retval.update({"x_label": "Track duration [frames]", "y_label": "ID"})
            retval.update(
                {
                    "Results": np.array(
                        [
                            [i, np.count_nonzero(tracks_layer.data[:, 0] == i), i]
                            for i in np.unique(tracks_layer.data[:, 0])
                        ]
                    )
                }
            )
        else:
            raise ValueError("No defined behaviour for given metric.")

        return retval

    def _start_export_worker(self):
        """
        Starts the worker to export the selected metrics
        """
        selected_metrics = []
        for checkbox in self.checkboxes:
            if checkbox.checkState():
                selected_metrics.append(checkbox.text())

        if self.parent.tracking_window.cached_tracks is not None:
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            msg.setText("Export is only possible if all tracks are displayed! Display all now?")
            msg.addButton("Display all && export", QMessageBox.AcceptRole)
            msg.addButton(QMessageBox.Cancel)
            retval = msg.exec()
            if retval != 0:
                return
            self.parent.tracking_window.display_cached_tracks()

        if len(selected_metrics) == 0:
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            msg.setText("Please select at least one metric to export!")
            msg.exec()
            return

        if self.parent.combobox_tracks.currentText() == "":
            msg = QMessageBox()
            msg.setWindowTitle("napari")
            msg.setText("No label layer to extract metrics found!")
            msg.exec()
            return

        dialog = QFileDialog()
        file = dialog.getSaveFileName(filter="*.csv")
        if file[0] == "":
            return

        worker = self._export(file, selected_metrics)

    @thread_worker(connect={"errored": handle_exception})
    def _export(self, file, metrics):
        """
        Exports the selected metrics as csv

        Parameters
        ----------
        file : str
            path to save the csv file
        metrics : list
            list of metrics to export
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        tracks = grab_layer(
            self.parent.viewer, self.parent.combobox_tracks.currentText()
        ).data
        direction = self._calculate_direction(tracks)
        self.direction = direction

        (
            filtered_mask,
            min_movement,
            min_duration,
        ) = self._filter_tracks_by_parameters(tracks)

        duration = np.array(
            [[i, np.count_nonzero(tracks[:, 0] == i)] for i in np.unique(tracks[:, 0])]
        )
        data = self._compose_csv_data(
            tracks, duration, filtered_mask, min_movement, min_duration, metrics
        )
        save_csv(file, data)
        QApplication.restoreOverrideCursor()

    def _filter_tracks_by_parameters(self, tracks):
        """
        Filters the tracks by the given parameters

        Parameters
        ----------
        tracks : nd array
            (N,4) shape array, which follows napari's trackslayer format (ID, z, y, x)

        Returns
        -------
        filtered_mask: nd array
            (N,) shape array, which contains the IDs of the tracks matching the given parameters
        min_movement: int
            minimum movement in pixels
        min_duration: int
            minimum duration in frames
        """
        distances = self._calculate_accumulated_distance(tracks)
        if self.lineedit_movement.text() == "":
            min_movement = 0
            movement_mask = np.unique(tracks[:, 0])
        else:
            try:
                min_movement = int(self.lineedit_movement.text())
            except ValueError as exc:
                raise ValueError("Movement minimum can't be converted to int") from exc
            movement_mask = distances[np.where(distances[:, 1] >= min_movement)[0], 0]

        if self.lineedit_track_duration.text() == "":
            min_duration = 0
            duration_mask = np.unique(tracks[:, 0])
        else:
            try:
                min_duration = int(self.lineedit_track_duration.text())
            except ValueError as exc:
                raise ValueError("Minimum duration can't be converted to int") from exc
            indices = np.where(
                np.unique(tracks[:, 0], return_counts=True)[1] >= min_duration
            )
            duration_mask = np.unique(tracks[:, 0])[indices]

        return np.intersect1d(movement_mask, duration_mask), min_movement, min_duration

    def _compose_csv_data(
        self,
        tracks,
        duration,
        filtered_mask,
        min_movement,
        min_duration,
        selected_metrics,
    ):
        """
        Composes the data for the csv file

        Parameters
        ----------
        tracks : nd array
            (N,4) shape array, which follows napari's trackslayer format (ID, z, y, x)
        duration : nd array
            (N,2) shape array, which contains the ID and duration of each track
        filtered_mask : nd array
            (N,) shape array, which contains the IDs of the tracks matching the given parameters
        min_movement : int
            minimum movement in pixels
        min_duration : int
            minimum duration in frames
        selected_metrics : list
            list of metrics to export

        Returns
        -------
        rows : list
            list of rows to export
        """
        metrics = ["", "Number of cells", "Average track duration [# frames]"]
        metrics.append("Standard deviation of track duration [# frames]")
        individual_metrics = ["ID", "Track duration [# frames]"]
        all_values = ["all"]
        all_values.extend(
            [
                len(np.unique(tracks[:, 0])),
                np.around(np.average(duration[:, 1]), 3),
                np.around(np.std(duration[:, 1]), 3),
            ]
        )
        valid_values = ["valid"]
        valid_values.extend(
            [
                len(filtered_mask),
                np.around(
                    np.average(
                        [
                            duration[i, 1]
                            for i in range(len(duration))
                            if duration[i, 0] in filtered_mask
                        ]
                    ),
                    3,
                ),
                np.around(
                    np.std(
                        [
                            duration[i, 1]
                            for i in range(len(duration))
                            if duration[i, 0] in filtered_mask
                        ]
                    ),
                    3,
                ),
            ]
        )
        if len(selected_metrics) > 0:
            (
                more_metrics,
                more_individual_metrics,
                more_all_values,
                more_valid_values,
                metrics_dict,
            ) = self._extend_metrics(tracks, selected_metrics, filtered_mask, duration)
            metrics.extend(more_metrics)
            individual_metrics.extend(more_individual_metrics)
            all_values.extend(more_all_values)
            valid_values.extend(more_valid_values)

        rows = [metrics, all_values, valid_values]
        rows.append([None])
        rows.append(
            [
                "Movement Threshold: {} pixels".format(str(min_movement)),
                "Duration Threshold: {} frames".format(str(min_duration)),
            ]
        )
        rows.append([None])

        if len(selected_metrics) > 0:
            rows.append(individual_metrics)
            valid_values, invalid_values = self._individual_metric_values(
                tracks, filtered_mask, metrics_dict
            )
            rows.append(["Cells matching the filters"])
            for value in valid_values:
                rows.append(value)

            if not np.array_equal(np.unique(tracks[:, 0]), filtered_mask):
                rows.append([None])
                rows.append([None])
                rows.append(["Cells not matching the filters"])

                [rows.append(value) for value in invalid_values]

        return rows

    def _extend_metrics(self, tracks, selected_metrics, filtered_mask, duration):
        """
        Extends the metrics for the export

        Parameters
        ----------
        tracks : nd array
            (N,4) shape array, which follows napari's trackslayer format (ID, z, y, x)
        selected_metrics : list
            list of metrics to export
        filtered_mask : nd array
            (N,) shape array, which contains the IDs of the tracks matching the given parameters
        duration : nd array
            (N,2) shape array, which contains the ID and duration of each track

        Returns
        -------
        metrics : list
            list of metrics to export
        """
        metrics, individual_metrics, all_values, valid_values = [[], [], [], []]
        metrics_dict = {}
        if "Speed" in selected_metrics:
            speed = self._calculate_speed(tracks)
            metrics.append("Average_speed [# pixels/frame]")
            individual_metrics.extend(
                [
                    "Average speed [# pixels/frame]",
                    "Standard deviation of speed [# pixels/frame]",
                ]
            )
            all_values.append(np.around(np.average(speed[:, 1]), 3))
            valid_values.append(
                np.around(
                    np.average(
                        [
                            speed[i, 1]
                            for i in range(len(speed))
                            if speed[i, 0] in filtered_mask
                        ]
                    ),
                    3,
                )
            )
            metrics_dict.update({"Speed": speed})

        if "Size" in selected_metrics:
            segmentation = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            ).data
            size = self._calculate_size(tracks, segmentation)
            metrics.extend(
                ["Average size [# pixels]", "Standard deviation of size [# pixels]"]
            )
            individual_metrics.extend(
                ["Average size [# pixels]", "Standard deviation of size [# pixels]"]
            )
            all_values.extend(
                [np.around(np.average(size[:, 1]), 3), np.around(np.std(size[:, 1]), 3)]
            )
            valid_values.extend(
                [
                    np.around(
                        np.average(
                            [
                                size[i, 1]
                                for i in range(len(size))
                                if size[i, 0] in filtered_mask
                            ]
                        ),
                        3,
                    ),
                    np.around(
                        np.std(
                            [
                                size[i, 1]
                                for i in range(len(size))
                                if size[i, 0] in filtered_mask
                            ]
                        ),
                        3,
                    ),
                ]
            )
            metrics_dict.update({"Size": size})

        if "Direction" in selected_metrics:
            metrics.extend(
                [
                    "Average direction [°]",
                    "Standard deviation of direction [°]",
                ]
            )
            individual_metrics.extend(["Direction [°]"])
            all_values.extend(
                [
                    np.around(np.average(self.direction[:, 3]), 3),
                    np.around(np.std(self.direction[:, 3]), 3),
                ]
            )
            valid_values.extend(
                [
                    np.around(
                        np.average(
                            [
                                self.direction[i, 3]
                                for i in range(len(self.direction))
                                if self.direction[i, 0] in filtered_mask
                            ]
                        ),
                        3,
                    ),
                    np.around(
                        np.std(
                            [
                                self.direction[i, 3]
                                for i in range(len(self.direction))
                                if self.direction[i, 0] in filtered_mask
                            ]
                        ),
                        3,
                    ),
                ]
            )
            metrics_dict.update({"Direction": self.direction})

        if "Euclidean distance" in selected_metrics:
            euclidean_distance = self._calculate_euclidean_distance(tracks)
            metrics.extend(["Average euclidean distance [# pixels]"])
            individual_metrics.extend(["Euclidean distance [# pixels]"])
            all_values.extend(
                [
                    np.around(np.average(euclidean_distance[:, 1]), 3),
                ]
            )
            valid_values.extend(
                [
                    np.around(
                        np.average(
                            [
                                euclidean_distance[i, 1]
                                for i in range(len(euclidean_distance))
                                if euclidean_distance[i, 0] in filtered_mask
                            ]
                        ),
                        3,
                    ),
                ]
            )
            metrics_dict.update({"Euclidean distance": euclidean_distance})

        if "Velocity" in selected_metrics:
            velocity = self._calculate_velocity(tracks)
            metrics.extend(
                [
                    "Average velocity [#pixels/frame]",
                    "Standard deviation of velocity [# pixels/frame]",
                ]
            )
            individual_metrics.extend(["Velocity [# pixels/frame]"])
            all_values.extend(
                [
                    np.around(np.average(velocity[:, 1]), 3),
                    np.around(np.std(velocity[:, 1]), 3),
                ]
            )
            valid_values.extend(
                [
                    np.around(
                        np.average(
                            [
                                velocity[i, 1]
                                for i in range(len(velocity))
                                if velocity[i, 0] in filtered_mask
                            ]
                        ),
                        3,
                    ),
                    np.around(
                        np.std(
                            [
                                velocity[i, 1]
                                for i in range(len(velocity))
                                if velocity[i, 0] in filtered_mask
                            ]
                        ),
                        3,
                    ),
                ]
            )
            metrics_dict.update({"Velocity": velocity})

        if "Accumulated distance" in selected_metrics:
            accumulated_distance = self._calculate_accumulated_distance(tracks)
            metrics.append("Average accumulated distance [# pixels]")
            individual_metrics.append("Accumulated distance [# pixels]")
            all_values.append(np.around(np.average(accumulated_distance[:, 1]), 3))
            valid_values.append(
                np.around(
                    np.average(
                        [
                            accumulated_distance[i, 1]
                            for i in range(len(accumulated_distance))
                            if accumulated_distance[i, 0] in filtered_mask
                        ]
                    ),
                    3,
                )
            )
            metrics_dict.update({"Accumulated distance": accumulated_distance})

            if "Euclidean distance" in selected_metrics:
                metrics.append("Average directness")
                individual_metrics.append("Directness")

                directness = []
                for i in range(len(np.unique(tracks[:, 0]))):
                    directness.append(
                        [
                            euclidean_distance[i, 0],
                            euclidean_distance[i, 1] / accumulated_distance[i, 1]
                            if accumulated_distance[i, 1] > 0
                            else 0,
                        ]
                    )

                directness = np.around(np.array(directness), 3)
                all_values.append(np.around((np.average(directness[:, 1])), 3))
                valid_values.append(
                    np.around(
                        np.average(
                            [
                                directness[i, 1]
                                for i in range(len(directness))
                                if directness[i, 0] in filtered_mask
                            ]
                        ),
                        3,
                    )
                )
                metrics_dict.update({"Directness": directness})

        if "Perimeter" in selected_metrics:
            segmentation = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            ).data
            perimeter = self.calculate_cell_perimeter(tracks, segmentation)
            metrics.extend(
                [
                    "Average perimeter [# pixels]",
                    "Standard deviation of perimeter [# pixels",
                ]
            )
            individual_metrics.extend(
                [
                    "Average perimeter [# pixels]",
                    "Standard deviation of perimeter [# pixels]",
                ]
            )
            all_values.extend(
                [
                    np.around(np.average(perimeter[:, 1]), 3),
                    np.around(np.std(perimeter[:, 1]), 3),
                ]
            )
            valid_values.extend(
                [
                    np.around(
                        np.average(
                            [
                                perimeter[i, 1]
                                for i in range(len(perimeter))
                                if perimeter[i, 0] in filtered_mask
                            ]
                        ),
                        3,
                    ),
                    np.around(
                        np.std(
                            [
                                perimeter[i, 1]
                                for i in range(len(perimeter))
                                if perimeter[i, 0] in filtered_mask
                            ]
                        ),
                        3,
                    ),
                ]
            )
            metrics_dict.update({"Perimeter": perimeter})

        if "Eccentricity" in selected_metrics:
            segmentation = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            ).data
            eccentricity = self.calculate_cell_eccentricity(tracks, segmentation)
            metrics.extend(
                [
                    "Average eccentricity [# pixels]",
                    "Standard deviation of eccentricity [# pixels]",
                ]
            )
            individual_metrics.extend(
                [
                    "Average eccentricity [# pixels]",
                    "Standard deviation of eccentricity [# pixels]",
                ]
            )
            all_values.extend(
                [
                    np.around(np.average(eccentricity[:, 1]), 3),
                    np.around(np.std(eccentricity[:, 1]), 3),
                ]
            )
            valid_values.extend(
                [
                    np.around(
                        np.average(
                            [
                                eccentricity[i, 1]
                                for i in range(len(eccentricity))
                                if eccentricity[i, 0] in filtered_mask
                            ]
                        ),
                        3,
                    ),
                    np.around(
                        np.std(
                            [
                                eccentricity[i, 1]
                                for i in range(len(eccentricity))
                                if eccentricity[i, 0] in filtered_mask
                            ]
                        ),
                        3,
                    ),
                ]
            )
            metrics_dict.update({"Eccentricity": eccentricity})

        return metrics, individual_metrics, all_values, valid_values, metrics_dict

    def _individual_metric_values(self, tracks, filtered_mask, metrics):
        """
        Calculate each selected metric for both valid tracks that match the filter criteria and invalid tracks that do not match the filter criteria
        """
        valid_values, invalid_values = [[], []]
        for i, id in enumerate(np.unique(tracks[:, 0])):
            value = [id]
            value.append(np.count_nonzero(tracks[:, 0] == id))
            if "Speed" in metrics:
                value.extend([metrics["Speed"][i][1], metrics["Speed"][i][2]])
            if "Size" in metrics:
                value.extend([metrics["Size"][i][1], metrics["Size"][i][2]])
            if "Direction" in metrics:
                value.extend(
                    [metrics["Direction"][i][3]]
                )  # using index 3 as 1 and 2 are solely used for plotting
            if "Euclidean distance" in metrics:
                value.append(metrics["Euclidean distance"][i][1])
            if "Velocity" in metrics:
                value.append(metrics["Velocity"][i][1])
            if "Accumulated distance" in metrics:
                value.append(metrics["Accumulated distance"][i][1])
            if "Directness" in metrics:
                value.append(metrics["Directness"][i][1])
            if "Perimeter" in metrics:
                value.extend([metrics["Perimeter"][i][1], metrics["Perimeter"][i][2]])
            if "Eccentricity" in metrics:
                value.extend(
                    [metrics["Eccentricity"][i][1], metrics["Eccentricity"][i][2]]
                )

            if id in filtered_mask:
                valid_values.append(value)
            else:
                invalid_values.append(value)

        return valid_values, invalid_values


def calculate_size_single_track(track, segmentation):
    """
    Calculates the size of a single tracked cell

    Parameters
    ----------
    track : nd array
        (N,4) shape array, which follows napari's trackslayer format (ID, z, y, x)
    segmentation : nd array
        (z,y,x) shape array, which follows napari's labelslayer format

    Returns
    -------
    sizes: nd array
        (N,3) shape array, which contains the ID, average size and standard deviation of size
    """
    id = track[0, 0]
    sizes = []
    for line in track:
        _, z, y, x = line
        seg_id = segmentation[z, y, x]
        sizes.append(len(np.where(segmentation[z] == seg_id)[0]))
    average = np.around(np.average(sizes), 3)
    std_deviation = np.around(np.std(sizes), 3)
    return [id, average, std_deviation]
