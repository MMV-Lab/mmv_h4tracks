import numpy as np
import multiprocessing
from multiprocessing import Pool
from scipy import ndimage

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QPushButton,
    QCheckBox,
    QGridLayout,
    QFileDialog,
    QTableWidget,
    QSizePolicy,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from napari.qt.threading import thread_worker
import napari

from ._grabber import grab_layer
from ._logger import notify
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
        self.setWindowTitle("Analysis")
        try:
            self.setStyleSheet(napari.qt.get_stylesheet(theme = "dark"))
        except TypeError:
            pass

        ### QObjects

        # Labels
        label_min_movement = QLabel("Movement Minmum")
        label_min_duration = QLabel("Minimum Track Length")
        label_metric = QLabel("Metric:")
        label_segmentation = QLabel("Segmentation")
        label_tracks = QLabel("Tracks")
        label_plotting = QLabel("Plotting")
        label_filter = QLabel("Track filter")
        label_export = QLabel("Export")
        label_evaluation = QLabel("Evaluation")
        label_eval_range = QLabel("Evaluate between current slice and slice:")

        # Buttons
        btn_plot = QPushButton("Plot")
        btn_export = QPushButton("Export")
        btn_evaluate_segmentation = QPushButton("Evaluate Segmentation")
        btn_evaluate_tracking = QPushButton("Evaluate Tracking")

        btn_plot.setToolTip("Only displayed tracks are plotted")

        btn_plot.clicked.connect(self._start_plot_worker)
        btn_export.clicked.connect(self._start_export_worker)
        
        btn_evaluate_segmentation.clicked.connect(self._evaluate_segmentation)
        btn_evaluate_tracking.clicked.connect(self._evaluate_tracking)

        # Comboboxes
        self.combobox_plots = QComboBox()
        self.combobox_plots.addItems(
            ["Speed", "Size", "Direction", "Euclidean distance", "Accumulated distance"]
        )
        self.combobox_segmentation = QComboBox()
        self.combobox_tracks = QComboBox()
        self.layer_comboboxes = [self.combobox_segmentation, self.combobox_tracks]
        for layer in self.viewer.layers:
            for combobox in self.layer_comboboxes:
                combobox.addItem(layer.name)
        
        # Horizontal lines
        line = QWidget()
        line.setFixedHeight(4)
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line.setStyleSheet("background-color: #c0c0c0")
        line2 = QWidget()
        line2.setFixedHeight(4)
        line2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line2.setStyleSheet("background-color: #c0c0c0")
        line3 = QWidget()
        line3.setFixedHeight(4)
        line3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line3.setStyleSheet("background-color: #c0c0c0")

        # Checkboxes
        checkbox_speed = QCheckBox("Speed")
        checkbox_size = QCheckBox("Size")
        checkbox_direction = QCheckBox("Direction")
        checkbox_euclidean_distance = QCheckBox("Euclidean distance")
        checkbox_accumulated_distance = QCheckBox("Accumulated distance")

        self.checkboxes = [
            checkbox_speed,
            checkbox_size,
            checkbox_direction,
            checkbox_euclidean_distance,
            checkbox_accumulated_distance,
        ]

        # Line Edits
        self.lineedit_movement = QLineEdit("")
        self.lineedit_track_duration = QLineEdit("")
        self.lineedit_limit_evaluation = QLineEdit("0")

        ### Organize objects via widgets
        content = QWidget()
        content.setLayout(QGridLayout())
        content.layout().addWidget(label_segmentation, 0, 0)
        content.layout().addWidget(self.combobox_segmentation, 0, 1)
        content.layout().addWidget(label_tracks, 1, 0)
        content.layout().addWidget(self.combobox_tracks, 1, 1)
        content.layout().addWidget(line, 2, 0, 1, 3)
        content.layout().addWidget(label_plotting , 3, 0)
        content.layout().addWidget(label_metric, 4, 0)
        content.layout().addWidget(self.combobox_plots, 4, 1)
        content.layout().addWidget(btn_plot, 4, 2)
        content.layout().addWidget(line2, 5, 0, 1, 3)
        content.layout().addWidget(label_filter, 6, 0)
        content.layout().addWidget(label_min_movement, 7, 0)
        content.layout().addWidget(self.lineedit_movement, 7, 1)
        content.layout().addWidget(label_min_duration, 8, 0)
        content.layout().addWidget(self.lineedit_track_duration, 8, 1)
        content.layout().addWidget(label_export, 9, 0)
        content.layout().addWidget(checkbox_speed, 10, 0)
        content.layout().addWidget(checkbox_size, 10, 1)
        content.layout().addWidget(checkbox_direction, 10, 2)
        content.layout().addWidget(checkbox_euclidean_distance, 11, 0)
        content.layout().addWidget(checkbox_accumulated_distance, 11, 1)
        content.layout().addWidget(btn_export, 11, 2)
        content.layout().addWidget(line3, 12, 0, 1, 3)
        content.layout().addWidget(label_evaluation, 13, 0)
        content.layout().addWidget(label_eval_range, 14, 0)
        content.layout().addWidget(self.lineedit_limit_evaluation, 14, 1)
        content.layout().addWidget(btn_evaluate_segmentation, 15, 0)
        content.layout().addWidget(btn_evaluate_tracking, 15, 1)
        """content.setLayout(QVBoxLayout())

        threshhold_grid = QWidget()
        threshhold_grid.setLayout(QGridLayout())
        threshhold_grid.layout().addWidget(label_min_movement, 0, 0)
        threshhold_grid.layout().addWidget(self.lineedit_movement, 0, 1)
        threshhold_grid.layout().addWidget(label_min_duration, 1, 0)
        threshhold_grid.layout().addWidget(self.lineedit_track_duration, 1, 1)

        content.layout().addWidget(threshhold_grid)

        extract_grid = QWidget()
        extract_grid.setLayout(QGridLayout())
        extract_grid.layout().addWidget(label_metric, 0, 0)
        extract_grid.layout().addWidget(self.combobox_plots, 0, 1)
        extract_grid.layout().addWidget(btn_plot, 0, 2)
        extract_grid.layout().addWidget(checkbox_speed, 1, 0)
        extract_grid.layout().addWidget(checkbox_size, 1, 1)
        extract_grid.layout().addWidget(checkbox_direction, 1, 2)
        extract_grid.layout().addWidget(checkbox_euclidean_distance, 2, 0)
        extract_grid.layout().addWidget(checkbox_accumulated_distance, 2, 1)
        extract_grid.layout().addWidget(btn_export, 2, 2)

        content.layout().addWidget(extract_grid)
        content.layout().addWidget(self.lineedit_limit_evaluation)

        evaluation = QWidget()
        evaluation.setLayout(QHBoxLayout())
        evaluation.layout().addWidget(btn_evaluate_segmentation)
        evaluation.layout().addWidget(btn_evaluate_tracking)

        content.layout().addWidget(evaluation)"""

        self.layout().addWidget(content)
        self.viewer.layers.events.inserted.connect(self.add_entry_to_comboboxes)
        self.viewer.layers.events.removed.connect(self.remove_entry_from_comboboxes)
        for layer in self.viewer.layers:
            layer.events.name.connect(self.rename_entry_in_comboboxes) # doesn't contain index
        self.viewer.layers.events.moving.connect(self.reorder_entry_in_comboboxes)

    def add_entry_to_comboboxes(self, event):
        for combobox in self.layer_comboboxes:
            combobox.addItem(event.value.name)
        event.value.events.name.connect(self.rename_entry_in_comboboxes) # contains index
        
    def remove_entry_from_comboboxes(self, event):
        for combobox in self.layer_comboboxes:
            combobox.removeItem(event.index)

    def rename_entry_in_comboboxes(self, event):
        if not hasattr(event, 'index'):
            event.index = self.viewer.layers.index(event.source.name)
        for combobox in self.layer_comboboxes:
            index = combobox.currentIndex()
            combobox.removeItem(event.index)
            combobox.insertItem(event.index, event.source.name)
            combobox.setCurrentIndex(index)
        
    def reorder_entry_in_comboboxes(self, event):
        for combobox in self.layer_comboboxes:
            index = combobox.currentIndex()
            item = combobox.itemText(event.index)
            combobox.removeItem(event.index)
            combobox.insertItem(event.new_index, item)
            combobox.setCurrentIndex(index)
            
    def _calculate_speed(self, tracks):
        for unique_id in np.unique(tracks[:, 0]):
            track = np.delete(tracks, np.where(tracks[:, 0] != unique_id), 0)
            distance = []
            for i in range(0, len(track) - 1):
                distance.append(
                    np.hypot(
                        track[i, 2] - track[i + 1, 2], track[i, 3] - track[i + 1, 3]
                    )
                )
            average = np.around(np.average(distance), 3)
            std_deviation = np.around(np.std(distance), 3)
            try:
                speeds = np.append(speeds, [[unique_id, average, std_deviation]], 0)
            except UnboundLocalError:
                speeds = np.array([[unique_id, average, std_deviation]])
        return speeds

    def _calculate_size(self, tracks, segmentation):
        unique_ids = np.unique(tracks[:, 0])
        track_and_segmentation = []
        for unique_id in unique_ids:
            track_and_segmentation.append(
                [tracks[np.where(tracks[:, 0] == unique_id)], segmentation]
            )

        if self.parent.rb_eco.isChecked():
            AMOUNT_OF_PROCESSES = np.maximum(1, int(multiprocessing.cpu_count() * 0.4))
        else:
            AMOUNT_OF_PROCESSES = np.maximum(1, int(multiprocessing.cpu_count() * 0.8))
        print("Running on {} processes max".format(AMOUNT_OF_PROCESSES))

        global func

        def func(track, segmentation):
            id = track[0, 0]
            sizes = []
            for line in track:
                _, z, y, x = line
                seg_id = segmentation[z, y, x]
                sizes.append(len(np.where(segmentation[z] == seg_id)[0]))
            average = np.around(np.average(sizes), 3)
            std_deviation = np.around(np.std(sizes), 3)
            return [id, average, std_deviation]

        with Pool(AMOUNT_OF_PROCESSES) as p:
            sizes = p.starmap(func, track_and_segmentation)

        return np.array(sizes)

    def _calculate_direction(self, tracks):
        for id in np.unique(tracks[:, 0]):
            track = np.delete(tracks, np.where(tracks[:, 0] != id), 0)
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
            distance = np.around(np.sqrt(np.square(x) + np.square(y)), 3)
            direction = np.around(direction, 3)
            try:
                retval = np.append(retval, [[id, x, y, direction, distance]], 0)
            except UnboundLocalError:
                retval = np.array([[id, x, y, direction, distance]])
        return retval

    def _calculate_euclidean_distance(self, tracks):
        for id in np.unique(tracks[:, 0]):
            track = np.delete(tracks, np.where(tracks[:, 0] != id), 0)
            x = track[-1, 3] - track[0, 3]
            y = track[0, 2] - track[-1, 2]
            euclidean_distance = np.around(np.sqrt(np.square(x) + np.square(y)), 3)
            directed_speed = np.around(euclidean_distance / len(track), 3)
            try:
                retval = np.append(retval, [[id, euclidean_distance, len(track), directed_speed]], 0)
            except UnboundLocalError:
                retval = np.array([[id, euclidean_distance, len(track), directed_speed]])
        return retval

    def _calculate_accumulated_distance(self, tracks):
        for id in np.unique(tracks[:, 0]):
            track = np.delete(tracks, np.where(tracks[:, 0] != id), 0)
            steps = []
            for i in range(0, len(track) - 1):
                steps.append(
                    np.hypot(
                        track[i, 2] - track[i + 1, 2], track[i, 3] - track[i + 1, 3]
                    )
                )
            accumulated_distance = np.around(np.sum(steps), 3)
            try:
                retval = np.append(retval, [[id, accumulated_distance, len(track)]], 0)
            except UnboundLocalError:
                retval = np.array([[id, accumulated_distance, len(track)]])
        return retval

    def _start_plot_worker(self):
        worker = self._sort_plot_data(self.combobox_plots.currentText())
        worker.returned.connect(self._plot)
        worker.start()

    def _plot(self, ret):
        fig = Figure(figsize=(6, 7))
        fig.patch.set_facecolor("#262930")
        axes = fig.add_subplot(111)
        axes.set_facecolor("#262930")
        axes.spines["bottom"].set_color("white")
        axes.spines["top"].set_color("white")
        axes.spines["right"].set_color("white")
        axes.spines["left"].set_color("white")
        axes.xaxis.label.set_color("white")
        axes.yaxis.label.set_color("white")
        axes.tick_params(axis="x", colors="white")
        axes.tick_params(axis="y", colors="white")

        title = ret["Name"]
        results = ret["Results"]

        data = axes.scatter(
            results[:, 1], results[:, 2], c=np.array([[0, 0.240802676, 0.70703125, 1]])
        )

        axes.set_title(title, {"fontsize": 18, "color": "white"})
        if not title == "Direction":
            axes.set_xlabel(ret["x_label"])
            axes.set_ylabel(ret["y_label"])

        canvas = FigureCanvas(fig)
        self.parent.plot_window = QWidget()
        self.parent.plot_window.setLayout(QVBoxLayout())
        self.parent.plot_window.layout().addWidget(QLabel(ret["Description"]))

        selector = Selector(self, axes, results) # TODO: fix selector?

        self.parent.plot_window.layout().addWidget(canvas)
        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(selector.apply)
        self.parent.plot_window.layout().addWidget(btn_apply)
        print("Showing plot window")
        self.parent.plot_window.show()

    @thread_worker
    def _sort_plot_data(self, metric):
        """
        Create dictionary holding metric data and results
        """
        try:
            tracks_layer = grab_layer(self.parent.viewer, self.combobox_tracks.currentText())
        except ValueError:
            notify("Please make sure to select the correct tracks layer!")
            return
        retval = {"Name": metric}
        if metric == "Speed":
            print("Plotting speed")
            retval.update(
                {"Description": "Scatterplot Standard Deviation vs Average: Speed"}
            )
            retval.update({"x_label": "Average", "y_label": "Standard Deviation"})
            retval.update({"Results": self._calculate_speed(tracks_layer.data)})
        elif metric == "Size":
            print("Plotting size")
            try:
                segmentation_layer = grab_layer(self.viewer, self.combobox_segmentation.currentText())
            except ValueError:
                notify("Please make sure to select the correct segmentation!")
                return
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
            print("Plotting direction")
            retval.update({"Description": "Scatterplot: Travel direction & Distance"})
            retval.update({"Results": self._calculate_direction(tracks_layer.data)})
        elif metric == "Euclidean distance":
            print("Plotting euclidean distance")
            retval.update({"Description": "Scatterplot x vs y"})
            retval.update({"x_label": "x", "y_label": "y"})
            retval.update(
                {"Results": self._calculate_euclidean_distance(tracks_layer.data)}
            )
        elif metric == "Accumulated distance":
            print("Plotting accumulated distance")
            retval.update({"Description": "Scatterplot x vs y"})
            retval.update({"x_label": "x", "y_label": "y"})
            retval.update(
                {"Results": self._calculate_accumulated_distance(tracks_layer.data)}
            )
        else:
            raise ValueError("No defined behaviour for given metric.")

        return retval

    def _start_export_worker(self):
        print("Exporting")
        selected_metrics = []
        for checkbox in self.checkboxes:
            if checkbox.checkState():
                selected_metrics.append(checkbox.text())

        if len(selected_metrics) == 0:
            print("Export stopped due to no selected metric")
            return

        dialog = QFileDialog()
        file = dialog.getSaveFileName(filter="*.csv")
        if file[0] == "":
            print("Export stopped due to no selected file")
            return

        worker = self._export(file, selected_metrics)
        worker.start()

    @thread_worker
    def _export(self, file, metrics):
        try:
            tracks = grab_layer(self.parent.viewer, self.combobox_tracks.currentText()).data
        except ValueError:
            notify("Please make sure to select the correct tracks layer!")
            return
        direction = self._calculate_direction(tracks)
        self.direction = direction

        try:
            (
                filtered_mask,
                min_movement,
                min_duration,
            ) = self._filter_tracks_by_parameters(tracks, direction)
        except ValueError:
            return

        duration = np.array(
            [[i, np.count_nonzero(tracks[:, 0] == i)] for i in np.unique(tracks[:, 0])]
        )

        data = self._compose_csv_data(
            tracks, duration, filtered_mask, min_movement, min_duration, metrics
        )
        save_csv(file, data)

    def _filter_tracks_by_parameters(self, tracks, direction):
        if self.lineedit_movement.text() == "":
            min_movement = 0
            movement_mask = np.unique(tracks[:, 0])
        else:
            try:
                min_movement = int(self.lineedit_movement.text())
            except ValueError:
                notify("Please use an integer instead of text for movement minimum")
                raise ValueError
            else:
                if min_movement != float(self.lineedit_movement.text()):
                    notify(
                        "Please use an integer instead of a float for movement minimum"
                    )
                    raise ValueError
                movement_mask = direction[
                    np.where(direction[:, 4] >= min_movement)[0], 0
                ]

        if self.lineedit_track_duration.text() == "":
            min_duration = 0
            duration_mask = np.unique(tracks[:, 0])
        else:
            try:
                min_duration = int(self.lineedit_track_duration.text())
            except ValueError:
                notify("Please use an integer instead of text for duration minimum")
                raise ValueError
            else:
                if min_duration != float(self.lineedit_track_duration.text()):
                    notify(
                        "Please use an integer instead of a float for duration minimum"
                    )
                    raise ValueError
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
                "Movement Threshold: {} pixels/frame".format(str(min_movement)),
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
            # [rows.append(value) for value in valid_values]
            for value in valid_values:
                rows.append(value)

            if not np.array_equal(np.unique(tracks[:, 0]), filtered_mask):
                rows.append([None])
                rows.append([None])
                rows.append(["Cells not matching the filters"])

                [rows.append(value) for value in invalid_values]

        return rows

    def _extend_metrics(self, tracks, selected_metrics, filtered_mask, duration):
        metrics, individual_metrics, all_values, valid_values = [[], [], [], []]
        metrics_dict = {}
        if "Speed" in selected_metrics:
            speed = self._calculate_speed(tracks)
            metrics.append("Average_speed [# pixels/frame]")
            individual_metrics.extend(["Average speed [# pixels/frame]", "Standard deviation of speed [# pixels/frame]"])
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
            try:
                segmentation = grab_layer(self.viewer, self.combobox_segmentation.currentText()).data
            except ValueError:
                notify("Please make sure the label layer exists!")
                return
            size = self._calculate_size(tracks, segmentation)
            metrics.extend(["Average size [# pixels]", "Standard deviation of size [# pixels]"])
            individual_metrics.extend(["Average size [# pixels]", "Standard deviation of size [# pixels]"])
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
                    "Average distance [# pixels]",
                ]
            )
            individual_metrics.extend(["Direction [°]", "Distance [# pixels]"])
            all_values.extend(
                [
                    np.around(np.average(self.direction[:, 3]), 3),
                    np.around(np.std(self.direction[:, 3]), 3),
                    np.around(np.average(self.direction[:, 4]), 3),
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
                    np.around(
                        np.average(
                            [
                                self.direction[i, 4]
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
            metrics.extend(["Average euclidean distance [# pixels]", "Average velocity [# pixels/frame]"])
            individual_metrics.extend(["Euclidean distance [# pixels]", "Velocity [# pixels/frame]"])
            all_values.extend(
                [
                    np.around(np.average(euclidean_distance[:, 1]), 3),
                    np.around(
                        np.average(
                            euclidean_distance[:, 1] / len(np.unique(tracks[:, 0]))
                        ),
                        3,
                    ),
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
                    np.around(
                        np.average(
                            [
                                euclidean_distance[i, 1] / duration[i, 1]
                                for i in range(len(euclidean_distance))
                                if euclidean_distance[i, 0] in filtered_mask
                            ]
                        ),
                        3,
                    ),
                ]
            )
            metrics_dict.update({"Euclidean distance": euclidean_distance})

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
                    directness.append([euclidean_distance[i, 0], euclidean_distance[i, 1] / accumulated_distance[i,1] if accumulated_distance[i,1] > 0 else 0])
                    
                directness = np.around(np.array(directness), 3)
                #all_values.append(np.around(np.average(directness[:, 1]), 3))
                all_values.append((np.average(directness[:, 1])))
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

        return metrics, individual_metrics, all_values, valid_values, metrics_dict

    def _individual_metric_values(self, tracks, filtered_mask, metrics):
        # raise NotImplementedError
        valid_values, invalid_values = [[], []]
        for id in np.unique(tracks[:, 0]):
            value = [id]
            value.append(np.count_nonzero(tracks[:, 0] == id))
            if "Speed" in metrics:
                value.extend([metrics["Speed"][id][1], metrics["Speed"][id][2]])
            if "Size" in metrics:
                value.extend([metrics["Size"][id][1], metrics["Size"][id][2]])
            if "Direction" in metrics:
                value.extend([metrics["Direction"][id][3], metrics["Direction"][id][4]])
            if "Euclidean distance" in metrics:
                value.extend(
                    [
                        metrics["Euclidean distance"][id][1],
                        metrics["Euclidean distance"][id][3],
                    ]
                )
            if "Accumulated distance" in metrics:
                value.append(metrics["Accumulated distance"][id][1])
            if "Directness" in metrics:
                value.append(metrics["Directness"][id][1])

            if id in filtered_mask:
                valid_values.append(value)
            else:
                invalid_values.append(value)

        return valid_values, invalid_values

    def _evaluate_segmentation(self):
        automatic_segmentation = self.parent.initial_layers[0]
        try:
            corrected_segmentation = grab_layer(self.viewer, self.combobox_segmentation.currentText()).data
        except ValueError:
            notify("Please make sure the label layer exists!")
            return
        current_frame = int(self.viewer.dims.point[0])
        frame_range = [current_frame]
        try:
            selected_limit = int(self.lineedit_limit_evaluation.text())
        except ValueError:
            notify("Please use integer instead of text")
            return
        if selected_limit > current_frame:
            frame_range.append(selected_limit)
        else:
            frame_range.insert(0, selected_limit)
        frame_range[1] += 1
        frames = [current_frame, frame_range, len(automatic_segmentation) - 1]
        ### FOR CURRENT FRAME
        intersection = np.sum( np.sum( np.logical_and( automatic_segmentation[current_frame], corrected_segmentation[current_frame])))
        union = np.sum( np.sum( np.logical_or( automatic_segmentation[current_frame], corrected_segmentation[current_frame])))
        single_iou = intersection / union
        single_dice = (2 * intersection) / (np.count_nonzero(automatic_segmentation[current_frame]) + np.count_nonzero(corrected_segmentation[current_frame]))
        single_f1 = (2 * intersection) / (intersection + union)
        ### FOR SOME FRAMES
        intersection = np.sum( np.sum( np.sum( np.logical_and( automatic_segmentation[list(range(frame_range[0], frame_range[1]))], corrected_segmentation[list(range(frame_range[0], frame_range[1]))]))))
        union = np.sum( np.sum( np.sum( np.logical_or( automatic_segmentation[list(range(frame_range[0], frame_range[1]))], corrected_segmentation[list(range(frame_range[0], frame_range[1]))]))))
        some_iou = intersection / union
        some_dice = (2 * intersection) / (np.count_nonzero(automatic_segmentation[list(range(frame_range[0], frame_range[1]))]) + np.count_nonzero(corrected_segmentation[list(range(frame_range[0], frame_range[1]))]))
        some_f1 = (2 * intersection) / (intersection + union)
        ### FOR ALL FRAMES
        intersection = np.sum( np.sum( np.sum( np.logical_and( automatic_segmentation, corrected_segmentation))))
        union = np.sum( np.sum( np.sum( np.logical_or( automatic_segmentation, corrected_segmentation))))
        all_iou = intersection / union
        all_dice = (2 * intersection) / (np.count_nonzero(automatic_segmentation) + np.count_nonzero(corrected_segmentation))
        all_f1 = (2 * intersection) / (intersection + union)
        results = np.asarray([[all_iou, all_dice, all_f1],
                              [some_iou, some_dice, some_f1],
                              [single_iou, single_dice, single_f1]
                            ])
        self._display_evaluation_result("Evaluation of Segmentation", results, frames)
        self.results_window.show()
        print("Opening results window")
        
    def adjust_track_centroids(self):
        try:
            tracks_layer = grab_layer(self.parent.viewer, self.combobox_tracks.currentText())
        except ValueError:
            notify("Please make sure to select the correct tracks layer!")
            return
        try:
            segmentation = grab_layer(self.viewer, self.combobox_segmentation.currentText()).data
        except ValueError:
            notify("Please make sure the label layer exists!")
            return
        tracks_old = tracks_layer.data
        tracks = tracks_layer.data
        for line in tracks:
            _,z,y,x = line
            segmentation_id = segmentation[z,y,x]
            if segmentation_id == 0:
                print("couldn't adjust centroid")
                continue
            centroid = ndimage.center_of_mass(
                segmentation[z], labels=segmentation[z], index=segmentation_id
            )
            y_new = int(np.rint(centroid[0]))
            x_new = int(np.rint(centroid[1]))
            if x_new != x or y_new != y:
                line[2] = y_new
                line[3] = x_new

        tracks_layer.data = tracks

    def _evaluate_tracking(self):
        self.adjust_track_centroids()
        automatic_tracks = self.parent.initial_layers[1]
        try:
            corrected_tracks = grab_layer(self.parent.viewer, self.combobox_tracks.currentText()).data
        except ValueError:
            notify("Please make sure to select the correct tracks layer!")
            return
        automatic_segmentation = self.parent.initial_layers[0]
        try:
            corrected_segmentation = grab_layer(self.viewer, self.combobox_segmentation.currentText()).data
        except ValueError:
            notify("Please make sure the label layer exists!")
            return
        
        fp = 0
        fn = 0
        de = 0
        ae = 0
        sc = 0

        if self.parent.rb_eco.isChecked():
            AMOUNT_OF_PROCESSES = np.maximum(1, int(multiprocessing.cpu_count() * 0.4))
        else:
            AMOUNT_OF_PROCESSES = np.maximum(1, int(multiprocessing.cpu_count() * 0.8))
        print("Running on {} processes max".format(AMOUNT_OF_PROCESSES))
        
        # create dict with double lookup for cells in gt and prediction
        # key naming schema: ({p/gt},z,y,x)
        lookup_dict = {}
        for slice in range(len(corrected_segmentation)):
            matches = []
            for cell in np.unique(corrected_segmentation[slice])[1:]:
                centroid = ndimage.center_of_mass(
                    corrected_segmentation[slice],
                    labels = corrected_segmentation[slice],
                    index = cell
                )
                y = int(np.rint(centroid[0]))
                x = int(np.rint(centroid[1]))
                match_id = automatic_segmentation[slice, y, x]
                matches.append((cell, match_id))
                if match_id != 0:
                    p_centroid = ndimage.center_of_mass(
                        automatic_segmentation[slice],
                        labels = automatic_segmentation[slice],
                        index = match_id
                    )
                    p_y = int(np.rint(p_centroid[0]))
                    p_x = int(np.rint(p_centroid[1]))
                    gt = (slice, y, x)
                    p = (slice, p_y, p_x)
                    lookup_dict[f'gt_{gt}'] = p
                    lookup_dict[f'p_{p}'] = gt
            matches = np.asarray(matches)
            uniques_gt = np.unique(matches[:,0], return_counts = True)
            uniques_pred = np.unique(matches[:,1], return_counts = True)
            counts_pred = uniques_pred[1]
            fn_current = 0
            if uniques_pred[0][0] == 0:
                fn_current = uniques_pred[1][0]
                counts_pred = counts_pred[1:]
            sc_current = sum(counts_pred - np.ones_like(counts_pred))
            fp += len(uniques_pred[0]) + fn_current + sc_current - len(uniques_gt[0])
            fn += fn_current
            sc += sc_current
            
        def connection_has_match(connection:tuple, prediction:bool):
            if not prediction:
                cell_gt_1 = connection[0]
                cell_gt_2 = connection[1]
                if not f"gt_{tuple(cell_gt_1)}" in lookup_dict or not f"gt_{tuple(cell_gt_2)}" in lookup_dict:
                    return False
                cell_p_1 = lookup_dict[f"gt_{tuple(cell_gt_1)}"]
                cell_p_2 = lookup_dict[f"gt_{tuple(cell_gt_2)}"]
                return is_connection(cell_p_1, cell_p_2, not prediction)
            else:
                cell_p_1 = connection[0]
                cell_p_2 = connection[1]
                if not f"p_{tuple(cell_p_1)}" in lookup_dict or not f"p_{tuple(cell_p_2)}" in lookup_dict:
                    return False
                cell_gt_1 = lookup_dict[f"p_{tuple(cell_p_1)}"]
                cell_gt_2 = lookup_dict[f"p_{tuple(cell_p_2)}"]
                return is_connection(cell_gt_1, cell_gt_2, not prediction)
    
        def is_connection(cell_1, cell_2, prediction:bool):
            cell_1 = np.asarray(cell_1)
            cell_2 = np.asarray(cell_2)
            if prediction:
                index = np.where(np.all(automatic_tracks[:,1:] == cell_1, axis = 1))[0]
                if len(index) == 0:
                    return False
                return np.array_equal(automatic_tracks[index[0]+1,1:], cell_2)
            index = np.where(np.all(corrected_tracks[:,1:] == cell_1, axis = 1))[0]
            if len(index) == 0:
                return False
            return np.array_equal(corrected_tracks[index[0]+1,1:], cell_2)

        for line in range(len(corrected_tracks) - 1):
            if corrected_tracks[line, 0] != corrected_tracks[line+1,0]:
                continue
            cell_1 = corrected_tracks[line, 1:]
            cell_2 = corrected_tracks[line+ 1, 1:]
            connection = (cell_1, cell_2)
            if not connection_has_match(connection, False):
                print(f"added edge for {connection}")
                ae += 1
                
        for line in range(len(automatic_tracks) -1):
            if automatic_tracks[line, 0] != automatic_tracks[line +1,0]:
                continue
            cell_1 = automatic_tracks[line, 1:]
            cell_2 = automatic_tracks[line+1,1:]
            connection = (cell_1, cell_2)
            if not connection_has_match(connection, True):
                print(f"deleted edge for {connection}")
                de += 1 

        print(f"False Negatives: {fn}")
        print(f"False Positives: {fp}")
        print(f"Split Cells: {sc}")
        print(f"Added Edges: {ae}")
        print(f"Deleted Edges: {de}")
        
        fault_value = fp + fn * 10 + de + ae * 1.5 + sc * 5
        
        print(f"Fault value: {fault_value}")
        #print(lookup_dict)
        #return
        
        
        
        
        
        
        
        """data = []
        for i in range(len(corrected_segmentation)):
            data.append((corrected_tracks[i], automatic_tracks[i]))
        
        def match_cells(labels1, labels2):
            "#""
            Takes two slices as argument. Both should be from the same timestep. Returns list of matching cell ids.
            "#""
            pairs = []
            paired = []
            
            for id in np.unique(labels1):
                if id == 0:
                    continue
                
                ## try to find pair by looking at the centroid
                centroid = ndimage.center_of_mass(
                    labels1, labels = labels1, index = id
                )
                match_id = labels2[int(np.rint(centroid[0])), int(np.rint(centroid[1]))]
                if match_id > 0:
                    pairs.append((id, match_id))
                    paired.append(match_id)
                    continue
                pairs.append((id, -1))
                
            for id in np.unique(labels2):
                if not id in paired:
                    pairs.append((-1, id))
                
            return pairs
        
        with Pool(AMOUNT_OF_PROCESSES) as p:
            matched_cells = p.starmap(match_cells, data)
            
        for line in matched_cells:
            arr = np.asarray(line)
            unique, counts = np.unique(arr[:,0], return_counts = True)
            fp += counts[0] # should always be counts of -1
            unique, counts = np.unique(arr[:,1], return_counts = True)
            fn += counts[0]
        
        data = []
        def extend_tracks(tracks, segmentation):
            extended_tracks = np.asarray([[]])
            for line in tracks:
                seg_id = segmentation[line[1], line[2], line[3]]
                np.append(extended_tracks, [line[0],line[1],line[2],line[3],seg_id], axis=0)
            return extended_tracks
        
        extended_corrected_tracks = extend_tracks(corrected_tracks, corrected_segmentation)
        extended_automatic_tracks = extend_tracks(automatic_tracks, automatic_segmentation)
        for i in range(len(extended_corrected_tracks)):
            if extended_corrected_tracks[i,0] == extended_corrected_tracks[i+1,0]:
                data.append((extended_corrected_tracks[i], extended_corrected_tracks[i+1]), extended_automatic_tracks, matched_cells)
                
        def match_track_connections(connection, tracks, matched_cells):
            connection_id1 = connection[0][4]
            match_id1 = matched_cells[matched_cells[0].index(connection_id1)][1]
            connection_id2 = connection[1][4]
            match_id2 = matched_cells[matched_cells[0].index(connection_id2)][1]
            index1 = np.where(tracks, tracks[:,4] == match_id1)
            index2 = np.where(tracks, tracks[:,4] == match_id2)
            if index1[0] == index2[0] + 1:
                return index1[0]
            return -1
        
        with Pool(AMOUNT_OF_PROCESSES) as p:
            matched_indices = p.starmap(match_track_connections, data)
            
        for index in matched_indices:
            if index == -1:
                ae += 1
                
        for index in range(len(automatic_tracks)):
            if not index in matched_indices:
                de += 1
        
        hit_cells = []
        for line in corrected_tracks:
            if not (line[1], line[2], line[3]) in hit_cells:
                hit_cells.append((line[1], line[2], line[3]))
            else:
                sc += 1
            
        print(f"False negatives: {fn}")
        print("False positives: %s" % fp)
        print("Deleted edges: %s" % de)
        print("Added edges: %s" % ae)
        print("Split cells: %s" % sc)
        
        fault_value = fp + fn * 10 + de + ae * 1.5 + sc * 5
        
        print("Fault value: %s" % fault_value)
        
        return
        
        
        
        current_frame = int(self.viewer.dims.point[0])
        frame_range = [current_frame]
        try:
            selected_limit = int(self.lineedit_limit_evaluation.text())
        except ValueError:
            notify("Please use integer instead of text")
            return
        if selected_limit > current_frame:
            frame_range.append(selected_limit)
        else:
            frame_range.insert(0, selected_limit)
            
        frames = [current_frame, frame_range, len(automatic_segmentation) - 1]
        self._display_evaluation_result("Evaluation of Tracking", results, frames)
        self.results_window.show()
        print("Opening results window")"""
        
            
    def _display_evaluation_result(self, title, results, frames):
        self.results_window = ResultsWindow(title, results, frames)
        
class ResultsWindow(QWidget):
    def __init__(self, title, results, frames):
        """
        Parameters
        ----------
        title : str
            Title of the window
        results : arr
            2d array holding the results
        frames : list
            list holding the selected, a list of both lower and higher bound frame, and the last frame
        """
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.setWindowTitle(title)
        table_widget = QTableWidget(4,4)
        self.layout().addWidget(table_widget)
        try:
            self.setStyleSheet(napari.qt.get_stylesheet(theme = "dark"))
        except TypeError:
            pass
        
        table_widget.setCellWidget(0,0,QLabel("Evaluation Interval"))
        table_widget.setCellWidget(0,1,QLabel("IoU Score"))
        table_widget.setCellWidget(0,2,QLabel("DICE Score"))
        table_widget.setCellWidget(0,3,QLabel("F1 score"))
        table_widget.setCellWidget(1,0,QLabel("0 - {}".format(frames[2])))
        table_widget.setCellWidget(1,1,QLabel(str(results[0,0])))
        table_widget.setCellWidget(1,2,QLabel(str(results[0,1])))
        table_widget.setCellWidget(1,3,QLabel(str(results[0,2])))
        table_widget.setCellWidget(2,0,QLabel("{} - {}".format(frames[1][0], frames[1][1] - 1)))
        table_widget.setCellWidget(2,1,QLabel(str(results[1,0])))
        table_widget.setCellWidget(2,2,QLabel(str(results[1,1])))
        table_widget.setCellWidget(2,3,QLabel(str(results[1,2])))
        table_widget.setCellWidget(3,0,QLabel(str(frames[0])))
        table_widget.setCellWidget(3,1,QLabel(str(results[2,0])))
        table_widget.setCellWidget(3,2,QLabel(str(results[2,1])))
        table_widget.setCellWidget(3,3,QLabel(str(results[2,2])))
