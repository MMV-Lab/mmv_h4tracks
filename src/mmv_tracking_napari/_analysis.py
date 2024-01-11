import multiprocessing
from multiprocessing import Pool
import numpy as np
from scipy import ndimage
import platform

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
from qtpy.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from napari.qt.threading import thread_worker
import napari

from mmv_tracking_napari import IOU_THRESHOLD
from ._grabber import grab_layer
from mmv_tracking_napari._logger import notify, handle_exception
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
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.parent = parent
        self.viewer = parent.viewer
        self.setLayout(QVBoxLayout())
        self.setWindowTitle("Analysis")
        try:
            self.setStyleSheet(napari.qt.get_stylesheet(theme="dark"))
        except TypeError:
            self.setStyleSheet(napari.qt.get_stylesheet(theme_id="dark"))

        ### QObjects

        # Labels
        label_min_movement = QLabel("Movement Minmum")
        label_min_duration = QLabel("Minimum Track Length")
        label_metric = QLabel("Metric:")
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

        btn_evaluate_segmentation.clicked.connect(self.call_evaluate_segmentation)
        btn_evaluate_tracking.clicked.connect(self.call_evaluate_tracking)

        # Comboboxes
        self.combobox_plots = QComboBox()
        self.combobox_plots.addItems(
            ["Speed", "Size", "Direction", "Euclidean distance", "Accumulated distance"]
        )

        # Horizontal lines
        line = QWidget()
        line.setFixedHeight(4)
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line.setStyleSheet("background-color: #c0c0c0")
        line2 = QWidget()
        line2.setFixedHeight(4)
        line2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line2.setStyleSheet("background-color: #c0c0c0")

        # Checkboxes
        checkbox_speed = QCheckBox("Speed")
        checkbox_size = QCheckBox("Size")
        checkbox_direction = QCheckBox("Direction")
        checkbox_euclidean_distance = QCheckBox("Euclidean distance")
        checkbox_accumulated_distance = QCheckBox("Accumulated distance")
        checkbox_velocity = QCheckBox("Velocity")

        self.checkboxes = [
            checkbox_speed,
            checkbox_size,
            checkbox_direction,
            checkbox_euclidean_distance,
            checkbox_accumulated_distance,
            checkbox_velocity,
        ]

        # Line Edits
        self.lineedit_movement = QLineEdit("")
        self.lineedit_track_duration = QLineEdit("")
        self.lineedit_limit_evaluation = QLineEdit("0")

        ### Organize objects via widgets
        content = QWidget()
        content.setLayout(QGridLayout())
        content.layout().addWidget(label_plotting, 3, 0)
        content.layout().addWidget(label_metric, 4, 0)
        content.layout().addWidget(self.combobox_plots, 4, 1)
        content.layout().addWidget(btn_plot, 4, 2)
        content.layout().addWidget(line, 5, 0, 1, -1)
        content.layout().addWidget(label_filter, 6, 0)
        content.layout().addWidget(label_min_movement, 7, 0)
        content.layout().addWidget(self.lineedit_movement, 7, 1)
        content.layout().addWidget(label_min_duration, 8, 0)
        content.layout().addWidget(self.lineedit_track_duration, 8, 1)
        content.layout().addWidget(label_export, 9, 0)
        content.layout().addWidget(checkbox_speed, 10, 0)
        content.layout().addWidget(checkbox_velocity, 10, 1)
        content.layout().addWidget(checkbox_size, 10, 2)
        content.layout().addWidget(checkbox_euclidean_distance, 11, 0)
        content.layout().addWidget(checkbox_accumulated_distance, 11, 1)
        content.layout().addWidget(checkbox_direction, 11, 2)
        content.layout().addWidget(btn_export, 10, 3,2,1)
        content.layout().addWidget(line2, 12, 0, 1, -1)
        content.layout().addWidget(label_evaluation, 13, 0)
        content.layout().addWidget(label_eval_range, 14, 0)
        content.layout().addWidget(self.lineedit_limit_evaluation, 14, 1)
        content.layout().addWidget(btn_evaluate_segmentation, 15, 0)
        content.layout().addWidget(btn_evaluate_tracking, 15, 1)

        self.layout().addWidget(content)

    def _calculate_speed(self, tracks):
        """
        Calculates the speed of each track

        Parameters
        ----------
        tracks : ??     # nd array
            ??          # (N,4) shape array, which follows napari's trackslayer format (ID, z, y, x)

        Returns
        -------
        nd array
            # ??  (N,3) shape array, which contains the ID, average speed and standard deviation of speed
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
        tracks : ??     # nd array
            ??          # (N,4) shape array, which follows napari's trackslayer format (ID, z, y, x)
        segmentation : ??   # nd array
            ??              # (z,y,x) shape array, which follows napari's labelslayer format

        Returns
        -------
        speeds: nd array
            # ??  (N,3) shape array, which contains the ID, average speed and standard deviation of speed
        """           
        track_and_segmentation = []
        for unique_id in np.unique(tracks[:, 0]):
            track_and_segmentation.append(
                [tracks[np.where(tracks[:, 0] == unique_id)], segmentation]
            )

        if self.parent.rb_eco.isChecked():
            AMOUNT_OF_PROCESSES = np.maximum(1, int(multiprocessing.cpu_count() * 0.4))
        else:
            AMOUNT_OF_PROCESSES = np.maximum(1, int(multiprocessing.cpu_count() * 0.8))
        print("Running on {} processes max".format(AMOUNT_OF_PROCESSES))

        with Pool(AMOUNT_OF_PROCESSES) as p:
            sizes = p.starmap(calculate_size_single_track, track_and_segmentation)

        return np.array(sizes)

    def _calculate_direction(self, tracks):
        """
        Calculates the angle [°] of each trajectory

        Parameters
        ----------
        tracks : ??     # nd array
            ??          # (N,4) shape array, which follows napari's trackslayer format (ID, z, y, x)

        Returns
        -------
        sizes: nd array
            # ??  (N,3) shape array, which contains the ID, average speed and standard deviation of speed
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
        tracks : ??     # nd array
            ??          # (N,4) shape array, which follows napari's trackslayer format (ID, z, y, x)

        Returns
        -------
        euclidean_distances: nd array
            # ??  (N,3) shape array, which contains the ID, average euclidean distance and standard deviation of euclidean distance
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
                euclidean_distances = np.array(
                    [[id, euclidean_distance, len(track)]]
                )
        return euclidean_distances
    
    def _calculate_velocity(self, tracks):
        euclidean_distances = self._calculate_euclidean_distance(tracks)
        for id in np.unique(tracks[:, 0]):
            euclidean_distance = np.delete(euclidean_distances, np.where(euclidean_distances[:,0] != id),0)
            track = np.delete(tracks, np.where(tracks[:, 0] != id), 0)
            directed_speed = np.around(euclidean_distance[0,1] / len(track), 3)
            try:
                directed_speeds = np.append(directed_speeds, [[id, directed_speed]], 0)
            except UnboundLocalError:
                directed_speeds = np.array([[id, directed_speed]])
        return directed_speeds

    def _calculate_accumulated_distance(self, tracks):
        """
        Calculates the accumulated distance of each trajectory

        Parameters
        ----------
        tracks : ??     # nd array
            ??          # (N,4) shape array, which follows napari's trackslayer format (ID, z, y, x)

        Returns
        -------
        accumulated_distances: nd array
            # ??  (N,3) shape array, which contains the ID, average accumulated distance and standard deviation of accumulated distance
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
                accumulated_distances = np.append(accumulated_distances, [[id, accumulated_distance, len(track)]], 0)
            except UnboundLocalError:
                accumulated_distances = np.array([[id, accumulated_distance, len(track)]])
        return accumulated_distances

    def _start_plot_worker(self):
        worker = self._sort_plot_data(self.combobox_plots.currentText())
        worker.returned.connect(self._plot)
        worker.start()

    def _plot(self, ret):
        """
        ??

        Parameters
        ----------
        ?? : ??         # ?? can we rename ret/what is ret?
            ??          

        """              
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

        axes.scatter(
            results[:, 1], results[:, 2], c=np.array([[0, 0.240802676, 0.70703125, 1]])
        )

        xabs_max = abs(max(axes.get_xlim(), key = abs))
        yabs_max = abs(max(axes.get_ylim(), key = abs))
        axes.set_xlim(xmin = -xabs_max, xmax = xabs_max)
        axes.set_ylim(ymin = -yabs_max, ymax = yabs_max)
        axes.set_title(title, {"fontsize": 18, "color": "white"})
        axes.set_xlabel(ret["x_label"])
        axes.set_ylabel(ret["y_label"])

        canvas = FigureCanvas(fig)
        self.parent.plot_window = QWidget()
        self.parent.plot_window.setLayout(QVBoxLayout())
        self.parent.plot_window.layout().addWidget(QLabel(ret["Description"]))
        self.selector = Selector(self, axes, results)

        self.parent.plot_window.layout().addWidget(canvas)
        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self.selector.apply)
        self.parent.plot_window.layout().addWidget(btn_apply)
        print("Showing plot window")
        self.parent.plot_window.show()

    @thread_worker(connect={"errored": handle_exception})
    def _sort_plot_data(self, metric):
        """
        Create dictionary holding metric data and results
        Parameters
        ----------
        metric : ??     
            ??       

        Returns
        -------
        retval: ??
            ??      
        """
        tracks_layer = grab_layer(
            self.parent.viewer, self.parent.combobox_tracks.currentText()
        )
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
            print("Plotting direction")
            retval.update({"Description": "Scatterplot: Travel direction & Distance"})
            retval.update({"Results": self._calculate_direction(tracks_layer.data)})
            retval.update({"x_label": "Δx", "y_label": "Δy"})
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

    @thread_worker(connect={"errored": handle_exception})
    def _export(self, file, metrics):
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

    def _filter_tracks_by_parameters(self, tracks):
        distances = self._calculate_accumulated_distance(tracks)
        if self.lineedit_movement.text() == "":
            min_movement = 0
            movement_mask = np.unique(tracks[:, 0])
        else:
            try:
                min_movement = int(self.lineedit_movement.text())
            except ValueError as exc:
                raise ValueError("Movement minimum can't be converted to int") from exc
            movement_mask = distances[
                np.where(distances[:, 1] >= min_movement)[0], 0
            ]

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
    ):                          # ?? muss ich mir in Ruhe anschauen
        
        """
        ??

        Parameters
        ----------
        ?? : ??     
            ??       

        Returns
        -------
        ??: ??
            ??      
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
            # [rows.append(value) for value in valid_values]
            for value in valid_values:
                rows.append(value)

            if not np.array_equal(np.unique(tracks[:, 0]), filtered_mask):
                rows.append([None])
                rows.append([None])
                rows.append(["Cells not matching the filters"])

                [rows.append(value) for value in invalid_values]

        return rows

    def _extend_metrics(self, tracks, selected_metrics, filtered_mask, duration):   # ?? muss ich mir in Ruhe anschauen, durch die Formatierung sind es leider ~230 Zeilen geworden,  
                                                                                    # ?? können wir noch irgendwo hier einsparen?
        """
        ??

        Parameters
        ----------
        ?? : ??     
            ??       

        Returns
        -------
        ??: ??
            ??      
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
            metrics.extend(
                [
                    "Average euclidean distance [# pixels]"
                ]
            )
            individual_metrics.extend(
                ["Euclidean distance [# pixels]"]
            )
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
            metrics.extend(["Average velocity [#pixels/frame]",
                    "Standard deviation of velocity [# pixels/frame]"])
            individual_metrics.extend(["Velocity [# pixels/frame]"])
            all_values.extend([np.around(np.average(velocity[:, 1]),3),
                               np.around(np.std(velocity[:,1]),3)])
            valid_values.extend([np.around(np.average([velocity[i, 1] for i in range(len(velocity)) if velocity[i, 0] in filtered_mask]),3),
                    np.around(
                        np.std(
                            [
                                velocity[i, 1]
                                for i in range(len(velocity))
                                if velocity[i, 0] in filtered_mask
                            ]
                        ),
                        3,
                    ),])
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
                # all_values.append(np.around(np.average(directness[:, 1]), 3))
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
        """
        Calculate each selected metric for both valid tracks that match the filter criteria and invalid tracks that do not match the filter criteria
        """
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
                value.extend([metrics["Direction"][id][3]])     # using index 3 as 1 and 2 are solely used for plotting
            if "Euclidean distance" in metrics:
                value.extend(
                    [
                        metrics["Euclidean distance"][id][1],
                        metrics["Euclidean distance"][id][3],   # using index 3 as 2 is solely used for plotting
                    ]
                )
            if "Velocity" in metrics:
                value.append(metrics["Velocity"][id][1])
            if "Accumulated distance" in metrics:
                value.append(metrics["Accumulated distance"][id][1])
            if "Directness" in metrics:
                value.append(metrics["Directness"][id][1])

            if id in filtered_mask:
                valid_values.append(value)
            else:
                invalid_values.append(value)

        return valid_values, invalid_values

    def call_evaluate_segmentation(self):
        try:
            gt_seg = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            ).data
        except ValueError as exc:
            handle_exception(exc)
        eval_seg = self.parent.initial_layers[0]
        results, frames = self.evaluate_segmentation(gt_seg, eval_seg)
        self._display_evaluation_result("Evaluation of Segmentation", results, frames)
        self.results_window.show()
        print("Opening results window")

    def evaluate_segmentation(self, gt_seg, eval_seg):
        # evaluates segmentation based on changes made by user
        current_frame = int(self.viewer.dims.point[0])
        try:
            selected_limit = int(self.lineedit_limit_evaluation.text())
        except ValueError:
            notify("Please use integer instead of text")
            return
        if selected_limit > current_frame:
            frame_range = list(range(current_frame, selected_limit + 1))
        else:
            frame_range = list(range(selected_limit, current_frame + 1))
        frames = [current_frame, (frame_range[0], frame_range[-1]), len(eval_seg) - 1]

        ### FOR CURRENT FRAME
        single_iou = self.get_iou(gt_seg[current_frame], eval_seg[current_frame])
        single_dice = self.get_dice(gt_seg[current_frame], eval_seg[current_frame])
        single_f1 = self.get_f1(gt_seg[current_frame], eval_seg[current_frame])

        ### FOR SOME FRAMES
        some_iou = self.get_iou(gt_seg[frame_range], eval_seg[frame_range])
        some_dice = self.get_dice(gt_seg[frame_range], eval_seg[frame_range])
        some_f1 = self.get_f1(gt_seg[frame_range], eval_seg[frame_range])

        ### FOR ALL FRAMES
        all_iou = self.get_iou(gt_seg, eval_seg)
        all_dice = self.get_dice(gt_seg, eval_seg)
        all_f1 = self.get_f1(gt_seg, eval_seg)

        results = np.asarray(
            [
                [all_iou, all_dice, all_f1],
                [some_iou, some_dice, some_f1],
                [single_iou, single_dice, single_f1],
            ]
        )
        return results, frames

    def get_intersection(self, seg1, seg2):
        # takes multidimensional numpy arrays and returns their intersection
        return np.sum(np.logical_and(seg1, seg2).flat)

    def get_union(self, seg1, seg2):
        # takes multidimensional numpy arrays and returns their union
        return np.sum(np.logical_or(seg1, seg2).flat)

    def get_iou(self, seg1, seg2):
        # calculate IoU for two given segmentations
        intersection = self.get_intersection(seg1, seg2)
        union = self.get_union(seg1, seg2)
        return intersection / union

    def get_dice(self, seg1, seg2):
        # calculate DICE score for two given segmentations
        intersection = self.get_intersection(seg1, seg2)
        union = self.get_union(seg1, seg2)
        return (2 * intersection) / (np.count_nonzero(seg1) + np.count_nonzero(seg2))

    def get_f1(self, seg1, seg2):
        # calculate F1 score for two given segmentations
        intersection = self.get_intersection(seg1, seg2)
        union = self.get_union(seg1, seg2)
        return (2 * intersection) / (intersection + union)

    def adjust_track_centroids(self):
        # adjusts tracks to centroid of cells
        print(f"segmentation: {self.parent.combobox_segmentation.currentText()}")
        print(f"tracks: {self.parent.combobox_tracks.currentText()}")
        try:
            tracks_layer = grab_layer(
                self.parent.viewer, self.parent.combobox_tracks.currentText()
            )
        except ValueError as exc:
            handle_exception(exc)
        try:
            segmentation = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            ).data
        except ValueError as exc:
            handle_exception(exc)
        tracks = tracks_layer.data
        for row_trackslayer in tracks:
            _, z, y, x = row_trackslayer
            segmentation_id = segmentation[z, y, x]
            if segmentation_id == 0:
                continue
            centroid = ndimage.center_of_mass(
                segmentation[z], labels=segmentation[z], index=segmentation_id
            )
            y_new = int(np.rint(centroid[0]))
            x_new = int(np.rint(centroid[1]))
            if x_new != x or y_new != y:
                row_trackslayer[2] = y_new
                row_trackslayer[3] = x_new

        tracks_layer.data = tracks

    def call_evaluate_tracking(self):
        """
        Evaluates manipulated tracks, compares them to ground truth and calculates fault value
        """ 
        automatic_tracks = self.parent.initial_layers[1]
        try:
            corrected_tracks = grab_layer(
                self.parent.viewer, self.parent.combobox_tracks.currentText()
            ).data
        except ValueError as exc:
            handle_exception(exc)
        automatic_segmentation = self.parent.initial_layers[0]
        try:
            corrected_segmentation = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            ).data
        except ValueError as exc:
            handle_exception(exc)
        fault_value = self.evaluate_tracking(
            gt_seg=corrected_segmentation,
            eval_seg=automatic_segmentation,
            gt_tracks=corrected_tracks,
            eval_tracks=automatic_tracks,
        )
        print(f"Fault value: {fault_value}")

    def evaluate_tracking(self, gt_seg, eval_seg, gt_tracks, eval_tracks):
        """
        ??

        Parameters
        ----------
        ?? : ??     
            ??       

        Returns
        -------
        ??: ??
            ??      
        """             
        self.adjust_track_centroids()
        fp = self.get_false_positives(gt_seg, eval_seg)
        fn = self.get_false_negatives(gt_seg, eval_seg)
        sc = self.get_split_cells(gt_seg, eval_seg)
        de = self.get_removed_edges(gt_seg, eval_seg, gt_tracks, eval_tracks)
        ae = self.get_added_edges(gt_seg, eval_seg, gt_tracks, eval_tracks)

        fault_value = fp + fn * 10 + de + ae * 1.5 + sc * 5 # here you can set own weights for the different faults

        return fault_value

        ### MATCHING FOR CELLS BASED ON IOU > .4, split cells >= .2!

    def get_false_positives(self, gt_seg, eval_seg):
        # calculates amount of false positives for given segmentation
        fp = 0
        if np.array_equal(gt_seg, eval_seg):
            return fp
        if self.parent.rb_eco.isChecked():
            AMOUNT_OF_PROCESSES = np.maximum(1, int(multiprocessing.cpu_count() * 0.4))
        else:
            AMOUNT_OF_PROCESSES = np.maximum(1, int(multiprocessing.cpu_count() * 0.8))

        segmentations = []
        for i in range(len(gt_seg)):
            segmentations.append([gt_seg[i], eval_seg[i]])

        with Pool(AMOUNT_OF_PROCESSES) as p:
            for result in p.starmap(get_false_positives_slice, segmentations):
                fp += result
        print(f"False Positives: {fp}")
        return fp

    def get_false_negatives(self, gt_seg, eval_seg):
        # calculates amount of false negatives for given segmentation
        fn = 0
        if np.array_equal(gt_seg, eval_seg):
            return fn
        if self.parent.rb_eco.isChecked():
            AMOUNT_OF_PROCESSES = np.maximum(1, int(multiprocessing.cpu_count() * 0.4))
        else:
            AMOUNT_OF_PROCESSES = np.maximum(1, int(multiprocessing.cpu_count() * 0.8))

        segmentations = []
        for i in range(len(gt_seg)):
            segmentations.append([gt_seg[i], eval_seg[i]])

        with Pool(AMOUNT_OF_PROCESSES) as p:
            for result in p.starmap(get_false_negatives_slice, segmentations):
                fn += result
        print(f"False Negatives: {fn}")
        return fn

    def get_split_cells(self, gt_seg, eval_seg):
        #  calculates amount of split cells for given segmentation
        sc = 0
        if np.array_equal(gt_seg, eval_seg):
            return sc
        if self.parent.rb_eco.isChecked():
            AMOUNT_OF_PROCESSES = np.maximum(1, int(multiprocessing.cpu_count() * 0.4))
        else:
            AMOUNT_OF_PROCESSES = np.maximum(1, int(multiprocessing.cpu_count() * 0.8))

        segmentations = []
        for i in range(len(gt_seg)):
            segmentations.append([gt_seg[i], eval_seg[i]])

        with Pool(AMOUNT_OF_PROCESSES) as p:
            for result in p.starmap(get_split_cells_layer, segmentations):
                sc += result
        print(f"Split Cells: {sc}")
        return sc

    def get_added_edges(self, gt_seg, eval_seg, gt_tracks, eval_tracks):
        # calculates amount of added edges for given segmentation and tracks
        self.adjust_track_centroids()
        ae = 0
        if np.array_equal(gt_tracks, eval_tracks):
            return ae
        connections = []
        # find all the connections between 2 adjacent frames in ground truth
        for i in range(len(gt_tracks) - 1):
            if gt_tracks[i][0] == gt_tracks[i + 1][0]:
                connections.append((gt_tracks[i][1:4], gt_tracks[i + 1][1:4]))

        # check if connection also exists in eval
        for connection in connections:
            gt_id1 = get_id_from_track(eval_seg, connection[0])
            gt_id2 = get_id_from_track(eval_seg, connection[1])
            id1 = get_match_cell(gt_seg, eval_seg, gt_id1, connection[0][0])
            id2 = get_match_cell(gt_seg, eval_seg, gt_id2, connection[1][0])
            if id1 == 0 or id2 == 0:
                ae += 1
                continue
            # calculate centroids of the candidate cells in eval
            # centroid1 is in earlier frame, centroid2 is in later frame
            centroid1 = ndimage.center_of_mass(
                eval_seg[connection[0][0]], labels=eval_seg[connection[0][0]], index=id1
            )
            centroid1 = [
                connection[0][0],
                int(np.rint(centroid1[0])),
                int(np.rint(centroid1[1])),
            ]
            centroid2 = ndimage.center_of_mass(
                eval_seg[connection[1][0]], labels=eval_seg[connection[1][0]], index=id2
            )
            centroid2 = [
                connection[1][0],
                int(np.rint(centroid2[0])),
                int(np.rint(centroid2[1])),
            ]
            # check if the centroids are connected
            if not is_connected(eval_tracks, centroid1, centroid2):
                ae += 1
        print(f"Added Edges: {ae}")
        return ae

    def get_removed_edges(self, gt_seg, eval_seg, gt_tracks, eval_tracks):
        # calculates amount of removed edges for given segmentation and tracks
        self.adjust_track_centroids()
        de = 0
        if np.array_equal(gt_tracks, eval_tracks):
            return de
        connections = []
        for i in range(len(eval_tracks) - 1):
            if eval_tracks[i][0] == eval_tracks[i + 1][0] and eval_tracks[i][1] + 1 == eval_tracks[i + 1][1]:
                connections.append((eval_tracks[i][1:4], eval_tracks[i + 1][1:4]))

        for connection in connections:
            # explain what eval_id is what
            eval_id1 = get_id_from_track(gt_seg, connection[0])
            eval_id2 = get_id_from_track(gt_seg, connection[1])
            id1 = get_match_cell(eval_seg, gt_seg, eval_id1, connection[0][0])
            id2 = get_match_cell(eval_seg, gt_seg, eval_id2, connection[1][0])
            if id1 == 0 or id2 == 0:
                de += 1
                continue
            # explain what centroid is what
            centroid1 = ndimage.center_of_mass(
                gt_seg[connection[0][0]], labels=gt_seg[connection[0][0]], index=id1
            )
            centroid1 = [
                connection[0][0],
                int(np.rint(centroid1[0])),
                int(np.rint(centroid1[1])),
            ]
            centroid2 = ndimage.center_of_mass(
                gt_seg[connection[1][0]], labels=gt_seg[connection[1][0]], index=id2
            )
            centroid2 = [
                connection[1][0],
                int(np.rint(centroid2[0])),
                int(np.rint(centroid2[1])),
            ]
            if not is_connected(gt_tracks, centroid1, centroid2):
                de += 1
        print(f"Deleted Edges: {de}")
        return de

    def _display_evaluation_result(self, title, results, frames):
        self.results_window = ResultsWindow(title, results, frames)


def get_specific_intersection(slice1, id1, slice2, id2):
    #Intersection of the given cells in the corresponding slices
    return np.sum((slice1 == id1) & (slice2 == id2))


def get_specific_union(slice1, id1, slice2, id2):
    return np.sum((slice1 == id1) | (slice2 == id2))


def get_specific_iou(slice1, id1, slice2, id2):
    intersection = get_specific_intersection(slice1, id1, slice2, id2)
    union = get_specific_union(slice1, id1, slice2, id2)
    return intersection / union


def get_id_from_track(label_layer, track):
    z, y, x = track
    return label_layer[z, y, x]


def get_match_cell(base_layer, compare_layer, base_id, slice_id):
    # base_layer is layer that has cell, compare_layer is layer that is searched for cell
    # return id of matched cell
    # check for highest iou (above threshold)
    base_slice = base_layer[slice_id]
    compare_slice = compare_layer[slice_id]
    indices_of_id = np.where(base_slice == base_id)
    compare_ids = np.unique(compare_slice[indices_of_id])
    ious = []
    for compare_id in compare_ids:
        if compare_id == 0:
            continue
        iou = [
            compare_id,
            get_specific_iou(compare_slice, compare_id, base_slice, base_id),
        ]
        ious.append(iou)
    ious = np.array(ious)
    if len(ious) == 0:
        return 0
    max_iou = max(ious[:, 1])
    if max_iou < IOU_THRESHOLD:
        return 0
    return int(ious[np.where(ious[:, 1] == max_iou), 0][0][0])


def is_connected(tracks, centroid1, centroid2):
    # centroid1 is in earlier frame, centroid2 is in later frame
    for i in range(len(tracks) - 1):
        # if (
        #     centroid1[0] == tracks[i, 1]
        #     and centroid1[1] == tracks[i, 2]
        #     and centroid1[2] == tracks[i, 3]
        # ):
        if all(centroid1[j] == tracks[i, j+1] for j in range(3)):   # ?? muss ich noch testen, aber vlt. sparen wir hier so ein bisschen was ein
            # return (                                              # sieht schonmal sehr cool aus, falls es funktioniert
            #     centroid2[0] == tracks[i + 1, 1]
            #     and centroid2[1] == tracks[i + 1, 2]
            #     and centroid2[2] == tracks[i + 1, 3]
            # )
            return all(centroid2[j] == tracks[i + 1, j+1] for j in range(3))
    return False


def calculate_size_single_track(track, segmentation):
    id = track[0, 0]
    sizes = []
    for line in track:
        _, z, y, x = line
        seg_id = segmentation[z, y, x]
        sizes.append(len(np.where(segmentation[z] == seg_id)[0]))
    average = np.around(np.average(sizes), 3)
    std_deviation = np.around(np.std(sizes), 3)
    return [id, average, std_deviation]


def get_false_positives_slice(gt_slice, eval_slice):
    # calculates false positives for the given pair of slices
    fp = 0
    if np.array_equal(gt_slice, eval_slice):
        print("layers are equal")
        return fp
    for id in np.unique(eval_slice):
        if id == 0:
            print("skipping check for background")
            continue
        # get IoU on all cells in gt at locations of id in eval
        indices_of_id = np.where(eval_slice == id)
        gt_ids = np.unique(gt_slice[indices_of_id])
        ious = []
        for gt_id in gt_ids:
            if gt_id == 0:
                # skipping IoU calculation with background
                continue
            iou = get_specific_iou(gt_slice, gt_id, eval_slice, id)
            ious.append(iou)
        if len(ious) < 1:
            fp += 1
            print("No overlapping cell, fp")
            continue
        if max(ious) > 0.4:
            print("overlap high enough")
            continue
        ious.remove(max(ious))
        if len(ious) < 1 or max(ious) < 0.2:
            print("single low overlap or multiple very low overlaps, fp")
            fp += 1
    return fp


def get_false_negatives_slice(gt_slice, eval_slice):
    # calculates false negatives for the given pair of slices
    fn = 0
    if np.array_equal(gt_slice, eval_slice):
        return fn
    for id in np.unique(gt_slice):
        if id == 0:
            continue
        # get IoU on all cells in eval at locations of id in gt
        indices_of_id = np.where(gt_slice == id)
        eval_ids = np.unique(eval_slice[indices_of_id])
        max_iou = -1
        highest_match_eval_id = -1
        for eval_id in eval_ids:
            if eval_id == 0:
                continue
            iou = get_specific_iou(gt_slice, id, eval_slice, eval_id)
            if iou > max_iou:
                max_iou = iou
                highest_match_eval_id = eval_id
        if not max_iou > 0.4:
            fn += 1
            print("Largest IoU too small")
            continue
        indices_of_reverse_id = np.where(eval_slice == eval_id)
        reverse_ids = np.unique(gt_slice[indices_of_reverse_id])
        ious = []
        for reverse_id in reverse_ids:
            if reverse_id == 0:
                continue
            iou = get_specific_iou(gt_slice, reverse_id, eval_slice, eval_id)
            ious.append(iou)
        if len(ious) < 2:
            print("Reverse IoU only has one match")
            continue
        if max(ious) > max_iou:
            fn += 1
            print("Largest IoU large enough, but reverse IoU has larger")
            continue
        max_reverse_iou = max(ious)
        ious.remove(max(ious))
        if max(ious) == max_reverse_iou:
            fn += 0.5
    if int(fn) != fn:
        raise ValueError("False negatives don't sum up to whole integer")
    return int(fn)


def get_split_cells_layer(gt_slice, eval_slice):
    sc = 0
    if np.array_equal(gt_slice, eval_slice):
        return sc
    for id in np.unique(eval_slice):
        if id == 0:
            continue
        # get IoU on all cells in gt at locations of id in eval
        indices_of_id = np.where(eval_slice == id)
        gt_ids = np.unique(gt_slice[indices_of_id])
        ious = []
        for gt_id in gt_ids:
            if gt_id == 0:
                continue
            iou = get_specific_iou(gt_slice, gt_id, eval_slice, id)
            ious.append(iou)
        if len(ious) < 2:  # or max(ious) > .4:
            print("cell has less than two matches")
            continue
        ious.remove(max(ious))
        if max(ious) >= 0.2:
            print("second match has large enough area, sc")
            sc += 1
    return sc


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
        self.setMinimumWidth(500)
        table_widget = QTableWidget(4, 4)
        self.layout().addWidget(table_widget)
        try:
            self.setStyleSheet(napari.qt.get_stylesheet(theme="dark"))
        except TypeError:
            pass

        table_widget.setCellWidget(0, 0, QLabel("Evaluation Interval"))
        table_widget.setCellWidget(0, 1, QLabel("IoU Score"))
        table_widget.setCellWidget(0, 2, QLabel("DICE Score"))
        table_widget.setCellWidget(0, 3, QLabel("F1 score"))
        table_widget.setCellWidget(1, 0, QLabel("0 - {}".format(frames[2])))
        table_widget.setCellWidget(1, 1, QLabel(str(results[0, 0])))
        table_widget.setCellWidget(1, 2, QLabel(str(results[0, 1])))
        table_widget.setCellWidget(1, 3, QLabel(str(results[0, 2])))
        table_widget.setCellWidget(
            2, 0, QLabel("{} - {}".format(frames[1][0], frames[1][1]))
        )
        table_widget.setCellWidget(2, 1, QLabel(str(results[1, 0])))
        table_widget.setCellWidget(2, 2, QLabel(str(results[1, 1])))
        table_widget.setCellWidget(2, 3, QLabel(str(results[1, 2])))
        table_widget.setCellWidget(3, 0, QLabel(str(frames[0])))
        table_widget.setCellWidget(3, 1, QLabel(str(results[2, 0])))
        table_widget.setCellWidget(3, 2, QLabel(str(results[2, 1])))
        table_widget.setCellWidget(3, 3, QLabel(str(results[2, 2])))
