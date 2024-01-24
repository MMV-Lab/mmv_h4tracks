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
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from napari.qt.threading import thread_worker
import napari

from ._grabber import grab_layer
from mmv_tracking_napari._logger import handle_exception
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
        label_min_movement = QLabel("Movement Minmum:")
        label_min_duration = QLabel("Minimum Track Length:")
        label_metric = QLabel("Metric:")

        # Buttons
        btn_plot = QPushButton("Plot")
        btn_export = QPushButton("Export")

        btn_plot.setToolTip("Only displayed tracks are plotted")

        btn_plot.clicked.connect(self._start_plot_worker)
        btn_export.clicked.connect(self._start_export_worker)

        # Comboboxes
        self.combobox_plots = QComboBox()
        self.combobox_plots.addItems(
            [
                "Speed",
                "Size",
                "Direction",
                "Euclidean distance",
                "Accumulated distance",
                "Velocity",
                "Perimeter",
                "Eccentricity",
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
        self.lineedit_track_duration = QLineEdit("")

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
        export.layout().addWidget(h_spacer_4, 7, 0, 1, -1)
        export.layout().addWidget(btn_export, 8, 0, 1, -1)

        ### Organize objects via widgets
        content = QWidget()
        content.setLayout(QVBoxLayout())
        content.layout().addWidget(plot)
        content.layout().addWidget(export)
        content.layout().addWidget(v_spacer)

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
        AMOUNT_OF_PROCESSES = self.parent.get_process_limit()
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
                euclidean_distances = np.array([[id, euclidean_distance, len(track)]])
        return euclidean_distances

    def _calculate_velocity(self, tracks):
        euclidean_distances = self._calculate_euclidean_distance(tracks)
        for id in np.unique(tracks[:, 0]):
            euclidean_distance = np.delete(
                euclidean_distances, np.where(euclidean_distances[:, 0] != id), 0
            )
            track = np.delete(tracks, np.where(tracks[:, 0] != id), 0)
            directed_speed = np.around(euclidean_distance[0, 1] / len(track), 3)
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
                accumulated_distances = np.append(
                    accumulated_distances, [[id, accumulated_distance, len(track)]], 0
                )
            except UnboundLocalError:
                accumulated_distances = np.array(
                    [[id, accumulated_distance, len(track)]]
                )
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

        xabs_max = abs(max(axes.get_xlim(), key=abs))
        yabs_max = abs(max(axes.get_ylim(), key=abs))
        axes.set_xlim(xmin=-xabs_max, xmax=xabs_max)
        axes.set_ylim(ymin=-yabs_max, ymax=yabs_max)
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
    ):  # ?? muss ich mir in Ruhe anschauen
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

    def _extend_metrics(
        self, tracks, selected_metrics, filtered_mask, duration
    ):  # ?? muss ich mir in Ruhe anschauen, durch die Formatierung sind es leider ~230 Zeilen geworden,
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
        valid_values, invalid_values = [[], []]
        for id in np.unique(tracks[:, 0]):
            value = [id]
            value.append(np.count_nonzero(tracks[:, 0] == id))
            if "Speed" in metrics:
                value.extend([metrics["Speed"][id][1], metrics["Speed"][id][2]])
            if "Size" in metrics:
                value.extend([metrics["Size"][id][1], metrics["Size"][id][2]])
            if "Direction" in metrics:
                value.extend(
                    [metrics["Direction"][id][3]]
                )  # using index 3 as 1 and 2 are solely used for plotting
            if "Euclidean distance" in metrics:
                value.extend(
                    [
                        metrics["Euclidean distance"][id][1],
                        metrics["Euclidean distance"][id][
                            3
                        ],  # using index 3 as 2 is solely used for plotting
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
