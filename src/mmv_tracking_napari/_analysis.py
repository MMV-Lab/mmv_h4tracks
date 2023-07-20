import numpy as np
import multiprocessing
from multiprocessing import Pool

from qtpy.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton,
                            QCheckBox, QHBoxLayout, QGridLayout, QFileDialog)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from napari.qt.threading import thread_worker

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
        self.setLayout(QVBoxLayout())
        self.setWindowTitle("Analysis")
        
        ### QObjects
        
        # Labels
        label_min_movement = QLabel("Movement Minmum")
        label_min_duration = QLabel("Minimum Track Length")
        label_metric = QLabel("Evaluation metrics:")
                
        # Buttons
        btn_plot = QPushButton("Plot")
        btn_export = QPushButton("Export")
        btn_evaluate_segmentation = QPushButton("Evaluate Segmentation")
        btn_evaluate_tracking = QPushButton("Evaluate Tracking")
        
        btn_plot.setToolTip("Only displayed tracks are plotted")
        
        btn_plot.clicked.connect(self._start_plot_worker)
        btn_export.clicked.connect(self._start_export_worker)
        
        # Comboboxes
        self.combobox_plots = QComboBox()
        self.combobox_plots.addItems(
            ["Speed", "Size", "Direction", "Euclidean distance", "Accumulated distance"]
        )
        
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
            checkbox_accumulated_distance
        ]
        
        # Line Edits
        self.lineedit_movement = QLineEdit("")
        self.lineedit_track_duration = QLineEdit("")
        lineedit_limit_evaluation = QLineEdit("0")
        
        ### Organize objects via widgets
        content = QWidget()
        content.setLayout(QVBoxLayout())
        
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
        content.layout().addWidget(lineedit_limit_evaluation)
        
        evaluation = QWidget()
        evaluation.setLayout(QHBoxLayout())
        evaluation.layout().addWidget(btn_evaluate_segmentation)
        evaluation.layout().addWidget(btn_evaluate_tracking)
        
        content.layout().addWidget(evaluation)
        
        self.layout().addWidget(content)
        
    def _calculate_speed(self, tracks):
        for unique_id in np.unique(tracks[:, 0]):
            track = np.delete(tracks, np.where(tracks[:, 0] != unique_id), 0)
            distance = []
            for i in range(0, len(track) - 1):
                distance.append(np.hypot(
                    track[i, 2] - track[i + 1, 2],
                    track[i, 3] - track[i + 1, 3]
                ))
            average = np.around(np.average(distance),3)
            std_deviation = np.around(np.std(distance),3)
            try:
                speeds = np.append(speeds, [[unique_id, average, std_deviation]], 0)
            except UnboundLocalError:
                speeds = np.array([[unique_id, average, std_deviation]])
        return speeds
    
    def _calculate_size(self, tracks, segmentation):
        unique_ids = np.unique(tracks[:, 0])
        track_and_segmentation = []
        for unique_id in unique_ids:
            track_and_segmentation.append([
                tracks[np.where(tracks[:,0] == unique_id)],
                segmentation
            ])
        
        if self.parent.rb_eco.isChecked():
            AMOUNT_OF_PROCESSES = np.maximum(1,int(multiprocessing.cpu_count() * 0.4))
        else:
            AMOUNT_OF_PROCESSES = np.maximum(1,int(multiprocessing.cpu_count() * 0.8))
        print("Running on {} processes max".format(AMOUNT_OF_PROCESSES))

        global func
        
        def func(track, segmentation):
            id = track[0, 0]
            sizes = []
            for line in track:
                _, z, y, x = line
                seg_id = segmentation[z, y, x]
                sizes.append(
                    len(np.where(segmentation[z] == seg_id)[0])
                )
            average = np.around(np.average(sizes), 3)
            std_deviation = np.around(np.std(sizes), 3)
            return [id, average, std_deviation]
        
        with Pool(AMOUNT_OF_PROCESSES) as p:
            sizes = p.starmap(func, track_and_segmentation)
            
        return np.array(sizes)
    
    def _calculate_direction(self, tracks):
        for id in np.unique(tracks[:, 0]):
            track = np.delete(tracks, np.where(tracks[:,0] != id), 0)
            x = track[-1, 3] - track[0, 3]
            y = track[0, 2] - track[-1, 2]
            if y > 0:
                if x == 0:
                    direction = np.pi / 2
                else:
                    direction = np.pi - np.arctan(np.abs(y/x))
            elif y < 0:
                if x == 0:
                    direction = 1.5 * np.pi
                else:
                    direction = np.pi - np.arctan(np.abs(y/x))
            else:
                if x < 0:
                    direction = np.pi
                else: direction = 0
            distance = np.around(np.sqrt(np.square(x) + np.square(y)), 3)
            direction = np.around(direction, 3)
            try:
                retval = np.append(retval, [[id, x, y, direction, distance]], 0)
            except UnboundLocalError:
                retval = np.array([[id, x, y, direction, distance]])
        return retval
    
    def _calculate_euclidean_distance(self, tracks):
        for id in np.unique(tracks[:, 0]):
            track = np.delete(tracks, np.where(tracks[:,0] != id), 0)
            x = track[-1, 3] - track[0, 3]
            y = track[0, 2] - track[-1, 2]
            euclidean_distance = np.around(np.sqrt(np.square(x) + np.square(y)), 3)
            try:
                retval = np.append(retval, [[id, euclidean_distance, 0]], 0)
            except UnboundLocalError:
                retval = np.array([[id, euclidean_distance, 0]])
        return retval
    
    def _calculate_accumulated_distance(self, tracks):
        for id in np.unique(tracks[:, 0]):
            track = np.delete(tracks, np.where(tracks[:, 0] != id), 0)
            steps = []
            for i in range(0, len(track) - 1):
                steps.append(np.hypot(
                    track[i, 2] - track[i + 1, 2],
                    track[i, 3] - track[i + 1, 3]
                ))
            accumulated_distance = np.around(np.sum(steps), 3)
            try:
                retval = np.append(retval, [[id, accumulated_distance, 0]], 0)
            except UnboundLocalError:
                retval = np.array([[id, accumulated_distance, 0]])
        return retval
    
    def _start_plot_worker(self):
        worker = self._sort_plot_data(self.combobox_plots.currentText())
        worker.returned.connect(self._plot)
        worker.start()
        
    def _plot(self, ret):
        fig = Figure(figsize = (6, 7))
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
        
        data = axes.scatter(results[:,1],results[:,2], c = np.array([[0,0.240802676,0.70703125,1]]))
        
        axes.set_title(title, {"fontsize": 18, "color": "white"})
        if not title == "Direction":
            axes.set_xlabel(ret["x_label"])
            axes.set_ylabel(ret["y_label"])
        
        canvas = FigureCanvas(fig)
        self.parent.plot_window = QWidget()
        self.parent.plot_window.setLayout(QVBoxLayout())
        self.parent.plot_window.layout().addWidget(QLabel(ret["Description"]))
        
        selector = Selector(self, axes, results)

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
        tracks_layer = grab_layer(self.parent.viewer, "Tracks")
        retval = {"Name" : metric}
        if metric == "Speed":
            print("Plotting speed")
            retval.update({"Description": "Scatterplot Standard Deviation vs Average: Speed"})
            retval.update({"x_label": "Average", "y_label": "Standard Deviation"})
            retval.update({"Results": self._calculate_speed(tracks_layer.data)})
        elif metric == "Size":
            print("Plotting size")
            segmentation_layer = grab_layer(self.parent.viewer, "Segmentation Data")
            retval.update({"Description": "Scatterplot Standard Deviation vs Average: Size"})
            retval.update({"x_label": "Average", "y_label": "Standard Deviation"})
            retval.update({"Results": self._calculate_size(
                tracks_layer.data, segmentation_layer.data
            )})
        elif metric == "Direction":
            print("Plotting direction")
            retval.update({"Description": "Scatterplot: Travel direction & Distance"})
            retval.update({"Results": self._calculate_direction(tracks_layer.data)})
        elif metric == "Euclidean distance":
            print("Plotting euclidean distance")
            retval.update({"Description": "Scatterplot x vs y"})
            retval.update({"x_label": "x", "y_label": "y"})
            retval.update({"Results": self._calculate_euclidean_distance(
                tracks_layer.data
            )})
        elif metric == "Accumulated distance":
            print("Plotting accumulated distance")
            retval.update({"Description": "Scatterplot x vs y"})
            retval.update({"x_label": "x", "y_label": "y"})
            retval.update({"Results": self._calculate_accumulated_distance(
                tracks_layer.data
            )})
        else:
            raise ValueError('No defined behaviour for given metric.')
            
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
        file = dialog.getSaveFileName(filter = "*.csv")
        if file[0] == "":
            print("Export stopped due to no selected file")
            return
        
        worker = self._export(file, selected_metrics)
        worker.start()
    
    @thread_worker
    def _export(self, file, metrics):
        tracks = grab_layer(self.parent.viewer, "Tracks").data
        direction = self._calculate_direction(tracks)
        self.direction = direction
        
        try:
            filtered_mask, min_movement, min_duration = self._filter_tracks_by_parameters(tracks, direction)
        except ValueError:
            return
        
        duration = np.array([[i, np.count_nonzero(tracks[:, 0] == i)]
                             for i in np.unique(tracks[:, 0])])
        
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
                    notify("Please use an integer instead of a float for movement minimum")
                    raise ValueError
                movement_mask = direction[np.where(direction[:,4] >= min_movement)[0], 0]
                
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
                    notify("Please use an integer instead of a float for duration minimum")
                    raise ValueError
                indices = np.where(
                    np.unique(tracks[:, 0], return_counts = true)[1] >= min_duration
                )
                duration_mask = np.unique(tracks[:, 0])[indices]
                
        return np.intersect1d(movement_mask, duration_mask), min_movement, min_duration
    
    def _compose_csv_data(
            self, tracks, duration, filtered_mask,
            min_movement, min_duration, selected_metrics
        ):
        metrics = ["", "Number of cells", "Average track duration"]
        metrics.append("Standard deviation of track duration")
        individual_metrics = ["ID", "Track duration"]
        all_values = ["all"]
        all_values.extend([
            len(np.unique(tracks[:, 0])),
            np.around(np.average(duration[:, 1]), 3),
            np.around(np.std(duration[:, 1]), 3)
        ])
        valid_values = ["valid"]
        valid_values.extend([
            len(filtered_mask),
            np.around(np.average([duration[i, 1]
                                  for i in range(len(duration))
                                  if duration[i, 0] in filtered_mask]), 3),
            np.around(np.std([duration[i, 1]
                              for i in range(len(duration))
                              if duration[i, 0] in filtered_mask]), 3)
        ])
        if len(selected_metrics) > 0:
            more_metrics, more_individual_metrics, more_all_values, more_valid_values, metrics_dict = self._extend_metrics(
                tracks, selected_metrics, filtered_mask, duration
            )
            metrics.extend(more_metrics)
            individual_metrics.extend(more_individual_metrics)
            all_values.extend(more_all_values)
            valid_values.extend(more_valid_values)
        
        rows = [metrics, all_values, valid_values]
        rows.append([None])
        rows.append(["Movement Threshold: " + str(min_movement), "Duration Threshold: " + str(min_duration)])
        rows.append([None])
        
        if len(selected_metrics) > 0:
            rows.append(individual_metrics)
            valid_values, invalid_values = self._individual_metric_values(
                tracks, filtered_mask, metrics_dict
            )
            
            rows.append(["Valid Cells"])
            #[rows.append(value) for value in valid_values]
            for value in valid_values: rows.append(value)
            
            if not np.array_equal(np.unique(tracks[:,0]), filtered_mask):
                rows.append([None])
                rows.append([None])
                rows.append(["Invalid cells"])
                
                [rows.append(value) for value in invalid_values]
            
        return rows
    
    def _extend_metrics(self, tracks, selected_metrics, filtered_mask, duration):
        metrics, individual_metrics, all_values, valid_values = [[], [], [], []]
        metrics_dict = {}
        if "Speed" in selected_metrics:
            speed = self._calculate_speed(tracks)
            metrics.append("Average_speed")
            individual_metrics.extend(["Average speed", "Standard deviation of speed"])
            all_values.append(np.around(np.average(speed[:, 1]), 3))
            valid_values.append(np.around(
                np.average([speed[i, 1]
                            for i in range(len(speed))
                            if speed[i, 0] in filtered_mask]), 3
            ))
            metrics_dict.update({"Speed": speed})
            
        if "Size" in selected_metrics:
            segmentation = grab_layer(self.parent.viewer, "Segmentation Data").data
            size = self._calculate_size(tracks, segmentation)
            metrics.extend(["Average size", "Standard deviation of size"])
            individual_metrics.extend(["Average size", "Standard deviation of size"])
            all_values.extend([
                np.around(np.average(size[:, 1]), 3),np.around(np.std(size[:, 1]), 3)
            ])
            valid_values.extend([
                np.around(np.average([size[i, 1]
                                      for i in range(len(size))
                                      if size[i, 0] in filtered_mask]), 3),
                np.around(np.std([size[i, 1]
                                  for i in range(len(size))
                                  if size[i, 0] in filtered_mask]), 3)
            ])
            metrics_dict.update({"Size": size})
        
        if "Direction" in selected_metrics:
            metrics.extend(
                ["Average direction", "Standard deviation of direction", "Average distance"]
                )
            individual_metrics.extend(["Direction", "Distance"])
            all_values.extend([
                np.around(np.average(self.direction[:, 3]), 3),
                np.around(np.std(self.direction[:, 3]), 3),
                np.around(np.average(self.direction[:, 4]), 3)
            ])
            valid_values.extend([
                np.around(np.average([self.direction[i, 3]
                                      for i in range(len(self.direction))
                                      if self.direction[i, 0] in filtered_mask]), 3),
                np.around(np.std([self.direction[i, 3]
                                  for i in range(len(self.direction))
                                  if self.direction[i, 0] in filtered_mask]), 3),
                np.around(np.average([self.direction[i, 4]
                                      for i in range(len(self.direction))
                                      if self.direction[i, 0] in filtered_mask]), 3)
            ])
            metrics_dict.update({"Direction": self.direction})

        if "Euclidean distance" in selected_metrics:
            euclidean_distance = self._calculate_euclidean_distance(tracks)
            metrics.extend(["Average euclidean distance", "Average directed speed"])
            individual_metrics.extend(["Euclidean distance", "Directed speed"])
            all_values.extend([
                np.around(np.average(euclidean_distance[:, 1]), 3),
                np.around(np.average(euclidean_distance[:, 1] / len(np.unique(tracks[:, 0]))), 3)
            ])
            valid_values.extend([
                np.around(
                    np.average([euclidean_distance[i, 1]
                                for i in range(len(euclidean_distance))
                                if euclidean_distance[i, 0] in filtered_mask]), 3),
                np.around(
                    np.average([euclidean_distance[i, 1] / duration[i, 1]
                                for i in range(len(euclidean_distance))
                                if euclidean_distance[i, 0] in filtered_mask]), 3)
            ])
            metrics_dict.update({"Euclidean Distance": euclidean_distance})
            
        if "Accumulated distance" in selected_metrics:
            accumulated_distance = self._calculate_accumulated_distance(tracks)
            metrics.append("Average accumulated distance")
            individual_metrics.append("Accumulated distance")
            all_values.append(np.around(np.average(accumulated_distance[:, 1]), 3))
            valid_values.append(
                np.around(np.average([accumulated_distance[i, 1]
                                      for i in range(len(accumulated_distance))
                                      if accumulated_distance[i, 0] in filtered_mask]), 3)
            )
            metrics_dict.update({"Accumulated Distance": accumulated_distance})
            
            if "Euclidean distance" in selected_metrics:
                metrics.append("Average directness")
                individual_metrics.append("Directness")
                
                """directness = []
                for i in range(len(np.unique(tracks[:, 0]))):
                    directness.append()
                    
                directness = np.array(directness)"""
                directness = np.array([euclidean_distance[i,0],euclidean_distance[i,1]/accumulated_distance[i,1] if accumulated_distance[i,1] > 0 else 0] for i in range(len(np.unique(tracks[:, 0]))))
                """directness = np.array([
                    euclidean_distance[i, 0],
                    euclidean_distance[i, 1] / accumulated_distance[i, 1]
                    if accumulated_distance[i, 1] > 0 else 0]
                for i in range(len(np.unique(tracks[:, 0])))
                )"""
                print(directness)
                all_values.append(np.around(np.average(directness[:, 1]), 3))
                valid_values.append(np.around(np.average([directness[i, 1]
                                                          for i in range(len(directness))
                                                          if directness[i, 0] in filtered_mask]), 3))
                metrics_dict.update({"Directness": directness})
                
        print(metrics_dict.keys())

        return metrics, individual_metrics, all_values, valid_values, metrics_dict
    
    def _individual_metric_values(self, tracks, filtered_mask, metrics):
        #raise NotImplementedError
        valid_values, invalid_values = [[], []]
        for id in np.unique(tracks[:, 0]):
            value = [id]
            value.append(np.count_nonzero(tracks[:, 0] == id))
            if "Speed" in metrics:
                value.extend([metrics["Speed"][id][1], metrics["Speed"][id][2]])
            if "Size" in metrics:
                value.extend([metrics["Size"][id][1], metrics["Size"][id][2]])
            if "Direction" in metrics:
                value.extend([metrics["Direction"][id][1], metrics["Direction"][id][2]])
            if "Euclidean distance" in metrics:
                value.extend([
                    metrics["Euclidean distance"][id][1], [metrics["Euclidean distance"][id][2]]
                ])
            if "Accumulated distance" in metrics:
                print(accumulated_distance)
                value.append(metrics["Accumulated distance"][id][1])
            if "Directness" in metrics:
                value.append(metrics["Directness"][id][1])
                
            if id in filtered_mask:
                valid_values.append(value)
            else:
                invalid_values.append(value)
        
        return valid_values, invalid_values
    
    def _evaluate_segmentation(self):
        raise NotImplementedError
    
    def _evaluate_tracking(self):
        raise NotImplementedError
    
    
        
        
        
        
        
        