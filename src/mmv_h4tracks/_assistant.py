import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QGroupBox,
    QSizePolicy,
    QGridLayout,
    QCheckBox,
)
from qtpy.QtGui import QDoubleValidator

import napari

from scipy.ndimage import label, center_of_mass
from tqdm import tqdm

from ._grabber import grab_layer
from ._logger import notify

DEFAULT_SPEED_THRESHOLD = 10
DEFAULT_SIZE_THRESHOLD = 5
DEFAULT_DISTANCE_THRESHOLD = 50
DEFAULT_SMALL_SIZE_THRESHOLD = 10


class AssistantWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.viewer = parent.viewer
        self.setup_ui()

    def setup_ui(self):
        self.setLayout(QVBoxLayout())
        try:
            self.setStyleSheet(napari.qt.get_stylesheet(theme="dark"))
        except TypeError:
            self.setStyleSheet(napari.qt.get_stylesheet(theme_id="dark"))

        ### QObjects

        # Labels
        label_speed = QLabel("Threshold speed change")
        label_size = QLabel("Threshold size change")
        label_distance = QLabel("Threshold edge distance")
        label_small_size = QLabel("Threshold size")
        label_FOI = QLabel("Frames of interest:")
        self.label_frames = QLabel()
        self.label_frames.setMaximumWidth(335)

        # while we support python<=3.11, we can't do multiline f-strings
        speed_tooltip = f"Threshold for the speed change within a track (max/min).\nDefault is {DEFAULT_SPEED_THRESHOLD}"
        size_tooltip = f"Threshold for the size change within a track (max/min).\nDefault is {DEFAULT_SIZE_THRESHOLD}"
        distance_tooltip = f"Threshold for the distance to the edge of the image at wich a track starts or ends.\nOnly applies when the track does not start in the first frame and does not end in the last frame.\nDefault is {DEFAULT_DISTANCE_THRESHOLD}"
        small_size_tooltip = f"Threshold for the size of a cell.\nOnly applies within a frame.\nDefault is {DEFAULT_SMALL_SIZE_THRESHOLD}"
        foi_tooltip = (
            "Frames of interest are the frames in which the outliers are marked."
        )

        label_speed.setToolTip(speed_tooltip)
        label_size.setToolTip(size_tooltip)
        label_distance.setToolTip(distance_tooltip)
        label_small_size.setToolTip(small_size_tooltip)
        label_FOI.setToolTip(foi_tooltip)
        self.label_frames.setToolTip(foi_tooltip)

        # Buttons
        btn_speed = QPushButton("Show speed outliers")
        btn_size = QPushButton("Show size outliers")
        btn_distance = QPushButton("Show noteworthy tracks")
        btn_relabel = QPushButton("Relabel all cells")
        btn_align_ids = QPushButton("Align segmentation IDs")
        btn_untracked = QPushButton("Show untracked cells")
        btn_tiny = QPushButton("Show small cells")
        # btn_delete_id = QPushButton("Delete ID")

        btn_speed.setToolTip(speed_tooltip)
        btn_size.setToolTip(size_tooltip)
        btn_distance.setToolTip(distance_tooltip)
        btn_tiny.setToolTip(small_size_tooltip)
        btn_relabel.setToolTip(
            "Make sure that each cell has a unique ID.\nThis can fix slip ups in the segmentation.\nReplaces the existing label layer."
        )
        btn_align_ids.setToolTip(
            "Align the IDs of the tracks with the segmentation.\nReplaces the existing label layer."
        )
        btn_untracked.setToolTip("Show cells that are not tracked")
        # btn_delete_id.setToolTip("Delete a specific ID from the whole movie")

        btn_speed.clicked.connect(self.show_speed_outliers_on_click)
        btn_size.clicked.connect(self.show_size_outliers_on_click)
        btn_distance.clicked.connect(self.show_abrupt_tracks_on_click)
        btn_relabel.clicked.connect(self.relabel_cells_on_click)
        btn_align_ids.clicked.connect(self.align_ids_on_click)
        btn_untracked.clicked.connect(self.show_untracked_cells_on_click)
        btn_tiny.clicked.connect(self.show_tiny_cells_on_click)
        # btn_delete_id.clicked.connect(self.delete_id_on_click)

        # LineEdits
        self.speed_lineedit = QLineEdit()
        self.speed_lineedit.setValidator(QDoubleValidator(0, 1000, 2))
        self.size_lineedit = QLineEdit()
        self.size_lineedit.setValidator(QDoubleValidator(0, 10000, 2))
        self.distance_lineedit = QLineEdit()
        self.distance_lineedit.setValidator(QDoubleValidator(0, 1000, 2))

        self.tiny_lineedit = QLineEdit()
        self.tiny_lineedit.setValidator(QDoubleValidator(0, 10000, 2))
        # self.delete_id_lineedit = QLineEdit()

        self.speed_lineedit.setPlaceholderText(str(DEFAULT_SPEED_THRESHOLD))
        self.size_lineedit.setPlaceholderText(str(DEFAULT_SIZE_THRESHOLD))
        self.distance_lineedit.setPlaceholderText(str(DEFAULT_DISTANCE_THRESHOLD))
        self.tiny_lineedit.setPlaceholderText(str(DEFAULT_SMALL_SIZE_THRESHOLD))
        # self.delete_id_lineedit.setPlaceholderText("ID to delete")

        # Checkboxes
        self.checkbox_hidden = QCheckBox("Include hidden tracks")
        self.checkbox_hidden.setChecked(True)
        self.checkbox_hidden.setToolTip(
            "Include tracks that are hidden in evaluating the untracked cells"
        )

        # Horizontal lines
        line = QWidget()
        line.setFixedHeight(4)
        line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line.setStyleSheet("background-color: #c0c0c0")

        # Spacers
        v_spacer = QWidget()
        v_spacer.setFixedWidth(4)
        v_spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        h_spacer = QWidget()
        h_spacer.setFixedHeight(10)
        h_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # QGroupBoxes
        filters = QGroupBox("Filters")
        filters_layout = QGridLayout()
        filters_layout.addWidget(label_distance, 0, 0, 1, 2)
        filters_layout.addWidget(self.distance_lineedit, 0, 2)
        filters_layout.addWidget(btn_distance, 0, 3)
        filters_layout.addWidget(label_speed, 1, 0, 1, 2)
        filters_layout.addWidget(self.speed_lineedit, 1, 2)
        filters_layout.addWidget(btn_speed, 1, 3)
        filters_layout.addWidget(label_size, 2, 0, 1, 2)
        filters_layout.addWidget(self.size_lineedit, 2, 2)
        filters_layout.addWidget(btn_size, 2, 3)
        filters_layout.addWidget(line, 3, 0, 1, -1)
        filters_layout.addWidget(label_small_size, 4, 0, 1, 2)
        filters_layout.addWidget(self.tiny_lineedit, 4, 2)
        filters_layout.addWidget(btn_tiny, 4, 3)
        filters_layout.addWidget(self.checkbox_hidden, 5, 0, 1, 2)
        filters_layout.addWidget(btn_untracked, 5, 2, 1, 2)
        filters_layout.addWidget(label_FOI, 6, 0)
        filters_layout.addWidget(self.label_frames, 6, 1, 1, 3)
        # filters_layout.addWidget(self.delete_id_lineedit, 4, 0, 1, 2) # can't just uncomment
        # filters_layout.addWidget(btn_delete_id, 4, 2)
        filters.setLayout(filters_layout)

        segmentation_adaptation = QGroupBox("Segmentation adaptation")
        segmentation_adaptation_layout = QVBoxLayout()
        segmentation_adaptation_layout.addWidget(h_spacer)
        segmentation_adaptation_layout.addWidget(btn_align_ids)
        segmentation_adaptation_layout.addWidget(btn_relabel)
        segmentation_adaptation.setLayout(segmentation_adaptation_layout)

        ### Layout
        content = QWidget()
        content.setLayout(QVBoxLayout())
        content.layout().addWidget(filters)
        content.layout().addWidget(segmentation_adaptation)
        content.layout().addWidget(v_spacer)
        self.layout().addWidget(content)

    def show_speed_outliers_on_click(self):
        self.label_frames.setText("")
        try:
            tracks_layer = grab_layer(
                self.viewer, self.parent.combobox_tracks.currentText()
            )
        except ValueError:
            print("No tracks layer found")
            return
        tracks = tracks_layer.data
        outliers = []
        try:
            threshold = float(self.speed_lineedit.text())
        except ValueError:
            threshold = DEFAULT_SPEED_THRESHOLD
        speeds = self.parent.analysis_window._calculate_speed(tracks)
        for result in speeds:
            if result[1] == 0:
                print(f"avoiding division by zero. would divide {result[3]}")
                continue
            if result[3] / result[1] > threshold:
                outliers.append(int(result[0]))
        print(outliers)
        print(len(outliers))
        self.display_outliers(outliers)

    def show_size_outliers_on_click(self):
        self.label_frames.setText("")
        try:
            label_layer = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            )
        except ValueError:
            print("No segmentation layer found")
            return
        segmentation = label_layer.data
        try:
            tracks_layer = grab_layer(
                self.viewer, self.parent.combobox_tracks.currentText()
            )
        except ValueError:
            print("No tracks layer found")
            return
        tracks = tracks_layer.data
        outliers = []
        try:
            threshold = float(self.size_lineedit.text())
        except ValueError:
            threshold = DEFAULT_SIZE_THRESHOLD
        sizes = self.parent.analysis_window._calculate_size(tracks, segmentation)
        for result in sizes:
            if result[4] / result[3] > threshold:
                outliers.append(int(result[0]))
        print(outliers)
        print(len(outliers))
        self.display_outliers(outliers)

    def show_abrupt_tracks_on_click(self):
        self.label_frames.setText("")
        try:
            label_layer = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            )
        except ValueError:
            print("No segmentation layer found")
            return
        segmentation = label_layer.data
        frames, y, x = segmentation.shape
        shape = (y, x)
        try:
            tracks_layer = grab_layer(
                self.viewer, self.parent.combobox_tracks.currentText()
            )
        except ValueError:
            print("No tracks layer found")
            return
        tracks = tracks_layer.data
        outliers = []
        try:
            threshold = float(self.distance_lineedit.text())
        except ValueError:
            threshold = DEFAULT_DISTANCE_THRESHOLD

        for id_ in np.unique(tracks[:, 0]):
            track = tracks[tracks[:, 0] == id_]
            if track[0, 1] > 0 and not self.close_to_edge(
                track[0, 2], track[0, 3], shape, threshold
            ):
                outliers.append(id_)
            elif track[-1, 1] < frames - 1 and not self.close_to_edge(
                track[-1, 2], track[-1, 3], shape, threshold
            ):
                outliers.append(id_)
        print(outliers)
        print(len(outliers))
        self.display_outliers(outliers)

    def close_to_edge(self, y, x, shape, threshold):
        y_edge = y < threshold or y > shape[0] - threshold
        x_edge = x < threshold or x > shape[1] - threshold
        return y_edge or x_edge

    def display_outliers(self, outliers):
        if len(outliers) == 0:
            print("No outliers found")
            notify("No outliers found")
            return
        self.parent.tracking_window.display_selected_tracks(outliers)
        outlier_text = ", ".join(map(str, outliers))
        self.parent.tracking_window.lineedit_filter.setText(outlier_text)

    def relabel_cells_on_click(self):
        self.label_frames.setText("")
        try:
            label_layer = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            )
        except ValueError:
            print("No segmentation layer found")
            return
        data = label_layer.data
        frames, _, _ = data.shape
        relabeled_data = np.zeros_like(data)
        for frame in tqdm(range(frames), desc="Processing frames"):
            current_frame = data[frame]
            unique_ids = np.unique(current_frame)
            unique_ids = unique_ids[unique_ids != 0]
            new_frame = np.zeros_like(current_frame)
            current_max_label = 0
            for uid in unique_ids:
                mask = current_frame == uid
                labeled_mask, _ = label(mask)
                labeled_mask[labeled_mask > 0] += current_max_label
                current_max_label = labeled_mask.max()
                new_frame += labeled_mask
            relabeled_data[frame] = new_frame

        label_layer.data = relabeled_data
        # self.viewer.layers.remove(label_layer.name)
        # self.viewer.add_labels(relabeled_data, name=label_layer.name)

    def align_ids_on_click(self):
        self.label_frames.setText("")
        try:
            label_layer = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            )
        except ValueError:
            print("No segmentation layer found")
        segmentation = label_layer.data
        new_segmentation = np.zeros_like(segmentation)
        tracks_name = self.parent.combobox_tracks.currentText()
        try:
            tracks_layer = grab_layer(self.viewer, tracks_name)
        except ValueError:
            print("No tracks layer found")
            return
        tracks = tracks_layer.data

        # Justin's code
        if 0 in tracks[:, 0]:
            tracks[:, 0] = tracks[:, 0] + 1

        new_segmentation[segmentation > 0] = segmentation[segmentation > 0] + np.max(
            [np.max(segmentation), np.max(tracks[:, 0])]
        )
        # segmentation[segmentation > 0] = segmentation[segmentation > 0] + np.max(
        #     segmentation
        # )

        for track_id in tqdm(np.unique(tracks[:, 0])):
            new_segmentation = self.adjust_segmentation_ids(
                new_segmentation, segmentation, tracks[tracks[:, 0] == track_id]
            )
        # for track_id in tqdm(np.unique(tracks[:, 0])):
        #     segmentation = self.adjust_segmentation_ids(
        #         segmentation, tracks[tracks[:, 0] == track_id]
        #     )

        # self.viewer.add_labels(segmentation, name="Aligned cells")
        label_layer.data = new_segmentation
        # self.viewer.layers.remove(label_layer.name)
        # self.viewer.add_labels(new_segmentation, name=label_layer.name)

    def adjust_segmentation_ids(self, segmentation, old_segmentation, tracks):
        for track in tracks:
            centroid = (track[1], track[2], track[3])
            if segmentation[centroid] == 0:
                print("centroid not in cell")
                frame = old_segmentation[track[1]]
                candidates = np.unique(
                    frame[
                        np.maximum(centroid[1] - 20, 0) : np.minimum(
                            centroid[1] + 20, frame.shape[0]
                        ),  # Candidates must be located within the edges.
                        np.maximum(centroid[2] - 20, 0) : np.minimum(
                            centroid[2] + 20, frame.shape[1]
                        ),
                    ]
                )
                cell_found = False
                for id_ in candidates[1:]:
                    candidate_centroid = center_of_mass(frame, labels=frame, index=id_)
                    if (
                        int(np.rint(candidate_centroid[0])) == centroid[1]
                        and int(np.rint(candidate_centroid[1])) == centroid[2]
                    ):
                        segmentation[track[1]][frame == id_] = track[0]
                        cell_found = True
                        break
                if not cell_found:
                    raise ValueError("Could not find cell")
            else:
                segmentation[track[1]][
                    segmentation[track[1]] == segmentation[track[1], track[2], track[3]]
                ] = track[0]
        return segmentation

    def show_untracked_cells_on_click(self):
        try:
            label_layer = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            )
        except ValueError:
            print("No segmentation layer found")
            return
        segmentation = label_layer.data
        try:
            tracks_layer = grab_layer(
                self.viewer, self.parent.combobox_tracks.currentText()
            )
        except ValueError:
            print("No tracks layer found")
            return
        if (
            self.checkbox_hidden.isChecked()
            and self.parent.tracking_window.cached_tracks is not None
        ):
            tracks = self.parent.tracking_window.cached_tracks
        else:
            tracks = tracks_layer.data
        untracked = []
        for frame in tqdm(range(segmentation.shape[0])):
            # for frame in range(segmentation.shape[0]):
            #     starttime1 = time.time()
            #     for id_ in set(np.unique(segmentation[frame])) - {0}:
            #         centroid = center_of_mass(
            #             label_layer.data[frame],
            #             labels=label_layer.data[frame],
            #             index=id_,
            #         )
            #         centroid = [
            #             frame,
            #             int(np.rint(centroid[0])),
            #             int(np.rint(centroid[1])),
            #         ]
            #         if not any(
            #             np.all(centroid == track[1:4]) for track in tracks
            #         ):
            #             untracked.append(centroid)
            #     print(f"center_of_mass took {time.time() - starttime1} seconds")
            #     print(untracked)

            tracked_centroids = [
                [entry[2], entry[3]] for entry in tracks if entry[1] == frame
            ]
            tracked_ids = [
                segmentation[frame][coord[0], coord[1]] for coord in tracked_centroids
            ]
            untracked_ids = set(np.unique(segmentation[frame])) - set(tracked_ids) - {0}
            centroids = center_of_mass(
                segmentation[frame],
                labels=segmentation[frame],
                index=list(untracked_ids),
            )
            for centroid in centroids:
                centroid = [frame, int(np.rint(centroid[0])), int(np.rint(centroid[1]))]
                untracked.append(centroid)

        if len(untracked) > 0:
            unique_frames = set([coord[0] for coord in untracked])
            self.label_frames.setText(", ".join(map(str, sorted(unique_frames))))
        else:
            self.label_frames.setText("")
        self.mark_outliers(untracked, "Untracked cells")

    def show_tiny_cells_on_click(self):
        try:
            label_layer = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            )
        except ValueError:
            print("No segmentation layer found")
            return
        segmentation = label_layer.data
        try:
            threshold = float(self.tiny_lineedit.text())
        except ValueError:
            threshold = DEFAULT_SMALL_SIZE_THRESHOLD
        tiny = []
        for frame in tqdm(range(segmentation.shape[0])):
            for id_ in set(np.unique(segmentation[frame])) - {0}:
                if np.sum(segmentation[frame] == id_) < threshold:
                    # centroid = center_of_mass(
                    #     label_layer.data[frame],
                    #     labels=label_layer.data[frame],
                    #     index=id_,
                    # )
                    # centroid = [
                    #     frame,
                    #     int(np.rint(centroid[0])),
                    #     int(np.rint(centroid[1])),
                    # ]
                    all_pixels = np.where(segmentation[frame] == id_)
                    for y, x in zip(all_pixels[0], all_pixels[1]):
                        tiny.append([frame, y, x])
                    # tiny.append(centroid)

        unique_frames = set([coord[0] for coord in tiny])
        # print(f"Found {len(tiny)} tiny cells in frames {sorted(unique_frames)}")
        if len(unique_frames) > 0:
            self.label_frames.setText(", ".join(map(str, sorted(unique_frames))))
        else:
            self.label_frames.setText("")

        self.mark_outliers(tiny, "Tiny cells")

    # def delete_id_on_click(self):
    #     try:
    #         label_layer = grab_layer(
    #             self.viewer, self.parent.combobox_segmentation.currentText()
    #         )
    #     except ValueError:
    #         print("No segmentation layer found")
    #         return
    #     segmentation = label_layer.data
    #     try:
    #         id_ = int(self.delete_id_lineedit.text())
    #     except ValueError:
    #         print("Invalid ID")
    #         return
    #     segmentation[segmentation == id_] = 0
    #     label_layer.data = segmentation
    #     print(f"Deleted ID {id_}")
    #     self.delete_id_lineedit.clear()

    def mark_outliers(self, outliers, layername):
        try:
            label_layer = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            )
        except ValueError:
            print("No segmentation layer found")
            return
        data = np.zeros_like(label_layer.data)
        indices = []
        for outlier in outliers:
            # indices.extend(self.get_cross(outlier))
            # indices.extend(self.get_circle(outlier))
            indices.extend(self.get_plus(outlier))
        indices = np.array(indices)
        indices = tuple(indices.T)
        if len(indices) == 0:
            return
        data[indices] = 9
        try:
            layer = self.viewer.layers[self.viewer.layers.index(layername)]
            layer.data = data
        except:
            self.viewer.add_labels(data, name=layername)

    def get_plus(self, centroid):
        try:
            label_layer = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            )
        except ValueError:
            print("No segmentation layer found")
            return
        _, max_y, max_x = label_layer.data.shape
        z, y, x = centroid
        plus_size = 35
        plus_thickness = 7
        half_size = plus_size // 2
        half_thickness = plus_thickness // 2
        coordinates = set()
        for dy in range(-half_size, half_size + 1):
            for dx in range(-half_thickness, half_thickness + 1):
                new_y = y + dy
                new_x = x + dx
                if 0 <= new_y < max_y and 0 <= new_x < max_x:
                    coordinates.add((z, new_y, new_x))

        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_thickness, half_thickness + 1):
                new_y = y + dy
                new_x = x + dx
                if 0 <= new_y < max_y and 0 <= new_x < max_x:
                    coordinates.add((z, new_y, new_x))

        return list(coordinates)

    # def get_cross(self, centroid):
    #     try:
    #         label_layer = grab_layer(
    #             self.viewer, self.parent.combobox_segmentation.currentText()
    #         )
    #     except ValueError:
    #         print("No segmentation layer found")
    #         return
    #     _, max_y, max_x = label_layer.data.shape
    #     z, y, x = centroid
    #     cross = []
    #     for i in range(26):
    #         if 0 < x - i < max_x and 0 < y - i < max_y:
    #             cross.append([z, y - i, x - i])
    #         if 0 < x + i < max_x and 0 < y - i < max_y:
    #             cross.append([z, y - i, x + i])
    #         if 0 < x - i < max_x and 0 < y + i < max_y:
    #             cross.append([z, y + i, x - i])
    #         if 0 < x + i < max_x and 0 < y + i < max_y:
    #             cross.append([z, y + i, x + i])

    #     return cross

    # def get_circle(self, centroid, R=10):
    #     try:
    #         label_layer = grab_layer(
    #             self.viewer, self.parent.combobox_segmentation.currentText()
    #         )
    #     except ValueError:
    #         print("No segmentation layer found")
    #         return

    #     _, max_y, max_x = label_layer.data.shape
    #     z, y, x = centroid
    #     circle = []
    #     R_squared = R * R

    #     for dy in range(-R, R + 1):
    #         dx_limit = int((R_squared - dy**2) ** 0.5)
    #         for dx in range(-dx_limit, dx_limit + 1):
    #             new_x, new_y = x + dx, y + dy
    #             if 0 <= new_x < max_x and 0 <= new_y < max_y:
    #                 circle.append([z, new_y, new_x])

    #     return circle
