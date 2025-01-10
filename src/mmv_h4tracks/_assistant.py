import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QGroupBox,
    QGridLayout,
)

import napari

from scipy.ndimage import label, center_of_mass
from tqdm import tqdm

from ._grabber import grab_layer
from ._logger import notify

DEFAULT_SPEED_THRESHOLD = 10
DEFAULT_SIZE_THRESHOLD = 5
DEFAULT_DISTANCE_THRESHOLD = 50


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
        label_speed = QLabel("Threshold Speed")
        label_size = QLabel("Threshold Size")
        label_distance = QLabel("Threshold Distance")

        # Buttons
        btn_speed = QPushButton("Show speed outliers")
        btn_size = QPushButton("Show size outliers")
        btn_distance = QPushButton("Show tracks starting/ending abruptly")
        btn_relabel = QPushButton("Relabel cells")
        btn_align_ids = QPushButton("Less flashing")
        btn_untracked = QPushButton("Show untracked cells")
        btn_tiny = QPushButton("Show small cells")
        btn_delete_id = QPushButton("Delete ID")

        btn_relabel.setToolTip("Make sure that each cell has a unique ID")
        btn_align_ids.setToolTip("Align the IDs of the tracks with the segmentation")
        btn_delete_id.setToolTip("Delete a specific ID from the whole movie")

        btn_speed.clicked.connect(self.show_speed_outliers_on_click)
        btn_size.clicked.connect(self.show_size_outliers_on_click)
        btn_distance.clicked.connect(self.show_abrupt_tracks_on_click)
        btn_relabel.clicked.connect(self.relabel_cells_on_click)
        btn_align_ids.clicked.connect(self.align_ids_on_click)
        btn_untracked.clicked.connect(self.show_untracked_cells_on_click)
        btn_tiny.clicked.connect(self.show_tiny_cells_on_click)
        btn_delete_id.clicked.connect(self.delete_id_on_click)

        # LineEdits
        self.speed_lineedit = QLineEdit()
        self.size_lineedit = QLineEdit()
        self.distance_lineedit = QLineEdit()
        self.tiny_lineedit = QLineEdit()
        self.delete_id_lineedit = QLineEdit()

        self.speed_lineedit.setPlaceholderText(str(DEFAULT_SPEED_THRESHOLD))
        self.size_lineedit.setPlaceholderText(str(DEFAULT_SIZE_THRESHOLD))
        self.distance_lineedit.setPlaceholderText(str(DEFAULT_DISTANCE_THRESHOLD))
        self.tiny_lineedit.setPlaceholderText("10")
        self.delete_id_lineedit.setPlaceholderText("ID to delete")

        # QGroupBoxes
        filters = QGroupBox("Filters")
        filters_layout = QGridLayout()
        filters_layout.addWidget(label_speed, 0, 0)
        filters_layout.addWidget(self.speed_lineedit, 0, 1)
        filters_layout.addWidget(btn_speed, 0, 2)
        filters_layout.addWidget(label_size, 1, 0)
        filters_layout.addWidget(self.size_lineedit, 1, 1)
        filters_layout.addWidget(btn_size, 1, 2)
        filters_layout.addWidget(label_distance, 2, 0)
        filters_layout.addWidget(self.distance_lineedit, 2, 1)
        filters_layout.addWidget(btn_distance, 2, 2)
        filters_layout.addWidget(self.tiny_lineedit, 3, 0, 1, 2)
        filters_layout.addWidget(btn_tiny, 3, 2)
        filters_layout.addWidget(self.delete_id_lineedit, 4, 0, 1, 2)
        filters_layout.addWidget(btn_delete_id, 4, 2)
        filters_layout.addWidget(btn_untracked, 5, 0)
        filters_layout.addWidget(btn_align_ids, 5, 1)
        filters_layout.addWidget(btn_relabel, 5, 2)
        filters.setLayout(filters_layout)

        ### Layout
        content = QWidget()
        content.setLayout(QVBoxLayout())
        content.layout().addWidget(filters)
        self.layout().addWidget(content)

    def show_speed_outliers_on_click(self):
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

        self.viewer.add_labels(relabeled_data, name="Relabeled cells")

    def align_ids_on_click(self):
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
            [np.max(segmentation), np.max(tracks[:,0])]
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
        self.viewer.add_labels(new_segmentation, name="Aligned cells")

    def adjust_segmentation_ids(self, segmentation, old_segmentation, tracks):
        for track in tracks:
            centroid = (track[1], track[2], track[3])
            if segmentation[centroid] == 0:
                print("centroid not in cell")
                frame = old_segmentation[track[1]]
                candidates = np.unique(
                    frame[
                        np.maximum(centroid[1] - 20, 0) : np.minimum(centroid[1] + 20, frame.shape[0]),        # Candidates must be located within the edges.
                        np.maximum(centroid[2] - 20, 0) : np.minimum(centroid[2] + 20, frame.shape[1]),
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

            tracked_centroids = [[entry[2], entry[3]] for entry in tracks if entry[1] == frame]
            tracked_ids = [segmentation[frame][coord[0], coord[1]] for coord in tracked_centroids]
            untracked_ids = set(np.unique(segmentation[frame])) - set(tracked_ids) - {0}
            centroids = center_of_mass(segmentation[frame], labels=segmentation[frame], index=list(untracked_ids))
            for centroid in centroids:
                centroid = [frame, int(np.rint(centroid[0])), int(np.rint(centroid[1]))]
                untracked.append(centroid)

        print(untracked)
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
            threshold = 10
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
        print(f"Found {len(tiny)} tiny cells in frames {sorted(unique_frames)}")
        
        self.mark_outliers(tiny, "Tiny cells")

    def delete_id_on_click(self):
        try:
            label_layer = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            )
        except ValueError:
            print("No segmentation layer found")
            return
        segmentation = label_layer.data
        try:
            id_ = int(self.delete_id_lineedit.text())
        except ValueError:
            print("Invalid ID")
            return
        segmentation[segmentation == id_] = 0
        label_layer.data = segmentation
        print(f"Deleted ID {id_}")
        self.delete_id_lineedit.clear()

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
            indices.extend(self.get_circle(outlier))
        indices = np.array(indices)
        indices = tuple(indices.T)
        data[indices] = 1
        try:
            layer = self.viewer.layers[self.viewer.layers.index(layername)]
            layer.data = data
        except:
            self.viewer.add_labels(data, name=layername)

    def get_cross(self, centroid):
        try:
            label_layer = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            )
        except ValueError:
            print("No segmentation layer found")
            return
        _, max_y, max_x = label_layer.data.shape
        z, y, x = centroid
        cross = []
        for i in range(26):
            if 0 < x - i < max_x and 0 < y - i < max_y:
                cross.append([z, y - i, x - i])
            if 0 < x + i < max_x and 0 < y - i < max_y:
                cross.append([z, y - i, x + i])
            if 0 < x - i < max_x and 0 < y + i < max_y:
                cross.append([z, y + i, x - i])
            if 0 < x + i < max_x and 0 < y + i < max_y:
                cross.append([z, y + i, x + i])

        return cross

    def get_circle(self, centroid, R=10):
        try:
            label_layer = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            )
        except ValueError:
            print("No segmentation layer found")
            return

        _, max_y, max_x = label_layer.data.shape
        z, y, x = centroid
        circle = []
        R_squared = R * R

        for dy in range(-R, R + 1):
            dx_limit = int((R_squared - dy**2) ** 0.5)
            for dx in range(-dx_limit, dx_limit + 1):
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < max_x and 0 <= new_y < max_y:
                    circle.append([z, new_y, new_x])

        return circle
