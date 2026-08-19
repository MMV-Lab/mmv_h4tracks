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
from collections import defaultdict

from scipy.ndimage import label, center_of_mass

from ._logger import notify
from ._qt_utils import apply_napari_dark_theme, layer_as_numpy
from ._constants import (
    DEFAULT_SPEED_THRESHOLD,
    DEFAULT_SIZE_THRESHOLD,
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_SMALL_SIZE_THRESHOLD,
)
from ._analysis import calculate_size_single_track
import mmv_h4tracks._processing as processing


class AssistantWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.viewer = parent.viewer
        self.setup_ui()

    def setup_ui(self):
        self.setLayout(QVBoxLayout())
        apply_napari_dark_theme(self)

        ### QObjects

        # Labels
        label_speed = QLabel("Threshold speed change")
        label_size = QLabel("Threshold size change")
        label_distance = QLabel("Threshold edge distance")
        label_small_size = QLabel("Threshold size")
        label_FOI = QLabel("Frames of interest:")

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

        # Buttons
        btn_speed = QPushButton("Show speed outliers")
        btn_size = QPushButton("Show size outliers")
        btn_distance = QPushButton("Show noteworthy tracks")
        btn_relabel = QPushButton("Relabel all cells")
        btn_align_ids = QPushButton("Align segmentation IDs")
        btn_untracked = QPushButton("Show untracked cells")
        btn_tiny = QPushButton("Show small cells")

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

        btn_speed.clicked.connect(self.show_speed_outliers_on_click)
        btn_size.clicked.connect(self.show_size_outliers_on_click)
        btn_distance.clicked.connect(self.show_abrupt_tracks_on_click)
        btn_relabel.clicked.connect(self.relabel_cells_on_click)
        btn_align_ids.clicked.connect(self.align_ids_on_click)
        btn_untracked.clicked.connect(self.show_untracked_cells_on_click)
        btn_tiny.clicked.connect(self.show_tiny_cells_on_click)

        # LineEdits
        self.speed_lineedit = QLineEdit()
        self.speed_lineedit.setValidator(QDoubleValidator(0, 1000, 2))
        self.size_lineedit = QLineEdit()
        self.size_lineedit.setValidator(QDoubleValidator(0, 10000, 2))
        self.distance_lineedit = QLineEdit()
        self.distance_lineedit.setValidator(QDoubleValidator(0, 1000, 2))
        self.FOI_lineedit = QLineEdit()
        self.FOI_lineedit.setReadOnly(True)
        self.FOI_lineedit.setMaximumWidth(335)
        self.FOI_lineedit.setToolTip(foi_tooltip)

        self.tiny_lineedit = QLineEdit()
        self.tiny_lineedit.setValidator(QDoubleValidator(0, 10000, 2))

        self.speed_lineedit.setPlaceholderText(str(DEFAULT_SPEED_THRESHOLD))
        self.size_lineedit.setPlaceholderText(str(DEFAULT_SIZE_THRESHOLD))
        self.distance_lineedit.setPlaceholderText(str(DEFAULT_DISTANCE_THRESHOLD))
        self.tiny_lineedit.setPlaceholderText(str(DEFAULT_SMALL_SIZE_THRESHOLD))

        # Checkboxes
        self.checkbox_hidden = QCheckBox("Include hidden tracks")
        # does not apply to aligning segmentation IDs
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
        filters_layout.addWidget(self.FOI_lineedit, 6, 1, 1, 3)
        filters.setLayout(filters_layout)

        segmentation_adaptation = QGroupBox("Segmentation adaptation")
        # Ensure enough height for the title (avoid clipping descenders in "Segmentation adaptation")
        segmentation_adaptation.setMinimumHeight(130)
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
        content.layout().addStretch(1)
        self.layout().addWidget(content)

    def show_speed_outliers_on_click(self):
        self.parent.callback_handler.remove_callback_viewer()
        self.FOI_lineedit.setText("")
        try:
            tracks_layer = self.parent.selected_tracks_layer()
        except ValueError:
            print("No tracks layer found")
            return
        tracks = np.asarray(tracks_layer.data)
        try:
            threshold = float(self.speed_lineedit.text())
        except ValueError:
            threshold = DEFAULT_SPEED_THRESHOLD
        n_tracks = len(np.unique(tracks[:, 0])) if tracks.size else 0
        processing.run_with_dock_progress(
            self.parent,
            self._worker_speed_outliers,
            tracks,
            threshold,
            desc="Speed outliers",
            total=n_tracks,
            on_returned=self.display_outliers,
            skip_none_result=True,
        )

    def _worker_speed_outliers(self, tracks, threshold, reporter):
        speeds = self.parent.analysis_window._calculate_speed(tracks, reporter)
        outliers = []
        for result in speeds:
            if result[1] == 0:
                print(f"avoiding division by zero. would divide {result[3]}")
                continue
            if result[3] / result[1] > threshold:
                outliers.append(int(result[0]))
        return outliers

    def show_size_outliers_on_click(self):
        self.parent.callback_handler.remove_callback_viewer()
        self.FOI_lineedit.setText("")
        try:
            label_layer = self.parent.selected_labels_layer()
        except ValueError:
            print("No segmentation layer found")
            return
        segmentation = np.asarray(layer_as_numpy(label_layer))
        try:
            tracks_layer = self.parent.selected_tracks_layer()
        except ValueError:
            print("No tracks layer found")
            return
        tracks = np.asarray(tracks_layer.data)
        try:
            threshold = float(self.size_lineedit.text())
        except ValueError:
            threshold = DEFAULT_SIZE_THRESHOLD
        n_tracks = len(np.unique(tracks[:, 0])) if tracks.size else 0
        processing.run_with_dock_progress(
            self.parent,
            self._worker_size_outliers,
            tracks,
            segmentation,
            threshold,
            desc="Size outliers",
            total=n_tracks,
            on_returned=self.display_outliers,
            skip_none_result=True,
        )

    def _worker_size_outliers(self, tracks, segmentation, threshold, reporter):
        outliers = []
        for unique_id in np.unique(tracks[:, 0]):
            track = tracks[tracks[:, 0] == unique_id]
            result = calculate_size_single_track(track, segmentation)
            if result[3] != 0 and result[4] / result[3] > threshold:
                outliers.append(int(result[0]))
            reporter.increment()
        return outliers

    def show_abrupt_tracks_on_click(self):
        self.parent.callback_handler.remove_callback_viewer()
        self.FOI_lineedit.setText("")
        try:
            label_layer = self.parent.selected_labels_layer()
        except ValueError:
            print("No segmentation layer found")
            return
        segmentation = np.asarray(layer_as_numpy(label_layer))
        frames, y, x = segmentation.shape
        shape = (y, x)
        try:
            tracks_layer = self.parent.selected_tracks_layer()
        except ValueError:
            print("No tracks layer found")
            return
        tracks = np.asarray(tracks_layer.data)
        try:
            threshold = float(self.distance_lineedit.text())
        except ValueError:
            threshold = DEFAULT_DISTANCE_THRESHOLD

        # graph maps track_id -> list of parent_ids (lineage)
        graph = dict(getattr(tracks_layer, "graph", {}) or {})
        n_tracks = len(np.unique(tracks[:, 0])) if tracks.size else 0
        processing.run_with_dock_progress(
            self.parent,
            self._worker_abrupt_tracks,
            tracks,
            frames,
            shape,
            threshold,
            graph,
            desc="Noteworthy tracks",
            total=n_tracks,
            on_returned=self.display_outliers,
            skip_none_result=True,
        )

    def _worker_abrupt_tracks(self, tracks, frames, shape, threshold, graph, reporter):
        children_dict = defaultdict(list)
        for track_id, parent_ids in graph.items():
            for parent_id in parent_ids:
                children_dict[parent_id].append(track_id)

        outliers = []
        for id_ in np.unique(tracks[:, 0]):
            track = tracks[tracks[:, 0] == id_]
            # Beginning check: exclude if track has a parent (came from division)
            if (
                track[0, 1] > 0
                and not self.close_to_edge(
                    track[0, 2], track[0, 3], shape, threshold
                )
                and id_ not in graph
            ):
                outliers.append(int(id_))
            # End check: exclude if track has children (split into multiple tracks)
            elif (
                track[-1, 1] < frames - 1
                and not self.close_to_edge(
                    track[-1, 2], track[-1, 3], shape, threshold
                )
                and id_ not in children_dict
            ):
                outliers.append(int(id_))
            reporter.increment()
        return outliers

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
        self.parent.callback_handler.remove_callback_viewer()
        self.FOI_lineedit.setText("")
        try:
            label_layer = self.parent.selected_labels_layer()
        except ValueError:
            notify("No segmentation layer found")
            return

        raw = label_layer.data
        if isinstance(raw, (list, tuple)):
            if not raw:
                return
            data = np.asarray(raw[0])
            multiscale_list = list(raw)
            write_multiscale = True
        else:
            data = np.asarray(layer_as_numpy(label_layer))
            multiscale_list = None
            write_multiscale = False

        original_dtype = data.dtype
        if data.ndim == 2:
            data_work = data[np.newaxis, ...]
            squeeze_output = True
        elif data.ndim == 3:
            data_work = data
            squeeze_output = False
        else:
            notify(
                f"Relabel all cells needs 2D or 3D label data; got shape {data.shape}."
            )
            return

        n_frames = data_work.shape[0]

        def _on_returned(result):
            out_to_store, write_ms, ms_list = result
            if write_ms:
                ms_list[0] = out_to_store
                label_layer.data = ms_list
            else:
                label_layer.data = out_to_store

        processing.run_with_dock_progress(
            self.parent,
            self._worker_relabel_cells,
            data_work,
            squeeze_output,
            original_dtype,
            write_multiscale,
            multiscale_list,
            desc="Relabel cells",
            total=n_frames,
            on_returned=_on_returned,
            skip_none_result=True,
        )

    def _worker_relabel_cells(
        self,
        data_work,
        squeeze_output,
        original_dtype,
        write_multiscale,
        multiscale_list,
        reporter,
    ):
        n_frames = data_work.shape[0]
        relabeled_data = np.zeros(data_work.shape, dtype=np.int32)
        for frame in range(n_frames):
            current_frame = data_work[frame]
            unique_ids = np.unique(current_frame)
            unique_ids = unique_ids[unique_ids != 0]
            new_frame = np.zeros(current_frame.shape, dtype=np.int32)
            current_max_label = 0
            for uid in unique_ids:
                mask = current_frame == uid
                labeled_mask, _ = label(mask)
                labeled_mask = labeled_mask.astype(np.int32, copy=False)
                labeled_mask[labeled_mask > 0] += current_max_label
                current_max_label = int(labeled_mask.max())
                new_frame += labeled_mask
            relabeled_data[frame] = new_frame
            reporter.increment()

        out = relabeled_data[0] if squeeze_output else relabeled_data
        if original_dtype.kind in "iu":
            info = np.iinfo(original_dtype)
            if info.min <= int(out.min()) <= int(out.max()) <= info.max:
                out_to_store = out.astype(original_dtype, copy=False)
            else:
                out_to_store = out
        else:
            out_to_store = out

        return out_to_store, write_multiscale, multiscale_list

    def align_ids_on_click(self, saving=False):
        """Align the IDs of the tracks with the segmentation
        new_segmentation is generated from reference_segmentation and tracks
        reference_segmentation is not changed"""
        self.parent.callback_handler.remove_callback_viewer()
        self.FOI_lineedit.setText("")
        try:
            label_layer = self.parent.selected_labels_layer()
        except ValueError:
            notify("No segmentation layer found")
            return
        reference_segmentation = np.asarray(layer_as_numpy(label_layer))
        try:
            tracks_layer = self.parent.selected_tracks_layer()
        except ValueError:
            notify("No tracks layer found")
            return
        if self.parent.tracking_window.cached_tracks is not None:
            tracks = self.parent.tracking_window.cached_tracks
        else:
            tracks = tracks_layer.data
        tracks = np.asarray(tracks).copy()

        if 0 in tracks[:, 0]:
            tracks[:, 0] = tracks[:, 0] + 1

        offset = np.max([np.max(reference_segmentation), np.max(tracks[:, 0])])

        tracks_by_frame = defaultdict(list)
        for track in tracks:
            tracks_by_frame[track[1]].append(track)

        n_frames = reference_segmentation.shape[0]
        total = (n_frames if saving else 1) + len(tracks_by_frame)

        def _on_returned(new_segmentation):
            label_layer.data = new_segmentation

        processing.run_with_dock_progress(
            self.parent,
            self._worker_align_ids,
            reference_segmentation,
            tracks,
            saving,
            offset,
            dict(tracks_by_frame),
            desc="Align IDs",
            total=total,
            on_returned=_on_returned,
            skip_none_result=True,
        )

    def _worker_align_ids(
        self, reference_segmentation, tracks, saving, offset, tracks_by_frame, reporter
    ):
        def inflate_labels(reference_segmentation, starting_offset=1):
            output = np.zeros_like(reference_segmentation, dtype=np.int32)
            frame_offset = starting_offset

            for z in range(reference_segmentation.shape[0]):
                frame = reference_segmentation[z]
                labels = np.unique(frame)
                labels = labels[labels != 0]
                if labels.size == 0:
                    reporter.increment()
                    continue

                mapping = np.arange(frame_offset, frame_offset + labels.size)
                frame_offset += labels.size

                sort_idx = np.searchsorted(labels, frame)
                mask = frame != 0
                output[z][mask] = mapping[sort_idx[mask]]
                reporter.increment()

            return output

        def inflate_labels_simple(reference_segmentation):
            output = np.zeros_like(reference_segmentation, dtype=np.int32)
            output[reference_segmentation > 0] = reference_segmentation[
                reference_segmentation > 0
            ] + np.max([np.max(reference_segmentation), np.max(tracks[:, 0])])
            return output

        if saving:
            new_segmentation = inflate_labels(reference_segmentation, starting_offset=offset)
        else:
            new_segmentation = inflate_labels_simple(reference_segmentation)
            reporter.increment()

        for z in sorted(tracks_by_frame):
            z_tracks = tracks_by_frame[z]
            ref_slice = reference_segmentation[z]
            new_slice = new_segmentation[z]

            needs_centroids = any(
                new_slice[track[2], track[3]] == 0 for track in z_tracks
            )
            centroids_dict = None
            if needs_centroids:
                labels = np.unique(ref_slice)
                labels = labels[labels != 0]
                centroids = center_of_mass(ref_slice, labels=ref_slice, index=labels)
                centroids_dict = {
                    label_id: (int(round(c[0])), int(round(c[1])))
                    for label_id, c in zip(labels, centroids)
                }

            for track_id, _, y, x in z_tracks:
                val = new_slice[y, x]
                if val == 0:
                    if centroids_dict is None:
                        raise ValueError("Centroids not computed")
                    for label_id, center in centroids_dict.items():
                        if center == (y, x):
                            new_slice[ref_slice == label_id] = track_id
                            break
                    else:
                        raise ValueError("Could not find cell")
                else:
                    new_slice[new_slice == val] = track_id

            reporter.increment()

        return new_segmentation

    def show_untracked_cells_on_click(self):
        self.parent.callback_handler.remove_callback_viewer()
        try:
            label_layer = self.parent.selected_labels_layer()
        except ValueError:
            notify("No segmentation layer found")
            return
        segmentation = np.asarray(layer_as_numpy(label_layer))
        try:
            tracks_layer = self.parent.selected_tracks_layer()
        except ValueError:
            notify("No tracks layer found")
            return
        if (
            self.checkbox_hidden.isChecked()
            and self.parent.tracking_window.cached_tracks is not None
        ):
            tracks = self.parent.tracking_window.cached_tracks
        else:
            tracks = tracks_layer.data
        tracks = np.asarray(tracks)
        n_frames = segmentation.shape[0]

        def _on_returned(untracked):
            if len(untracked) > 0:
                unique_frames = set([coord[0] for coord in untracked])
                self.FOI_lineedit.setText(", ".join(map(str, sorted(unique_frames))))
            else:
                self.FOI_lineedit.setText("")
            self.mark_outliers(untracked, "Untracked cells")

        processing.run_with_dock_progress(
            self.parent,
            self._worker_untracked_cells,
            segmentation,
            tracks,
            desc="Untracked cells",
            total=n_frames,
            on_returned=_on_returned,
            skip_none_result=True,
        )

    def _worker_untracked_cells(self, segmentation, tracks, reporter):
        untracked = []
        for frame in range(segmentation.shape[0]):
            tracked_centroids = [
                [entry[2], entry[3]] for entry in tracks if entry[1] == frame
            ]
            tracked_ids = [
                segmentation[frame][coord[0], coord[1]] for coord in tracked_centroids
            ]
            untracked_ids = set(np.unique(segmentation[frame])) - set(tracked_ids) - {0}
            if untracked_ids:
                centroids = center_of_mass(
                    segmentation[frame],
                    labels=segmentation[frame],
                    index=list(untracked_ids),
                )
                for centroid in centroids:
                    centroid = [
                        frame,
                        int(np.rint(centroid[0])),
                        int(np.rint(centroid[1])),
                    ]
                    if centroid[1:] not in tracked_centroids:
                        untracked.append(centroid)
            reporter.increment()
        return untracked

    def show_tiny_cells_on_click(self):
        self.parent.callback_handler.remove_callback_viewer()
        try:
            label_layer = self.parent.selected_labels_layer()
        except ValueError:
            notify("No segmentation layer found")
            return
        segmentation = np.asarray(layer_as_numpy(label_layer))
        try:
            threshold = float(self.tiny_lineedit.text())
        except ValueError:
            threshold = DEFAULT_SMALL_SIZE_THRESHOLD
        n_frames = segmentation.shape[0]

        def _on_returned(tiny):
            unique_frames = set([coord[0] for coord in tiny])
            if len(unique_frames) > 0:
                self.FOI_lineedit.setText(", ".join(map(str, sorted(unique_frames))))
            else:
                self.FOI_lineedit.setText("")
            self.mark_outliers(tiny, "Tiny cells")

        processing.run_with_dock_progress(
            self.parent,
            self._worker_tiny_cells,
            segmentation,
            threshold,
            desc="Small cells",
            total=n_frames,
            on_returned=_on_returned,
            skip_none_result=True,
        )

    def _worker_tiny_cells(self, segmentation, threshold, reporter):
        tiny = []
        for frame in range(segmentation.shape[0]):
            for id_ in set(np.unique(segmentation[frame])) - {0}:
                if np.sum(segmentation[frame] == id_) < threshold:
                    all_pixels = np.where(segmentation[frame] == id_)
                    for y, x in zip(all_pixels[0], all_pixels[1]):
                        tiny.append([frame, y, x])
            reporter.increment()
        return tiny

    def mark_outliers(self, outliers, layername):
        try:
            label_layer = self.parent.selected_labels_layer()
        except ValueError:
            notify("No segmentation layer found")
            return
        data = np.zeros_like(label_layer.data)
        indices = []
        for outlier in outliers:
            indices.extend(self.get_plus(outlier))
        indices = np.array(indices)
        indices = tuple(indices.T)
        if len(indices) == 0:
            return
        data[indices] = 9
        try:
            layer = self.viewer.layers[self.viewer.layers.index(layername)]
            layer.data = data
        except ValueError:
            self.viewer.add_labels(data, name=layername)

    def get_plus(self, centroid):
        try:
            label_layer = self.parent.selected_labels_layer()
        except ValueError:
            notify("No segmentation layer found")
            return []
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
