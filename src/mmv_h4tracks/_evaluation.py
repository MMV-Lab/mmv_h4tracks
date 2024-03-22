import numpy as np
import math
from multiprocessing import Pool

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QGridLayout,
    QLabel,
    QSizePolicy,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QAbstractScrollArea,
)
from scipy import ndimage

from ._logger import notify
from ._grabber import grab_layer
from mmv_h4tracks._logger import handle_exception
from mmv_h4tracks import IOU_THRESHOLD


class EvaluationWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.parent = parent
        self.viewer = parent.viewer

        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI of the evaluation window."""
        # Labels
        evaluation_label = QLabel("Frames to evaluate:")

        # Buttons
        evaluate_segmentation = QPushButton("Evaluate Segmentation")
        evaluate_tracking = QPushButton("Evaluate Tracking")

        evaluate_segmentation.clicked.connect(self.evaluate_segmentation)
        evaluate_tracking.clicked.connect(self.evaluate_tracking)

        # Lineedits
        self.evaluation_limit_lower = QLineEdit()
        self.evaluation_limit_upper = QLineEdit()

        # Table
        segmentation_table = QTableWidget(2, 3)
        segmentation_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        segmentation_table.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        segmentation_table.setHorizontalHeaderLabels(
            ["IoU Score", "DICE Score", "F1 Score"]
        )
        segmentation_table.setVerticalHeaderLabels(["Range", "All"])
        segmentation_table.setItem(0, 0, QTableWidgetItem())
        segmentation_table.setItem(0, 1, QTableWidgetItem())
        segmentation_table.setItem(0, 2, QTableWidgetItem())
        segmentation_table.setItem(1, 0, QTableWidgetItem())
        segmentation_table.setItem(1, 1, QTableWidgetItem())
        segmentation_table.setItem(1, 2, QTableWidgetItem())

        tracking_table = QTableWidget(5, 5)
        tracking_table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        tracking_table.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        height = tracking_table.rowHeight(0)
        tracking_table.setColumnWidth(0, 125)
        tracking_table.setColumnWidth(1, 40)
        tracking_table.setColumnWidth(2, height)
        tracking_table.setColumnWidth(3, 115)
        tracking_table.setColumnWidth(4, 45)
        tracking_table.setCellWidget(0, 0, QLabel("False Positives"))
        tracking_table.setCellWidget(1, 0, QLabel("False Negatives"))
        tracking_table.setCellWidget(2, 0, QLabel("Split Cells"))
        tracking_table.setCellWidget(0, 3, QLabel("Added Edges"))
        tracking_table.setCellWidget(1, 3, QLabel("Removed Edges"))
        tracking_table.setCellWidget(4, 0, QLabel("<b>Total Fault Value</b>"))
        tracking_table.setItem(0, 1, QTableWidgetItem())
        tracking_table.setItem(1, 1, QTableWidgetItem())
        tracking_table.setItem(2, 1, QTableWidgetItem())
        tracking_table.setItem(0, 4, QTableWidgetItem())
        tracking_table.setItem(1, 4, QTableWidgetItem())
        tracking_table.setItem(4, 1, QTableWidgetItem())
        tracking_table.setItem(4, 3, QTableWidgetItem())

        # Spacer
        self.v_spacer = QWidget()
        self.v_spacer.setFixedWidth(4)
        self.v_spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        h_spacer_1 = QWidget()
        h_spacer_1.setFixedHeight(0)
        h_spacer_1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_spacer_2 = QWidget()
        h_spacer_2.setFixedHeight(0)
        h_spacer_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_spacer_3 = QWidget()
        h_spacer_3.setFixedHeight(0)
        h_spacer_3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # QGroupBoxes
        evaluation = QGroupBox("_________")
        evaluation_layout = QGridLayout()
        evaluation_layout.addWidget(h_spacer_1, 0, 0, 1, -1)
        evaluation_layout.addWidget(evaluation_label, 1, 0)
        evaluation_layout.addWidget(self.evaluation_limit_lower, 1, 1)
        evaluation_layout.addWidget(QLabel("-"), 1, 2)
        evaluation_layout.addWidget(self.evaluation_limit_upper, 1, 3)
        evaluation_layout.addWidget(evaluate_segmentation, 2, 0)
        evaluation_layout.addWidget(evaluate_tracking, 2, 1, 1, -1)

        evaluation.setLayout(evaluation_layout)

        self.segmentation_results = QGroupBox("Segmentation Results")
        self.segmentation_results.hide()
        segmentation_results_layout = QGridLayout()
        segmentation_results_layout.addWidget(h_spacer_2, 0, 0)
        segmentation_results_layout.addWidget(segmentation_table, 1, 0)

        self.segmentation_results.setLayout(segmentation_results_layout)

        self.tracking_results = QGroupBox("Tracking Results")
        self.tracking_results.hide()
        tracking_results_layout = QGridLayout()
        tracking_results_layout.addWidget(h_spacer_3, 0, 0)
        tracking_results_layout.addWidget(tracking_table, 1, 0)

        self.tracking_results.setLayout(tracking_results_layout)

        # Build the main layout
        content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.addWidget(evaluation)
        content_layout.addWidget(self.v_spacer)

        content.setLayout(content_layout)
        layout = QVBoxLayout()
        layout.addWidget(content)
        self.setLayout(layout)

    def evaluate_segmentation(self):
        """
        Evaluate the segmentation results against the curated segmentation.
        """
        try:
            gt_seg = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            ).data
        except ValueError as exc:
            handle_exception(exc)
            return
        eval_seg = self.parent.initial_layers[0]
        if eval_seg is None:
            notify("Segmentation and Tracks must be imported from zarr currently! (Drag and drop will be supported in the future). As a work-around for now export your data as zarr and import it.")
            return
            
        self.evaluate_curated_segmentation(gt_seg, eval_seg)
        content = self.layout().itemAt(0).widget()
        if not self.segmentation_results.isVisible():
            content.layout().replaceWidget(self.v_spacer, self.segmentation_results)
            content.layout().addWidget(self.v_spacer)
            self.segmentation_results.show()

    def evaluate_curated_segmentation(self, gt_seg, eval_seg):
        """Evaluate the curated ground truth segmentation against the automatically generated segmentation."""
        try:
            lower_bound = int(self.evaluation_limit_lower.text())
        except ValueError:
            lower_bound = 0
        try:
            upper_bound = int(self.evaluation_limit_upper.text())
        except ValueError:
            upper_bound = gt_seg.shape[0] - 1
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound
        if lower_bound < 0:
            lower_bound = 0
        if upper_bound >= gt_seg.shape[0]:
            upper_bound = gt_seg.shape[0] - 1

        if lower_bound == upper_bound:
            return

        ### Calculate the scores
        # IoU
        range_iou = self._calculate_iou(
            gt_seg[lower_bound:upper_bound + 1], eval_seg[lower_bound:upper_bound + 1]
        )
        all_iou = self._calculate_iou(gt_seg, eval_seg)

        # DICE
        range_dice = self._calculate_dice(
            gt_seg[lower_bound:upper_bound + 1], eval_seg[lower_bound:upper_bound + 1]
        )
        all_dice = self._calculate_dice(gt_seg, eval_seg)

        # F1
        range_f1 = self._calculate_f1(
            gt_seg[lower_bound:upper_bound + 1], eval_seg[lower_bound:upper_bound + 1]
        )
        all_f1 = self._calculate_f1(gt_seg, eval_seg)

        ### Update the table
        table = self.segmentation_results.layout().itemAt(1).widget()
        table.setVerticalHeaderItem(
            0, QTableWidgetItem(f"{lower_bound} - {upper_bound}")
        )
        table.item(0, 0).setText(f"{round_half_up(range_iou, 3):.3f}")
        table.item(0, 1).setText(f"{round_half_up(range_dice, 3):.3f}")
        table.item(0, 2).setText(f"{round_half_up(range_f1, 3):.3f}")
        table.item(1, 0).setText(f"{round_half_up(all_iou, 3):.3f}")
        table.item(1, 1).setText(f"{round_half_up(all_dice, 3):.3f}")
        table.item(1, 2).setText(f"{round_half_up(all_f1, 3):.3f}")

    def _calculate_iou(self, gt_seg, eval_seg):
        """Calculate the IoU score for two given segmentations."""
        intersection = np.sum(np.logical_and(gt_seg, eval_seg).flat)
        union = np.sum(np.logical_or(gt_seg, eval_seg).flat)
        return intersection / union

    def _calculate_dice(self, gt_seg, eval_seg):
        """Calculate the DICE score for two given segmentations."""
        intersection = np.sum(np.logical_and(gt_seg, eval_seg).flat)
        return (
            2 * intersection / (np.count_nonzero(gt_seg) + np.count_nonzero(eval_seg))
        )

    def _calculate_f1(self, gt_seg, eval_seg):
        """Calculate the F1 score for two given segmentations."""
        intersection = np.sum(np.logical_and(gt_seg, eval_seg).flat)
        union = np.sum(np.logical_or(gt_seg, eval_seg).flat)
        return 2 * intersection / (intersection + union)

    def evaluate_tracking(self):
        """Evaluate the tracking results against the curated tracks."""
        try:
            gt_tracks_layer = grab_layer(
                self.viewer, self.parent.combobox_tracks.currentText()
            )
            gt_seg = grab_layer(
                self.viewer, self.parent.combobox_segmentation.currentText()
            ).data
        except ValueError as exc:
            handle_exception(exc)
            return
        eval_tracks = self.parent.initial_layers[1]
        eval_seg = self.parent.initial_layers[0]
        if eval_tracks is None or eval_seg is None:
            notify("Segmentation and Tracks must be imported from zarr currently! (Drag and drop will be supported in the future). As a work-around for now export your data as zarr and import it.")
            return
        self.evaluate_curated_tracking(gt_tracks_layer, gt_seg, eval_tracks, eval_seg)

        content = self.layout().itemAt(0).widget()
        if not self.tracking_results.isVisible():
            content.layout().replaceWidget(self.v_spacer, self.tracking_results)
            content.layout().addWidget(self.v_spacer)
            self.tracking_results.show()

    def evaluate_curated_tracking(self, gt_tracks_layer, gt_seg, eval_tracks, eval_seg):
        """Evaluate the curated ground truth tracking against the automatically generated tracking."""
        try:
            lower_bound = int(self.evaluation_limit_lower.text())
        except ValueError:
            lower_bound = 0
        try:
            upper_bound = int(self.evaluation_limit_upper.text())
        except ValueError:
            upper_bound = gt_seg.shape[0] - 1
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound
        if lower_bound < 0:
            lower_bound = 0
        if upper_bound >= gt_seg.shape[0]:
            upper_bound = gt_seg.shape[0] -1

        if lower_bound == upper_bound:
            return

        self.adjust_centroids(gt_seg, gt_tracks_layer, (lower_bound, upper_bound))

        gt_tracks = gt_tracks_layer.data
        mask = (gt_tracks[:, 1] >= lower_bound) & (gt_tracks[:, 1] < upper_bound)
        gt_tracks = gt_tracks[mask]
        gt_tracks[:,1] -= lower_bound
        gt_seg = gt_seg[lower_bound : upper_bound + 1]
        mask = (eval_tracks[:, 1] >= lower_bound) & (eval_tracks[:, 1] < upper_bound)
        eval_tracks = eval_tracks[mask]
        eval_tracks[:,1] -= lower_bound
        eval_seg = eval_seg[lower_bound : upper_bound + 1]
        fp = self.get_segmentation_fault(gt_seg, eval_seg, get_false_positives)
        fn = self.get_segmentation_fault(gt_seg, eval_seg, get_false_negatives)
        sc = self.get_segmentation_fault(gt_seg, eval_seg, get_split_cells)
        de, ae = self.get_track_fault(gt_seg, gt_tracks, eval_seg, eval_tracks, lower_bound)

        fv = fp + fn * 10 + sc * 5 + de + ae * 1.5

        ### Update the table
        table = self.tracking_results.layout().itemAt(1).widget()
        table.item(0, 1).setText(str(fp))
        table.item(1, 1).setText(str(fn))
        table.item(2, 1).setText(str(sc))
        table.item(0, 4).setText(str(ae))
        table.item(1, 4).setText(str(de))

        table.item(4, 1).setText(str(fv))
        table.item(4, 3).setText(f"for slices {lower_bound} - {upper_bound}")

    def adjust_centroids(self, segmentation, tracks_layer, bounds):
        """Adjust the centroids of the tracks to the current segmentation."""
        tracks = tracks_layer.data
        for row in tracks:
            _, z, y, x = row
            if z < bounds[0] or z >= bounds[1]:
                continue
            segmentation_id = segmentation[z, y, x]
            if segmentation_id == 0:
                continue
            centroid = ndimage.center_of_mass(
                segmentation[z], labels=segmentation[z], index=segmentation_id
            )
            row[2] = int(np.rint(centroid[0]))
            row[3] = int(np.rint(centroid[1]))

        tracks_layer.data = tracks

    def get_segmentation_fault(self, gt_seg, eval_seg, evaluation_function):
        """Calculate the segmentation fault value.
        Designed for either false positives, false negatives or split cells."""
        faults = 0
        if np.array_equal(gt_seg, eval_seg):
            return faults
        AMOUNT_OF_PROCESSES = self.parent.get_process_limit()

        slice_pairs = []
        for i in range(len(gt_seg)):
            slice_pairs.append((gt_seg[i], eval_seg[i]))
        with Pool(AMOUNT_OF_PROCESSES) as p:
            faults = sum(p.starmap(evaluation_function, slice_pairs))

        return faults

    def get_track_fault(self, gt_seg, gt_tracks, eval_seg, eval_tracks, lower_bound=0):
        """
        Calculate the track fault value.
        Calculates both deleted edges and added edges.
        """
        de, ae = 0, 0
        if np.array_equal(gt_tracks, eval_tracks):
            return de, ae
        gt_connections = []
        eval_connections = []
        fault_value = [de, ae]
        for connections, tracks, segmentations, fault_index in zip(
            [gt_connections, eval_connections],
            [(gt_tracks, eval_tracks), (eval_tracks, gt_tracks)],
            [(gt_seg, eval_seg), (eval_seg, gt_seg)],
            [1, 0],
        ):
            base_tracks, comparison_tracks = tracks
            base_segmentation, comparison_segmentation = segmentations
            for i in range(len(base_tracks) - 1):
                if (
                    base_tracks[i][0] == base_tracks[i + 1][0]
                    and base_tracks[i][1] + 1 == base_tracks[i + 1][1]
                ):
                    connections.append((base_tracks[i][1:4], base_tracks[i + 1][1:4]))
            for connection in connections:
                old_cell_base = base_segmentation[
                    connection[0][0], connection[0][1], connection[0][2]
                ]
                new_cell_base = base_segmentation[
                    connection[1][0], connection[1][1], connection[1][2]
                ]
                old_cell_comparison = get_matching_cell(
                    base_segmentation,
                    comparison_segmentation,
                    old_cell_base,
                    connection[0][0],
                )
                new_cell_comparison = get_matching_cell(
                    base_segmentation,
                    comparison_segmentation,
                    new_cell_base,
                    connection[1][0],
                )

                if old_cell_comparison == 0 or new_cell_comparison == 0:
                    fault_value[fault_index] += 1
                    continue

                old_centroid_in_slice = ndimage.center_of_mass(
                    comparison_segmentation[connection[0][0]],
                    labels=segmentations[1][connection[0][0]],
                    index=old_cell_comparison,
                )
                old_centroid = [
                    connection[0][0],
                    int(np.rint(old_centroid_in_slice[0])),
                    int(np.rint(old_centroid_in_slice[1])),
                ]

                new_centroid_in_slice = ndimage.center_of_mass(
                    comparison_segmentation[connection[1][0]],
                    labels=segmentations[1][connection[1][0]],
                    index=new_cell_comparison,
                )
                new_centroid = [
                    connection[1][0],
                    int(np.rint(new_centroid_in_slice[0])),
                    int(np.rint(new_centroid_in_slice[1])),
                ]

                # Check if the centroids are connected
                if not is_connected(comparison_tracks, old_centroid, new_centroid):
                    fault_value[fault_index] += 1
                    continue

        de, ae = fault_value
        return de, ae


def round_half_up(n, decimals=0):
    """Round a number to a given number of decimals."""
    multiplier = 10**decimals
    return math.floor(n * multiplier + 0.5) / multiplier


def get_false_positives(gt_slice, eval_slice):
    """Calculate the number of false positives in a given slice.
    Counts as false positive if:
    - A cell is present in the evaluated slice but not in the ground truth slice.
    - The highest IoU score of a cell in the evaluated slice is below 0.2.
    - The highest IoU score of a cell in the evaluated slice is below 0.4 and there are multiple candidate cells.
    """
    fp = 0
    if np.array_equal(gt_slice, eval_slice):
        return fp

    for cell in np.unique(eval_slice):
        if cell == 0:
            continue

        # Calculate the IoU scores for the current cell
        iou_scores = []
        for gt_cell in np.unique(gt_slice):
            if gt_cell == 0:
                continue
            intersection = np.sum(
                np.logical_and(gt_slice == gt_cell, eval_slice == cell).flat
            )
            if intersection == 0:
                continue
            union = np.sum(np.logical_or(gt_slice == gt_cell, eval_slice == cell).flat)
            iou_scores.append(intersection / union)
        iou_scores.sort(reverse=True)
        if (
            len(iou_scores) < 1
            or iou_scores[0] < 0.4
            and not (len(iou_scores) > 1 and iou_scores[1] < 0.2)
        ):
            fp += 1
    return fp


def get_false_negatives(ground_truth_slice, evaluated_slice):
    """Calculate the number of false negatives in a given slice.
    Counts as false negative if:
    - A cell is present in the ground truth slice but not in the evaluated slice.
    - The highest IoU score of a cell in the ground truth slice is below 0.4.
    - The cell in the evaluation slice with the hightest IoU with the ground truth cell has a higher IoU with a different ground truth cell.
    Counts as half a false negative if:
    - The cell in the evaluation slice with the hightest IoU with the ground truth cell has an equal IoU with a different ground truth cell.
    """
    # Initialize the count of false negatives
    false_negatives = 0

    # If the ground truth and evaluated slices are identical, return 0
    if np.array_equal(ground_truth_slice, evaluated_slice):
        return false_negatives

    # Iterate over unique identifiers in the ground truth slice
    for identifier in np.unique(ground_truth_slice):
        if identifier == 0:
            continue

        # Get the Intersection over Union (IoU) on all cells in the evaluated slice at locations of the identifier in the ground truth
        id_locations = np.where(ground_truth_slice == identifier)
        evaluated_identifiers = np.unique(evaluated_slice[id_locations])

        max_intersection_over_union = -1
        best_match_evaluated_id = -1

        for evaluated_id in evaluated_identifiers:
            if evaluated_id == 0:
                continue

            intersection = np.sum(
                np.logical_and(
                    ground_truth_slice == identifier, evaluated_slice == evaluated_id
                ).flat
            )
            union = np.sum(
                np.logical_or(
                    ground_truth_slice == identifier, evaluated_slice == evaluated_id
                ).flat
            )
            intersection_over_union = intersection / union

            if intersection_over_union > max_intersection_over_union:
                max_intersection_over_union = intersection_over_union
                best_match_evaluated_id = evaluated_id

        if not max_intersection_over_union > 0.4:
            false_negatives += 1
            # Largest IoU too small
            continue

        reverse_id_locations = np.where(evaluated_slice == best_match_evaluated_id)
        reverse_identifiers = np.unique(ground_truth_slice[reverse_id_locations])

        reverse_intersections_over_union = []

        for reverse_id in reverse_identifiers:
            if reverse_id == 0:
                continue

            intersection = np.sum(
                np.logical_and(
                    evaluated_slice == best_match_evaluated_id,
                    ground_truth_slice == reverse_id,
                ).flat
            )
            union = np.sum(
                np.logical_or(
                    evaluated_slice == best_match_evaluated_id,
                    ground_truth_slice == reverse_id,
                ).flat
            )
            reverse_intersection_over_union = intersection / union
            reverse_intersections_over_union.append(reverse_intersection_over_union)

        if len(reverse_intersections_over_union) < 2:
            # Reverse IoU only has one match
            continue

        if max(reverse_intersections_over_union) > max_intersection_over_union:
            false_negatives += 1
            # Largest IoU large enough, but reverse IoU has larger
            continue

        max_reverse_intersection_over_union = max(reverse_intersections_over_union)
        reverse_intersections_over_union.remove(max(reverse_intersections_over_union))

        if max(reverse_intersections_over_union) == max_reverse_intersection_over_union:
            false_negatives += 0.5

    if int(false_negatives) != false_negatives:
        raise ValueError("False negatives don't sum up to whole integer")

    return int(false_negatives)


def get_split_cells(gt_slice, eval_slice):
    """Calculate the number of split cells in a given slice.
    Counts as split cell if:
    - A cell in the ground truth slice has two or more cells in the evaluated slice with an IoU score above 0.2.
    """
    sc = 0
    if np.array_equal(gt_slice, eval_slice):
        return 0

    for cell in np.unique(eval_slice):
        if cell == 0:
            continue

        # Calculate the IoU scores for the current cell
        iou_scores = []
        for gt_cell in np.unique(gt_slice):
            if gt_cell == 0:
                continue
            intersection = np.sum(
                np.logical_and(eval_slice == cell, gt_slice == gt_cell).flat
            )
            if intersection == 0:
                continue
            union = np.sum(np.logical_or(eval_slice == cell, gt_slice == gt_cell).flat)
            iou_scores.append(intersection / union)
        iou_scores.sort(reverse=True)
        if len(iou_scores) > 1 and iou_scores[1] > 0.2:
            sc += 1
    return sc


def get_matching_cell(base_layer, comparison_layer, base_id, z):
    """Find the cell in the comparison layer with the highest IoU score with the cell in the base layer.
    IoU has to be above IOU_THRESHOLD."""
    base_slice = base_layer[z]
    comparison_slice = comparison_layer[z]

    base_locations = np.where(base_slice == base_id)
    comparison_ids = np.unique(comparison_slice[base_locations])

    max_intersection_over_union = 0
    best_match_id = 0

    for comparison_id in comparison_ids:
        if comparison_id == 0:
            continue

        intersection = np.sum(
            np.logical_and(
                base_slice == base_id, comparison_slice == comparison_id
            ).flat
        )
        union = np.sum(
            np.logical_or(base_slice == base_id, comparison_slice == comparison_id).flat
        )
        intersection_over_union = intersection / union

        if intersection_over_union > max_intersection_over_union:
            max_intersection_over_union = intersection_over_union
            best_match_id = comparison_id

    if max_intersection_over_union < IOU_THRESHOLD:
        return 0

    return best_match_id


def is_connected(tracks, old_centroid, new_centroid):
    """Check if the old and new centroids are connected in the tracks."""
    for i in range(len(tracks) - 1):
        if all(old_centroid[j] == tracks[i, j + 1] for j in range(3)):
            return all(new_centroid[j] == tracks[i + 1, j + 1] for j in range(3))
    return False
