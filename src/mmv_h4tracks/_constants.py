"""Constants for the mmv_h4tracks module."""
# Evaluation thresholds
IOU_THRESHOLD = 0.4
IOU_LOW_THRESHOLD = 0.2

# Tracking parameters
APPROX_INF = 65535
MAX_MATCHING_DIST = 45

# Assistant constants
DEFAULT_SPEED_THRESHOLD = 10
DEFAULT_SIZE_THRESHOLD = 5
DEFAULT_DISTANCE_THRESHOLD = 50
DEFAULT_SMALL_SIZE_THRESHOLD = 10

# Standard button texts
LINK_TEXT = "Link tracks"
UNLINK_TEXT = "Unlink tracks"
CONFIRM_TEXT = "Confirm"

# Dock status line (below progress bar)
STATUS_READY = "Ready"
STATUS_INITIALIZING = "Initializing…"
STATUS_AWAITING_USER = "Waiting for user input…"
STATUS_ADDING_CUSTOM_MODEL = "Adding custom model…"
STATUS_CLICK_TRACK_CELL = "Click a cell to track…"
STATUS_CLICK_REMOVE_CELL = "Click a cell to remove…"
STATUS_CLICK_SELECT_ID = "Click a cell to select its ID…"
STATUS_CLICK_SEPARATE = "Click a cell to separate…"
STATUS_CLICK_MERGE_FIRST = "Click the first cell to merge…"
STATUS_CLICK_MERGE_SECOND = "Click the second cell to merge into…"
STATUS_CLICK_LINK_CELLS = "Click cells to link, then Confirm…"
STATUS_CLICK_UNLINK_CELLS = "Click cells to unlink, then Confirm…"
STATUS_LINK_SELECT_MISSING = "Select a cell in each missing frame"
LINK_STATUS_MAX_SELECTED_FRAMES = 6

# Default napari Tracks layer name when creating/replacing tracks
DEFAULT_TRACKS_LAYER_NAME = "Tracks"

# Minimum acceptable track length
MIN_TRACK_LENGTH = 5

# Fraction of overlapping label pixels required to continue an overlap track
MIN_OVERLAP = 0.7

# Prefix for custom cellpose models
CUSTOM_MODEL_PREFIX = "custom_"
