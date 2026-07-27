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

# Default napari Tracks layer name when creating/replacing tracks
DEFAULT_TRACKS_LAYER_NAME = "Tracks"

# Minimum acceptable track length
MIN_TRACK_LENGTH = 5

# Fraction of overlapping label pixels required to continue an overlap track
MIN_OVERLAP = 0.7

# Prefix for custom cellpose models
CUSTOM_MODEL_PREFIX = "custom_"
