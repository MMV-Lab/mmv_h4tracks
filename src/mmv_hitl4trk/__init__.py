
__version__ = "1.0.0"

# Evaluation threshold
IOU_THRESHOLD = .4

# Tracking parameters
APPROX_INF = 65535
MAX_MATCHING_DIST = 45

from ._widget import MMVHITL4TRK

__all__ = ["MMVHITL4TRK"]
