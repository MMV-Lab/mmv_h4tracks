__version__ = "1.3.0"

__all__ = ["MMVH4TRACKS"]


def __getattr__(name: str):
    if name == "MMVH4TRACKS":
        from ._widget import MMVH4TRACKS

        return MMVH4TRACKS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
