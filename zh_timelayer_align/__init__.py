from .timelayer import CUT, apply_timelayer

__all__ = ["CUT", "apply_timelayer", "CtcAligner"]


def __getattr__(name):  # lazy: importing the rules must not require torch
    if name == "CtcAligner":
        from .aligner import CtcAligner
        return CtcAligner
    raise AttributeError(name)
