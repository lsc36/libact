"""
Concrete labeler classes.
"""

from .ideal_labeler import IdealLabeler
try:
    from .interactive_labeler import InteractiveLabeler
except ImportError as import_error:
    raise ImportError("Error importing matplotlib."
                      "InteractiveLabeler not supported.") from import_error
