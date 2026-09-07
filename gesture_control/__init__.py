"""Control the Windows mouse by pointing at the screen with your hand."""

import os

# OpenCV, MediaPipe and TensorFlow-Lite all log heavily on import, and probing
# camera indices produces a WARN per backend per index.  These have to be set
# before those libraries load, so they live here rather than in a function.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

__version__ = "1.0.0"
