"""Shared test configuration.

TensorFlow's startup banner is written to stderr before any of our code runs
and buries short test output, so it is silenced here rather than in each test.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
