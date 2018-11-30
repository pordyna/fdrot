# cython: language_level=3
"""c_defs.pxd
This file is a part of the Fdrot package.
"""
ctypedef struct Interval:
    Py_ssize_t start
    Py_ssize_t stop


ctypedef enum UpDown:
    up
    down

ctypedef enum Side:
    left
    right

