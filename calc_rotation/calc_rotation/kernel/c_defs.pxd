# cython: language_level=3

ctypedef struct Interval:
    Py_ssize_t start
    Py_ssize_t stop

cdef inline double d_square(double x):
    return x*x

ctypedef enum UpDown:
    up
    down

ctypedef enum Side:
    left
    right

cdef struct EvenOddControls:
    Py_ssize_t offset
    Py_ssize_t write_offset
    Py_ssize_t y_loop_start
    float r_offset
    float x_offset
    float y_offset

