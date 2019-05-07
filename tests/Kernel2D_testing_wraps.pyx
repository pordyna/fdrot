"""This is for unit tests."""
from fdrot.Kernel2D cimport *
from fdrot.c_defs cimport *
from typing import Tuple

def  wrap_interpolate_up(Kernel2D kernel, Py_ssize_t zz, Py_ssize_t rr, double radius):
    return kernel.interpolate_up(zz, rr, radius)

def  wrap_interpolate_down(Kernel2D kernel, Py_ssize_t zz, Py_ssize_t rr, double radius):
    return kernel.interpolate_down(zz, rr, radius)

def  wrap_translator(Kernel2D kernel, Py_ssize_t start, Py_ssize_t stop):
    cdef Interval interval
    side = kernel.translator(start, stop, &interval)
    return interval.start, interval.stop, side

def  wrap_if_split(Kernel2D kernel, Py_ssize_t start, Py_ssize_t stop):
    return kernel.if_split(start, stop)