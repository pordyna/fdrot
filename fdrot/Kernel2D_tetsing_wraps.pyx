"""This is for unit tests."""
from .Kernel2D cimport *
from .c_defs cimport *
from typing import Tuple

def  wrap_interpolate_up(kernel, Py_ssize_t zz, Py_ssize_t rr, double radius) :
    return kernel.interpolate_up(zz, rr, radius)

def  wrap_interpolate_down(kernel, Py_ssize_t zz, Py_ssize_t rr, double radius):
    return kernel.interpolate_down(zz, rr, radius)

def  wrap_translator(kernel, Py_ssize_t start, Py_ssize_t stop):
    cdef Interval interval
    kernel.translator(start, stop, &interval)
    return interval.start, interval.stop

def  wrap_if_split(kernel, Py_ssize_t start, Py_ssize_t stop):
    return kernel.if_split(start, stop)
def  wrap_x_loop(kernel, Py_ssize_t zz, Py_ssize_t yy, Py_ssize_t x_start , Py_ssize_t x_stop, incl_down,
                 max_x_at_y, max_xx_at_y, double [:,:] output):
    #Setup needed attributes:
    kernel.max_x_at_y = max_x_at_y
    kernel.max_xx_at_y = max_xx_at_y
    return kernel.x_loop(zz=zz, yy=yy, x_start=x_start, x_stop=x_stop, output=output, incl_down=incl_down)

def  wrap_inside_y_loop(kernel, Py_ssize_t zz, Py_ssize_t yy ,  interval: Tuple[int,int],  double [:,:] output,
                         Py_ssize_t max_xx_at_y, max_x_at_y):
    cdef Interval interval_c
    interval_c.start, interval_c.stop = interval
    kernel.max_x_at_y = max_x_at_y
    kernel.max_xx_at_y = max_xx_at_y
    return kernel.inside_y_loop(zz=zz, yy=yy, interval=interval, output=output)

def  wrap_sum_line_over_pulse(kernel, Py_ssize_t yy, Py_ssize_t leading_start, Py_ssize_t leading_stop, str up_down,
                              Py_ssize_t max_xx_at_y, max_x_at_y, x_line = None):
    if x_line is not None:
        kernel.x_line = x_line
    cdef UpDown up_down_c
    if up_down == 'up':
        up_down_c = up
    if up_down == 'down':
        up_down_c = down
    else:
        raise ValueError("up_down has to be \"up\" or \"down\"")
    kernel.max_x_at_y = max_x_at_y
    kernel.max_xx_at_y = max_xx_at_y
    kernel.sum_line_over_pulse(yy=yy, leading_start=leading_start, leading_stop=leading_stop, up_down=up_down_c)
def  wrap_write_out(kernel, Py_ssize_t zz, Py_ssize_t yy, double summed_line, UpDown up_down):
    pass