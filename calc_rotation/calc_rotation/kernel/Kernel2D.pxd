# cython: language_level=3

from .c_defs cimport Interval
from .c_defs cimport UpDown
from .c_defs cimport  EvenOddControls


cdef class Kernel2D:
    cdef:
        double [:, ::1] input_d
        double [:, ::1] output_d
        bint interpolation
        bint inc_sym
        bint inc_sym_only_vertical_middle
        bint add # here?
        bint _odd
        Py_ssize_t r_len
        Py_ssize_t s_len
        Py_ssize_t r_len_half
        double factor
        EvenOddControls controls
        double [::1] pulse
        Py_ssize_t pulse_len
        Py_ssize_t pulse_step
        double [:,::1] x_line
        Interval global_interval
    cdef double _interpolate_up(self, Py_ssize_t zz, Py_ssize_t rr, double radius)
    cdef double _interpolate_down(self, Py_ssize_t zz, Py_ssize_t rr, double radius)
    cdef int translator(self, Py_ssize_t start, Py_ssize_t stop, Interval* p_converted) except-1
    cdef bint if_split(self, Py_ssize_t start, Py_ssize_t stop)
    cdef _x_loop(self, Py_ssize_t zz, Py_ssize_t yy,
                 Py_ssize_t x_start , Py_ssize_t x_stop, double [:,:] output, bint incl_down=*)
    cdef _inside_y_loop (self, Py_ssize_t zz, Py_ssize_t yy ,  Interval interval,  double [:,:] output)
    cdef double _sum_line_over_pulse(self, Py_ssize_t leading_start, Py_ssize_t leading_stop, UpDown up_down)
    cdef _write_out(self, Py_ssize_t zz, Py_ssize_t yy, double summed_line, UpDown up_down)
    cpdef rotate_slice(self, Py_ssize_t leading_start, Py_ssize_t leading_stop)


