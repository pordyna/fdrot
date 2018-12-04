# cython: language_level=3

from .c_defs cimport Interval
from .c_defs cimport UpDown

cdef class Kernel2D:
    cdef:
        double [:, ::1] input_d
        double [:, ::1] output_d
        bint interpolation
        bint inc_sym
        bint inc_sym_only_vertical_middle
        bint add
        bint _odd
        Py_ssize_t r_len
        Py_ssize_t s_len
        Py_ssize_t r_len_half
        double factor
        Py_ssize_t offset
        Py_ssize_t write_offset
        Py_ssize_t y_loop_start
        float r_offset
        float x_offset
        float y_offset
        float max_radius_sqd
        Py_ssize_t x_count_offset
        Py_ssize_t max_xx_at_y
        double max_x_at_y
        bint edge
        double [::1] pulse
        Py_ssize_t pulse_len
        Py_ssize_t pulse_step
        double [:,::1] x_line
        Interval global_interval
        double y_sqrd
    cdef double interpolate_up(self, Py_ssize_t zz, Py_ssize_t rr, double radius) except *
    cdef double interpolate_down(self, Py_ssize_t zz, Py_ssize_t rr, double radius) except *
    cdef int translator(self, Py_ssize_t start, Py_ssize_t stop, Interval* p_converted) except-1
    cdef bint if_split(self, Py_ssize_t start, Py_ssize_t stop) except *
    cdef short x_loop(self, Py_ssize_t zz, Py_ssize_t yy,
                 Py_ssize_t x_start , Py_ssize_t x_stop, double [:,:] output, bint incl_down=*) except -1
    cdef short inside_y_loop (self, Py_ssize_t zz, Py_ssize_t yy ,  Interval interval,  double [:,:] output) except -1
    cdef double sum_line_over_pulse(self, Py_ssize_t yy, Py_ssize_t leading_start, Py_ssize_t leading_stop, UpDown up_down) except *
    cdef short write_out(self, Py_ssize_t zz, Py_ssize_t yy, double summed_line, UpDown up_down) except -1
    cpdef short propagate_step(self, Py_ssize_t leading_start, Py_ssize_t leading_stop) except -1


