from libc.math cimport sqrt
from libc.math cimport abs
from cython.view cimport array as cvarray
# from .c_rotation cimport Interval

ctypedef struct Interval:
    Py_ssize_t start
    Py_ssize_t stop
cimport cython
@cython.profile(False)
cdef inline double d_square(double x): # maybe use instead of x**2, should be faster
    return x*x

cdef inline double sum_arr(cython.floating [::1] arr):
    cdef double sum = 0
    cdef ssize_t ii
    for ii in range(arr.shape[0]):
        sum+= arr[ii]

cdef struct EvenOddControls:
    Py_ssize_t offset
    Py_ssize_t write_offset
    Py_ssize_t y_loop_start
    float r_offset
    float x_offset
    float y_offset

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
        double [::1] x_line

    def __cinit__(self, double [:, ::1] input_d, double [:, ::1] output_d, double [::1] pulse , Py_ssize_t pulse_step=1,
                  bint interpolation=0, bint inc_sym=0, bint inc_sym_only_vertical_middle=0, bint add=1):

        self.input_d = input_d
        self.output_d = output_d
        self.interpolation = interpolation
        self.inc_sym = inc_sym
        self.inc_sym_only_vertical_middle = inc_sym_only_vertical_middle
        self.add = add

        self.r_len = input_d.shape[1]
        self.s_len = input_d.shape[0]

        assert self.r_len == output_d.shape[0]
        assert self.s_len == output_d.shape[1]

        self.r_len_half = self.r_len // 2

        if self.r_len%2 == 0:
            self._odd = 0
            self.controls.offset = 1
            self.controls.write_offset = 1
            self.controls.y_loop_start =  0
            self.controls.r_offset = 0.5
            self.controls.x_offset = - 0.5
            self.controls.y_offset = 0.5
        else:
            self._odd = 1
            self.controls.offset = 0
            self.controls.write_offset = 0
            self.controls.y_loop_start =  1
            self.controls.r_offset = 0
            self.controls.x_offset = 0
            self.controls.y_offset = 0

        # Pulse:
        self.pulse = pulse
        self.pulse_len = pulse.shape[0]
        self.pulse_step = pulse_step
        self.x_line = cvarray(shape=(self.r_len,), itemsize=sizeof(double), format="d")


    cdef double _interpolate_up(self, Py_ssize_t zz, Py_ssize_t rr, double radius):
        cdef double value_1, value_2
        cdef double value_diff

        value_1 = self.input_d[zz, self.r_len_half + rr]
        if not self.interpolation:
            return value_1

        value_2 = self.input_d[zz, self.r_len_half + rr +1]
        value_diff = value_2 - value_1
        return value_1 + value_diff * (radius - (rr + self.controls.r_offset ))


    cdef double _interpolate_down(self, Py_ssize_t zz, Py_ssize_t rr, double radius):
        cdef double value_1, value_2
        cdef double value_diff

        value_1 = self.input_d[zz, self.r_len_half - self.controls.offset + rr]
        if not self.interpolation:
            return value_1

        value_2 = self.input_d[zz, self.r_len_half  - self.controls.offset + rr +1]
        value_diff = value_2 - value_1
        return value_1 + value_diff * (radius - (rr + self.controls.r_offset ))


    cdef int translator(self, Py_ssize_t start, Py_ssize_t stop, Interval* p_converted) except-1 :
        """ start and stop : 0 ist the first one
        """
        if start == stop:
            raise ValueError("Interval must be longer than 0")

        cdef Py_ssize_t modifier

        if self._odd:
            modifier = 1
        else:
            modifier = 0
        if start < self.r_len_half + modifier and stop <= self.r_len_half + modifier:
            p_converted[0].start = self.r_len_half - start + modifier
            p_converted[0].stop = self.r_len_half - stop + modifier
            return 0
        if start >= self.r_len_half  and stop > self.r_len_half:
            p_converted[0].start = stop - self.r_len_half + 1
            p_converted[0].stop = start -self. r_len_half + 1
            return 1
        else:
            raise ValueError("Interval goes over the middle point. You have to split it first.")


    cdef bint if_split(self, Py_ssize_t start, Py_ssize_t stop): # maybe inline it
        # half = r_len // 2
        cdef Py_ssize_t modifier
        if self._odd:
            modifier = 1
        else:
            modifier = 0
        if start < self.r_len_half and stop > self.r_len_half + modifier:
            return True
        else:
            return False


    cdef _x_loop(self, Py_ssize_t yy, Py_ssize_t x_start , Py_ssize_t x_stop, double [:] output):
        cdef Py_ssize_t xx, rr
        cdef double radius
        for xx in range(x_start, x_stop, -1):
            radius = sqrt(d_square(yy + self.controls.y_offset) + d_square(xx + self.controls.x_offset))
            rr = int(radius)


    cdef _inside_y_loop (self, Py_ssize_t yy , Interval interval,  double [:] output):
        cdef x_start
        if self.r_len_half - yy <= interval.start:
            x_start  = self.r_len_half - yy
        else:
            x_start = interval.start
        self._x_loop(yy, x_start, interval.stop, output)


    cdef rotate_slice(self, Py_ssize_t leading_start, Py_ssize_t leading_stop, bint left):
        cdef Interval first_interval
        cdef Interval second_interval
        cdef total_start

        cdef enum side:
            left = 0
            right = 1

        total_start = leading_start - self.pulse_len
        cdef bint split = self.if_split(total_start, leading_stop)
        if split:
            self.translator(total_start, self.r_len_half, &first_interval)
            self.translator(self.r_len_half, leading_stop, &second_interval)
        else:
            side = self.translator(total_start, leading_stop, &first_interval)

        cdef Py_ssize_t zz, yy, pp
        cdef Py_ssize_t n_slices
        cdef Py_ssize_t pulse_idx
        cdef Py_ssize_t a, b
        cdef double summed_over_pulse
        for zz in range(self.s_len):
            for yy in range(self.controls.y_loop_start, self.r_len_half + self.controls.y_loop_start):
                if split:
                    self._inside_y_loop(yy, first_interval, self.x_line[0:self.r_len_half])
                    self._inside_y_loop(yy, second_interval, self.x_line[self.r_len_half:0:-1])
                else:
                    if side is left:
                        self._inside_y_loop(yy, first_interval, self.x_line[0:self.r_len_half])
                    if side is right:
                        self._inside_y_loop(yy, first_interval, self.x_line[self.r_len_half:0:-1])


