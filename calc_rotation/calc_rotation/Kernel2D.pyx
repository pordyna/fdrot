from libc.math cimport sqrt
from cython.view cimport array as cvarray
cimport cython

# from .c_rotation cimport Interval

ctypedef struct Interval:
    Py_ssize_t start
    Py_ssize_t stop

@cython.profile(False)
cdef inline double d_square(double x): # maybe use instead of x**2, should be faster
    return x*x

cdef inline double sum_arr(cython.floating [::1] arr):
    cdef double sum = 0
    cdef ssize_t ii
    for ii in range(arr.shape[0]):
        sum+= arr[ii]
    return sum

ctypedef enum UpDown:
    up = 0
    down = 1

ctypedef enum Side:
    left = 0
    right = 1

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
        double [:,::1] x_line
        Interval global_interval

    def __cinit__(self, double [:, ::1] input_d, double [:, ::1] output_d, double [::1] pulse , double factor,
                  Py_ssize_t global_start, Py_ssize_t global_end, Py_ssize_t pulse_step=1,
                  bint interpolation=0, bint inc_sym=0, bint inc_sym_only_vertical_middle=0, bint add=1):

        self.input_d = input_d
        self.output_d = output_d
        self.interpolation = interpolation
        self.inc_sym = inc_sym
        self.inc_sym_only_vertical_middle = inc_sym_only_vertical_middle
        self.add = add
        self.factor = factor
        self.r_len = input_d.shape[1]
        self.s_len = input_d.shape[0]

        self.global_interval.start = global_start
        self.global_interval.stop =  global_end

        assert self.r_len == output_d.shape[0]
        assert self.s_len == output_d.shape[1]

        self.r_len_half = self.r_len // 2

        if self.r_len%2 == 0:
            self._odd = False
            self.controls.offset = 1
            self.controls.write_offset = 1
            self.controls.y_loop_start =  0
            self.controls.r_offset = 0.5
            self.controls.x_offset = - 0.5
            self.controls.y_offset = 0.5
        else:
            self._odd = True
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
        self.x_line = cvarray(shape=(2, self.r_len), itemsize=sizeof(double), format="d")


    cdef double _interpolate_up(self, Py_ssize_t zz, Py_ssize_t rr, double radius):
        cdef double value_1, value_2
        cdef double value_diff

        value_1 = self.input_d[zz, self.r_len_half + rr]
        if not self.interpolation or rr == self.r_len_half - 1:
            return value_1

        value_2 = self.input_d[zz, self.r_len_half + rr +1]
        value_diff = value_2 - value_1
        return value_1 + value_diff * (radius - (rr + self.controls.r_offset ))


    cdef double _interpolate_down(self, Py_ssize_t zz, Py_ssize_t rr, double radius):
        cdef double value_1, value_2
        cdef double value_diff

        value_1 = self.input_d[zz, self.r_len_half - self.controls.offset + rr]
        if not self.interpolation or rr == self.r_len_half - 1:
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


    cdef _x_loop(self, Py_ssize_t zz, Py_ssize_t yy,
                 Py_ssize_t x_start , Py_ssize_t x_stop, double [:,:] output, bint incl_down=True):
        cdef Py_ssize_t xx, rr
        cdef double radius
        cdef double val_up, val_down
        for xx in range(x_start, x_stop, -1):
            radius = sqrt(d_square(yy + self.controls.y_offset) + d_square(xx + self.controls.x_offset))
            rr = int(radius)
            val_up = self._interpolate_up(zz, rr, radius) * (yy + self.controls.y_offset) / radius
            self.x_line[0, xx] = val_up
            if incl_down:
                val_down = self._interpolate_down(zz, rr, radius) * (yy + self.controls.y_offset) / radius
                self.x_line[1, xx] = val_down


    cdef _inside_y_loop (self, Py_ssize_t zz, Py_ssize_t yy ,  Interval interval,  double [:,:] output):
        cdef x_start
        if self.r_len_half - yy <= interval.start:
            x_start  = self.r_len_half - yy
        else:
            x_start = interval.start
        self._x_loop(zz, yy, x_start, interval.stop, output)


    cdef double _sum_line_over_pulse(self, Py_ssize_t leading_start, Py_ssize_t leading_stop, UpDown up_down):
        cdef double summed_over_pulse = 0
        cdef Interval slice_interval
        cdef ssize_t nn
        cdef double [::1] line = self.x_line[<ssize_t>up_down, :]
        for nn in range(self.pulse_len):
                # nn = 0 : incoming slice
                # nn = pulse_len -1 : tail slice
                slice_interval.start = leading_start - nn
                slice_interval.stop = leading_stop - nn
                # keep interval inside the global boundaries.
                if slice_interval.start < self.global_interval.start:
                    slice_interval.start = self.global_interval.start
                if slice_interval.stop > self.global_interval.stop:
                    slice_interval.stop = self.global_interval.stop
                # skip a slice, if it's  not included in this time step.
                # That could happen, if the pulse is not yet, or not anymore, completely inside the simulation box (target).
                if slice_interval.start <= slice_interval.stop:
                    continue
                pulse_idx = self.pulse_len - nn - 1
                summed_over_pulse += (self.pulse[pulse_idx]
                                      * sum_arr(line[slice_interval.start:slice_interval.stop]))
        return summed_over_pulse


    cdef _write_out(self, Py_ssize_t zz, Py_ssize_t yy, double summed_line, UpDown up_down):
        if self.add and up_down == up:
            self.output_d[self.r_len_half - self.controls.write_offset -yy, self.s_len - 1 - zz] += self.factor * summed_line
        if self.add and up_down == down:
            self.output_d[self.r_len_half +  yy,  self.s_len - 1 - zz]  += self.factor * summed_line
        if not self.add and up_down == up:
            self.output_d[self.r_len_half - self.controls.write_offset -yy, self.s_len - 1 - zz] = self.factor * summed_line
        if not self.add and up_down == down:
            self.output_d[self.r_len_half +  yy,  self.s_len - 1 - zz]  = self.factor * summed_line


    cpdef rotate_slice(self, Py_ssize_t leading_start, Py_ssize_t leading_stop):
        cdef Interval first_interval
        cdef Interval second_interval
        cdef Py_ssize_t total_start
        cdef double summed_up
        cdef double summed_down
        cdef Side side
        cdef float save

        total_start = leading_start - self.pulse_len
        cdef bint split = self.if_split(total_start, leading_stop)
        if split:
            self.translator(total_start, self.r_len_half, &first_interval)
            self.translator(self.r_len_half, leading_stop, &second_interval)
        else:
            side = <Side>self.translator(total_start, leading_stop, &first_interval)

        cpdef Py_ssize_t zz, yy
        for zz in range(self.s_len):
            for yy in range(self.controls.y_loop_start, self.r_len_half + self.controls.y_loop_start):
                if split:
                    self._inside_y_loop(zz, yy,  first_interval, self.x_line[:, self.r_len_half - self.controls.offset ::-1])
                    self._inside_y_loop(zz, yy, second_interval, self.x_line[:, self.r_len_half::1])
                else:
                    if side is left:
                        self._inside_y_loop(zz, yy, first_interval, self.x_line[:, self.r_len_half - self.controls.offset ::-1])
                    if side is right:
                        self._inside_y_loop(zz, yy, first_interval, self.x_line[:, self.r_len_half::1])
                summed_up = self._sum_line_over_pulse(leading_start, leading_stop, up)
                summed_down = self._sum_line_over_pulse(leading_start, leading_stop, down)
                self._write_out(zz, yy, summed_up, up)
                self._write_out(zz, yy, summed_down, down)

            if self._odd:
                save = self.controls.y_offset
                self.controls.y_offset = 0.25
                if split:
                    self._x_loop(zz, 0,  first_interval.start, first_interval.stop,
                                 self.x_line[:, self.r_len_half - self.controls.offset ::-1], incl_down=False)
                    self._x_loop(zz, 0, first_interval.start, first_interval.stop,
                                 self.x_line[:, self.r_len_half::1], incl_down=False)
                else:
                    if side is left:
                        self._x_loop(zz, 0, first_interval.start, first_interval.stop,
                                     self.x_line[:, self.r_len_half - self.controls.offset ::-1], incl_down=False)
                    if side is right:
                        self._x_loop(zz, 0, first_interval.start, first_interval.stop,
                                            self.x_line[:, self.r_len_half::1], incl_down=False)
                self.controls.y_offset = save
                summed = self._sum_line_over_pulse(leading_start, leading_stop, up)
                self._write_out(zz, 0, summed, up)
