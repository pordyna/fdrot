# cython: language_level=3
# cython: profile = False
# cython: initializedcheck = False
"""Kernel2D.pyx
This file is a part of an Cython extension in the Fdrot package.

It provides the the tools for a time resolved propagation of a pulse through a simulation box
and calculating the Faraday effect.
"""

# TODO: Try using the fused types again for the input and the output, so that single precision data is acceptable.
from libc.math cimport sqrt
from cython.view cimport array as cvarray
cimport cython
import numpy as np
from  .c_defs cimport *

cdef inline double sum_arr(cython.floating [::1] arr):
    """Sums the elements of an array."""
    cdef double sum_d = 0
    cdef ssize_t ii
    for ii in range(arr.shape[0]):
        sum_d+= arr[ii]
    return sum_d

@cython.profile(False)
cdef inline double d_square(double x):
    """Returns x^2"""
    return x*x

# Attributes declarations are in the pxd file.
@cython.final
cdef class Kernel2D:
    """ ... """
    def __cinit__(self, double [:, ::1] input_d, double [:, ::1] output_d, double [::1] pulse , double factor,
                  Py_ssize_t global_start, Py_ssize_t global_end, Py_ssize_t pulse_step=1,
                  bint interpolation=0, bint inc_sym=0, bint add=1):
        """

        :param input_d: Bz * n_e
        :param output_d: Output array. Here the calculated rotation effect is stored.
        :param pulse: The form of the pulse. The sum of this array should be 1. (It's normed.)
        :param factor: The constant factor (Should include dx as well.)
        :param global_start: The cell from which the propagation begins (This cell is also included).
        :param global_end: The cell at which the propagation stops (This cell is not included).
        :param pulse_step: For now it should stay 1.
        :param interpolation: Turns of and off the linear interpolation of the radial data.
        :param inc_sym: For the static case. If set to True, the output is multiplied by two,
         so that one can integrate only over the first half of the path. The distribution is symmetric
          in the static case.
        :param add: If True, the summed effect is added to the output, if False it overwrites the data there.
        """
        self.input_d = input_d
        self.output_d = output_d
        self.interpolation = interpolation
        self.inc_sym = inc_sym
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
            self.offset = 1
            self.write_offset = 1
            self.y_loop_start =  0
            self.r_offset = 0.5
            self.x_offset = - 0.5
            self.y_offset = 0.5
        else:
            self._odd = True
            self.offset = 0
            self.write_offset = 0
            self.y_loop_start =  1
            self.r_offset = 0
            self.x_offset = -1 # changed from 0 to -1, I think it should be that way
            self.y_offset = 0

        # Pulse:
        self.pulse = pulse
        self.pulse_len = pulse.shape[0]
        self.pulse_step = pulse_step
        self.x_line = cvarray(shape=(2, self.r_len), itemsize=sizeof(double), format="d")

    @property
    def input(self):
        return np.asarray(self.input_d)

    @input.setter
    def input(self, double [:, ::1] input_d):
        assert input_d.shape[0] == self.s_len
        assert input_d.shape[1] == self.r_len
        self.input_d = input_d

    # Following two properties are included only for debugging purposes, and should be  removed in the final release.
    @property
    def output(self):
        return np.asarray(self.output_d)

    @output.setter # not really necessary
    def output(self, double [:, ::1] output_d):
        assert output_d.shape[1] == self.s_len
        assert output_d.shape[0] == self.r_len
        self.output_d = output_d

    @property # not  necessary
    def line(self):
        return np.asarray(self.x_line)

    # TODO: Maybe don't interpolate over the rotational axis?
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline double interpolate_up(self, Py_ssize_t zz, Py_ssize_t rr, double radius):
        cdef double value_1, value_2
        cdef double value_diff

        value_1 = self.input_d[zz, self.r_len_half + rr]
        if not self.interpolation or rr == self.r_len_half - 1:
            return value_1

        value_2 = self.input_d[zz, self.r_len_half + rr +1]
        value_diff = value_2 - value_1
        return value_1 + value_diff * (radius - (rr + self.r_offset ))

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double interpolate_down(self, Py_ssize_t zz, Py_ssize_t rr, double radius):
        cdef double value_1, value_2
        cdef double value_diff

        value_1 = self.input_d[zz, self.r_len_half - (self.offset + rr)] #check if it's right
        if not self.interpolation or rr == self.r_len_half - 1:
            return value_1

        value_2 = self.input_d[zz, self.r_len_half  - (self.offset + rr +1)] #check if it's right
        value_diff = value_2 - value_1
        return value_1 + value_diff * (radius - (rr + self.r_offset ))


    cdef int translator(self, Py_ssize_t start, Py_ssize_t stop, Interval* p_converted) except-1 :
        """ Translates interval to the coordinates used in the integration process. 
          
        The integration is performed over the distance from the rotational axis. This translates the start
         and end points to this variable. The interval can't go over the axis. 
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
            return 0 # has to be in Side
        if start >= self.r_len_half  and stop > self.r_len_half:
            p_converted[0].start = stop - self.r_len_half -1 + modifier
            p_converted[0].stop = start -self. r_len_half -1 + modifier
            return 1 # has to be in Side
        else:
            raise ValueError("Interval goes over the middle point. You have to split it first.")


    cdef bint if_split(self, Py_ssize_t start, Py_ssize_t stop):
        """ chekcs if the interval goes over the middle axis and has to be split."""
        cdef Py_ssize_t modifier
        if self._odd:
            modifier = 1
        else:
            modifier = 0
        if start < self.r_len_half and stop > self.r_len_half + modifier:
            return True
        else:
            return False

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef x_loop(self, Py_ssize_t zz, Py_ssize_t yy,
                 Py_ssize_t x_start , Py_ssize_t x_stop, double [:,:] output, bint incl_down=True):
        """Calculates the rotation values for the each integration step in the line."""
        cdef Py_ssize_t xx, rr
        cdef double radius
        cdef double val_up, val_down

        for xx in range(x_start, x_stop, -1):
            radius = sqrt(d_square(yy + self.y_offset) + d_square(xx + self.x_offset))
            rr = int(radius)
            val_up = self.interpolate_up(zz, rr, radius) * (yy + self.y_offset) / radius
            output[0, xx-1] = val_up
            if incl_down:
                val_down = self.interpolate_down(zz, rr, radius) * (yy + self.y_offset) / radius
                output[1, xx-1] = val_down

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inside_y_loop (self, Py_ssize_t zz, Py_ssize_t yy ,  Interval interval,  double [:,:] output):
        cdef Py_ssize_t x_start
        # calculate the start value for the X-loop.
        if self.r_len_half - yy <= interval.start:
            x_start  = self.r_len_half - yy
        else:
            x_start = interval.start
        # Run the loop.
        self.x_loop(zz, yy, x_start, interval.stop, output)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double sum_line_over_pulse(self, Py_ssize_t leading_start, Py_ssize_t leading_stop, UpDown up_down):
        """ Integrates the effect on one line over the pulse.
        
                
        :param leading_start: Interval start for the leading slice of the pulse (in a single time step).
        :param leading_stop:  Interval stop for the leading slice of the pulse (in a single time step).
        :param up_down: Switch between the lines in the upper and down parts of the distribution. 
        :return: Summed rotation.
        """
        cdef double summed_over_pulse = 0
        cdef Interval slice_interval
        cdef ssize_t nn
        cdef double [::1] line = self.x_line[<Py_ssize_t>up_down, :]
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
                # That could happen, if the pulse is not yet, or not anymore,
                # completely inside the simulation box (target).
                if slice_interval.stop <= slice_interval.start:
                    continue
                pulse_idx = self.pulse_len - nn - 1
                summed_over_pulse += (self.pulse[pulse_idx]
                                      * sum_arr(line[slice_interval.start:slice_interval.stop]))
        return summed_over_pulse

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef write_out(self, Py_ssize_t zz, Py_ssize_t yy, double summed_line, UpDown up_down):
        """Writes the effect summed over a line to the output.
        
        It adds or overwrites the value, depending on `self.add`. 
        """
        if self.add and up_down == up:
            self.output_d[self.r_len_half - self.write_offset -yy, self.s_len - 1 - zz] += self.factor * summed_line
        if self.add and up_down == down:
            self.output_d[self.r_len_half +  yy,  self.s_len - 1 - zz]  += self.factor * summed_line
        if not self.add and up_down == up:
            self.output_d[self.r_len_half - self.write_offset -yy, self.s_len - 1 - zz] = self.factor * summed_line
        if not self.add and up_down == down:
            self.output_d[self.r_len_half +  yy,  self.s_len - 1 - zz]  = self.factor * summed_line

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef propagate_step(self, Py_ssize_t leading_start, Py_ssize_t leading_stop):
        """It propagates the pulse through a single time step.
        
        :param leading_start: Interval start for the leading slice of the pulse (in a single time step).
        :param leading_stop:  Interval stop for the leading slice of the pulse (in a single time step).
        """
        cdef Interval first_interval
        cdef Interval second_interval
        cdef Py_ssize_t total_start
        cdef double summed_up
        cdef double summed_down
        cdef Side side
        cdef float save

        total_start = leading_start - self.pulse_len
        cdef bint split = self.if_split(total_start, leading_stop)
        # Check if the interval goes over the middle part & translate the interval(s).
        if split:
            self.translator(total_start, self.r_len_half, &first_interval)
            self.translator(self.r_len_half, leading_stop, &second_interval)
        else:
            # Also check on each side of the axis th interval is.
            side = <Side>self.translator(total_start, leading_stop, &first_interval)

        cpdef Py_ssize_t zz, yy
        for zz in range(self.s_len):
            # Calculate the effect for all cells needed for this times tep:
            for yy in range(self.y_loop_start, self.r_len_half + self.y_loop_start):
                # the array to save the line
                if split:
                    self.inside_y_loop(zz, yy,  first_interval, self.x_line[:, self.r_len_half - self.offset ::-1])
                    self.inside_y_loop(zz, yy, second_interval, self.x_line[:, self.r_len_half:])
                else:
                    if side is left:
                        self.inside_y_loop(zz, yy, first_interval, self.x_line[:, self.r_len_half - self.offset ::-1])
                    if side is right:
                        self.inside_y_loop(zz, yy, first_interval, self.x_line[:, self.r_len_half::1])
                # Integrated over the pulse:
                summed_up = self.sum_line_over_pulse(leading_start, leading_stop, up)
                summed_down = self.sum_line_over_pulse(leading_start, leading_stop, down)
                #Add the effect from this time step to the output.
                self.write_out(zz, yy, summed_up, up)
                self.write_out(zz, yy, summed_down, down)

            # Extra calculation for yy=0 in the odd case. It's moved a half of the cell from
            # the middle (yy=0), so that it's not equal 0. yy=0 was already excluded from the loop above
            # by the `y_loop_start` attribute.
            if self._odd:
                save = self.y_offset
                self.y_offset = 0.25
                if split:
                    self.x_loop(zz, 0,  first_interval.start, first_interval.stop,
                                 self.x_line[:, self.r_len_half - self.offset ::-1], incl_down=False)
                    self.x_loop(zz, 0, first_interval.start, first_interval.stop,
                                 self.x_line[:, self.r_len_half::1], incl_down=False)
                else:
                    if side is left:
                        self.x_loop(zz, 0, first_interval.start, first_interval.stop,
                                     self.x_line[:, self.r_len_half - self.offset ::-1], incl_down=False)
                    if side is right:
                        self.x_loop(zz, 0, first_interval.start, first_interval.stop,
                                            self.x_line[:, self.r_len_half::1], incl_down=False)
                self.y_offset = save
                summed = self.sum_line_over_pulse(leading_start, leading_stop, up)
                self.write_out(zz, 0, summed, up)