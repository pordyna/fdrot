# cython: language_level=3
# cython: profile = False
# cython: initializedcheck = False
"""This is a Cython extension for handling 2D inputs."""

from libc.math cimport sqrt
from libc.math cimport ceil


cimport cython
import numpy as np

from cython.view cimport array as cvarray

from .c_defs cimport *

"""
This file is a part of an Cython extension in the fdrot package.

Authors: Paweł Ordyna
"""

# TODO: Try using the fused types again for the input and the output,
#  so that single precision data is acceptable.


cdef inline double sum_arr(cython.floating [::1] arr):
    """Sums elements of an array."""
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

    """This is the propagator for 2D inputs.

    In the propagation process a 2D simulation output is extrapolated on
    a cylinder by assuming a half-turn symmetry of the distribution.

    **'__cinit__' method:**
    Initializes a Kernel2D object.

    Args:
        input_d: Simulation data: :math:`Bz \\cdot n_e`.
        output_d: Output array. Here the calculated rotation effect is
          stored.
        pulse: The form of the pulse. The sum of this array should be
          ~1(normed).
        factor: The constant factor (Should include dx as well.).
        global_start: The cell from which the propagation begins (This
          cell is also included).
        global_end: The cell at which the propagation stops (This cell
          is not included).
        interpolation: If True the linear interpolation is on,
              otherwise the nearest neighbour method is used.
        add: If True, the summed effect is added to the output if False,
          it overwrites the data there.
        inc_sym_only_vertical_middle: If the transverse shape of
              the input is odd, integration steps directly at the middle
              cross-section of the cylinder are only half-long so that
              they are not evaluated at the y-axis but
              :math:`0.25 \\Delta x` away from it. If this is set to
              True value at such integration step is doubled to include
              the missing half of the cell. Usually that is the desired
              behavior. If one wants to  propagate the pulse only till
              that middle cross-section and later multiple the output
              by two to include the other half, which gives an identical
              contribution if the fields are static (only one step in
              the sequence), one should set it to False.
    """

    def __cinit__(self, double [:, ::1] input_d,
                  double [:, ::1] output_d, double [::1] pulse , double factor,
                  Py_ssize_t global_start, Py_ssize_t global_end,
                  bint interpolation=0, bint add=1,
                  bint inc_sym_only_vertical_middle = 1):

        self.input_d = input_d
        self.output_d = output_d
        self.interpolation = interpolation
        self.add = add
        self.inc_sym_only_vertical_middle = inc_sym_only_vertical_middle
        self.factor = factor
        self.r_len = input_d.shape[1]
        self.s_len = input_d.shape[0]

        self.global_interval.start = global_start
        self.global_interval.stop =  global_end

        assert (self.r_len == output_d.shape[0],
                'r_len:{}, output_d.shape[0]:{}'
                ''.format(self.r_len, output_d.shape[0]))
        assert (self.s_len == output_d.shape[1],
                'r_len:{}, output_d.shape[0]:{}'
                ''.format(self.s_len, output_d.shape[1]))

        self.r_len_half = self.r_len // 2

        if self.r_len%2 == 0:
            self._odd = False
            self.offset = 1
            self.write_offset = 1
            self.y_loop_start =  0
            self.r_offset = 0.5
            self.x_offset = 0
            self.y_offset = 0.5
            self.max_radius_sqd = d_square(self.r_len_half)
            self.x_count_offset = 0
        else:
            self._odd = True
            self.offset = 0
            self.write_offset = 0
            self.y_loop_start =  1
            self.r_offset = 0
            self.x_offset = -0.5 
            self.y_offset = 0
            self.max_radius_sqd = d_square(self.r_len_half + 0.5)
            self.x_count_offset = 1

        # Pulse:
        self.pulse = pulse
        self.pulse_len = pulse.shape[0]
        self.x_line = cvarray(shape=(2, self.r_len), itemsize=sizeof(double),
                              format="d")

    @property
    def input(self):
        """(Type: ndarray) :math:`Bz \\cdot n_e`

        Use this property to acquire or update input_d.
        """
        return np.asarray(self.input_d)

    @input.setter
    def input(self, double [:, ::1] input_d):
        assert input_d.shape[0] == self.s_len
        assert input_d.shape[1] == self.r_len
        self.input_d = input_d

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double interpolate_up(self, Py_ssize_t zz, Py_ssize_t rr,
                               double radius) except *:
        """Returns an interpolated value from the upper part.
        
        Args:
            zz: Data slice from which the value should be interpolated. 
            rr: This should be just floor(radius).
            radius: Radius at which the value should be interpolated.
        """

        cdef double value_1, value_2
        cdef double value_diff
        value_1 = self.input_d[zz, self.r_len_half + rr]
        if not self.interpolation or self.r_len_half + rr == self.r_len - 1:
            return value_1
        value_2 = self.input_d[zz, self.r_len_half + rr +1]
        value_diff = value_2 - value_1
        return value_1 + value_diff * (radius - (rr + self.r_offset ))

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double interpolate_down(self, Py_ssize_t zz, Py_ssize_t rr,
                                 double radius) except *:
        """Returns an interpolated value from the lower part.
        
        Args:
            zz: Data slice from which the value should be interpolated. 
            rr: This should be just floor(radius).
            radius: Radius at which the value should be interpolated.
        """

        cdef double value_1, value_2
        cdef double value_diff

        value_1 = self.input_d[zz, self.r_len_half - (self.offset + rr)]

        if not self.interpolation or self.r_len_half - (self.offset + rr) == 0:
            return value_1

        value_2 = self.input_d[zz, self.r_len_half  - (self.offset + rr +1)]
        value_diff = value_2 - value_1
        return value_1 + value_diff * (radius - (rr + self.r_offset ))

    cdef int translator(self, Py_ssize_t start, Py_ssize_t stop,
                        Interval* p_converted) except-1 :
        """ Translates interval to the local coordinates.  
          
        The integration is performed over x that is defined as the 
        distance from the cylinder axis. This translates the start
        and end points from the global coordinate system, where x=0 is 
        at the beginning of the cylinder (There were the X-ray enters
        the volume), to this local coordinate.
        
        This coordinate system is valid only in either the left or the
        right half of the cylinder (negative values are not allowed) so
        the global interval, one that should be translated, can't go
        over the cylinder axis. 
         
        Args:
            start: Begin of the interval.
            stop: End of the interval.
            p_converted: Pointer to an interval to write the result.
             
        Raises:
             ValueError: If start=stop, or the interval goes over the 
               middle axis. 
        """

        if start == stop:
            raise ValueError("Interval must be longer than 0")

        cdef Py_ssize_t modifier

        if self._odd:
            modifier = 1
        else:
            modifier = 0
        if (start < self.r_len_half + modifier
                and stop <= self.r_len_half + modifier):
            p_converted[0].start = self.r_len_half - start + modifier
            p_converted[0].stop = self.r_len_half - stop + modifier
            assert p_converted[0].stop >=0
            return 0 # has to be in Side
        if start >= self.r_len_half  and stop > self.r_len_half:
            p_converted[0].start = stop - self.r_len_half
            p_converted[0].stop = start - self.r_len_half
            assert p_converted[0].stop >=0
            return 1 # has to be in Side
        else:
            raise ValueError("Interval goes over the middle point. You have "
                             "to split it first.")


    cdef bint if_split(self, Py_ssize_t start, Py_ssize_t stop) except *:
        """ Checks if the interval goes over the middle axis.
        
        Args:
            start: Begin of the interval.
            stop: End of the interval.
            
        Returns:
            bool: True if it has to be split, False otherwise.
        """

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
    cdef short x_loop(self, Py_ssize_t zz, Py_ssize_t yy,
                 Py_ssize_t x_start , Py_ssize_t x_stop, double [:,:] output,
                      bint incl_down=True) except -1 :
        """Calculates the rotation values on a line along x.
        
        Values of single integration steps are calculated at a given 
        height in a specified interval and written in an array, so that
        they can be
        summed up later. This function works within one half of a
        cylinder slice (split at the y-axis).
         
        A value on the edge of the cylinder is calculated for a shorter
        integration step (cut by the outer cylinder wall). Also in the
        case of an odd r_len the value directly at the axis is 
        calculated for a half-long integration step, as the step is then
        cut in the middle of the cylinder.
        
        Args:
            zz: Current xy-slice along z.
            yy: Current iteration in the yy loop.
            x_start: The outermost position for which the value should
              be calculated. In local coordinates so a higher number 
              describes a cell further away from the cylinder axis. 
            x_stop: (x_stop - 1) is the innermost cell for which the 
              value should be calculated. Also in local coordinates. 
            output: An array view specifying where the values should be
              written. The shape of the first dimension should be 2; 
              That gives two lines, one for the values in the upper part
              and one for the values in the lower part. The integration
              steps are written out one by one starting ([:,0]) with the
              value for the outermost position and ending ([:,-1])with
              the innermost one. 
            incl_down: If False the values are calculated and written 
              out only for the upper part of the distribution.
        """
        cdef Py_ssize_t xx, rr
        cdef double radius
        cdef double val_up, val_down
        cdef double step_size
        # Check for edge an axis cases.
        # Perform the extra calculations, if they are needed.
        # Edge:
        # If the cell on the circle is included,  calculate a non full step,
        # cut by the arc at sqt(R^2 - y^2).
        if self.edge:
            xx = x_start
            x_start -= 1
            step_size = self.max_x_at_y - (<double>self.max_xx_at_y -1
                                           + self.x_offset)
            radius = sqrt(self.y_sqrd + d_square(xx  - 1 + self.x_offset
                                                 + 0.5 * step_size))
            rr = int(radius)
            val_up = (self.interpolate_up(zz, rr, radius)
                      * (yy + self.y_offset) / radius)
            val_up *= step_size
            output[0, xx-1] = val_up
            if incl_down:
                val_down = (self.interpolate_down(zz, rr, radius)
                            * (yy + self.y_offset) / radius)
                val_down *= step_size
                output[1, xx-1] = val_down
        # Axis:
        # In the odd case, the cell in the middle, at the symmetry axis,
        # is only half that long.
        # The other half is on the right side.
        # If `inc_sym_only_vertical_middle` is set to True, the outcome is
        # multiplied by two.
        # That is needed if we are going to continue with a next interval,
        # starting at the next cell,
        # and not at this axis anymore. For example when a bigger interval
        # was split in to two.
        if x_stop == 0 and self._odd:
            xx = 1
            x_stop += 1
            radius = sqrt(self.y_sqrd + d_square(0.25))
            rr = int(radius)
            val_up = (self.interpolate_up(zz, rr, radius)
                      * (yy + self.y_offset) / radius)
            if not self.inc_sym_only_vertical_middle:
                val_up *= 0.5
            output[0, xx-1] = val_up
            if incl_down:
                val_down = (self.interpolate_down(zz, rr, radius)
                            * (yy + self.y_offset) / radius)
                if not self.inc_sym_only_vertical_middle:
                    val_down *= 0.5
                output[1, xx-1] = val_down

        # Do other steps:
        for xx in range(x_start, x_stop, -1):
            radius = sqrt(self.y_sqrd + d_square(xx  - 0.5 + self.x_offset))
            # if rr==150:
            #     print('xx', xx, 'yy', yy)
            rr = int(radius)
            val_up = (self.interpolate_up(zz, rr, radius)
                      * (yy + self.y_offset) / radius)
            output[0, xx-1] = val_up
            if incl_down:
                val_down = (self.interpolate_down(zz, rr, radius)
                            * (yy + self.y_offset) / radius)
                output[1, xx-1] = val_down
        return 0

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef short inside_y_loop(self, Py_ssize_t zz, Py_ssize_t yy ,
                             Interval interval, double [:,:] output,
                             bint incl_down=True) except -1:
        """Intermediate step between the y and the x-loop.
          
          Args:
            zz: Current xy-slice along z.
            yy: Current iteration in the yy loop.
            interval: Interval in local coordinates in which the
              integration is performed. (! start > stop).
            output: An array view specifying where the values should be
              written. The shape of the first dimension should be 2; 
              That gives two lines, one for the values in the upper part
              and one for the values in the lower part. The integration
              steps are written out one by one starting ([:,0]) with the
              value for the outermost position and ending ([:,-1])with
              the innermost one. 
            incl_down: If False the values are calculated and written 
              out only for the upper part of the distribution.
              
          Raises:
               ValueError: If interval.start <= interval.stop (after 
               eventual cutting at the cylinder wall).
          """
        cdef Py_ssize_t x_start
        # Determine the start value for the X-loop.
        if  interval.start >= self.max_xx_at_y:
            x_start = self.max_xx_at_y
            self.edge = True
        else:
            x_start = interval.start
            self.edge = False
        # Run the loop.
        if x_start <= interval.stop:
            raise ValueError('interval.start hast to be grater than '
                             'interval.stop since the x-loop moves from the '
                             'outside of the cylinder to the inside of it.')
        self.x_loop(zz, yy, x_start, interval.stop, output, incl_down)
        return 0

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double sum_line_over_pulse(self, Py_ssize_t yy,
                                    Py_ssize_t leading_start,
                                    Py_ssize_t leading_stop,
                                    UpDown up_down) except* :
        """ Integrates the effect on one line over the pulse.
        
        Args:
            yy: Current y-loop iteration.     
            leading_start: Interval start for the leading slice of the 
              pulse.
            leading_stop:  Interval stop for the leading slice of the
              pulse.
            up_down: Switch between the lines in the upper and the
              lower part of the distribution. 
            
        Returns:
            Summed rotation.
        """
        cdef double summed_over_pulse = 0
        cdef Interval slice_interval
        cdef ssize_t nn
        cdef double [::1] line = self.x_line[<Py_ssize_t>up_down, :]
        cdef Interval global_at_y

        global_at_y.start = (self.r_len_half - self.max_xx_at_y
                             + self.x_count_offset)
        global_at_y.stop = self.r_len_half + self.max_xx_at_y

        if self.global_interval.start > global_at_y.start:
            global_at_y.start = self.global_interval.start
        if self.global_interval.stop < global_at_y.stop:
            global_at_y.stop = self.global_interval.stop

        for nn in range(self.pulse_len):
                # nn = 0 : incoming slice
                # nn = pulse_len -1 : tail slice
                slice_interval.start = leading_start - nn
                slice_interval.stop = leading_stop - nn
                # keep interval inside the global boundaries.
                if slice_interval.start < global_at_y.start:
                    slice_interval.start = global_at_y.start
                if slice_interval.stop > global_at_y.stop:
                    slice_interval.stop = global_at_y.stop

                # skip a slice, if it's  not included in this time step.
                # That could happen, if the pulse is not yet, or not anymore,
                # completely inside the simulation box (target).
                if slice_interval.stop <= slice_interval.start:
                    continue
                pulse_idx = self.pulse_len - nn - 1
                summed_over_pulse += (self.pulse[pulse_idx]
                                      * sum_arr(
                            line[slice_interval.start:slice_interval.stop]))
        return summed_over_pulse

    # TODO: maybe remove ecxcept ? I think this was only for boundschecks.
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef short write_out(self, Py_ssize_t zz, Py_ssize_t yy,
                         double summed_line, UpDown up_down) except -1:
        """Writes the effect, summed over a line, to the output.
        
        It adds or overwrites the value, depending on `self.add`. 
        
        Args:
            zz: Current xy-slice along z.
            yy: Current iteration in the yy loop.
            summed_line: integrated value (without the integration
              factor).
            up_down: Switch between writing to the upper and the lower
              part of the output. 
        """
        if self.add and up_down == up:
            self.output_d[self.r_len_half - self.write_offset -yy,
                          self.s_len - 1 - zz] += self.factor * summed_line
        if self.add and up_down == down:
            self.output_d[self.r_len_half +  yy,
                          self.s_len - 1 - zz]  += self.factor * summed_line
        if not self.add and up_down == up:
            self.output_d[self.r_len_half - self.write_offset -yy,
                          self.s_len - 1 - zz] = self.factor * summed_line
        if not self.add and up_down == down:
            self.output_d[self.r_len_half +  yy,
                          self.s_len - 1 - zz]  = self.factor * summed_line
        return 0
#

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(True)
    cpdef short propagate_step(self, Py_ssize_t leading_start,
                               Py_ssize_t leading_stop) except -1:
        """It propagates the pulse through a single time step.
        
        This is the method that should be used from outside the
        extension (Python).
        
        Args:
            leading_start: Interval start for the leading slice of the
              pulse (cell included).
            leading_stop:  Interval stop for the leading slice of the
              pulse (cell not included).
        """

        cdef Interval first_interval
        cdef Interval second_interval
        cdef Py_ssize_t total_start
        cdef double summed_up
        cdef double summed_down
        cdef Side side
        cdef float save

        total_start = leading_start - self.pulse_len + 1
        cdef bint split = self.if_split(total_start, leading_stop)
        # Check if the interval goes over the middle part & translate
        # the interval(s).
        if split:
            self.translator(total_start, self.r_len_half, &first_interval)
            self.translator(self.r_len_half, leading_stop, &second_interval)
        else:
            # Also check on each side of the axis th interval is.
            side = <Side>self.translator(total_start, leading_stop,
                                         &first_interval)

        cpdef Py_ssize_t zz, yy
        for zz in range(self.s_len):
            # Calculate the effect for all cells needed for this times tep:
            for yy in range(self.y_loop_start,
                            self.r_len_half + self.y_loop_start):
                self.x_line[:,:] = 0

                self.y_sqrd = d_square(yy + self.y_offset)
                self.max_x_at_y = sqrt(self.max_radius_sqd - self.y_sqrd)
                self.max_xx_at_y = <Py_ssize_t>ceil(self.max_x_at_y
                                                    - self.x_offset)
                # the array to save the line
                if split:
                    self.inside_y_loop(zz, yy, first_interval,
                                       self.x_line[:, (self.r_len_half
                                                       - self.offset) ::-1])
                    self.inside_y_loop(zz, yy, second_interval,
                                       self.x_line[:, self.r_len_half:])
                else:
                    if side is left:
                        self.inside_y_loop(zz, yy, first_interval,
                                           self.x_line
                                           [:,(self.r_len_half
                                               - self.offset) ::-1])
                    if side is right:
                        self.inside_y_loop(zz, yy, first_interval,
                                           self.x_line[:, self.r_len_half:])
                summed_up = self.sum_line_over_pulse(yy, leading_start,
                                                     leading_stop, up)
                summed_down = self.sum_line_over_pulse(yy, leading_start,
                                                       leading_stop, down)
                #Add the effect from this time step to the output.
                self.write_out(zz, yy, summed_up, up)
                self.write_out(zz, yy, summed_down, down)

            # Extra calculation for yy=0 in the odd case. It's moved a half of
            # the cell from the middle (yy=0), so that it's not equal 0. yy=0
            # was already excluded from the loop above by the `y_loop_start`
            # attribute.
            if self._odd:
                save = self.y_offset
                self.y_offset = 0.25
                self.y_sqrd = 0.25**2
                if split:
                    self.inside_y_loop(zz, 0,  first_interval,
                                 self.x_line[:, (self.r_len_half
                                                 - self.offset) ::-1],
                                       incl_down=False)
                    self.inside_y_loop(zz, 0, second_interval,
                                 self.x_line[:, self.r_len_half::1],
                                       incl_down=False)
                else:
                    if side is left:
                        self.inside_y_loop(zz, 0, first_interval,
                                     self.x_line[:,
                                     self.r_len_half - self.offset ::-1],
                                           incl_down=False)
                    if side is right:
                        self.inside_y_loop(zz, 0, first_interval,
                                            self.x_line[:, self.r_len_half::1],
                                           incl_down=False)
                self.y_offset = save
                summed = self.sum_line_over_pulse( 0 , leading_start,
                                                   leading_stop, up)
                self.write_out(zz, 0, summed, up)
        return 0