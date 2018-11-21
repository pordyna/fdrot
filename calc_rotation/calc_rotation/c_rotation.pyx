# cython: profile=False
# cython: linetrace=False
# cython: binding=False
cimport cython
import  numpy as np
cimport numpy as np
from libc.math cimport sqrt
# from cython cimport floating
# from libc.math cimport ceil
# from typing import Tuple

ctypedef struct Interval:
    Py_ssize_t start
    Py_ssize_t stop

@cython.profile(False)
cdef inline double d_square(double x): # maybe use instead of x**2, should be faster
    return x*x
# TODO introduce fused types and casting if needed. So that input, output can be of single precision.

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int kernel_2d_perp(double [:,::1] input_data, double [:,::1] output_data, Py_ssize_t interval_start,
                         Py_ssize_t interval_stop, double factor=1, bint interpolation=0, bint inc_sym=0,
                        bint inc_sym_only_vertical_middle=0, bint add=1):
    # array shape:
    cdef Py_ssize_t r_len, s_len
    r_len = input_data.shape[1]
    s_len = input_data.shape[0]
    cdef Py_ssize_t r_len_half = r_len // 2
    # iterators:
    cdef Py_ssize_t yy, xx ,rr, zz
    # 'up' and 'down' stand for the upper and lower parts of the distribution.
    cdef double radius
    cdef double rr_exact
    cdef double summed_line_up, summed_line_down
    # linear interpolation:
    cdef double interpolated_value_up, interpolated_value_down
    cdef double value_diff_up, value_diff_down
    # changed double* to double
    cdef double value_1_down
    cdef double value_1_up
    cdef double value_2_down
    cdef double value_2_up

    # When using without without slicing (just one step), multiply the outcome with 2,
    # to include the other, identical, part.
    if inc_sym:
        factor *= 2
    # for the middle line in the odd case
    cdef float extra_factor = 0.5
    # Handling both odd and even number of cells in a slice.
    cdef Py_ssize_t offset, loop_start, write_offset
    cdef float r_offset
    cdef bint middle_line_splitting = 0
    cdef bint odd
    if  r_len%2 == 0:
        odd = 0
        offset = 1
        write_offset = 1
        loop_start = 0
        r_offset = 0.5
    else:
        odd = 1
        offset = 0
        write_offset = 0
        loop_start = 1
        r_offset = 0
        if interval_stop == 0:
            interval_stop = 1
            middle_line_splitting = 1
        # There is a middle line, in the odd case which doesn't belong to neither the lower
        # nor the upper distribution. It is summed here, and the main loop starts with y=1
        # (loop_start).
        summed_line_up = 0

        # The first loop is over slices.
        for zz in range(s_len):
            # second one goes over x. We start, where the X-ray enters the cylinder.
            # That is at the full radius, and we proceed towards the middle of the way
            # through the cylinder.
            for xx in range(interval_start , interval_stop , -1):
                # This is the radius at the middle point of the cell (point P), halfway through
                # a single step (xx +1) in the propagation direction (X-direction).
                radius = sqrt(d_square(0.25) + d_square(xx - r_offset))
                rr = int(radius)
                rr_exact = rr
                # Value at floor(radius).
                value_1_up = input_data[zz,r_len_half + rr]
                if not interpolation or r_len_half - 1 == rr:
                    interpolated_value_up = value_1_up
                else:
                    # Interpolation between two data points.
                    value_2_up = input_data[zz,r_len_half + rr + 1]
                    value_diff_up = value_2_up - value_1_up
                    interpolated_value_up = (value_1_up +
                                                 value_diff_up
                                                 * (radius - rr_exact))
                # Add to the integral.
                summed_line_up += (interpolated_value_up
                                   * (0.25)/radius)
            # Write the output.
            if add:
                output_data[r_len_half, s_len -1 - zz] += <double>factor * summed_line_up
            else:
                output_data[r_len_half, s_len -1 - zz] = <double>factor * summed_line_up

    # Main part:
    # The first loop is over slices.
    for zz in range(s_len):
        # The second one is over the Y-Axis. That upwards, is away from the middle of the circle.
        # The X-Ray beam propagates along the X-Axis.
        for yy in range(loop_start, r_len_half + loop_start ):
            summed_line_up = 0
            summed_line_down = 0
            # In a case, that we integrate only over a part of the way, the first iteration of the
            # next loop has to be set extra.
            if r_len_half - yy <= interval_start:
                x_start  = r_len_half - yy
            else:
                x_start = interval_start
            # The most inner one goes over x. We start, where the X-ray enters the cylinder.
            # That is at the full radius, and we proceed towards the middle of the way
            # through the cylinder.
            for xx in range(x_start, interval_stop , -1):
                # This is the radius at the middle point of the cell (point P), halfway through
                # a single step (xx +1) in the propagation direction (X-direction).
                # The r_offset is not applied to the radius, it self, yet.
                radius = sqrt(d_square(yy + r_offset) + d_square(xx - r_offset))
                # Value at floor(radius).
                rr = int(radius)
                value_1_up = input_data[zz,r_len_half + rr]
                value_1_down = input_data[zz, r_len_half - offset - rr]
                if not interpolation or r_len_half - 1 == rr:
                    interpolated_value_up = value_1_up
                    interpolated_value_down = value_1_down
                else:
                    if odd:
                        rr_exact = rr
                    else:
                        rr_exact = rr + 0.5 # change it back to just r_offset no if statement
                    # Interpolation between two data points.
                    value_2_up = input_data[zz, r_len_half + rr +1]
                    value_2_down = input_data[zz, r_len_half  - offset - rr - 1]

                    value_diff_up = value_2_up - value_1_up
                    value_diff_down = value_2_down - value_1_down

                    interpolated_value_up = (value_1_up +
                                             value_diff_up
                                             * (radius - rr_exact))
                    interpolated_value_down = (value_1_down +
                                               value_diff_down
                                               * (radius - rr_exact))
                # Add to the integrals.
                summed_line_up += (interpolated_value_up
                                   * (yy + 0.5)/radius)
                summed_line_down += (interpolated_value_down
                                     * (yy + 0.5)/radius)
            if middle_line_splitting:
                if inc_sym_only_vertical_middle:
                    extra_factor = 1
                radius = sqrt(d_square(yy) + d_square(0.25))
                rr = int(radius)
                value_1_up = input_data[zz,r_len_half + rr]
                value_1_down = input_data[zz, r_len_half - offset - rr]
                if not interpolation or r_len_half - 1 == rr:
                    interpolated_value_up = value_1_up
                    interpolated_value_down = value_1_down
                else:
                    rr_exact = rr #
                    # Interpolation between two data points.
                    value_2_up = input_data[zz,r_len_half + rr +1]
                    value_2_down = input_data[zz,r_len_half  - offset - rr - 1]

                    value_diff_up = value_2_up - value_1_up
                    value_diff_down = value_2_down - value_1_down

                    interpolated_value_up = extra_factor * factor * (value_1_up +
                                             value_diff_up
                                             * (radius - rr_exact))
                    interpolated_value_down = extra_factor * factor * (value_1_down +
                                               value_diff_down
                                               * (radius - rr_exact))
                # Add to the integrals.
                summed_line_up += (interpolated_value_up
                                   * (yy + 0.5)/radius)
                summed_line_down += (interpolated_value_down
                                     * (yy + 0.5)/radius)
            # Write the output.
            if add:
                output_data[r_len_half - write_offset -yy, s_len - 1 - zz] += <double>factor * summed_line_up
                output_data[r_len_half +  yy,  s_len - 1 - zz]  += <double>factor * summed_line_down
            else:
                output_data[r_len_half - write_offset -yy, s_len - 1 - zz] = <double>factor * summed_line_up
                output_data[r_len_half +  yy,  s_len - 1 - zz]  = <double>factor * summed_line_down

    return 0

def rotation_static_2d(np.ndarray input_data, interpolation=False):
    try:
        input_data.flags
    except AttributeError:
        print('input_data should be a numpy array!')
        raise
    if not input_data.flags['C_CONTIGUOUS']:
        print('input_data has to be a C - contiguous array.')
        raise ValueError
    if input_data.dtype != np.float64:
        print('input should be an ndarray of type np.float64.')
        print('Converting to float64 and continuing...')
        input_data = input_data.astype(np.float64)

    output = np.zeros(input_data.size, dtype=np.float64, order='C').reshape(
                    input_data.shape[1], input_data.shape[0])


    cdef Py_ssize_t start = input_data.shape[1]
    cdef double [:,::1] output_view = output
    cdef double [:,::1] input_view = input_data
    kernel_2d_perp(input_view, output_view, start, 0, 1,
                   interpolation=interpolation, add=0, inc_sym_only_vertical_middle=0, inc_sym=1)
    return output



cdef int translator(Py_ssize_t start, Py_ssize_t stop, Py_ssize_t half, bint odd, Interval* p_converted) except-1 :
    """ start and stop : 0 ist the first one
    """
    if start == stop:
        raise ValueError("Interval must be longer than 0")

    cdef Py_ssize_t modifier

    if odd:
        modifier = 1
    else:
        modifier = 0
    if start < half + modifier and stop <= half + modifier:
        p_converted[0].start = half - start + modifier
        p_converted[0].stop = half - stop + modifier
        return 0
    if start >= half  and stop > half:
        p_converted[0].start = stop - half + 1
        p_converted[0].stop = start - half + 1
        return 0
    else:
        raise ValueError("Interval goes over the middle point. You have to split it first.")



cdef bint if_split(Py_ssize_t start, Py_ssize_t stop, Py_ssize_t half, bint odd): # maybe inline it
    # half = r_len // 2
    cdef Py_ssize_t modifier
    if odd:
        modifier = 1
    else:
        modifier = 0
    if start < half and stop > half + modifier:
        return True
    else:
        return False

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int add_elementwise(double [:,::1] a, double [:, ::1] b) except -1:
    cdef Py_ssize_t a_x, a_y, b_x, b_y
    a_x = a.shape[0]
    a_y = a.shape[1]
    b_x = b.shape[0]
    b_y = b.shape[1]
    if a_x != b_x or a_y != b_y:
        raise ValueError("The arrays have to have the same shape.")
    for xx in range(a_x):
        for yy in range(a_y):
            a[xx, yy] = a[xx ,yy] + b[xx, yy]
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
def  one_step_extended_pulse(double [:,::1] output, double[:,::1] step, double [::1] pulse,
                         Py_ssize_t x_start_glob, Py_ssize_t x_end_glob, Py_ssize_t leading_interval_start, Py_ssize_t leading_interval_end, double factor, bint interpolation):
    cdef Py_ssize_t n, r_len, interval_start, interval_stop, nn
    n = pulse.size # Number of slices in the beam package.
    r_len = step.shape[1] # Full length of the radial data.
    cdef Py_ssize_t half = r_len // 2
    # Check, if radial data is of an odd length.
    cdef bint odd = 0
    if r_len%2 !=2:
       odd = True
    # Iterations over the sliced mask for the X-Ray beam.
    cdef double [:, ::1] inter_output = np.zeros(output.size, order='C', dtype=np.float64).reshape(output.shape[0],
                                                                                                    output.shape[1])
    cdef Interval current_interval
    cdef double weighted_factor
    for nn in range(n):
        interval_start = leading_interval_start - nn
       # we can't just take "interval_end -nn", because the last interval can be trimmed.
        interval_stop = leading_interval_end - nn
        # keep interval inside the global boundaries.
        if interval_start < x_start_glob:
            interval_start = x_start_glob
        if interval_stop > x_end_glob:
            interval_stop = x_end_glob
        # skip a slice, if it's  not included in this time step.
        # That could happen, if the pulse is not yet, or not anymore, completely inside the simulation box (target).
        if interval_stop <= interval_start:
            continue
        # Calculating rotation for the current slice in the current step.
        # The Kernel is designed for integration over the left part of the cylinder, or it's parts.
        # The  symmetry allows us to use it for the Right side as well.
        # If the step covers a part of both the left and the right side, it has to be split.
        # Coordinates in the kernel are defined differently than the starting and endpoints here, so they have
        # to be converted and eventually mirrored, if the interval is on the right side.
        weighted_factor = pulse[nn] * factor
        if if_split(interval_start, interval_stop, half, odd):
           translator(interval_start, half, half, odd, &current_interval)
           kernel_2d_perp(step, inter_output ,current_interval.start, current_interval.stop,
                          factor=weighted_factor, interpolation=interpolation, inc_sym=0, inc_sym_only_vertical_middle=1,
                          add=False)
           translator(half, interval_stop, half, odd, &current_interval)
           kernel_2d_perp(step, inter_output, current_interval.start, current_interval.stop,
                          factor=weighted_factor, interpolation=interpolation, inc_sym=0, inc_sym_only_vertical_middle=1,
                          add=True)
        else:
           translator(interval_start, interval_stop, half, odd, &current_interval)
           kernel_2d_perp(step, inter_output, current_interval.start, current_interval.stop,
                          factor=weighted_factor, interpolation=interpolation, inc_sym=0, inc_sym_only_vertical_middle=1,
                          add=False)
        add_elementwise(output, inter_output)        #

@cython.boundscheck(False)
@cython.wraparound(False)
def  new_one_step_extended_pulse(double [:,::1] output, double[:,::1] step, double [::1] pulse,
                         Py_ssize_t x_start_glob, Py_ssize_t x_end_glob, Py_ssize_t leading_interval_start, Py_ssize_t leading_interval_end, double factor, bint interpolation):
    cdef Py_ssize_t n, r_len, interval_start, interval_stop, nn
    n = pulse.size # Number of slices in the beam package.
    r_len = step.shape[1] # Full length of the radial data.
    cdef Py_ssize_t half = r_len // 2
    # Check, if radial data is of an odd length.
    cdef bint odd = 0
    if r_len%2 !=2:
       odd = True
    # Iterations over the sliced mask for the X-Ray beam.
    cdef double [:, ::1] inter_output = np.zeros(output.size, order='C', dtype=np.float64).reshape(output.shape[0],
                                                                                                    output.shape[1])
    cdef Interval normal_long_interval
    cdef Interval ker_long_interval

    normal_long_interval.start = leading_interval_start - n
    normal_long_interval.stop = leading_interval_end

    if if_split(normal_long_interval.start, normal_long_interval.stop, half, odd):
        translator(normal_long_interval.start, half, half, &ker_long_interval)

    cdef double weighted_factor
    for nn in range(n):
        interval_start = leading_interval_start - nn
        interval_stop = leading_interval_end - nn
        # keep interval inside the global boundaries.
        if interval_start < x_start_glob:
            interval_start = x_start_glob
        if interval_stop > x_end_glob:
            interval_stop = x_end_glob
        # skip a slice, if it's  not included in this time step.
        # That could happen, if the pulse is not yet, or not anymore, completely inside the simulation box (target).
        if interval_stop <= interval_start:
            continue
        # Calculating rotation for the current slice in the current step.
        # The Kernel is designed for integration over the left part of the cylinder, or it's parts.
        # The  symmetry allows us to use it for the Right side as well.
        # If the step covers a part of both the left and the right side, it has to be split.
        # Coordinates in the kernel are defined differently than the starting and endpoints here, so they have
        # to be converted and eventually mirrored, if the interval is on the right side.
        weighted_factor = pulse[nn] * factor
        if if_split(interval_start, interval_stop, half, odd):
           translator(interval_start, half, half, odd, &current_interval)
           kernel_2d_perp(step, inter_output ,current_interval.start, current_interval.stop,
                          factor=weighted_factor, interpolation=interpolation, inc_sym=0, inc_sym_only_vertical_middle=1,
                          add=False)
           translator(half, interval_stop, half, odd, &current_interval)
           kernel_2d_perp(step, inter_output, current_interval.start, current_interval.stop,
                          factor=weighted_factor, interpolation=interpolation, inc_sym=0, inc_sym_only_vertical_middle=1,
                          add=True)
        else:
           translator(interval_start, interval_stop, half, odd, &current_interval)
           kernel_2d_perp(step, inter_output, current_interval.start, current_interval.stop,
                          factor=weighted_factor, interpolation=interpolation, inc_sym=0, inc_sym_only_vertical_middle=1,
                          add=False)
        add_elementwise(output, inter_output)
