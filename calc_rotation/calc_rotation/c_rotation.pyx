cimport cython
import  numpy as np
from libc.math cimport sqrt
from cython cimport floating
from libc.math cimport ceil
from typing import Tuple

ctypedef struct Interval:
    Py_ssize_t start
    Py_ssize_t end

ctypedef fused floating2:
    float
    double


# cdef inline double d_square(double x): # maybe use instead of x**2, should be faster
#   return x*x
# TODO introduce used types and casting if needed. So that input, output can be of single precision.

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
# TODO add a bint for including symmetric part only for the middle, vertical, line.

cdef int kernel_2d_perp(floating [:,::1] input_data, floating2 [:,::1] output_data, Py_ssize_t interval_start,
                         Py_ssize_t interval_stop, double factor=1, bint interpolation=0, bint inc_sym=0,
                        bint inc_sym_only_vertical_middle=0):
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
    cdef double* value_1_down
    cdef double* value_1_up
    cdef double* value_2_down
    cdef double* value_2_up

    # When using without without slicing (just one step), multiply the outcome with 2,
    # to include the other, identical, part.
    if inc_sym:
        factor *= 2

    # Handling both odd and even number of cells in a slice.
    cdef Py_ssize_t offset, loop_start
    cdef float r_offset
    cdef bint middle_line_splitting = 0
    cdef bint odd
    if  r_len%2 == 0:
        odd = 0
        offset = 1
        loop_start = 0
        r_offset = 0.5
    else:
        odd = 1
        offset = 0
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
                radius = sqrt((0.25)**2 + (xx - r_offset)**2)
                rr_exact = rr
                rr = int(radius)
                # Value at floor(radius).
                value_1_up = &input_data[zz,r_len_half + rr]
                if not interpolation or r_len_half - 1 == rr:
                    interpolated_value_up = value_1_up[0]
                else:
                    # Interpolation between two data points.
                    value_2_up = &input_data[zz,r_len_half + rr + 1]
                    value_diff_up = value_2_up[0] - value_1_up[0]
                    interpolated_value_up = (value_1_up[0] +
                                                 value_diff_up
                                                 * (radius - rr_exact))
                # Add to the integral.
                summed_line_up += (interpolated_value_up
                                   * (0.25)/radius)
            # Write the output.
            output_data[r_len_half, s_len -1 - zz] += <double>factor * summed_line_up

    # Main part:
    # The first loop is over slices.
    for zz in range(s_len):
        # The second one is over the Y-Axis. That upwards, is away from the middle of the circle.
        # The X-Ray beam propagates along the X-Axis.
        for yy in range(loop_start, r_len_half):
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
                radius = sqrt((yy + 0.5)**2 + (xx - r_offset)**2)
                # Value at floor(radius).
                rr = int(radius)
                value_1_up = &input_data[zz,r_len_half + rr]
                value_1_down = &input_data[zz, r_len_half - offset - rr]
                if not interpolation or r_len_half - 1 == rr:
                    interpolated_value_up = value_1_up[0]
                    interpolated_value_down = value_1_down[0]
                else:
                    if odd:
                        rr_exact = rr
                    else:
                        rr_exact = sqrt(0.5**2 + (rr + 0.5)**2)
                    # Interpolation between two data points.
                    value_2_up = &input_data[zz,r_len_half + rr +1]
                    value_2_down = &input_data[zz,r_len_half  - offset - rr - 1]

                    value_diff_up = value_2_up[0] - value_1_up[0]
                    value_diff_down = value_2_down[0] - value_1_down[0]

                    interpolated_value_up = (value_1_up[0] +
                                             value_diff_up
                                             * (radius - rr_exact))
                    interpolated_value_down = (value_1_down[0] +
                                               value_diff_down
                                               * (radius - rr_exact))
                # Add to the integrals.
                summed_line_up += (interpolated_value_up
                                   * (yy + 0.5)/radius)
                summed_line_down += (interpolated_value_down
                                     * (yy + 0.5)/radius)
            if middle_line_splitting:
                cdef float extra_factor = 0.5
                if inc_sym_only_vertical_middle:
                    extra_factor = 1
                radius = sqrt((yy + 0.5)**2 + 0.25**2)
                rr = int(radius)
                value_1_up = &input_data[zz,r_len_half + rr]
                value_1_down = &input_data[zz, r_len_half - offset - rr]
                if not interpolation or r_len_half - 1 == rr:
                    interpolated_value_up = value_1_up[0]
                    interpolated_value_down = value_1_down[0]
                else:
                    rr_exact = rr #
                    # Interpolation between two data points.
                    value_2_up = &input_data[zz,r_len_half + rr +1]
                    value_2_down = &input_data[zz,r_len_half  - offset - rr - 1]

                    value_diff_up = value_2_up[0] - value_1_up[0]
                    value_diff_down = value_2_down[0] - value_1_down[0]

                    interpolated_value_up = extra_factor * (value_1_up[0] +
                                             value_diff_up
                                             * (radius - rr_exact))
                    interpolated_value_down = extra_factor * (value_1_down[0] +
                                               value_diff_down
                                               * (radius - rr_exact))
                # Add to the integrals.
                summed_line_up += (interpolated_value_up
                                   * (yy + 0.5)/radius)
                summed_line_down += (interpolated_value_down
                                     * (yy + 0.5)/radius)
            # Write the output.
            output_data[r_len_half - offset -yy, s_len - 1 - zz] += <double>factor * summed_line_up
            output_data[r_len_half +  yy,  s_len - 1 - zz]  += <double>factor * summed_line_down

    return 0


def rotation_static_2d(input_data, interpolation=False, output_dtype=np.float64):
    try:
        input_data.flags
    except AttributeError:
        print('input_data should be a numpy array!')
        raise
    if input_data.flags['F_CONTIGUOUS']:
        print('input_data should be stored in a raw major, C contiguous.')
        raise ValueError
    elif not input_data.flags['C_CONTIGUOUS']:
        print('input_data has to be a contiguous array.')
        raise ValueError
    if input_data.dtype != np.float64 and input_data.dtype != np.float32:
        print('input should be an ndarray of type np.float64, or np.float32')
        print('Converting to float64 and continuing...')
        input_data = input_data.astype(np.float64)

    output = np.zeros(input_data.size, dtype=output_dtype, order='C').reshape(
                    input_data.shape[1], input_data.shape[0])

    kernel_2d_perp(input_data, output, input_data.shape[1], 0, 1,
                   interpolation=interpolation, inc_sym=1)
    return output


cdef Interval translator(Py_ssize_t start, Py_ssize_t stop, Py_ssize_t half, bint odd) except -1:
    """ start and stop : 0 ist the first one

    """
    if start == stop:
        raise ValueError("Interval must be longer than 0")

    cdef Py_ssize_t modifier
    if odd:
         modifier = 1
    else:
         modifier = 0
    cdef Iterval converted
     if start < half + modifier and stop <= half + modifier:
         converted.start = half - start + modifier
         converted.stop = half - stop + modifier
     if start >= half  and stop > half:
         converted.start = stop - half + 1
         converted.stop = start - half + 1
     else:
        raise ValueError("Interval goes over the middle point. You have to split it first.")

     return converted

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

def  one_step_extended_pulse(floating [:,:,::1] output, floating2 [:,::1] step, Py_ssize_t full_step_size,
                         Py_ssize_t x_start_glob, Py_ssize_t x_end_glob, Py_ssize_t leading_interval_start, double factor, bint interpolation):
     cdef Py_ssize_t n, r_len, interval_start, interval_stop
     cdef Py_ssize_t n = output.shape[0] # Number of slices in the beam package.
     cdef Py_ssize_t r_len = step.shape[1] # Full length of the radial data.
     cdef Py_ssize_t half = r_len // 2
     # Check, if radial data is of an odd length.
     cdef bint odd = False
     if r_len%2 !=2:
        odd = True
     # Iterations over the sliced mask for the X-Ray beam.
     for nn in range(n)
         interval_start = leading_interval_start - nn
         # we can't just take "interval_end -nn", because the last interval can be trimmed.
         interval_stop = interval_start + full_step_size - nn
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
         cdef Interval current_interval
         if if_split(interval_start, interval_stop, half, odd):
            current_interval = translator(interval_start, half)
            kernel_2d_perp(step, output[n,:,:] ,current_interval.start, current_interval.stop,
                           factor=factor, interpolation=interpolation, inc_sym=0, inc_sym_only_vertical_middle=1)
            current_interval = translator(half, interval_stop)
            kernel_2d_perp(step, output[n,:,:], current_interval.start, current_interval.stop,
                           factor=factor, interpolation=interpolation, inc_sym=0, inc_sym_only_vertical_middle=1)
         else:
            current_interval = translator(interval_start, interval_stop)
            kernel_2d_perp(step, output[n,:,:], current_interval.start, current_interval.stop,
                           factor=factor, interpolation=interpolation, inc_sym=0, inc_sym_only_vertical_middle=1)
