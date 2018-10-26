cimport cython
# cimport numpy as np
import  numpy as np
from libc.math cimport sqrt


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int rotation_slice(double [:,::1] input_data, double [:,::1] output_data, Py_ssize_t interval_start,
                         Py_ssize_t interval_stop, double factor=1, bint interpolation=0, bint inc_sym=0):
    cdef Py_ssize_t r_len, s_len
    r_len = input_data.shape[1]
    s_len = input_data.shape[0]
    cdef Py_ssize_t r_len_half = r_len // 2

    cdef Py_ssize_t yy, xx ,rr
    cdef double radius
    cdef double summed_line_up, summed_line_down
    cdef double interpolated_value_up, interpolated_value_down
    cdef double value_diff_up, value_diff_down

    cdef double* value_1_down
    cdef double* value_1_up
    cdef double* value_2_down
    cdef double* value_2_up

    # when using without without slicing (just one step), multiply the outcome with 2,
    # to include the other identical part.
    if inc_sym:
        factor *= factor

    cdef Py_ssize_t offset, loop_start
    if  r_len%2 == 0:
        offset = 1
        loop_start = 0
    else:
        offset = 0
        loop_start = 1

        yy = 0
        summed_line_up = 0
        # just the middle line
        if not interpolation:
            for zz in range(s_len):
                for xx in range(interval_start - yy, interval_stop , -1):
                    radius = sqrt((yy + 0.25)**2 + (xx - 0.5)**2)
                    rr = int(radius)
                    summed_line_up += (input_data[zz,r_len_half  + rr]
                                        * (yy + 0.25)/radius)
                output_data[r_len_half - yy ,s_len -1 - zz] = factor * summed_line_up
        else:
            for zz in range(s_len):
                for xx in range(interval_start - yy, interval_stop , -1):
                    radius = sqrt((yy + 0.25)**2 + (xx - 0.5)**2)
                    rr = int(radius)
                    value_1_up = &input_data[zz,r_len_half + rr]
                    if r_len_half - 1 == rr:
                        interpolated_value_up = value_1_up[0]
                    else:
                        value_2_up = &input_data[zz,r_len_half + rr + 1]
                        value_diff_up = value_2_up[0] - value_1_up[0]
                        interpolated_value_up = (value_1_up[0] +
                                                     value_diff_up
                                                     * (radius - rr))
                    summed_line_up += (interpolated_value_up
                                       * (yy + 0.25)/radius)
                output_data[r_len_half - yy ,s_len -1 - zz] = factor * summed_line_up

    if interpolation:
        for zz in range(s_len):
            for yy in range(loop_start, r_len_half):
                summed_line_up = 0
                summed_line_down = 0
                if r_len_half - yy <= interval_start:
                    x_start  = r_len_half - yy
                else:
                    x_start = interval_start
                for xx in range(x_start, interval_stop , -1):
                    radius = sqrt((yy + 0.5)**2 + (xx - 0.5)**2)
                    rr = int(radius)
                    value_1_up = &input_data[zz,r_len_half + rr]
                    value_1_down = &input_data[zz, r_len_half - offset - rr]
                    if r_len_half - 1 == rr:
                        interpolated_value_up = value_1_up[0]
                        interpolated_value_down = value_1_down[0]
                    else:
                        value_2_up = &input_data[zz,r_len_half + rr +1]
                        value_2_down = &input_data[zz,r_len_half  - offset - rr - 1]

                        value_diff_up = value_2_up[0] - value_1_up[0]
                        value_diff_down = value_2_down[0] - value_1_down[0]

                        interpolated_value_up = (value_1_up[0] +
                                                 value_diff_up
                                                 * (radius - rr))
                        interpolated_value_down = (value_1_down[0] +
                                                   value_diff_down
                                                   * (radius - rr))

                    summed_line_up += (interpolated_value_up
                                       * (yy + 0.5)/radius)
                    summed_line_down += (interpolated_value_down
                                         * (yy + 0.5)/radius)
                output_data[r_len_half - offset -yy, s_len - 1 - zz] = factor * summed_line_up
                output_data[r_len_half +  yy,  s_len - 1 - zz]  = factor * summed_line_down
    else:
        for zz in range(s_len):
            for yy in range(loop_start, r_len_half):
                summed_line_up = 0
                summed_line_down = 0
                if r_len_half - yy <= interval_start:
                    x_start  = r_len_half - yy
                else:
                    x_start = interval_start
                for xx in range(x_start, interval_stop, -1):
                    radius = sqrt((yy + 0.5)**2 + (xx - 0.5)**2)
                    rr = int(radius)

                    value_1_up = &input_data[zz,r_len_half + rr]
                    value_1_down = &input_data[zz, r_len_half - offset - rr]

                    interpolated_value_up = value_1_up[0]
                    interpolated_value_down = value_1_down[0]

                    summed_line_up += (interpolated_value_up
                                       * (yy + 0.5)/radius)
                    summed_line_down += (interpolated_value_down
                                         * (yy + 0.5)/radius)
                output_data[r_len_half - offset -yy ,s_len - 1 - zz] = factor * summed_line_up
                output_data[r_len_half +  yy, s_len - 1 - zz]  = 2 * factor * summed_line_down

    return 0


def rotation(input_data, interpolation=False):
    try:
        input_data.flags
    except AttributeError:
        print('input_data should be a numpy array!')
        raise
    if input_data.flags['F_CONTIGUOUS']:
        print('input_data should be stored in a raw major, C contigous.')
        raise ValueError
    elif not input_data.flags['C_CONTIGUOUS']:
        print('input_data has to be a contigous array.')
        raise ValueError
    if input_data.dtype != np.float64:
        print('input should be an ndarray of type np.float64.')
        print('Converting to float64 and continuing...')
        input_data = input_data.astype(np.float64)

    output = np.zeros(input_data.size, dtype=np.float64, order='C').reshape(
                    input_data.shape[1], input_data.shape[0])

    cdef double [:,::1] input_data_view = input_data
    cdef double [:,::1] output_view = output
    rotation_slice(input_data_view, output_view, (input_data.shape[1], 0),
                   interpolation=interpolation, inc_sym=1)
    return output

def rotation_timeresolved( steps, intervals, interpolation=False):
    if  len(steps) != len(intervals):
        raise ValueError
    cdef double [:,::1] step_view
    output = np.zeros(steps[0].size, dtype=np.float64,
                      order = "C").reshape(steps[0].shape[1], steps[0].shape[0])
    end_output = np.zeros_like(output)
    for ii in range(len(steps)):
        try:
            steps[ii].flags
        except AttributeError:
            print('input_data should be a numpy array!')
            raise
        if steps[ii].flags['F_CONTIGUOUS']:
            print('input_data should be stored in a raw major, C contigous.')
            raise ValueError
        elif not steps[ii].flags['C_CONTIGUOUS']:
            print('input_data has to be a contigous array.')
            raise ValueError
        if steps[ii].dtype != np.float64:
            print('input should be an ndarray of type np.float64.')
            print('Converting to float64 and continuing...')
            input_data = steps[ii].astype(np.float64)
        rotation_slice(steps[ii], output, intervals[ii], interpolation)
        end_output += output
    return end_output

