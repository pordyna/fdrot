import numpy as np
from FilesList import FilesList
from SimSequence import SimSequence
from typing import Union, Iterable, Callable, Tuple, Sequence, MutableSequence, Optional, Any
from warnings import warn
from os import path

def const_velocity(files: FilesList, vel: float, inc_time: float, start_x: float, end_x: float,
                   iter_step: int = 1, ignore_missing_first_step: bool = False, ignore_missing_last_step: bool = False,
                   tail_cut_threshold: float = 1e-4) -> SimSequence:
    """ determines the proper sequence from ..., loads the files_B to the memory and returns a SimSequence object."""

    # adapting the time resolution:
    if iter_step != 1 and iter_step % 2 != 0 or iter_step < 1:
        raise ValueError('`iter_step` hast to be an even integer or one, and it cant be negative or 0')
    single_time_step = files.single_time_step * iter_step
    # get the first time step to use:
    length = end_x - start_x
    prop_time = length / vel
    step_length = vel * single_time_step
    first_step = int(round(inc_time / single_time_step))
    # part of a single time step, which is preceding the "1 + the first step".
    front_tail = 0.5 - (first_step - inc_time / single_time_step)
    whole_steps = int((prop_time - front_tail * single_time_step) / single_time_step)
    end_tail = (prop_time - front_tail * single_time_step) % single_time_step
    last_step = first_step + whole_steps + 1
    # first and/or last step omission for very short tails
    omitting_front = False
    omitting_end = False
    if front_tail < tail_cut_threshold:
        first_step = first_step + 1
        omitting_front = True
    if end_tail < tail_cut_threshold:
        last_step = last_step - 1
        omitting_end = True

    max_id = files.ids[-1]  # it is always sorted in an ascending order.
    min_id = files.ids[0]

    steps = np.arange(first_step, last_step + 1, dtype=np.uint16)
    if iter_step == 1:
        steps_ids = steps
    else:
        steps_ids = steps * iter_step + iter_step / 2

    # check for another missing steps:
    for step_id in steps_ids:
        missing = []
        if step_id not in files.ids:
            missing.append(step_id)
        if not missing:
            print('Following time steps are needed for this Sequence, but not listed in files_B.ids.')
            print(missing)
            raise ValueError()

    # Check if the first, or the last step is missing. only for iter_step = 1
    #  First step:
    if iter_step == 1:
        if first_step < min_id:
            if first_step < min_id - 1:
                raise ValueError('More than one step, at the the beginning of the sequence, is not available.'
                                 ' Try increasing the x-ray delay, or provide the missing data.')
            elif not ignore_missing_first_step:
                if omitting_front:
                    missing = step_length + front_tail * step_length
                else:
                    missing = front_tail * step_length
                raise ValueError('First step in the sequence is not available.  {:.3f} microns are not covered by data.'
                                 ' The propagation length in  a single time step is'
                                 ' {:.3f} microns, so {:2.2}% are missing. Run again with `ignore_missing_first_step` set to True'
                                 ' to use the first available time step instead the missing data.'
                                 .format(missing, step_length, missing / step_length * 100))
            else:
                steps_ids = steps_ids[1:]
        else:
            ignore_missing_first_step = False
            # It's for flow control. If user sets it to True, but it's not needed, this sets it back to False.
        # Last step:
        if last_step > max_id:
            if last_step > max_id - 1:
                raise ValueError('More than one step, at the the end of the sequence, is not available.'
                                 ' Try reducing the x-ray delay, or provide the missing data.')

            elif not ignore_missing_last_step:
                if omitting_end:
                    missing = end_tail * step_length + step_length
                else:
                    missing = end_tail * step_length
                raise ValueError('Last step in the sequence is not available. The propagation length exceeds the length'
                                 ' covered by data by {:.3f} microns. The propagation length in a single time step is'
                                 ' {:.3f} microns, so {:2.2}% are missing. Run again with `ignore_missing_last_step` set to True'
                                 ' to use the last available time step instead the missing data.'
                                 .format(missing, step_length, missing / step_length * 100))
            else:
                steps_ids = steps_ids[:-1]
        else:
            ignore_missing_first_step = False
            # It's for flow control. If user sets it to True, but it's not needed, this sets it back to False.

    # check for (another) missing steps:
    for step_id in steps_ids:
        missing = []
        if step_id not in files.ids:
            missing.append(step_id)
        if not missing:
            print('Following time steps are needed for this Sequence, but not listed in files_B.ids.')
            print(missing)
            raise ValueError

    slices = [len(steps_ids)]
    idx_step_length = step_length / files.x_step
    start_first = int(start_x / files.x_step)
    stop_first = (steps_ids[0] + 1) * idx_step_length
    slices[0] = (start_first, stop_first)
    for ii in range(1, len(slices) - 1):
        prev = slices[ii - 1][1]
        slices[ii] = (prev, prev + idx_step_length)
    start_last = slices[-2][1]
    end_last = int(end_x / files.x_step) + 1  # last is not included
    slices[-1] = (start_last, end_last)

    return SimSequence(slices, steps_ids)
