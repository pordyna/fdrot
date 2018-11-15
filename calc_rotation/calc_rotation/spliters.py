from scipy.constants import speed_of_light


# this is wrong:
def cell_at_step_end(step, single_step, grid_unit, inc_time, start):
    time_at_step_end = (step + 0.5) * single_step
    prop_time_from_start = time_at_step_end - inc_time
    number_of_prop_cells = (prop_time_from_start * speed_of_light) / grid_unit
    number_of_prop_cells = int(round(number_of_prop_cells))
    end_of_the_slice = start + number_of_prop_cells
    return end_of_the_slice


def cells_perp(start, end, inc_time, iter_step, beam_length_cells, iteration_dt, grid_unit):
    # steps in time and slices in length.
    # the X-Ray comes also in slices,
    # Let's calculate the simulation box slices for the first X-Ray slice.
    # Slices are full, not trimmed by the global start and end points.
    # Let's find the first needed slice first.
    single_step = iteration_dt * iter_step
    first_step = int(round(inc_time / single_step))
    #  offset from the first step middle point
    #offset = inc_time / single_step - first_step
    #time_progress_in_first_step = (0.5 + offset)* single_step
    #time_at_1_step_start = inc_time - time_progress_in_first_step
    #cell_at_1_step_start = start - int(round(( time_progress_in_first_step * speed_of_light) / grid_unit))

    # Now the last step:
    # Cell in there the first X-Ray slice is, when the last X-Ray slice is at the end.
    stop = end + (beam_length_cells - 1)
    propagation_time = (stop - start) * grid_unit / speed_of_light
    last_step = int(round((inc_time + propagation_time)/single_step))
    # Let's fill the slices. End of a slice should be always the start of the next one.
    # When the 0 step is included we get negative time and negative start.
    # That's how it should be. In the end we cut it at start anyway.
    number_of_steps = last_step - first_step + 1
    slices = [None] * number_of_steps
    slice_start = cell_at_step_end(first_step - 1, single_step, grid_unit, inc_time, start)
    for ii, step in enumerate(range(first_step, last_step + 1)):
        slice_end = cell_at_step_end(step, single_step, grid_unit, inc_time, start) + 1
        slices[ii] = (slice_start, slice_end)
        slice_start = slice_end

    # No convert steps to iterations.
    first_iteration = first_step * iter_step
    number_of_iterations = number_of_steps * iter_step
    iterations = (first_iteration, iter_step, number_of_steps)

    return iterations, slices, start, end


def micron_perp(start, end, inc_time, iter_step, beam_length_cells, iteration_dt, grid_unit):
    start_cells = int(round(start / grid_unit))
    end_cells = int(round(end / grid_unit))
    return cells_perp(start_cells, end_cells, inc_time, iter_step, beam_length_cells, iteration_dt, grid_unit)
