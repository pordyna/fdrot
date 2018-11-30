"""spliters.py
This module is a part of the Fdrot package.
It provides tools for splitting the calculation in to time steps,
choice of corresponding iterations and  spacial intervals.
"""

from typing import Tuple, List
from scipy.constants import speed_of_light


# this is wrong: # Did I just forget to remove this comment? TODO: check it.
def cell_at_step_end(step: int, single_step: float, grid_unit: float, inc_time: float, start: int) -> int:
    """ Returns the last cell in a time step.

    Args:
        step: Step number.
        single_step: Duration of one step.
        grid_unit: Length of one cell.
        inc_time: Time at which the front of the pulse arrives at start (counting from 0th iteration).
        start: Reference cell. Position of pulse front at the incoming time.
    """
    time_at_step_end = (step + 0.5) * single_step
    prop_time_from_start = time_at_step_end - inc_time
    number_of_prop_cells = (prop_time_from_start * speed_of_light) / grid_unit
    number_of_prop_cells = int(round(number_of_prop_cells))
    end_of_the_slice = start + number_of_prop_cells
    return end_of_the_slice


def cells_perp(start: int, end: int, inc_time: float, iter_step: int,
               beam_length_cells: int, iteration_dt: float, grid_unit: float
               )-> Tuple[Tuple[int, int, int], List[Tuple[int, int]], int, int]:
    """ Generates a sequence of time steps for a given start time and the first and last cell.

    All arguments with unit have to be in their base SI unit.
    Args:
        start: Cell from which the calculation should start.
        end: Cell at which the calculation should stop.
        inc_time: Time at which the front of the pulse arrives at start (counting from 0th iteration).
        iter_step: Number of simulation iterations in one time step.
        beam_length_cells: Length of the beam in cells.
        iteration_dt: Duration of a single iteration.
        grid_unit: Length of one cell.

    Returns:
        * Iteration to start, iteration step, number of iterations.
        * List of spacial intervals (for each step);
            Only intervals for the first slice of the pulse (pulse front).
        * the start argument.
        * the end argument.
    """
    # steps in time and slices in length.
    # the X-Ray comes also in slices,
    # Let's calculate the simulation box slices for the first X-Ray slice.
    # Slices are full, not trimmed by the global start and end points.
    # Let's find the first needed slice first.
    single_step = iteration_dt * iter_step
    first_step = int(round(inc_time / single_step))

    # Now the last step:
    # Cell in there the first X-Ray slice is, when the last X-Ray slice is at the end.
    stop = end + (beam_length_cells - 1)
    propagation_time = (stop - start) * grid_unit / speed_of_light
    last_step = int(round((inc_time + propagation_time)/single_step))
    # Let's fill the slices. End of a slice should be always the start of the next one.
    # When the 0 step is included we get negative time and negative start.
    # That's how it should be. In the end we cut it at start anyway.
    number_of_steps = last_step - first_step + 1
    slices: List[Tuple[int, int]] = [None] * number_of_steps
    slice_start = cell_at_step_end(first_step - 1, single_step, grid_unit, inc_time, start)
    for ii, step in enumerate(range(first_step, last_step + 1)):
        slice_end = cell_at_step_end(step, single_step, grid_unit, inc_time, start) + 1
        slices[ii] = (slice_start, slice_end)
        slice_start = slice_end

    # Now convert steps to iterations.
    first_iteration = first_step * iter_step
    iterations = (first_iteration, iter_step, number_of_steps)

    return iterations, slices, start, end


def micron_perp(start: float, end: float, inc_time: float, iter_step: int,
                beam_length_cells: int, iteration_dt: float, grid_unit: int
                )-> Tuple[Tuple[int, int, int], List[Tuple[int, int]], int, int]:
    """ Like cells_perp, only with start and end points in microns.

    It finds the first and last cell from the given input, before running cells_perp.
    All  other arguments with unit have to be in their base SI unit.
    Args:
        start: Distance from the beginning of the simulation box to the point,
          where the calculation should start (microns).
        end: Distance from the beginning of the simulation box to the point,
          where the calculation should stop (microns).
        [...] as in cells_perp

    Returns: Same as in cells_perp.
    """
    start, end = start * 10e-15, end * 10e-15  # convert to meters
    start_cells = int(round(start / grid_unit))
    end_cells = int(round(end / grid_unit))
    return cells_perp(start_cells, end_cells, inc_time, iter_step, beam_length_cells, iteration_dt, grid_unit)
