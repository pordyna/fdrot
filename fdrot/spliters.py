"""
This module provides tools for splitting the calculation in to time
steps, choosing corresponding iterations and spacial intervals.
"""

from typing import Tuple, List
import decimal
from decimal import Decimal
from scipy.constants import speed_of_light

"""This module is a part of the fdrot package.

Authors: Paweł Ordyna
"""


def cell_at_step_end(step: int, single_step: Decimal, grid_unit: Decimal,
                     inc_time: Decimal, start: int) -> int:
    """ Returns the last cell in a time step (for the pulse front).

    Args:
        step
        single_step: Duration of one step (in fs).
        grid_unit: Length of one cell (in nm).
        inc_time: Time at which the front of the pulse arrives at the
          beginning of the first step (counting from 0th iteration)
          in (fs).
        start: Reference cell. Position of the pulse front at the
          incoming time.
    """

    c = Decimal(speed_of_light)  # * 1e-6  # m/s --> nm/fs

    time_at_step_end = (step + Decimal('0.5')) * single_step
    prop_time_from_start = time_at_step_end - inc_time
    number_of_prop_cells = (prop_time_from_start * c) / grid_unit

    # Now proper rounding. The round function rounds *.5 towards the
    # closets even integer. So round(0.5) = 0 round(-0.5) = 1
    # That usually helps avoiding a preference towards the higher numbers.
    # Here we want to always round in one direction at *.5
    # (towards +infinity) 0.5 should become 1 but -0.5 should become 0.
    # But -0.6 should be -1.
    # It would work also with rounding towards - infinity, it just has to
    # be consistent. The desired behavior is introduced with the decimal
    # module.
    if step >= 0:
        rounding = 'ROUND_HALF_UP'
    else:
        rounding = 'ROUND_HALF_DOWN'
    # context_save = decimal.getcontext()
    number_of_prop_cells = number_of_prop_cells.quantize(Decimal('10')**(-3))
    number_of_prop_cells = number_of_prop_cells.to_integral_exact(
        rounding=rounding)
    number_of_prop_cells = int(number_of_prop_cells)
    end_of_the_slice = start + number_of_prop_cells

    # for >0 because we start counting at 0
    # for <0 we start counting at -1 but we propagate from the wrong side
    # so we are calculating the first cell of (step + 1)
    end_of_the_slice -= 1

    return end_of_the_slice


def cells_perp(start: int, end: int, inc_time: float, iter_step: int,
               pulse_length_cells: int, iteration_dt: float, grid_unit: float
               ) -> Tuple[Tuple[int, int, int],
                          List[Tuple[int, int]], int, int]:
    """ Generates a sequence of time steps.

    All arguments with a unit have to be in their base SI unit.

    Args:
        start: Cell from which the calculation should start.
        end: Cell at which the calculation should stop.
        inc_time: Time at which the front of the pulse arrives at start
          (counting from the 0th iteration).
        iter_step: Number of simulation iterations in one time step.
        pulse_length_cells: Length of the beam in cells.
        iteration_dt: Duration of a single iteration.
        grid_unit: Length of one cell.

    Returns:
        * (First iteration, iteration step, total number of iterations).
        * List of spacial intervals (for each step) for the first slice
          of the pulse (pulse front).
        * The 'start' parameter.
        * The 'end' parameter.
    """

    # inc_time *= 1e15  # s to fs
    # iteration_dt *= 1e15  # s to fs
    # grid_unit *= 1e9  # m to nm
    decimal.setcontext(decimal.Context(prec=28,
                                       traps=[decimal.Overflow,
                                              decimal.InvalidOperation,
                                              decimal.DivisionByZero]))

    inc_time = Decimal(inc_time)
    iteration_dt = Decimal(iteration_dt)
    grid_unit = Decimal(grid_unit)

    # steps in time and slices in length.
    # the X-Ray comes also in slices,
    # Let's calculate the simulation box slices for the first X-Ray slice.
    # Slices are full, not trimmed by the global start and end points.
    # Let's find the first needed slice first.

    single_step = iteration_dt * iter_step
    first_step = inc_time / single_step
    first_step = int(first_step.to_integral_value(rounding='ROUND_HALF_UP'))

    # Now the last step:
    # Cell in there the first X-Ray slice is, when the last X-Ray slice is at
    # the end.
    stop = end + (pulse_length_cells - 1)
    # Let's fill the slices. End of a slice should be always the start of the
    # next one.
    # When the 0 step is included we get negative time and negative start.
    # That's how it should be. In the end we cut it at start anyway.
    slice_start = cell_at_step_end(first_step - 1, single_step, grid_unit,
                                   inc_time, start) + 1
    slice_end = cell_at_step_end(first_step, single_step, grid_unit,
                                 inc_time, start) + 1
    slices = [(slice_start, slice_end)]
    step = first_step + 1
    while slices[-1][1] < stop:
        slice_start = slice_end
        slice_end = cell_at_step_end(step, single_step, grid_unit, inc_time,
                                     start) + 1
        slices.append((slice_start, slice_end))
        step += 1
    number_of_steps = len(slices)
    # Now convert steps to iterations.
    first_iteration = first_step * iter_step
    iterations = (first_iteration, iter_step, number_of_steps)

    return iterations, slices, start, end
