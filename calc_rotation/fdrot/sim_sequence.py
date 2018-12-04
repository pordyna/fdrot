"""sim_sequence.py
This module is a part of the Fdrot package.
It provides tools for managing and creating sequences of time steps
and propagating over them.
"""

from typing import Union, Tuple, Sequence, MutableSequence, Mapping
from warnings import warn

import numpy as np

from .sim_data import GenericList
from .c_rotation import one_step_extended_pulse
from . import spliters
from .Kernel2D import Kernel2D

SimFiles = Union[GenericList, Mapping[str, GenericList]]
# TODO: maybe change Mapping[str, GenericList] to a named tuple?


# naming: maybe different class name? It's not a Sequence like typing.Sequence.
class SimSequence:
    """ A sequence of time steps, mapped to the simulation data. Stores some simulation parameters.

    Attributes:
        first_iteration: Iteration at which the sequence starts.
        iter_step: Number of iterations in a time step.
        number_of_steps: Number of time steps.
        global_start: The cell from which the propagation begins (This cell is also included).
        global_end: The cell at which the propagation stops (This cell is not included).
        slices: Sequence of intervals, for the leading slice of the pulse, one for each step. It is always the whole
          range, in which the pulse slice would be in the given time step; it is not cut by the global boundaries.
        files: It can be a single files list (see the sim_data module), if it provides all the fields,
         or a mapping of lists to the fields. Acceptable keys are 'Bz' and 'n_e'.
        pulse_length_cells: Length of the pulse in cells.
    """
    def __init__(self, files: SimFiles, pulse_length_cells: int, iteration: Tuple[int, int, int],
                 slices: Sequence[Tuple[int,int]], global_start: int,
                 global_end: int) -> None:
        """ Initializes a SimSequence object.

        Args have the same meaning as the corresponding class attributes.
        first_iteration, iter_step and number_of_steps are passed in the iteration tuple.
        """
        self.first_iteration, self.iter_step, self.number_of_steps = iteration
        self.slices = slices
        self.global_start = global_start
        self.global_end = global_end
        self.files = files
        self.pulse_length_cells = pulse_length_cells
        if not self.check_iterations():
            raise ValueError("Iterations missing in the file index.")

    def step_to_iter(self, step: int) -> int:
        """ Returns the iteration number for the given step."""
        if step < 0 or step >= self.number_of_steps:
            raise ValueError("`step` is out of scope.")

        return self.first_iteration + step * self.iter_step

    def get_files(self, field: str) -> GenericList:
        """ Returns a files list for the given field."""
        if isinstance(self.files, Mapping):
            return self.files[field]
        else:
            return self.files

    def check_iterations(self) -> bool:
        """Checks if all needed iterations are listed.

        Returns: True if nothing is missing, False otherwise.
        """
        missing = []
        for step in range(self.number_of_steps):
            idd = self.step_to_iter(step)
            for field in ('Bz', 'n_e'):
                if idd not in self.get_files(field).ids:
                    missing.append((step, idd, field))
        ok = not bool(missing)
        if not ok:
            print("Those iterations are missing, (step, iteration, field):")
            print(missing)
        return ok

    def get_data(self, field: str, steps: Union[int, Sequence[int], str],
                 make_contiguous: bool = True)-> Union[np.ndarray, MutableSequence[np.ndarray]]:
        """Returns the field data for one or more steps.

        If make_contiguous is set to True, the 'C_CONTIGUOUS' flag is checked and, if needed, a
        contiguous (in the row major) copy is returned.
         """
        try:
            steps.strip()
        except AttributeError:
            try:
                steps[0]
            except TypeError:
                steps = [steps]
        else:
            if steps == 'all':
                steps = range(self.number_of_steps)
            else:
                raise ValueError('Second positional argument has to be an integer,'
                                 ' a sequence of integers or \'all\'.')
        data = [len(steps)]
        for ii,  step in enumerate(steps):
            data[ii] = self.get_files(field).open(self.step_to_iter(step), field)
            # This must be removed when single precision is also supported. Problems with fused types.
            if data[ii].dtype != np.float64:
                data[ii] = data[ii].astype(np.float64)
        if make_contiguous:
            for ii, step in enumerate(data):
                if not  step.flags['C_CONTIGUOUS']:
                    warn('At least one array was not C_Contiguous.')
                    data[ii] = np.ascontiguousarray(step)
        if len(data) == 1:
            return data[0]
        return data

    # TODO: implement the factor. Has to include the integration constants and dx.
    def rotation_2d_perp(self, pulse: np.ndarray, interpolation: bool = True) -> np.ndarray:
        """Propagates the pulse and calculates the integrated faraday rotation.

        The effect is integrated over the pulse.
        Args:
            pulse: An array containing the weights of pulse slices. That is a discrete, normed (to 1) pulse.
            interpolation: Turns interpolation on and off.
        Returns: rotation data
        """
        if pulse.size != self.pulse_length_cells:
            raise ValueError("This sequence was generated for a different pulse length ")

        # need one files list to get the correct output shape:
        files_bz = self.get_files('Bz')
        # create output:
        # TODO: When fused types start to work, allow single precision output.
        output = np.zeros((files_bz.sim_box_shape[0] * files_bz.sim_box_shape[1]), dtype=np.float64)
        output = output.reshape((files_bz.sim_box_shape[1], files_bz.sim_box_shape[0]))
        factor = 1.0  # It should be sth else. Not implemented yet.

        # start the Kernel:
        step_data = self.get_data('Bz', 0) * self.get_data('n_e', 0)
        kernel = Kernel2D(step_data, output, pulse, factor, self.global_start, self.global_end,
                          interpolation=interpolation, inc_sym=0, add=1)
        # rotate with the first slice:
        # TODO: maybe put it in the loop, by allowing kernel initialization without specifying the input?
        kernel.propagate_step(self.slices[0][0], self.slices[0][1])

        # do the other steps:
        for step in range(1, self.number_of_steps):
            step_data = self.get_data('Bz', step) * self.get_data('n_e', step)
            step_interval = self.slices[step]
            kernel.input = step_data
            kernel.propagate_step(step_interval[0], step_interval[1])

        return output


def _get_params(files: GenericList):
    dt = files.single_time_step
    grid_unit = files.grid_unit
    return dt, grid_unit


def seq_cells(start: int, end: int, inc_time: float, iter_step: int, pulse_length_cells: int, files: SimFiles
              )-> SimSequence:
    """Creates a SimSequence for the given parameters. Start, end in cells.

    Args:
        start: Cell from which the calculation should start. This cell is included.
        end: Cell at which the calculation should stop. This cell is not included.
        inc_time: Time at which the front of the pulse arrives at start (counting from 0th iteration).
        iter_step: Number of simulation iterations in one time step.
        pulse_length_cells: Length of the beam in cells.
        files: It can be a single files list (see the sim_data module), if it provides all the fields,
         or a mapping of lists to the fields. Acceptable keys are 'Bz' and 'n_e'.
    """
    # Handling the case with mapping:
    if isinstance(files, Mapping):
        params = _get_params(list(files.values())[0])
    else:
        params = _get_params(files)

    args = spliters.cells_perp(start, end, inc_time, iter_step, pulse_length_cells, *params)
    return SimSequence(files, pulse_length_cells, *args)


def seq_microns(start: int, end: int, inc_time: float, iter_step: int, pulse_length_cells: int, files: SimFiles):
    """Creates a SimSequence for the given parameters. Start, end in microns.

       Args:
           start: Distance from the beginning of the simulation box to the point,
            where the calculation should start (microns).
           end: Distance from the beginning of the simulation box to the point,
            where the calculation should stop (microns).
           inc_time: Time at which the front of the pulse arrives at start (counting from 0th iteration).
           iter_step: Number of simulation iterations in one time step.
           pulse_length_cells: Length of the beam in cells.
           files: It can be a single files list (see the sim_data module), if it provides all the fields,
            or a mapping of lists to the fields. Acceptable keys are 'Bz' and 'n_e'.
       """
    if isinstance(files, Mapping):
        params = _get_params(list(files.values())[0])
    else:
        params = _get_params(files)
    args = spliters.micron_perp(start, end, inc_time, iter_step, pulse_length_cells, *params)
    return SimSequence(files, pulse_length_cells, *args)
