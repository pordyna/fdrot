"""sim_sequence.py
This module is a part of the Fdrot package.
It provides tools for managing and creating sequences of time steps
and propagating over them.
"""

from typing import Union, Tuple, Sequence, MutableSequence, Mapping, NamedTuple, Optional
from warnings import warn
import math

import numpy as np
from scipy.constants import electron_mass, speed_of_light, elementary_charge, epsilon_0
from .sim_data import GenericList
from . import spliters
from .Kernel2D import Kernel2D
from .kernel3d import kernel3d

SimFiles = Union[GenericList, Mapping[str, GenericList]]
# TODO: maybe change Mapping[str, GenericList] to a named tuple?

class AxisOrder(NamedTuple):
    x: int
    y: int
    z: Optional[int]  # In case we need it for the 2D case as well.

def _switch_axis(array: np.ndarray, current_order: AxisOrder, desired_order: AxisOrder) -> np.ndarray:
    return np.moveaxis(array, current_order, desired_order)

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

        if isinstance(self.files, Mapping):
            shapes = []
            for field in self.files.values():
                shapes.append(field.sim_box_shape)
            assert len(set(shapes)) == 1, "All file lists should have the same `sim_box_shape`."

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
            if isinstance(self.files, Mapping):
                for field, file in self.files.items():
                    if idd not in file.ids:
                        missing.append((step, idd, field))
        ok = not bool(missing)
        if not ok:
            print("Those iterations are missing, (step, iteration, field):")
            print(missing)
        return ok

    def get_data(self, field: str, steps: Union[int, Sequence[int], str],
                 make_contiguous: bool = True,
                 cast_to: Optional[np.dtype] = None,
             dim1_cut: Optional[Tuple[int, int]] = None,
             dim2_cut: Optional[Tuple[int, int]] = None,
             dim3_cut: Optional[Tuple[int, int]] = None )-> Union[np.ndarray, MutableSequence[np.ndarray]]:
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
            if cast_to is not None:
                if data[ii].dtype < cast_to:
                    data[ii] = data[ii].astype(cast_to)
                if data[ii].dtype > cast_to:
                    raise TypeError("Data for the iteration {} can't be safely cast to the desired type.".format(ii))
        if make_contiguous:
            for ii, step in enumerate(data):
                if not  step.flags['C_CONTIGUOUS']:
                    warn('At least one array was not C_Contiguous.')
                    data[ii] = np.ascontiguousarray(step)
        if len(data) == 1:
            return data[0]
        return data

    def integration_factor(self, wavelength):
        critical_density = electron_mass * epsilon_0 * ((2 * math.pi * speed_of_light)
                                                        / (elementary_charge * wavelength))**2
        files_bz = self.get_files('Bz')
        delta_x = files_bz.grid_unit
        return elementary_charge / (2 * speed_of_light * electron_mass) / critical_density * delta_x

    def rotation_2d_perp(self, pulse: np.ndarray, wavelength: float, interpolation: bool = True) -> np.ndarray:
        """Propagates the pulse and calculates the integrated faraday rotation.

        The effect is integrated over the pulse.
        Args:
            pulse: An array containing the weights of pulse slices. That is a discrete, normed (to 1) pulse.
            interpolation: Turns interpolation on and off.
        Returns: rotation data
        """

        # need one files list to get the correct output shape:
        files_bz = self.get_files('Bz')
        assert files_bz.data_dim == 2, "This method works only  with 2 dimensional data."

        if pulse.size != self.pulse_length_cells:
            raise ValueError("This sequence was generated for a different pulse length ")
        if pulse.dtype < np.dtype('float64'):
            pulse = pulse.astype(np.float64)

        # create output:
        # TODO: When fused types start to work, allow single precision output.
        output = np.zeros((files_bz.sim_box_shape[0] * files_bz.sim_box_shape[1]), dtype=np.float64)
        output = output.reshape((files_bz.sim_box_shape[1], files_bz.sim_box_shape[0]))
        factor = self.integration_factor(wavelength)

        # start the Kernel:
        step_data = self.get_data('Bz', 0) * self.get_data('n_e', 0)
        kernel = Kernel2D(step_data, output, pulse, factor, self.global_start, self.global_end,
                          interpolation=interpolation, inc_sym=False, add=True)
        # rotate with the first slice:
        # TODO: maybe put it in the loop, by allowing kernel initialization without specifying the input?
        kernel.propagate_step(self.slices[0][0], self.slices[0][1])

        # do the other steps:
        for step in range(1, self.number_of_steps):
            step_data = (self.get_data('Bz', step, cast_to=np.dtype('float64'))
                         * self.get_data('n_e', step, cast_to=np.dtype('float64')))
            step_interval = self.slices[step]
            kernel.input = step_data
            kernel.propagate_step(step_interval[0], step_interval[1])

        return output

    def rotation_3d_perp(self, pulse, wavelength: float, second_axis_output: str, x_ray_axis: str) -> np.ndarray:

        acceptable_names = ['x', 'y', 'z']
        if second_axis_output not in acceptable_names:
            raise ValueError("`first_axis_output` hast to be 'x' or 'y' or 'z'.")
        if x_ray_axis not in acceptable_names:
            raise ValueError("`x_ray_axis` hast to be 'x' or 'y' or 'z'.")
        b_field_component: str = 'B' + x_ray_axis
        # Axis order in the loaded data.
        b_axis_order = self.get_files(b_field_component).axis_map
        n_e_axis_order = self.get_files('n_e').axis_map
        # x_ray_axis has to be the last one, second_axis_output has to be the second

        last_axis = acceptable_names
        for ax in [second_axis_output, x_ray_axis]:
            idx = last_axis.index(ax)
            last_axis.pop(idx)
        assert len(last_axis) == 1
        last_axis = last_axis[0]
        desired_order = {x_ray_axis: 3, second_axis_output: 2, last_axis: 1}
        desired_order = AxisOrder(**desired_order)
        output = np.zeros()
        for step in range(self.number_of_steps):
            # TODO add chunks.
            data_b = self.get_data(b_field_component, step)  # cast_to ? needed if we introduce cython.
            data_n = self.get_data('n_e', step)
            if b_axis_order != desired_order:
                data_b = _switch_axis(data_b, b_axis_order, desired_order)
            if n_e_axis_order != desired_order:
                data_n = _switch_axis(data_n, n_e_axis_order, desired_order)
            data = data_b * data_n
            step_interval = self.slices[step]
            kernel3d(pulse, data, output, self.global_start, self.global_end, step_interval[0], step_interval[1])
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
         or a mapping of lists to the fields. Acceptable keys are 'Bx', 'By', 'Bz' and 'n_e'.
    """
    # TODO do I need this iter_step?
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
            or a mapping of lists to the fields. Acceptable keys are 'Bx', 'By', 'Bz' and 'n_e'.
       """
    if isinstance(files, Mapping):
        params = _get_params(list(files.values())[0])
    else:
        params = _get_params(files)
    args = spliters.micron_perp(start, end, inc_time, iter_step, pulse_length_cells, *params)
    return SimSequence(files, pulse_length_cells, *args)
