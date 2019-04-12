"""
This module provides tools for managing and creating sequences of time
steps and propagating over them.
"""

from typing import (Union, Tuple, Sequence, Mapping, NamedTuple,
                    Optional, Callable)
from warnings import warn
from functools import partial
import math

import numpy as np
from scipy.constants import (electron_mass, speed_of_light,
                             elementary_charge, epsilon_0)

from .sim_data import GenericList
from . import spliters
from .Kernel2D import Kernel2D
from .kernel3d import kernel3d

"""
This module is a part of the fdrot package.

Authors: Paweł Ordyna
"""

SimFiles = Union[GenericList, Mapping[str, GenericList]]


class AxisOrder(NamedTuple):
    x: int
    y: int
    z: Optional[int] = None  # In case we need it for the 2D case as well.


def _switch_axis(array: np.ndarray, current_order: AxisOrder,
                 desired_order: AxisOrder) -> np.ndarray:
    # Handling 2D case. Otherwise numpy.moveaxis gets None
    # as an axis position.
    if current_order.z is None and desired_order.z is None:
        current_order = (current_order[0], current_order[1])
        desired_order = (desired_order[0], desired_order[1])
    return np.moveaxis(array, current_order, desired_order)


class SimSequence:

    """Provides tools for FR calculation over a time steps sequence.

    It stores the sequence of time steps and the intervals passed by
    the pulse front in these intervals. It provides methods for the
    Faraday Rotation calculation from 2D and 3D data inputs.

    Attributes:
        files: It can be a single files list (see the sim_data module)
          if it provides all the fields, or a mapping of lists to the
          fields. Acceptable keys in  a such dictionary are 'Bx', 'By',
          'Bz' and 'n_e'.
        first_iteration: Iteration at which the sequence starts.
        iter_step: Number of iterations in a time step.
        number_of_steps: Number of time steps.
        global_start: The cell from which the propagation begins (This
          cell is also included).
        global_end: The cell at which the propagation stops (This cell
          is not included).
        slices: Sequence of intervals, for the leading slice of the
          pulse, one for each step. It is always the whole range in
          which the pulse slice would be in the given time step; It is
          not cut by the global boundaries.
        pulse_length_cells: Length of the pulse in cells.
        propagation_axis: Axis along which the pulse should be
          propagated. Either 'x', 'y', or 'z'.
        cell_length_in_prop_direction: Grid spacing in the propagation
          direction.
        axis_map (dict): keys [x, y, z] possible values [1, 2, 3].
    """

    def __init__(self, files: SimFiles, pulse_length_cells: int,
                 propagation_axis: str, iteration: Tuple[int, int, int],
                 slices: Sequence[Tuple[int, int]], global_start: int,
                 global_end: int) -> None:
        """ Initializes a SimSequence object.

        Args:
            iteration: (first_iteration, iter_step,  number_of_steps)
              See the class doc-string.
        """
        self.first_iteration, self.iter_step, self.number_of_steps = iteration
        self.slices: Sequence[Tuple[int, int]] = slices
        self.global_start: int = global_start
        self.global_end: int = global_end
        self.files: SimFiles = files
        self.pulse_length_cells: int = pulse_length_cells
        self._acceptable_names = ['x', 'y', 'z']
        if propagation_axis not in self._acceptable_names:
            raise ValueError("`x_ray_axis` hast to be 'x' or 'y' or 'z'.")
        self.propagation_axis: str = propagation_axis
        
        if not self.check_iterations():
            raise ValueError("Iterations missing in the file index.")

        if isinstance(self.files, Mapping):
            shapes = []
            axis_orders = []
            grids = []
            for file_index in self.files.values():
                shapes.append(file_index.sim_box_shape)
                axis_orders.append(AxisOrder(**file_index.axis_order))
                grids.append(file_index.grid)
            assert len(set(shapes)) == 1, "All file lists should have the " \
                                          "same `sim_box_shape` attr. ."
            assert len(set(axis_orders)) == 1, "All file lists should have " \
                                               "the same `axis_map` attr. ."
            assert len(set(grids)) == 1, "All file lists should have the" \
                                         " same `grid` attr. ."

        # n_e is always needed.
        self.sim_box_shape = self.get_files('n_e').sim_box_shape
        self.axis_map = self.get_files('n_e').axis_map
        ax: int = self.axis_map[self.propagation_axis]
        self.cell_length_in_prop_direction = self.get_files('n_e').grid[ax]

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
            else:
                if idd not in self.files.ids:
                    missing.append((step, idd))
        ok = not bool(missing)
        if not ok:
            print("Those iterations are missing, (step, iteration, field):")
            print(missing)
        return ok

    def get_data(self, field: str, step: int,
                 transform: Optional[Callable[[np.ndarray],
                                              np.ndarray]] = None,
                 make_contiguous: bool = True,
                 cast_to: Optional[np.dtype] = None,
                 dim_cut: Optional[Union[Sequence[Tuple[int, int]],
                                         None]] = None) -> np.ndarray:
        """Returns field data for a given step.

        Args:
            field: Field for which the data should be returned (e.g)
              'Bz' or 'n_e'.
            step: Step in the sequence for which the data should be
              returned.
            transform: Set this to transform the data in any way before
              a contiguity check is performed. Use it for any
              transformation that could change contiguity.
            make_contiguous:  If True, the 'C_CONTIGUOUS' flag is
              checked and, if needed, a C-contiguous  copy is returned.
            cast_to: The data in the array can be casted to a specified
              type. If None, no casting is performed.
            dim_cut: Set this to obtain a chunk of data. It should be a
              sequence with three elements, each for every axis. Specify
              the cut along every axis; For (a,b) a is included, b is
              not. If None is set for an axis instead of a tuple, this
              axis won't be cut.
         """

        data = self.get_files(field).open(self.step_to_iter(step), field,
                                          *dim_cut)
        if cast_to is not None:
            if data.dtype < cast_to:
                data = data.astype(cast_to)
            if data.dtype > cast_to:
                raise TypeError("Data for the iteration {} can't be safely"
                                "cast to the desired type.".format(step))
        if transform is not None:
            data = transform(data)
        if make_contiguous:
            if not data.flags['C_CONTIGUOUS']:
                warn('The array was not C-contiguous.')
                data = np.ascontiguousarray(data)
        return data

    def integration_factor(self, wavelength):
        """ Calculates the integration factor.

        .. math::
            \\frac{e_0}{2 c e_m} \\frac{\\Delta x}{n_c}
        """

        critical_density = (electron_mass * epsilon_0
                            * ((2 * math.pi * speed_of_light)
                               / (elementary_charge * wavelength))**2)
        delta_x = self.cell_length_in_prop_direction
        return (elementary_charge / (2 * speed_of_light * electron_mass)
                / critical_density * delta_x)

    def rotation_2d_perp(self, pulse: np.ndarray, wavelength: float,
                         interpolation: bool = True,
                         cut_second_axis: Optional[Tuple[int, int]] = None,
                         inc_sym_only_vertical_middle: bool = True
                         ) -> np.ndarray:
        """Propagates the pulse and calculates faraday rotation (in 2D).

        The effect is integrated over the pulse.

        Args:
            pulse: An array containing the weights of pulse slices. The
              pulse is  discrete and normed (to 1).
            wavelength: X-ray wavelength in meters.
            interpolation: If True the linear interpolation is on,
              otherwise the nearest neighbour method is used.
            cut_second_axis: If set the rotation will be calculated
              only in the specified interval, the simulation box is cut
              along the z axis (cylinder axis). Useful if the simulation
              box is bigger than the target.
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

        Returns: rotation profile
        """

        # need one files list to get the correct output shape:
        files_bz = self.get_files('n_e')
        assert files_bz.data_dim == 2, "This method works only  with 2 " \
                                       "dimensional data."

        if pulse.size != self.pulse_length_cells:
            raise ValueError("This sequence was generated for a different "
                             "pulse length ")
        if pulse.dtype < np.dtype('float64'):
            pulse = pulse.astype(np.float64)

        # check if axis swap is needed:
        # swap axis and swap sim_box_shape
        labels = ['x', 'y']
        labels.pop(labels.index(self.propagation_axis))
        ax_2 =labels[0]
        desired_order = {self.propagation_axis: 1, ax_2: 0}
        desired_order = AxisOrder(**desired_order)
        order_in_index = AxisOrder(**self.axis_map)

        if desired_order != order_in_index:
            sim_box_shape_0 = files_bz.sim_box_shape[1]
            sim_box_shape_1 = files_bz.sim_box_shape[0]
            transform = partial(_switch_axis,  current_order=order_in_index,
                                desired_order=desired_order)
        else:
            sim_box_shape_0 = files_bz.sim_box_shape[0]
            sim_box_shape_1 = files_bz.sim_box_shape[1]
            transform = None

        # cutting along the second axis (not the radius)
        dim_cut = [None, None]
        if cut_second_axis is not None:
            dim_cut[self.axis_map[ax_2]] = cut_second_axis
            # the output array has to be smaller:
            sim_box_shape_0 = cut_second_axis[1] - cut_second_axis[0]
        # create output:
        output = np.zeros((sim_box_shape_0 * sim_box_shape_1),
                          dtype=np.float64)
        output = output.reshape((sim_box_shape_1, sim_box_shape_0))
        factor = self.integration_factor(wavelength)

        # start the Kernel:
        Bz = self.get_data('Bz', 0, cast_to=np.dtype('float64'),
                           transform=transform, dim_cut=dim_cut)
        ne = self.get_data('n_e', 0, cast_to=np.dtype('float64'),
                           transform=transform, dim_cut=dim_cut)
        step_data = Bz*ne
        kernel = Kernel2D(step_data, output, pulse, factor, self.global_start,
                          self.global_end, interpolation=interpolation,
                          add=True, inc_sym_only_vertical_middle=
                          inc_sym_only_vertical_middle)
        # rotate with the first slice:
        kernel.propagate_step(self.slices[0][0], self.slices[0][1])
        # do the other steps:
        for step in range(1, self.number_of_steps):
            step_data = (self.get_data('Bz', step, cast_to=np.dtype('float64'),
                                       transform=transform, dim_cut=dim_cut)
                         * self.get_data('n_e', step,
                                         cast_to=np.dtype('float64'),
                                         transform=transform, dim_cut=dim_cut))
            step_interval = self.slices[step]
            kernel.input = step_data
            kernel.propagate_step(step_interval[0], step_interval[1])
        return output

    def rotation_3d_perp(self, pulse, wavelength: float,
                         second_axis_output: str, n=1,
                         global_cut_output_first: Optional[
                             Tuple[int, int]] = None,
                         global_cut_output_second: Optional[
                             Tuple[int, int]] = None) -> np.ndarray:
        """Propagates the pulse and calculates faraday rotation (in 3D).

        The effect is integrated over the pulse.

        Args:
            pulse: An array containing the weights of pulse slices. The
              pulse is  discrete and normed (to 1).
            wavelength: X-ray wavelength in meters.
            second_axis_output: Defines the output orientation. Either
              'x', 'y' or 'z'.
            global_cut_output_first: It is possible to use only a
              specific chunk of data for the calculation. This defines
              this chunk along the axis that is neither the propagation
              axis or the one set in second_axis_output.
            global_cut_output_second: This defines the chunk of data to
              use along the axis set in second_axis_output.

        Returns: rotation profile
        """
        if second_axis_output not in self._acceptable_names:
            raise ValueError("`second_axis_output` hast to be 'x' or 'y' or "
                             "'z'.")
        b_field_component: str = 'B' + self.propagation_axis
        # x_ray_axis has to be the last one,
        # second_axis_output has to be the second.

        # Find which axis is the first axis of the output:
        last_axis = self._acceptable_names.copy()
        for axis in [second_axis_output, self.propagation_axis]:
            idx = last_axis.index(axis)
            last_axis.pop(idx)
        assert len(last_axis) == 1
        last_axis = last_axis[0]
        # Set the desired axis order
        desired_order = {self.propagation_axis: 1, second_axis_output: 2,
                         last_axis: 0}
        desired_order = AxisOrder(**desired_order)
        order_in_index = AxisOrder(**self.axis_map)

        # Get output shape. Set axis transformation if needed.
        output_first_idx = self.axis_map[last_axis]
        output_second_idx = self.axis_map[second_axis_output]
        if self.axis_map != desired_order:
            transform = partial(_switch_axis, current_order=order_in_index,
                                desired_order=desired_order)
            output_dim_0 = self.sim_box_shape[output_first_idx]
            output_dim_1 = self.sim_box_shape[output_second_idx]
        else:
            transform = None
            output_dim_0 = self.sim_box_shape[0]
            output_dim_1 = self.sim_box_shape[1]

        # Specify slicing for the data being loaded.
        dim_cut = [None, None, None]
        # Let's set slicing in the propagation direction.
        # Firstly one have to find the propagation axis in the data
        # before the transform (axis swap).
        prop_ax_idx = self.axis_map[self.propagation_axis]
        # slicing is set separately for each dimension in a tuple (a,b)
        # and it corresponds to  [a:b] in numpy or the [a,b[ interval.
        # Here a & b are global_start and global_end, as we don't need the data
        # from outside this scope.
        dim_cut[prop_ax_idx] = (self.global_start, self.global_end)

        dim_cut[output_second_idx] = global_cut_output_second
        if global_cut_output_first is not None:
            dim_cut[output_first_idx] = global_cut_output_first
            output_dim_0 = (dim_cut[output_first_idx][1]
                            - dim_cut[output_first_idx][0])
        if global_cut_output_second is not None:
            dim_cut[output_second_idx] = global_cut_output_second
            output_dim_1 = (dim_cut[output_second_idx][1]
                            - dim_cut[output_second_idx][0])

        # Create output:
        output = np.zeros((output_dim_0, output_dim_1), dtype=np.float64)
        
        verbose = np.arange(0, self.number_of_steps +1, n)
        # Begin calculation:
        for step in range(self.number_of_steps):
            if step in verbose:
                print(('step {} out of {} started'
                       ).format(step + 1, self.number_of_steps))
            # TODO add chunks.
            data_b = self.get_data(b_field_component, step,
                                   transform=transform, make_contiguous=True,
                                   dim_cut=dim_cut)
            data_n = self.get_data('n_e', step, transform=transform,
                                   make_contiguous=True, dim_cut=dim_cut)
            data = data_b * data_n
            step_interval = self.slices[step]
            local_start = 0
            local_end = self.global_end - self.global_start
            step_start = step_interval[0] - self.global_start
            step_stop = step_interval[1] - self.global_start

            kernel3d(pulse, data, output, local_start, local_end,
                     step_start, step_stop)
        output *= self.integration_factor(wavelength)
        return output


def _get_params_and_check(files: GenericList, propagation_axis: str,
                          start: int, end: int):
    dt = files.single_time_step
    ax = files.axis_map[propagation_axis]
    grid_unit = files.grid[ax]
    if start >= files.sim_box_shape[ax] or end > files.sim_box_shape[ax]:
        raise ValueError("(start, end) outside the simulation box.")
    return dt, grid_unit


def seq_cells(start: int, end: int, inc_time: float, iter_step: int,
              pulse_length_cells: int, files: SimFiles,
              propagation_axis: str)-> SimSequence:
    """Creates a SimSequence for the given parameters.

    Args:
        start: Cell from which the calculation should start. This cell
          is included.
        end: Cell at which the calculation should stop. This cell is not
          included.
        inc_time: Time at which the front of the pulse arrives at start
          (counting from 0th iteration).
        iter_step: Number of simulation iterations in one time step.
        pulse_length_cells: Length of the beam in cells.
        files: It can be a single files list (see the sim_data module)
          if it provides all the fields, or a mapping of lists to the
          fields. Acceptable keys in  a such dictionary are 'Bx', 'By',
          'Bz' and 'n_e'.
        propagation_axis: Axis along which the pulse should be
          propagated. Either 'x', 'y', or 'z'.

    Raises:
          ValueError: If either `start` or `end` is outside the
            simulation box.

    Returns: X-ray propagation sequence
    """

    # Handling the case with mapping:
    if isinstance(files, Mapping):
        params = _get_params_and_check(list(files.values())[0],
                                       propagation_axis, start, end)
    else:
        params = _get_params_and_check(files, propagation_axis, start, end)

    args = spliters.cells_perp(start, end, inc_time, iter_step,
                               pulse_length_cells, *params)
    return SimSequence(files, pulse_length_cells, propagation_axis, *args)
