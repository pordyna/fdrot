from .sim_data.FilesLists import GenericList
import numpy as np
from typing import Union, Tuple, Sequence, MutableSequence, Optional, Mapping
from warnings import warn
from .c_rotation import one_step_extended_pulse  # ignore this error somehow
from . import spliters

SimFiles = Union[GenericList, Mapping[str, GenericList]]


class SimSequence:
    """

    """
    def __init__(self, files: SimFiles, iteration: Tuple[int, int, int],
                 slices: Sequence[Tuple[int,int]], global_start: int , global_end: int, pulse_length_cells: int) -> None:
        """ """
        self.first_iteration = iteration[0]
        self.iter_step = iteration[1]
        self.number_of_steps = iteration[2]
        self.slices = slices
        self.global_start = global_start
        self.global_end = global_end
        self.files = files
        self.pulse_length_cells = pulse_length_cells
        if not self.check_iterations():
            raise ValueError("Iterations missing in the file index.")
    def step_to_iter(self, step: int) -> int:
        if step < 0 or step >= self.number_of_steps:
            raise ValueError("`step` is out of scope.")
        return self.first_iteration + step * self.iter_step

    def get_files(self, field: str) -> GenericList:
        if isinstance(self.files, Mapping):
            return self.files[field]
        else:
            return self.files

    def check_iterations(self):
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
                raise ValueError('Second positional argument has to be an integer, a sequence of integers or \'all\'.')
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

    def rotation_2d_perp(self, interpolation: bool = True) -> np.ndarray:
        # create output:
        files_bz = self.get_files('Bz')
        output = np.zeros((files_bz.sim_box_shape[0] * files_bz.sim_box_shape[1] *
                           self.pulse_length_cells), dtype=np.float64)

        output = output.reshape((self.pulse_length_cells, files_bz.sim_box_shape[1], files_bz.sim_box_shape[0]))
        factor = 1 # It should be sth else.
        for step in range(self.number_of_steps):
            step_data = self.get_data('Bz', step) * self.get_data('n_e', step)
            step_interval = self.slices[step]
            one_step_extended_pulse(output, step_data, self.global_start, self.global_end, step_interval[0],
                                    step_interval[1], factor, interpolation)
        return output


def _get_params(files: GenericList):
    dt = files.single_time_step
    grid_unit = files.grid_unit
    return dt, grid_unit


def seq_cells(start: int, end: int, inc_time: float, iter_step: int, pulse_length_cells: int, files: SimFiles):
    if isinstance(files, Mapping):
        params = _get_params(list(files.values())[0])
    else:
        params = _get_params(files)

    args = spliters.cells_perp(start, end, inc_time, iter_step, pulse_length_cells, *params)
    return SimSequence(files, args[0], args[1], args[2], args[3], pulse_length_cells)


def seq_microns(start: int, end: int, inc_time: float, iter_step: int, pulse_length_cells: int, files: SimFiles):
    if isinstance(files, Mapping):
        params = _get_params(list(files.values())[0])
    else:
        params = _get_params(files)

    args = spliters.micron_perp(start, end, inc_time, iter_step, pulse_length_cells, *params)
    return SimSequence(files, args[0], args[1], args[2], args[3], pulse_length_cells)
