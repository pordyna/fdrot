from FilesList import FilesList
import numpy as np
from typing import Union, Tuple, Sequence, MutableSequence, Optional
from c_rotation import one_step_extended_pulse  # ignore this error somehow


class SimSequence:
    """

    """
    def __init__(self, files: Union[], iteration: Tuple[int, int, int], first_slice: Tuple[int, int],
                 last_slice: Optional[Tuple[int, int]] = None, full_slice_length: Optional[int] = None) -> None:
        """ """
        self.number_of_steps = iteration[2]
        if self.number_of_steps > 1 and last_slice is None:
            raise TypeError('`last_slice`, can\'t be None, if there is more than one iteration in the sequence.')
        if self.number_of_steps > 2 and full_slice_length is None:
            raise TypeError('`full_slice_length`, can\'t be None, if there are more than 2 iterations in the sequence.')
        self.first_iteration = iteration[0]
        self.iter_step = iteration[1]
        self.number_of_steps = iteration[2]
        self.first_slice = first_slice
        self.last_slice = last_slice
        self.full_slice_length = full_slice_length
        self.files_B = files_B
        self.files_ne = files_ne

    def step_to_iter(self, step: int) -> int:
        if step < 0 or step >= self.number_of_steps:
            raise ValueError("`step` is out of scope.")
        return self.first_iteration + step * self.iter_step

    def get_interval(self, step: int) -> Tuple[int,int]:
        if step == 0:
            return self.first_slice
        if step == self.number_of_steps -1:
            return self.last_slice
        if step < 0 or step >= self.number_of_steps:
            raise ValueError("`step` is out of scope.")
        else:
            stop = self.first_slice[1] + self.full_slice_length * step
            start = stop - self.full_slice_length
            return start, stop

    def get_data(self, steps: Union[int, Sequence[int], str], make_contiguous: bool = True)-> MutableSequence[np.ndarray]:
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
                raise ValueError('First positional argument has to be an integer, a sequence of integers or \'all\'.')
        data = [len(steps)]
        for ii,  step in enumerate(steps):
            data[ii] = self.files_B.open(self.step_to_iter(step))

        if make_contiguous:
            for ii, step in enumerate(data):
                if not  step.flags['C_CONTIGUOUS']:
                    data[ii] = np.ascontiguousarray(step)
        return data

    def rotation_2d_perp(self, pulse_length_in_cells: int) -> np.ndarray:
        #create output:
        output = np.zeros((self.files_B.sim_box_shape[0] * self.files_B.sim_box_shape[1] *
                           pulse_length_in_cells), dtype=np.float64)

        output = output.reshape((pulse_length_in_cells, self.files_B.sim_box_shape[1], self.files_B.sim_box_shape[0]))

        for step in range(self.number_of_steps):
            step_data = self.get_data(step)
            step_interval = self.get_interval(step)

           #one_step_extended_pulse(step_data, , output)
