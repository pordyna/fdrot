from FilesList import FilesList
import numpy as np
from typing import Union, Tuple, Sequence, MutableSequence

class SimSequence:
    """

    """
    def __init__(self, slices_x: Sequence[Tuple[int,int]], iterations: Sequence, files: FilesList) -> None:
        """ """
        self.slices = slices_x # list of tuples
        self.min_x = self.slices[0]
        self.max_x = self.slices[-1]
        self.iterations = iterations
        self.files = files

    def get_data(self, steps: Union[int, Sequence[int], str], make_contiguous: bool = True)->  MutableSequence[np.ndarray]:
        try:
            steps.strip()
        except AttributeError:
            try:
                steps[0]
            except TypeError:
                steps = [steps]
        else:
            if steps == 'all':
                steps = self.iterations
            else:
                raise ValueError('First positional argument has to be an integer, a sequence of integers or \'all\'.')
        data = [len(steps)]
        for ii,  step in enumerate(steps):
            data[ii] = self.files.open(self.iterations[step])

        if make_contiguous:
            for ii, step in enumerate(data):
                if not  step.flags['C_CONTIGUOUS']:
                    data[ii] = np.ascontiguousarray(step)
        return data
    def rotation_2d_perp(self, pulse_length_in_cells: int) -> np.ndarray:
        #create output:
        output = np.zeros((self.files.sim_box_shape[0] * self.files.sim_box_shape[1] *
                           pulse_length_in_cells), dtype=np.float64)
        output = output.reshape((pulse_length_in_cells, self.files.sim_box_shape[1], self.files.sim_box_shape[0]))

        for ii, slice in enumerate(self.slices):
            step_data = self.get_data(ii)
            some_cython_function_for_a_single_step(step_data, slice, output)
