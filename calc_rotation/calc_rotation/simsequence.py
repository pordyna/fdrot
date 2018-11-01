import numpy as np
from os.path import isfile
from typing import Union, Iterable, Callable, Tuple, Sequence, MutableSequence, Optional
from warnings import warn

class SimSequence:
    """

    """
    def __init__(self, slices_x: Sequence[Tuple[int]], slices_views: MutableSequence[np.ndarray],
                 full_length: int) -> None:
        """ """
        self.slices = slices_x # list of tuples
        self.views = slices_views # list of numpy slices_views
        self.full_length = full_length # integer # why do i needed it?

    def make_contiguous(self):
        for ii, view in enumerate(self.views):
            if not  view.flags['C_CONTIGUOUS']:
                self.views[ii] = np.ascontiguousarray(view)


class FilesList:
    """

    """
    def __init__(self, path: str, single_time_step: Union[float, int], ids: Sequence[int],
                 name_front: str, name_end: str, export_func: Callable[..., np.ndarray]) -> None:
        """
        """
        # TODO check if path ends with '/', add if not. (change to property?, same for ids sorting?)
        # TODO move descriptions to the class doc_string.
        self.path = path# path to files
        self.single_time_step = single_time_step # duration of a single time step in fs. # to a child class?
        self.ids = ids  # An array of available time steps (integers).
        self.name_front = name_front # the constant part of a filename  before the number (id)
        self.name_end = name_end  # the constant part of a filename  after the number (id) (incl. '.extension')
        self.export_func = export_func # a function which opens the file. it should take the file
            #  as the first argument, additional arguments are acceptable. It should return the data as an ndarray.

    @property
    def ids(self):
        return self.ids
    @ids.setter
    def ids(self, id_list):
        u_sorted = np.unique(id_list)
        # check the uniqueness.
        if len(id_list) != u_sorted.size():
            warn('The IDs should be unique. Removing obsolete values and continuing.')
        self.ids = u_sorted

    @property
    def path(self):
        return self.path
    @path.setter
    def path(self, s : str) -> None:
        s = s.strip()
        if s[-1] != '/' :
            s = s + '/'
        self.path = s

    def check(self, noprint=False):
        bad_ids = []
        for id in self.ids:
            path = self.path + self.name_front + str(id) + self.name_end
            if not isfile(path):
                bad_ids.append(id)
        ok = bool(bad_ids)
        if not noprint:
            if not ok:
                print("Some files are missing. Ids of the missing files are:")
                print(bad_ids)
            if ok:
                print("All files exist.")
        return ok
    
def const_velocity(files: FilesList, vel: float, inc_time: float, start_x: float, end_x: float,
                   iter_step: int = 1, ignore_missing_step: bool = False) -> SimSequence:
    """ determines the proper sequence from ..., loads the files to the memory and return an Sequence object."""
    # get the first time step to use:
    first_step = int(inc_time / files.single_time_step)
    # propagation time:
    length = end_x - start_x
    prop_time = length / vel  # units?
    step_length = vel * files.single_time_step
    last_step = first_step + int(prop_time / files.single_time_step)
    max_id = max(files.ids)
    if last_step > max_id:
        if last_step > max_id - 1:
            raise ValueError('More than one step, at the the end of the sequence, is not available.'
                             ' Try reducing the x-ray delay, or provide the missing data.')
        elif not ignore_missing_step:
            missing = length % step_length
            raise ValueError('Last step in the sequence is not available. The propagation length exceeds '.format())
