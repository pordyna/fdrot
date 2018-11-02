import numpy as np
from os.path import isfile
from typing import Union, Iterable, Callable, Tuple, Sequence, MutableSequence, Optional, Any
from warnings import warn

class SimSequence:
    """

    """
    def __init__(self, slices_x: Sequence[Tuple[int,int]], slices_views: MutableSequence[np.ndarray]) -> None:
        """ """
        self.slices = slices_x # list of tuples
        self.views = slices_views # list of numpy slices_views

    def make_contiguous(self):
        for ii, view in enumerate(self.views):
            if not  view.flags['C_CONTIGUOUS']:
                self.views[ii] = np.ascontiguousarray(view)


class FilesList:
    """

    """
    def __init__(self, path: str, single_time_step: float, ids: Sequence[int], x_step: float,
                 name_front: str, name_end: str, export_func: Callable[[str, ...], np.ndarray], args_export: Optional[Sequence[Any]] = None) -> None:
        """
        """
        # TODO move descriptions to the class doc_string.
        self.path = path# path to files
        self.single_time_step = single_time_step # duration of a single time step in fs. # to a child class?
        self.ids = ids  # An array of available time steps (integers).
        self.name_front = name_front.strip() # the constant part of a filename  before the number (id)
        self.name_end = name_end.strip()  # the constant part of a filename  after the number (id) (incl. '.extension')
        self.x_step = x_step # x step in microns
        self.export_func = export_func # a function which opens the file. it should take the path to the file
            #  as the first argument, additional arguments are acceptable  It should return the data as an ndarray.
        #
        if args_export is None:
            self. args_export = ()
        else:
            self.args_export = args_export
        # check if files are there:
        isok, bad = self.check(noprint=True)
        if not isok:
            print('File check at the initialization performed. Some files are not there. Ids of the missing files are:')
            print(bad)

    @property
    def ids(self):
        return self.ids
    @ids.setter
    def ids(self, id_list: Sequence) -> None:
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

    def full_path(self, id: int) -> str:
        if id not in self.ids:
            raise ValueError('The id has to be listed in `ids` Attribute!')
        full_path = self.path + self.name_front + str(id) + self.name_end
        return full_path

    def check(self, noprint: bool = False) -> Tuple[bool, list]:
        bad_ids = []
        for id in self.ids:
            if not isfile(self.full_path(id)):
                bad_ids.append(id)
        ok = bool(bad_ids)
        if not noprint:
            if not ok:
                print("Some files are missing. Ids of the missing files are:")
                print(bad_ids)
            if ok:
                print("All files exist.")
        return ok, bad_ids



    def open(self, id:int) -> np.ndarray:
        return self.export_func(self.full_path(id), *self.args_export)

# class OpenPmdList(FilesList):    coming soon :)
 # also introducing functions, which index files from a directory and create an FilesList/Open... object, would be nice.


def const_velocity(files: FilesList, vel: float, inc_time: float, start_x: float, end_x: float,
                   iter_step: int = 1, ignore_missing_first_step: bool = False, ignore_missing_last_step: bool = False, tail_cut_threshold: float = 1e-4) -> SimSequence:
    """ determines the proper sequence from ..., loads the files to the memory and returns a SimSequence object."""

    # adapting the time resolution:
    if iter_step != 1 and iter_step % 2 != 0 or iter_step < 1:
        raise ValueError('`iter_step` hast to be an even integer or one, and it cant be negative or 0')
    single_time_step = files.single_time_step * iter_step
    # get the first time step to use:
    length = end_x - start_x
    prop_time = length / vel
    step_length = vel * single_time_step
    first_step = int(round(inc_time / single_time_step))
    # part of a single time step, which is preceding the "1 + the first step".
    front_tail = 0.5 - (first_step - inc_time / single_time_step)
    whole_steps = int((prop_time - front_tail * single_time_step) / single_time_step)
    end_tail = (prop_time - front_tail * single_time_step) % single_time_step
    last_step = first_step + whole_steps + 1
    # first and/or last step omission for very short tails
    omitting_front = False
    omitting_end = False
    if front_tail < tail_cut_threshold:
        first_step = first_step + 1
        omitting_front = True
    if end_tail < tail_cut_threshold:
        last_step = last_step -1
        omitting_end = True

    max_id = files.ids[-1]  # it is always sorted in an ascending order.
    min_id = files.ids[0]
    
    steps = np.arange(first_step, last_step + 1, dtype=np.uint16)
    if iter_step == 1:
        steps_ids = steps
    else:
        steps_ids = steps * iter_step  + iter_step / 2

    # check for another missing steps:
    for step_id in steps_ids:
        missing = []
        if step_id not in files.ids:
            missing.append(step_id)
        if not missing:
            print('Following time steps are needed for this Sequence, but not listed in files.ids.')
            print(missing)
            raise ValueError()

    # Check if the first, or the last step is missing. only for iter_step = 1
    #  First step:
    if iter_step == 1:
        if first_step < min_id:
            if first_step < min_id -1:
                raise ValueError('More than one step, at the the beginning of the sequence, is not available.'
                                 ' Try increasing the x-ray delay, or provide the missing data.')
            elif not ignore_missing_first_step:
                if  omitting_front:
                    missing = step_length + front_tail * step_length
                else:
                    missing = front_tail * step_length
                raise ValueError('First step in the sequence is not available.  {:.3f} microns are not covered by data.'
                                 ' The propagation length in  a single time step is'
                                 ' {:.3f} microns, so {:2.2}% are missing. Run again with `ignore_missing_first_step` set to True'
                                 ' to use the first available time step instead the missing data.'
                                 .format(missing, step_length, missing / step_length * 100))
            else: steps_ids = steps_ids[1:]
        else:
            ignore_missing_first_step = False
            # It's for flow control. If user sets it to True, but it's not needed, this sets it back to False.
        # Last step:
        if last_step > max_id:
            if last_step > max_id - 1:
                raise ValueError('More than one step, at the the end of the sequence, is not available.'
                                 ' Try reducing the x-ray delay, or provide the missing data.')

            elif not ignore_missing_last_step:
                if omitting_end:
                    missing = end_tail * step_length + step_length
                else:
                    missing = end_tail * step_length
                raise ValueError('Last step in the sequence is not available. The propagation length exceeds the length'
                                 ' covered by data by {:.3f} microns. The propagation length in a single time step is'
                                 ' {:.3f} microns, so {:2.2}% are missing. Run again with `ignore_missing_last_step` set to True'
                                 ' to use the last available time step instead the missing data.'
                                 .format(missing, step_length, missing / step_length * 100))
            else: steps_ids = steps_ids[:-1]
        else:
            ignore_missing_first_step = False
            # It's for flow control. If user sets it to True, but it's not needed, this sets it back to False.
        
    # check for (another) missing steps:
    for step_id in steps_ids:
        missing = []
        if step_id not in files.ids:
            missing.append(step_id)
        if not missing:
            print('Following time steps are needed for this Sequence, but not listed in files.ids.')
            print(missing)
            raise ValueError()
    
    steps_data = [len(steps_ids)]
    for ii, step_id in enumerate(steps_ids):
        steps_data[ii] = files.open(step_id)
    slices = [len(steps_data)]
    idx_step_length = step_length / files.x_step
    start_first = int(start_x / files.x_step)
    stop_first = (steps_ids[0] + 1) * idx_step_length
    slices[0] = (start_first, stop_first)
    for ii in range(1, len(slices) - 1):
        prev = slices[ii-1][1]
        slices[ii] = (prev, prev + idx_step_length)
    start_last = slices[-2][1]
    end_last = int(end_x / files.x_step)
    slices[-1] = (start_last, end_last)

    return SimSequence(slices, steps_data)

    # move to SimSequence
    for step in steps_data:
        if step.shape != steps_data[0].shape:
            raise ValueError('steps')
  


    # what do I need to get the indexing for the  slices.
    # sim box shape. #resolutions: one step in x in microns.
    #
    # in 'steps_overx' we need sth that copes with slices on the right side and slices on both sides.
    # just some simple translation and dividing the middle one in two should work
