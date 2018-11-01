import numpy as np
from os.path import isfile
class SimSequence:
    """

    """
    def __init__(self, slices, views, full_length=None):
        """ """
        self.slices = slices # list of tuples
        self.views = views # list of numpy views
        self.full_length = full_length # integer

    def make_contiguous(self):
        for view in self.views
            if not  view.flags['C_CONTIGUOUS']:
                view = np.ascontiguousarray(view)

class FilesList:
    """

    """
    def __init__(self, path, single_time_step, ids, fname_front, fname_end, export_func):
        # TODO check if path ends with '/', add if not. (change to property?)
        self.path = path# path to files
        self.single_time_step = single_time_step # duration of a single time step in fs. # to a child class?
        self.ids = ids #  available time steps (list of integers).
        self.fname_front = fname_front # the constant part of a filename string before the number (id)
        self.fname_end = fname_end  # the constant part of a filename string after the number (id) (incl. '.extension')
        self.export_func = export_func # a function which opens the file. it should take the file
            #  as the first argumens, additional arguments are acceptable. It should return the data as an ndarray.

    def check(self, noprint=False):
        bad_ids = []
        for id in self.ids:
            path = self.path + self.fname_front + str(id) + self.fname_end
            if not isfile(path):
                bad_ids.append(id)
        ok = bool(bad_ids)
        if not noprint:
            if not ok:
                print("Some files are missing. Ids of the missing files are:")
                print(bad_ids)
            if  ok:
                print("All files exist.")
        return ok
def const_velocity(files: FilesList, vel: float, inc_time: float, length: float) -> SimSequence:
    """ determines the proper sequence from ..., loads the files to the memory and return an Sequence object."""
    # get the first time step to use:
    files.si
