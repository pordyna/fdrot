import numpy as np
import calc_rotation


def open_file(path, *args):
    return np.load(path)

def main():
    dt = 1
    grid_unit = 1
    path = '/home/pawel/Work/HOME/Faraday_Rotation/data/steps_sample_1/'
    file_list = calc_rotation.sim_data.FilesLists.UniversalSingle(path, 'Bz', dt, grid_unit, 'step_', '.npy',
                                       open_file)
    path = '/home/pawel/Work/HOME/Faraday_Rotation/data/steps_sample_2/'
    file_list2 = calc_rotation.sim_data.FilesLists.UniversalSingle(path, 'n_e', dt, grid_unit, 'step_', '.npy',
                                       open_file)
    sequence = calc_rotation.SimSequence.SimSequence({'Bz': file_list, 'n_e': file_list2},  (0,1, 1), [(0,31)],0 ,30, 1)

    rotated = sequence.new_rotation_2d_perp(np.ones(1, dtype=np.float64), interpolation=False)

if __name__ == "__main__":
    main()
