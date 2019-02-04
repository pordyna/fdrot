import openpmd_api
import fdrot
from os.path import join
import numpy as np


if __name__ == "__main__":
    base_path = '/home/pawel/Work/HZDR_files/net/gssnfs/bigdata/hplsim/scratch/ordyna35/runs/'
    sim = 'FoilLCT_3D_03'
    inner_path = 'simOutput/h5/'
    section_spc = 'simData_%T.h5'

    path = join(base_path, sim)
    path = join(path, inner_path)
    full_path = join(path, section_spc)

    series = openpmd_api.Series(full_path, openpmd_api.Access_Type.read_only)
    files = fdrot.sim_data.OpenPMDList(series, data_stored=['Bz', 'Bx', 'By', 'n_e'])

    inc_time = 4500 * files.single_time_step
    sequence = fdrot.sim_sequence.seq_cells(start=0,
                                            end=64,
                                            inc_time=inc_time,
                                            iter_step=250,
                                            pulse_length_cells=1,
                                            files=files,
                                            propagation_axis='x')

    from scipy.constants import physical_constants, c

    h = physical_constants['Planck constant in eV s'][0]
    E = 6.5e3  # eV
    wvl = h * c / E

    rotated = sequence.rotation_3d_perp(pulse=np.ones(1, dtype=np.float32), second_axis_output='y', wavelength=wvl,
                                        global_cut_output_first=None, global_cut_output_second=(90, 186))
