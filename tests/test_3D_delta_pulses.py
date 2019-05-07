
import numpy as np
import os
import fdrot
from scipy.constants.constants import c


def test_3d_delta_pulses(tmp_path):
    tmpdir = str(tmp_path)
    shape = (300, 50, 25)
    grid = [1e-8] * 3  # (m)
    dt = 300 / 7.525 * grid[0] / c  # (s) that way it needs 7.5 steps - 8
    # iterations

    for ii in range(0, 8):
        for field, arr in [('Bx', np.random.rand(*shape)),
                           ('n_e', np.random.rand(*shape))]:
            np.save(os.path.join(tmpdir, field + str(ii)), arr)

    files = {}
    for field in ['Bx', 'n_e']:
        files[field] = fdrot.sim_data.UniversalSingle(
            path=tmpdir,
            data_stored=field,
            single_time_step=dt,
            grid=grid,
            name_front=field,
            name_end='.npy',
            sim_box_shape=shape,
            export_func=lambda path, *args: np.load(path),
            axis_order=('z', 'y', 'x')
        )

    sequence = fdrot.sim_sequence.seq_cells(start=0,
                                            end=shape[2],
                                            inc_time=0,
                                            iter_step=1,
                                            pulse_length_cells=13,
                                            files=files,
                                            propagation_axis='x')

    pulse = np.arange(13) + 1
    pulse = pulse / np.sum(pulse)
    pulse = pulse.astype(np.float64)

    rotated = sequence.rotation_3d_perp(pulse, wavelength=1,
                                        second_axis_output='y')

    rotated_compare = np.zeros_like(rotated)
    dt_p = grid[0] / c
    delta_pulse = np.ones(1, dtype=np.float64)
    # from the front to the back of the pulse
    for ii, p in enumerate(reversed(pulse)):
        sequence_p = fdrot.sim_sequence.seq_cells(start=0,
                                                  end=shape[2],
                                                  inc_time=ii * dt_p,
                                                  iter_step=1,
                                                  pulse_length_cells=1,
                                                  files=files,
                                                  propagation_axis='x')
        for_one_p = sequence_p.rotation_3d_perp(delta_pulse, wavelength=1,
                                                second_axis_output='y')
        rotated_compare += for_one_p * p

    rel_err = (rotated - rotated_compare) / rotated
    assert np.max(np.abs(rel_err)) < 1e-15  # epsilon fo doubles is close
    # to 1e-16.