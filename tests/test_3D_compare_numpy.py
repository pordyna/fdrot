import numpy as np
import fdrot
import pytest
import os

@pytest.mark.parametrize('prop_axis,sec_axis',
                         [('x', 'y'), ('y', 'x'), ('z', 'x'),
                          pytest.param('a', 'x',
                                       marks=pytest.mark.xfail(
                                           raises=(ValueError, KeyError))),
                          pytest.param('x', 'a',
                                       marks=pytest.mark.xfail(
                                           raises=ValueError))
                          ])
@pytest.mark.parametrize('interval',
                         [(10, 90), (0, 100), (30, 77),
                          pytest.param((0, 101),
                                       marks=pytest.mark.xfail(
                                           raises=(ValueError, KeyError)))])
def test_3d_compare_numpy(prop_axis, sec_axis, interval, tmp_path):
    tmpdir = str(tmp_path)
    shape = (100, 100, 100)
    for field, arr in [('Bx', np.random.rand(*shape)),
                       ('Bz', np.random.rand(*shape)),
                       ('By', np.random.rand(*shape)),
                       ('n_e', np.random.rand(*shape))]:
        np.save(os.path.join(tmpdir, field + '1'), arr)

    dt = 1
    grid = (1, 1, 1)

    files = {}
    for field in ['Bx', 'Bz', 'By', 'n_e']:
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

    sequence = fdrot.sim_sequence.seq_cells(start=interval[0],
                                            end=interval[1],
                                            inc_time=1,
                                            iter_step=1,
                                            pulse_length_cells=1,
                                            files=files,
                                            propagation_axis=prop_axis)

    rotated = sequence.rotation_3d_perp(np.ones(1, dtype=np.float64),
                                        wavelength=1,
                                        second_axis_output=sec_axis)

    B = np.load(os.path.join(tmpdir, 'B' + prop_axis + '1.npy'))
    n_e = np.load(os.path.join(tmpdir, 'n_e1.npy'))
    slc = [slice(None)] * 3
    slc[files['n_e'].axis_map[prop_axis]] = slice(interval[0], interval[1])

    th = (np.sum(B[tuple(slc)] * n_e[tuple(slc)],
                 axis=files['n_e'].axis_map[prop_axis])
          * sequence.integration_factor(1))


    assert np.allclose(rotated, th, atol=1e-15, rtol=1e-15)
