from functools import partial
from scipy.constants import c
import numpy as np
from fdrot.spliters import cell_at_step_end
from fdrot.spliters import cells_perp
from decimal import Decimal


class TestCellAtStepEnd:
    single_step = Decimal('10')
    inc_time = Decimal('0')

    def test_one(self):
        grid_unit = Decimal(5 * c)
        assert cell_at_step_end(step=0, single_step=self.single_step,
                                grid_unit=grid_unit,
                                inc_time=self.inc_time, start=0) == 0

    def test_two(self):
        grid_unit = Decimal(5 * c)
        assert cell_at_step_end(step=0, single_step=self.single_step,
                                grid_unit=grid_unit,
                                inc_time=self.inc_time, start=10) == 10

    def test_three(self):
        grid_unit = Decimal(5 * c)
        assert cell_at_step_end(step=1, single_step=self.single_step,
                                grid_unit=grid_unit,
                                inc_time=self.inc_time, start=0) == 2

    def test_four(self):
        grid_unit = Decimal(1 * c)
        assert cell_at_step_end(step=-1, single_step=Decimal('3'),
                                grid_unit=grid_unit,
                                inc_time=self.inc_time, start=0) == -2

    def test_five(self):
        grid_unit = Decimal(1 * c)
        assert cell_at_step_end(step=0, single_step=Decimal('3'),
                                grid_unit=grid_unit,
                                inc_time=self.inc_time, start=0) == 1

    def test_six(self):
        grid_unit = Decimal(1 * c)
        assert cell_at_step_end(step=1, single_step=Decimal('3'),
                                grid_unit=grid_unit,
                                inc_time=self.inc_time, start=0) == 4


class TestCellsPerp:

    def test_one(self):
        """The step length (dt/c) is a  multiple of the cell length"""
        preset = partial(cells_perp, start=0, end=20, inc_time=0,
                         iter_step=1, pulse_length_cells=10,
                         grid_unit=1e-17*c)
        for ii in range(1, 20, 2):
            iterations, slices, start, end = preset(iteration_dt=ii * 1e-17)
            for aa, sl in enumerate(slices):
                assert sl[1] - sl[0] == ii
                if aa < len(slices) - 1:  # not the last one
                    assert sl[1] == slices[aa + 1][0]
            assert iterations[0] == 0
            assert iterations[1] == 1
            number_of_steps = ((20 + 10 - 1 - ii/2) // ii) + 1
            assert (iterations[2] == number_of_steps
                    or iterations[2] == number_of_steps + 1)
            assert start == 0
            assert end == 20

    def test_two(self):
        """The step length (dt/c) is a  multiple of the cell length"""
        preset = partial(cells_perp, start=0, end=20, inc_time=0,
                         iter_step=1, pulse_length_cells=10,
                         grid_unit=1e-17*c)
        for ii in range(2, 20, 2):
            iterations, slices, start, end = preset(iteration_dt=ii * 1e-17)
            for aa, sl in enumerate(slices):
                assert sl[1] - sl[0] == ii
                if aa < len(slices) - 1:  # not the last one
                    assert sl[1] == slices[aa + 1][0]
            assert iterations[0] == 0
            assert iterations[1] == 1
            number_of_steps = ((20 + 10 - 1 - ii/2) // ii) + 1
            assert (iterations[2] == number_of_steps
                    or iterations[2] == number_of_steps + 1)
            assert start == 0
            assert end == 20

    def test_three(self):
        preset = partial(cells_perp, start=0, end=20, inc_time=0,
                         iter_step=1, pulse_length_cells=10,
                         grid_unit=1e-17*c)
        for val in np.random.random(10) + 1:
            iterations, slices, start, end = preset(iteration_dt=val * 1e-17)
            for aa, sl in enumerate(slices):
                dist = sl[1] - sl[0]
                assert (dist == int(val) or dist == int(val) + 1)
                if aa < len(slices) - 1:  # not the last one
                    assert sl[1] == slices[aa + 1][0]
            assert slices[0][0] <= 0
            assert slices[-1][-1] >= 20 + 10 - 1
            assert iterations[0] == 0
            assert iterations[1] == 1
            if len(slices) >= 2:
                assert slices[-2][1] <= 20 + 10 - 1
            assert start == 0
            assert end == 20

