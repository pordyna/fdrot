"""test_Kernel2D.py
"""
from math import isclose
import random
import numpy as np
import pytest
from fdrot.Kernel2D import Kernel2D
from fdrot.test_wraps import wrap_interpolate_up
from fdrot.test_wraps import wrap_interpolate_down
from fdrot.test_wraps import wrap_translator
from fdrot.test_wraps import wrap_if_split


class TestKernel2D:
    def test_interpolate(self):
        for offset, r_offset, shape in [(1, 0.5, (10, 20)), (0, 0, (10, 21))]:
            random_data = np.random.rand(*shape)
            output = np.zeros(tuple(reversed(shape)))
            pulse = np.ones([1])
            kernel = Kernel2D(random_data, output, pulse, 1, 0, shape[1])
            z = np.arange(0, shape[0])

            radi = np.random.random_sample(10) * (shape[1] / 2
                                                 - 0.2) + 0.1
            zz = random.choice(z)
            for radius in radi:
                rr = int(radius)
                assert (wrap_interpolate_up(kernel, zz, rr, radius) ==
                        random_data[zz, shape[1] // 2 + rr])
                assert (wrap_interpolate_down(kernel, zz, rr, radius) ==
                        random_data[zz, shape[1] // 2 - (offset + rr)])

            kernel = Kernel2D(random_data, output, pulse, 1, 0, shape[1],
                              interpolation=True)
            for radius in radi:
                rr = int(radius)
                result = wrap_interpolate_up(kernel, zz, rr, radius)
                val_1 = random_data[zz, shape[1] // 2 + rr]
                try:
                    assert shape[1] // 2 + rr + 1 >= 0
                    val_2 = random_data[zz, shape[1] // 2 + rr + 1]
                except (IndexError, AssertionError):
                    assert result == val_1
                else:
                    compare = (val_1 + (val_2 - val_1)
                               * (radius - (rr + r_offset)))
                    assert isclose(result, compare)

                result = wrap_interpolate_down(kernel, zz, rr, radius)
                val_1 = random_data[zz, shape[1] // 2 - (offset + rr)]
                try:
                    assert shape[1] // 2 - (offset + rr + 1) >= 0
                    val_2 = random_data[zz, shape[1] // 2 - (offset + rr + 1)]
                except (IndexError, AssertionError):
                    assert result == val_1
                else:
                    compare = (val_1 + (val_2 - val_1)
                               * (radius - (rr + r_offset)))
                    assert isclose(result, compare)

    def test_translator_ifsplit(self):
        for modifier, shape in [(0, (10, 20)), (1, (10, 21))]:
            data = np.ones(shape)
            output = np.zeros(tuple(reversed(shape)))
            kernel = Kernel2D(data, output, np.ones(1), 1, 0, shape[1])
            assert (wrap_translator(kernel, 2, 8)
                    == (shape[1] // 2 - 2 + modifier,
                        shape[1] // 2 - 8 + modifier, 0))
            assert wrap_if_split(kernel, 2, 8) is False
            assert (wrap_translator(kernel, 12, 18)
                    == (18 - shape[1] // 2,
                        12 - shape[1] // 2, 1))
            assert wrap_if_split(kernel, 12, 18) is False
            if shape[1] == 20:
                assert (wrap_translator(kernel, 0, 9)
                        == (shape[1] // 2 - 0 + modifier,
                            shape[1] // 2 - 9 + modifier, 0))
                assert wrap_if_split(kernel, 0, 9) is False
                with pytest.raises(ValueError):
                    wrap_translator(kernel, 0, 11)
                assert wrap_if_split(kernel, 0, 11) is True
                with pytest.raises(ValueError):
                    wrap_translator(kernel, 9, 19)
                assert wrap_if_split(kernel, 9, 19) is True
            if shape[1] == 21:
                assert (wrap_translator(kernel, 0, 10)
                        == (shape[1] // 2 - 0 + modifier,
                            shape[1] // 2 - 10 + modifier, 0))
                assert wrap_if_split(kernel, 0, 11) is False
                with pytest.raises(ValueError):
                    wrap_translator(kernel, 0, 12)
                assert wrap_if_split(kernel, 0, 12) is True
                with pytest.raises(ValueError):
                    wrap_translator(kernel, 9, 21)
                assert wrap_if_split(kernel, 9, 21) is True
            with pytest.raises(ValueError):
                wrap_translator(kernel, 1, 1)
