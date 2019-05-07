import pytest
from unittest.mock import patch
from fdrot.sim_sequence import SimSequence
from fdrot.sim_data import GenericList


class TestSimSequence:
    def test_step_to_iter(self):
        with patch.object(SimSequence, '__init__', lambda *args: None):
            seq = SimSequence()
            seq.number_of_steps = 10
            seq.first_iteration = 2
            seq.iter_step = 3

            for step in range(10):
                assert seq.step_to_iter(step) == 2 + 3 * step
            with pytest.raises(ValueError):
                seq.step_to_iter(-1)
            with pytest.raises(ValueError):
                seq.step_to_iter(10)

    def test_get_files(self):
        with patch.object(SimSequence, '__init__', lambda *args: None):
            seq = SimSequence()
            seq.files = {'a': 1, 'b': 2}
            assert seq.get_files('a') == 1
            assert seq.get_files('b') == 2
            with patch.object(GenericList, '__init__', lambda *args: None):
                files = GenericList()
                seq.files = files
                assert seq.get_files('a') == files

    def test_check_iterations(self):
        with patch.object(SimSequence, '__init__', lambda *args: None):
            seq = SimSequence()
        seq.number_of_steps = 10
        seq.first_iteration = 2
        seq.iter_step = 3
        with patch.object(GenericList, '__init__', lambda *args: None):
            files = GenericList()
            files_1 = GenericList()
            files_2 = GenericList()
        files.ids = list(range(2, 10*3 + 2 + 1, 3))
        files_1.ids = list(range(2, 10*3 + 2 + 1, 3))
        files_2.ids = list(range(2, 10 * 3 + 2 + 1, 3))
        seq.files = {'a': files_1, 'b': files_2}
        assert seq.check_iterations() is True
        seq.files = files
        assert seq.check_iterations() is True
        a = files.ids
        a.pop(5)
        files_1.ids = a
        assert seq.check_iterations() is False
        a = files.ids
        a.pop(5)
        files.ids = a
        seq.files = files
        assert seq.check_iterations() is False
