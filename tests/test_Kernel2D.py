"""test_Kernel2D.py
"""
import numpy as np
from fdrot.Kernel2D import Kernel2D


def setup_kernel(odd, sample, pulse, factor, global_start, global_end, **kwargs):
    if odd:
        data = np.load("samples/odd_" + sample + ".npy")
    else:
        data = np.load("samples/even_" +sample + ".npy")
    output = np.zeros_like(data)
    kernel = Kernel2D(data, output, pulse, factor, global_start, global_end, **kwargs)
    return kernel

class TestKernel2D:
    def test_kernel_init_odd():
        setup_kernel(True, "const", )