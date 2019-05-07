fdrot package
=============

Subpackages
-----------

.. toctree::

    fdrot.sim_data

Module contents
---------------

.. automodule:: fdrot
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: fdrot.kernel3d(pulse: numpy.ndarray, input_arr: numpy.ndarray, output: numpy.ndarray, global_start: int, global_stop: int, leading_start: int, leading_stop: int) -> None

Submodules
----------

fdrot.sim\_sequence module
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fdrot.sim_sequence
    :members:
    :undoc-members:
    :show-inheritance:

fdrot.spliters module
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: fdrot.spliters
    :members:
    :undoc-members:
    :show-inheritance:

fdrot.Kernel2D cython extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 .. automodule:: fdrot.Kernel2D
    :no-members:

    .. autoclass:: fdrot.Kernel2D.Kernel2D(self, double [:, ::1] input_d, double [:, ::1] output_d, double [::1] pulse , double factor, Py_ssize_t global_start, Py_ssize_t global_end, bint interpolation=0, bint add=1, bint inc_sym_only_vertical_middle = 1)
        :undoc-members:
        :show-inheritance:


