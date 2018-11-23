from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize


extensions = [Extension('calc_rotation.c_rotation', ['calc_rotation/c_rotation.pyx']),
                        Extension('calc_rotation.c_rotation_new', ['calc_rotation/Kernel2D.pyx'])]
                         # ,define_macros=[('CYTHON_TRACE', '1')])]
setup(name='Fdrot',
    ext_modules=cythonize(extensions, annotate=True, gdb_debug=False,
    packages = find_packages(), version='0.0.3'
)

