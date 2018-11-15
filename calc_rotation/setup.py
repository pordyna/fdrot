from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [Extension('calc_rotation.c_rotation', ['calc_rotation/c_rotation.pyx'])]
setup(name='Fdrot',
    ext_modules=cythonize(extensions, annotate=True, gdb_debug=False),
    packages = find_packages(), version='0.0.2'
)
