from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
from mpi4py import get_config

# Safely retrieve MPI configurations with defaults
mpi_config = get_config()
mpi_compile_args = mpi_config.get('include_dirs', [])
mpi_link_args = mpi_config.get('libraries', [])
mpi_library_dirs = mpi_config.get('library_dirs', [])
mpi_extra_compile_args = mpi_config.get('extra_compile_args', [])
mpi_extra_link_args = mpi_config.get('extra_link_args', [])

extensions = [
    Extension(
        "parameters",
        ["parameters.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],  # Enable optimizations
    ),
    Extension(
        "fluid_dynamics",
        ["fluid_dynamics.pyx"],
        include_dirs=[np.get_include()] + mpi_compile_args,
        library_dirs=mpi_library_dirs,
        libraries=mpi_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3", "-march=native"] + mpi_extra_compile_args,  # Optimizations and MPI flags
        extra_link_args=mpi_extra_link_args,
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level=3,
        compiler_directives={"boundscheck": False, "wraparound": False}
    ),
)
