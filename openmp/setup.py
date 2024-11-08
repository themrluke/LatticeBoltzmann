from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "parameters",
        ["parameters.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],  # Enable optimization
    ),
    Extension(
        "fluid_dynamics",
        ["fluid_dynamics.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3", "-fopenmp"],  # Enable optimization
        extra_link_args=["-fopenmp"]
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level=3,
        compiler_directives={"boundscheck": False, "wraparound": False}
    ),
)
