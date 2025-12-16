from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy


extensions = [
    Extension(
        name="fast_ops",
        sources=["src/fast_ops.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    name="scratch-cnn",
    version="0.1.0",
    description="Scratch CNN with optional Cython acceleration",
    python_requires=">=3.9",
    package_dir={"": "src"},
    py_modules=["cnn_base"],
    ext_modules=cythonize(extensions, language_level="3"),
)
