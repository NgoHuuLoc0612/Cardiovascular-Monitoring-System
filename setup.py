"""
Build script for C++ Cardiovascular DSP Engine
Compiles cardio_engine.cpp as a Python C Extension
"""
from distutils.core import setup, Extension

module = Extension(
    'cardio_engine',
    sources=['cardio_engine.cpp'],
    extra_compile_args=['-O3', '-march=native', '-std=c++17', '-ffast-math'],
    language='c++'
)

setup(
    name='CardioEngine',
    version='1.0.0',
    description='C++ DSP engine for cardiovascular signal processing',
    ext_modules=[module]
)
