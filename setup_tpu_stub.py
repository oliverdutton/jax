"""
Setup script to build the TPU custom call stub extension.

Usage:
    python setup_tpu_stub.py build_ext --inplace
"""

from setuptools import setup, Extension
import sys

# Define the extension module
tpu_stub_ext = Extension(
    'tpu_stub_extension',
    sources=['tpu_stub_extension.c'],
    extra_compile_args=['-O2', '-Wall'],
)

setup(
    name='tpu_stub_extension',
    version='0.1.0',
    description='Stub implementation of tpu_custom_call for CPU backend',
    ext_modules=[tpu_stub_ext],
    python_requires='>=3.7',
)
