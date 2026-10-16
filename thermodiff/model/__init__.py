"""User friendly front-end for thermodynamic model code generation.

This package implements the ``.model`` file format: a small, declarative
description of a thermodynamic model (variables, parameters, intermediate
expressions and a final function) that is compiled into a self contained
Fortran module with the function value and its analytical derivatives.

The main entry point is :func:`compile_model`.
"""

from thermodiff.model.compile import compile_model, compile_model_file
from thermodiff.model.errors import ModelError
from thermodiff.model.ir import build_ir
from thermodiff.model.parser import read_sections

__all__ = [
    "compile_model",
    "compile_model_file",
    "build_ir",
    "read_sections",
    "ModelError",
]
