# Thermodiff

![logo](https://github.com/SalvadorBrandolin/thermodiff/blob/4005fa272baec3df8315654a0357acc8e14ef5b6/media/logo.jpg?raw=true)

[![DOI](https://zenodo.org/badge/1006736196.svg)](https://doi.org/10.5281/zenodo.23065622)

thermodiff is a lightweight Python package designed to simplify the symbolic differentiation and manipulation of thermodynamic expressions using SymPy.

It was created to streamline everyday thermodynamic derivations with a clean and straightforward syntax tailored to practical research workflows.

While the package currently focuses on core, simple utility functions, suggestions and improvements are always welcome! If you have ideas for simple features that could be useful, feel free to open an issue or submit a pull request

## Installation

```shell
pip install git+https://github.com/SalvadorBrandolin/thermodiff
```

## Tutorial
On the [tutorial](https://salvadorbrandolin.github.io/thermodiff/) you can
check all the basics functionalities of this tool and an example of use.

## Model files

Models can also be described in a small declarative `.model` file and
compiled into a Fortran module with the function and all its derivatives:

```shell
thermodiff-model model.model -o model.f90
```

See the [model files](https://salvadorbrandolin.github.io/thermodiff/model.html)
documentation and the corresponding tutorial.

## Graphical front-end

An optional Streamlit application builds the model with buttons, editable
tables and menus, compiling as you edit.  It shows the generated Fortran
module and a typeset preview of every derivative:

```shell
pip install 'thermodiff[gui]'
thermodiff-gui
```

See the [GUI documentation](https://salvadorbrandolin.github.io/thermodiff/gui.html).

