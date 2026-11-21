# Thermodiff

![logo](https://github.com/ipqa-research/thermodiff/blob/main/media/logo.jpg?raw=true)

thermodiff is a lightweight Python package designed to simplify the symbolic differentiation and manipulation of thermodynamic expressions using SymPy.

It was created to streamline everyday thermodynamic derivations with a clean and straightforward syntax tailored to practical research workflows.

While the package currently focuses on core, simple utility functions, suggestions and improvements are always welcome! If you have ideas for simple features that could be useful, feel free to open an issue or submit a pull request

## Installation

Install the package from PyPI:

```shell
pip install thermodiff
```

Releases are published from the upstream repository
[ipqa-research/thermodiff](https://github.com/ipqa-research/thermodiff).

To install the optional graphical front-end (see below):

```shell
pip install 'thermodiff[gui]'
```

For development, install from a clone of the repository in editable mode
together with the development requirements:

```shell
pip install -e . -r requirements-dev.txt
```

## Graphical front-end

An optional Streamlit application builds the model with buttons, editable
tables and menus, compiling as you edit.  It shows the generated Fortran
module and a typeset preview of every derivative:

```shell
pip install 'thermodiff[gui]'
thermodiff-gui
```

## Tutorial
On the [tutorial](https://ipqa-research.github.io/thermodiff/) you can
check all the basics functionalities of this tool and an example of use.

See the [GUI documentation](https://ipqa-research.github.io/thermodiff/gui.html).

## Model files

Models can also be described in a small declarative `.model` file and
compiled into a Fortran module with the function and all its derivatives:

```shell
thermodiff-model model.model -o model.f90
```

See the [model files](https://ipqa-research.github.io/thermodiff/model.html)
documentation and the corresponding tutorial.


### Code generation presets

`thermodiff-model`, `compile_model`, `compile_model_file`, `emit_module` and
the GUI take a preset of code-generation options:

- `minimal`: the plain generator, the output of earlier versions.
- `standard` (the default): the recommended set (chain rule, symmetry,
  loop nesting and sum sharing).
- `fastest`: the fastest generated code in the met4 benchmark campaign, with
  slower generation and last-bit rounding differences.

Choose it with `--preset` on the command line or `preset=` in Python;
individual flags and keyword arguments override the preset only when given.
The gain of `fastest` comes from kernels specialised to one derivative set,
so use it with `optional_derivatives=False` (the default); with
`optional_derivatives=True` a warning lists the options that become inactive:

```python
from thermodiff.model import compile_model

code = compile_model(text, derivatives="dn,dndn", preset="fastest")
```
