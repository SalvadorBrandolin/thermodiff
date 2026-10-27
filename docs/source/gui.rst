Graphical front-end
===================

A small `Streamlit <https://streamlit.io>`_ application is included as an
optional extra.  It offers buttons, drop-down menus and editable tables for
the ``.model`` language instead of a bare text file: you describe the model,
the application compiles it while you edit and shows

* the generated Fortran module (ready to download), and
* a typeset preview of the function and of every derivative.

Installation
------------

The GUI is kept out of the default installation.  Install it with::

    pip install 'thermodiff[gui]'

Usage
-----

Start the application with::

    thermodiff-gui

Your browser opens on a form with the model name, the Fortran ``use``
statements and editable tables for the variables, the parameters and the
intermediate expressions.  A drop-down menu loads the bundled examples
(Peng-Robinson, ideal gas enthalpy, mole fraction and a minimal template) and
a button downloads the generated ``.f90`` file.

Errors are reported with the offending line of the assembled ``.model``
file, so the strict validation of the command line compiler is available
here too.

Python API
----------

The widgets live in ``thermodiff.gui.app``, which is a Streamlit script.
Everything that can be used or tested without a running Streamlit server is
collected in ``thermodiff.gui.logic``:

.. automodule:: thermodiff.gui.logic
   :members:
   :undoc-members:
   :show-inheritance:
