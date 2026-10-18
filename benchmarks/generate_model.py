"""Print the Fortran for one of the benchmark models.

Only the long lived public API (``DiffPlz``, ``sum_components`` and
``to_fortran``) is used, so this script runs unchanged in every revision
of the project that is compared.

The models are built here rather than read from ``examples/`` on purpose:
the oldest revisions do not ship a ``.model`` front-end.
"""

import argparse

import sympy as sp

from thermodiff import DiffPlz, R, T, V, k, l, n, sum_components, to_fortran


def build_pr():
    """Peng-Robinson, with a two index attractive term."""
    nc = sp.Symbol("nc", integer=True)
    ac = sp.IndexedBase("ac", shape=(nc,))
    tc = sp.IndexedBase("Tc", shape=(nc,))
    bb = sp.IndexedBase("b", shape=(nc,))
    kij = sp.IndexedBase("kij", shape=(nc, nc))

    ai = ac[k] * (T / tc[k]) ** 2
    aj = ac[l] * (T / tc[l]) ** 2
    amix = sum_components(
        sum_components(n[k] * n[l] * sp.sqrt(ai * aj) * (1 - kij[k, l]), l), k
    )
    bmix = sum_components(bb[l] * n[l], l)
    expr = R * T / (V - bmix) - amix / (V * (V + bmix) + bmix * (V - bmix))
    return DiffPlz(expr, internal_functions=None, indexes=[k, l], name="f")


def build_toy():
    """A small model the very first revision can already compile."""
    expr = T * sum_components(n[k] ** 2, k) + V * sum_components(n[k], k)
    return DiffPlz(expr, internal_functions=None, indexes=[k], name="f")


#: model name -> (builder, subroutine name)
MODELS = {
    "pr": (build_pr, "pr_derivs"),
    "toy": (build_toy, "f_derivs"),
}


def main(argv=None):
    """Print the Fortran source of the requested model."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("model", choices=sorted(MODELS))
    args = parser.parse_args(argv)

    builder, subroutine = MODELS[args.model]
    print(
        to_fortran(
            builder(),
            name="f",
            subroutine_name=subroutine,
        )
    )


if __name__ == "__main__":
    main()
