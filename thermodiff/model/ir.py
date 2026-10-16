"""Symbolic interpretation of a parsed ``.model`` file.

This module turns the raw declarations of :mod:`thermodiff.model.parser`
into a SymPy expression by:

* resolving declared names (arrays, scalars, dimensions and names brought
  in by ``use`` statements),
* substituting the ``expressions`` block into the ``function`` expression,
* validating that every index is balanced,
* renaming the user indices so they never clash with the ``i``/``j``
  indices reserved by :class:`~thermodiff.diffplz.DiffPlz`.
"""

from dataclasses import dataclass, field

import sympy as sp

from thermodiff import thermovars
from thermodiff.model.errors import ModelError
from thermodiff.model.parser import ExpressionParser

__all__ = ["Declaration", "ModelIR", "build_ir", "split_decl_name"]


#: Thermodynamic variables that ``DiffPlz`` knows how to differentiate.
_VARIABLE_SYMBOLS = {
    "T": thermovars.T,
    "V": thermovars.V,
    "P": thermovars.P,
}

#: Prefix used for the internal index symbols.  It avoids clashing with the
#: ``i`` and ``j`` indices reserved for the derivatives.
_INDEX_PREFIX = "model_idx_"


@dataclass
class Declaration:
    """A resolved declaration.

    Attributes
    ----------
    base : str
        Base name of the declared object.
    rank : int
        Number of indices (0 for scalars).
    letters : tuple of str
        Subscript letters written in the declaration, if any.
    dims : tuple of str
        Dimension names of the declaration.
    line : int
        Line number of the declaration.
    """

    base: str
    rank: int
    letters: tuple
    dims: tuple
    line: int


@dataclass
class ModelIR:
    """The symbolic interpretation of a ``.model`` file."""

    model_name: str
    uses: list = field(default_factory=list)
    imported: list = field(default_factory=list)
    dimensions: list = field(default_factory=list)
    variables: list = field(default_factory=list)
    parameters: list = field(default_factory=list)
    declared_variables: set = field(default_factory=set)
    function: sp.Expr | None = None
    indexes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def split_decl_name(token, dims):
    """Split a declaration token into base name, letters and dimensions.

    ``k_ij`` declared with two dimensions becomes the base ``k`` with the
    (decorative) subscripts ``ij``.  A trailing ``_letters`` is only
    treated as subscripts when its length matches the number of declared
    dimensions, or when no dimensions are given at all.

    Parameters
    ----------
    token : str
        Name as written in the file, for example ``k_ij``.
    dims : tuple of str
        Declared dimensions.

    Returns
    -------
    base : str
        Base name.
    letters : tuple of str
        Subscript letters, if any.
    dims : tuple of str
        The dimensions to use (``nc`` by default for each letter).
    """
    if "_" in token:
        base, _, suffix = token.rpartition("_")
        if base and suffix.isalpha():
            if dims:
                if len(suffix) == len(dims):
                    return base, tuple(suffix), tuple(dims)
            else:
                return base, tuple(suffix), ("nc",) * len(suffix)
    return token, (), tuple(dims)


def _imported_names(use_text):
    """Return the names declared after ``only:`` in a ``use`` line."""
    lowered = use_text.lower()
    marker = lowered.find("only")
    if marker < 0:
        return []
    start = marker + len("only")
    rest = use_text[start:].lstrip()
    if rest.startswith(":"):
        rest = rest[1:]
    names = [name.strip() for name in rest.split(",")]
    return [name for name in names if name]


class _Resolver:
    """Resolve identifiers found in expressions to SymPy objects."""

    def __init__(self, default_upper):
        self.scalars = {}
        self.bases = {}
        self.indexed_bases = {}
        self.pool = {}
        self.default_upper = default_upper

    def index(self, letter):
        """Return (creating if needed) the internal index for ``letter``."""
        if letter not in self.pool:
            self.pool[letter] = sp.Idx(_INDEX_PREFIX + letter)
        return self.pool[letter]

    def add_scalar(self, name, symbol):
        self.scalars[name] = symbol

    def add_base(self, name, rank, indexed_base):
        self.bases[name] = rank
        self.indexed_bases[name] = indexed_base

    def resolve(self, token):
        """Return the SymPy object for an identifier ``token``."""
        if token in self.scalars:
            return self.scalars[token]
        if token in self.bases:
            raise ModelError(
                f"array {token!r} requires {self.bases[token]} subscript(s)"
            )
        if "_" in token:
            base, _, suffix = token.rpartition("_")
            if base in self.bases and suffix.isalpha():
                rank = self.bases[base]
                if len(suffix) != rank:
                    raise ModelError(
                        f"array {base!r} expects {rank} subscript(s), "
                        f"got {len(suffix)}"
                    )
                indexed_base = self.indexed_bases[base]
                indices = [self.index(char) for char in suffix]
                return sp.Indexed(indexed_base, *indices)
        raise ModelError(f"undefined name {token!r}")


def _register_dimension(name, resolver, dimensions):
    """Register a dimension name and return its symbol."""
    if name == "nc":
        symbol = thermovars.nc
        resolver.add_scalar("nc", symbol)
        return symbol
    if name in resolver.scalars:
        return resolver.scalars[name]
    symbol = sp.Symbol(name, integer=True)
    resolver.add_scalar(name, symbol)
    dimensions.append(name)
    return symbol


def _build_declaration(raw, resolver, dimensions, filename, kind):
    """Turn a raw declaration into a :class:`Declaration`."""
    base, letters, dims = split_decl_name(raw.name, raw.dims)
    for dim in dims:
        _register_dimension(dim, resolver, dimensions)
    rank = len(dims)
    if letters and len(letters) != rank:
        raise ModelError(
            f"declaration {raw.name!r} has inconsistent subscripts/dimensions",
            raw.line,
            filename,
        )
    declaration = Declaration(base, rank, letters, dims, raw.line)
    if kind == "variables":
        _register_variable(declaration, resolver, filename)
    return declaration


def _indexed_base(name, rank, dims, resolver):
    """Create an :class:`sympy.IndexedBase` with a well defined shape.

    A shape is required so that the Fortran printer can inspect bare
    indexed objects without raising.
    """
    if dims:
        shape = tuple(resolver.scalars[dim] for dim in dims)
    else:
        shape = (resolver.default_upper,) * rank
    return sp.IndexedBase(name, shape=shape)


def _register_variable(declaration, resolver, filename):
    """Store a variable in the resolver."""
    base = declaration.base
    rank = declaration.rank
    if base in resolver.bases or base in resolver.scalars:
        raise ModelError(
            f"{base!r} is declared more than once",
            declaration.line,
            filename,
        )
    if base == "n":
        if rank != 1:
            raise ModelError(
                "'n' must be declared with a single dimension (rank 1)",
                declaration.line,
                filename,
            )
        resolver.add_base("n", 1, thermovars.n)
        return
    if base in _VARIABLE_SYMBOLS:
        if rank != 0:
            raise ModelError(
                f"variable {base!r} must be a scalar",
                declaration.line,
                filename,
            )
        resolver.add_scalar(base, _VARIABLE_SYMBOLS[base])
        return
    if rank == 0:
        resolver.add_scalar(base, sp.Symbol(base))
    else:
        resolver.add_base(
            base,
            rank,
            _indexed_base(base, rank, declaration.dims, resolver),
        )


def _register_parameter(declaration, resolver, filename):
    """Store a parameter in the resolver."""
    base = declaration.base
    if base in resolver.scalars or base in resolver.bases:
        raise ModelError(
            f"{base!r} is declared more than once",
            declaration.line,
            filename,
        )
    if declaration.rank == 0:
        resolver.add_scalar(base, sp.Symbol(base))
    else:
        resolver.add_base(
            base,
            declaration.rank,
            _indexed_base(base, declaration.rank, declaration.dims, resolver),
        )


def _inline(expr, defs, resolver):
    """Substitute every known intermediate definition into ``expr``."""
    for _ in range(len(defs) + 1):
        replacements = {}
        for base, (rank, letters, definition) in defs.items():
            if rank == 0:
                symbol = sp.Symbol(base)
                if expr.has(symbol):
                    replacements[symbol] = definition
            else:
                for occurrence in expr.atoms(sp.Indexed):
                    if (
                        str(occurrence.base) == base
                        and len(occurrence.indices) == rank
                    ):
                        mapping = {
                            resolver.index(letters[k]): occurrence.indices[k]
                            for k in range(rank)
                        }
                        replacements[occurrence] = definition.xreplace(mapping)
        if not replacements:
            break
        expr = expr.xreplace(replacements)
    return expr


def _free_indices(expr):
    """Return the free indices of ``expr``."""
    bound = {
        limit[0] for total in expr.atoms(sp.Sum) for limit in total.limits
    }
    return {idx for idx in expr.atoms(sp.Idx) if idx not in bound}


def _check_balance(expr, letters, resolver, line, filename, name):
    """Ensure the free indices of ``expr`` match the declared subscripts."""
    expected = {resolver.index(letter) for letter in letters}
    found = _free_indices(expr)
    if found != expected:
        missing = sorted(str(idx) for idx in expected - found)
        extra = sorted(str(idx) for idx in found - expected)
        details = []
        if missing:
            details.append("missing subscripts " + ", ".join(missing))
        if extra:
            details.append("unexpected free indices " + ", ".join(extra))
        raise ModelError(
            f"index mismatch in {name!r}: " + "; ".join(details),
            line,
            filename,
        )


def build_ir(raw, filename=None):
    """Build the symbolic representation of a parsed ``.model`` file.

    Parameters
    ----------
    raw : RawModel
        Result of :func:`thermodiff.model.parser.read_sections`.
    filename : str, optional
        File name used for error messages.

    Returns
    -------
    ModelIR
        The symbolic model ready to be differentiated.
    """
    resolver = _Resolver(default_upper=thermovars.nc)
    dimensions = []
    # ``nc`` is provided by the runtime and is not an extra argument.
    resolver.add_scalar("nc", thermovars.nc)

    if raw.target is not None and raw.target != "fortran":
        raise ModelError(
            f"unsupported target {raw.target!r}; only 'fortran' is supported",
            None,
            filename,
        )

    variables = []
    for raw_decl in raw.variables:
        declaration = _build_declaration(
            raw_decl, resolver, dimensions, filename, "variables"
        )
        variables.append(declaration)

    parameters = []
    for raw_decl in raw.parameters:
        declaration = _build_declaration(
            raw_decl, resolver, dimensions, filename, "parameters"
        )
        _register_parameter(declaration, resolver, filename)
        parameters.append(declaration)

    imported = []
    for use_text, _line in raw.uses:
        for name in _imported_names(use_text):
            if name in resolver.scalars or name in resolver.bases:
                continue
            resolver.add_scalar(name, sp.Symbol(name))
            imported.append(name)

    declared_variables = {declaration.base for declaration in variables}

    defs = {}
    for raw_def in raw.expressions:
        base, letters, def_dims = split_decl_name(raw_def.lhs, ())
        if base in defs or base in resolver.scalars or base in resolver.bases:
            raise ModelError(
                f"{base!r} is defined more than once",
                raw_def.line,
                filename,
            )
        parser = ExpressionParser(
            raw_def.rhs, resolver, raw_def.line, filename
        )
        expression = parser.parse()
        expression = _inline(expression, defs, resolver)
        _check_balance(
            expression, letters, resolver, raw_def.line, filename, base
        )
        defs[base] = (len(letters), letters, expression)
        if not letters:
            resolver.add_scalar(base, sp.Symbol(base))
        else:
            resolver.add_base(
                base,
                len(letters),
                _indexed_base(base, len(letters), def_dims, resolver),
            )

    parser = ExpressionParser(
        raw.function.rhs, resolver, raw.function.line, filename
    )
    function = parser.parse()
    function = _inline(function, defs, resolver)

    leftovers = sorted(
        {
            str(occurrence.base)
            for occurrence in function.atoms(sp.Indexed)
            if str(occurrence.base) in defs
        }
        | {
            str(symbol)
            for symbol in function.atoms(sp.Symbol)
            if str(symbol) in defs
        }
    )
    if leftovers:
        raise ModelError(
            "intermediate expressions were not fully substituted: "
            + ", ".join(leftovers),
            raw.function.line,
            filename,
        )

    model_name = (raw.model or "model").strip()
    warnings = _unused_warnings(function, parameters)

    return ModelIR(
        model_name=model_name,
        uses=[use_text for use_text, _ in raw.uses],
        imported=imported,
        dimensions=dimensions,
        variables=variables,
        parameters=parameters,
        declared_variables=declared_variables,
        function=function,
        indexes=list(resolver.pool.values()),
        warnings=warnings,
    )


def _unused_warnings(function, parameters):
    """Return warning strings for declared but unused parameters."""
    used_symbols = {str(symbol) for symbol in function.atoms(sp.Symbol)}
    used_bases = {str(indexed.base) for indexed in function.atoms(sp.Indexed)}
    warnings = []
    for declaration in parameters:
        if declaration.rank == 0:
            present = declaration.base in used_symbols
        else:
            present = declaration.base in used_bases
        if not present:
            warnings.append(
                f"parameter {declaration.base!r} is declared but never used"
            )
    return warnings
