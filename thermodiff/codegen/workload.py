"""Workload mode: fixed-output kernels behind one dispatcher.

``optional_derivatives=True`` gives one kernel for any output set, but it
pays a fixed cost on every call and disables the loop passes of the
``fastest`` preset.  The workload mode keeps that single public entry point
and makes it cheap:

* one *fixed-output* kernel per requested output set
  (``optional_derivatives=False``, specialised by the generator);
* the ``optional_derivatives=True`` kernel as a fallback;
* a dispatcher with the public signature of the optional kernel.  It builds
  a bit mask from ``present()`` and selects the kernel whose output set is
  *exactly* the present set with ``select case``; any other combination
  goes to the fallback.  Covering a request with a larger kernel would need
  scratch arrays for the extra outputs, which costs more than it saves.

Optionally (``fixed_nc``) every fixed-output kernel is also copied once per
listed number of components, with ``nc`` turned into a named constant so
that the compiler unrolls the loops, drops the array descriptors and keeps
the temporaries on the stack.  The dispatcher selects the copy by ``nc``.

The module is built on top of :func:`~thermodiff.codegen.pipeline.to_fortran`:
the kernels are generated one by one, parsed for their dummy arguments and
rewritten textually (names, array dummies, directives).  The dummy lists
are read from the generated code, so extra integer dimensions and the
placeholder arguments are forwarded without special cases.

Performance pitfalls handled here (measured by the ``yaeos`` benchmarks):

* the dispatcher's *optional* array dummies are not ``contiguous``: an
  absent or disassociated actual argument would make gfortran try to pack
  it and crash; the *inputs* keep ``contiguous`` (they are never absent);
* the fixed-output kernels get explicit-shape dummies (``n(nc)``,
  ``dBij(nc, nc)``), so forwarding the dispatcher's arrays makes no copy
  and the descriptor handling disappears (about 7 % of a small kernel);
* ``!GCC$ ATTRIBUTES NOINLINE`` on the fixed-output kernels: at ``-O3``
  gfortran inlines them into the dispatcher and the code loses the
  contiguity facts (up to 18x slower).
"""

import dataclasses
import re
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from thermodiff.codegen.options import (
    CodegenOptions,
    Preset,
    resolve_codegen_options,
)
from thermodiff.codegen.pipeline import to_fortran

__all__ = ["workload_module", "workload_procedures"]

#: Name of the component-count dummy of the generated kernels.
NC_NAME = "nc"
#: Dummy that keeps the ``nc`` slot of a fixed-nc copy.
NC_DUMMY_NAME = "nc_in"
#: Largest number of outputs: the mask is a default (32-bit) integer.
MAX_MASK_OUTPUTS = 30
#: Local variable of the dispatcher that holds the bit mask.
MASK_NAME = "present_mask_"
#: Column at which wrapped calls and headers continue.
WRAP_WIDTH = 100
#: gfortran directive that keeps a kernel out of the dispatcher.
NOINLINE_DIRECTIVE = "!GCC$ ATTRIBUTES NOINLINE :: {name}"

_HEADER = re.compile(r"^\s*subroutine\s+(\w+)\s*\(", re.I)
_IMPLICIT = re.compile(r"^\s*implicit\s+none\b", re.I)
_USE = re.compile(r"^\s*use\s", re.I)
_CONTAINS = re.compile(r"^\s*contains\b", re.I)
_DECL = re.compile(
    r"^(?P<indent>\s*)(?P<head>(?:real|integer)\b[^:]*?)::(?P<items>.*)$",
    re.I,
)
_ITEM = re.compile(r"^\s*(\w+)\s*(?:\((.*)\))?\s*$", re.S)


@dataclasses.dataclass
class Dummy:
    """A dummy argument declaration.

    Attributes
    ----------
    name : str
        Dummy argument name.
    spec : str
        Type specification, for example ``real(8)`` or ``integer``.
    attrs : list of str
        Attributes in their original order (``contiguous``,
        ``intent(out)``, ...).
    dims : list of str
        Dimension specifications (``":"`` for assumed shape); empty for
        scalars.
    """

    name: str
    spec: str
    attrs: list[str]
    dims: list[str]

    @property
    def rank(self) -> int:
        """Number of array dimensions (0 for scalars)."""
        return len(self.dims)

    @property
    def is_output(self) -> bool:
        """Whether the dummy is ``intent(out)``."""
        return "intent(out)" in self._attr_set

    @property
    def is_optional(self) -> bool:
        """Whether the dummy is ``optional``."""
        return "optional" in self._attr_set

    @property
    def _attr_set(self) -> set[str]:
        return {re.sub(r"\s+", "", attr).lower() for attr in self.attrs}

    def render(self) -> str:
        """Return the declaration without indentation."""
        text = ", ".join([self.spec, *self.attrs]) + " :: " + self.name
        if self.dims:
            text += "(" + ", ".join(self.dims) + ")"
        return text


@dataclasses.dataclass
class KernelInterface:
    """What the workload mode needs to know about a generated kernel.

    Attributes
    ----------
    name : str
        Subroutine name.
    args : list of str
        Dummy argument names in declaration order.
    dummies : dict
        Lower-case name to :class:`Dummy`.
    lines : list of str
        Source lines of the kernel.
    header_end : int
        Index of the first line after the subroutine header.
    decl_lines : dict
        Lower-case dummy name to the index of its declaration line.
    indent : str
        Indentation of the ``subroutine`` line.
    uses : list of str
        ``use`` statements of the specification part.
    """

    name: str
    args: list[str]
    dummies: dict[str, Dummy]
    lines: list[str]
    header_end: int
    decl_lines: dict[str, int]
    indent: str
    uses: list[str]

    @property
    def outputs(self) -> list[str]:
        """Names of the ``intent(out)`` dummies, in argument order."""
        return [
            arg for arg in self.args if self.dummies[arg.lower()].is_output
        ]

    @property
    def has_nc(self) -> bool:
        """Whether the kernel receives the number of components."""
        return NC_NAME in self.dummies

    @property
    def has_extra_dimensions(self) -> bool:
        """Whether integer dummies other than ``nc`` are present."""
        return any(
            dummy.spec == "integer" and dummy.name.lower() != NC_NAME
            for dummy in self.dummies.values()
        )


def _split_top_level(text: str) -> list[str]:
    """Split ``text`` at the commas that are outside parentheses."""
    pieces: list[str] = []
    depth = 0
    current = ""
    for char in text:
        depth += (char == "(") - (char == ")")
        if char == "," and depth == 0:
            pieces.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        pieces.append(current.strip())
    return pieces


def _read_header(lines: list[str]) -> tuple[str, list[str], int, int]:
    """Return the name, arguments, first and end line of the header."""
    for start, line in enumerate(lines):
        match = _HEADER.match(line)
        if match:
            break
    else:
        raise ValueError("no subroutine found in the kernel source")
    text = ""
    end = start
    while text.count("(") == 0 or text.count("(") != text.count(")"):
        text += " " + lines[end].strip().rstrip("&").lstrip("&")
        end += 1
    inside = text[text.index("(") + 1 : text.rindex(")")]
    return match.group(1), _split_top_level(inside), start, end


def _parse_declaration(line: str) -> list[Dummy]:
    """Parse a type declaration line into one record per declared name."""
    match = _DECL.match(line)
    if not match:
        return []
    head = _split_top_level(match.group("head"))
    dummies = []
    for item in _split_top_level(match.group("items")):
        parsed = _ITEM.match(item)
        if not parsed:
            raise ValueError(f"cannot parse the declaration {line.strip()!r}")
        dims = _split_top_level(parsed.group(2)) if parsed.group(2) else []
        dummies.append(Dummy(parsed.group(1), head[0], head[1:], dims))
    return dummies


def _count_uses(lines: list[str], start: int) -> int:
    """Count the ``use`` lines that precede ``implicit none``."""
    count = 0
    for line in lines[start:]:
        if _IMPLICIT.match(line):
            break
        count += bool(_USE.match(line))
    return count


def parse_kernel(source: str) -> KernelInterface:
    """Read the interface of a generated kernel.

    Parameters
    ----------
    source : str
        Text of one generated subroutine.

    Returns
    -------
    KernelInterface
        Name, dummy list and the declaration of every dummy.

    Raises
    ------
    ValueError
        If a dummy has no declaration or shares a line with a local.
    """
    lines = source.splitlines()
    name, args, start, header_end = _read_header(lines)
    wanted = {arg.lower() for arg in args}
    dummies: dict[str, Dummy] = {}
    decl_lines: dict[str, int] = {}
    for index in range(header_end, len(lines)):
        if _CONTAINS.match(lines[index]):
            break
        found = _parse_declaration(lines[index])
        mine = [dummy for dummy in found if dummy.name.lower() in wanted]
        if mine and len(mine) != len(found):
            raise ValueError(f"{name}: dummy and local on one line")
        for dummy in mine:
            if dummy.name.lower() not in dummies:
                dummies[dummy.name.lower()] = dummy
                decl_lines[dummy.name.lower()] = index
    missing = sorted(wanted - set(dummies))
    if missing:
        raise ValueError(f"{name}: no declaration for {missing}")
    uses = [line for line in lines[header_end:] if _USE.match(line)]
    return KernelInterface(
        name,
        args,
        dummies,
        lines,
        header_end,
        decl_lines,
        _indent_of(lines[start]),
        [line.strip() for line in uses[: _count_uses(lines, header_end)]],
    )


def _wrap_items(prefix: str, items: Sequence[str], suffix: str) -> list[str]:
    """Join ``prefix item, item ... suffix`` with ``&`` continuations."""
    lines: list[str] = []
    current = prefix
    indent = " " * (len(prefix) - len(prefix.lstrip()) + 5)
    for position, item in enumerate(items):
        piece = item + ("," if position < len(items) - 1 else suffix)
        if len(current) + len(piece) + 1 > WRAP_WIDTH and current.strip():
            lines.append(current.rstrip() + " &")
            current = indent
        current += piece + " "
    lines.append(current.rstrip())
    return lines


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def render_kernel(
    iface: KernelInterface,
    *,
    name: str,
    args: Sequence[str],
    edit: Callable[[Dummy], Dummy],
    after_implicit: Sequence[str] = (),
) -> str:
    """Rewrite a kernel with a new name, argument list and declarations.

    Parameters
    ----------
    iface : KernelInterface
        Parsed kernel.
    name : str
        New subroutine name.
    args : sequence of str
        Argument list of the new header.
    edit : callable
        Maps each dummy declaration to its new declaration.
    after_implicit : sequence of str
        Lines inserted right after ``implicit none`` (indentation is
        added).

    Returns
    -------
    str
        The kernel text.
    """
    lines = _wrap_items(f"{iface.indent}subroutine {name}(", list(args), ")")
    by_line: dict[int, list[Dummy]] = {}
    for key, index in iface.decl_lines.items():
        by_line.setdefault(index, []).append(iface.dummies[key])
    old_end = re.compile(rf"^(\s*end\s+subroutine\s+){iface.name}\b", re.I)
    body_indent = ""
    for index in range(iface.header_end, len(iface.lines)):
        line = iface.lines[index]
        if index in by_line:
            indent = _indent_of(line)
            lines.append(
                "\n".join(
                    indent + edit(dummy).render() for dummy in by_line[index]
                )
            )
            continue
        lines.append(old_end.sub(rf"\g<1>{name}", line))
        if _IMPLICIT.match(line) and not body_indent:
            body_indent = _indent_of(line)
            lines.extend(body_indent + text for text in after_implicit)
    return "\n".join(lines)


def _drop_contiguous(dummy: Dummy) -> Dummy:
    """Return ``dummy`` without the ``contiguous`` attribute."""
    attrs = [attr for attr in dummy.attrs if attr.lower() != "contiguous"]
    return dataclasses.replace(dummy, attrs=attrs)


def _drop_contiguous_if_optional(dummy: Dummy) -> Dummy:
    """Keep ``contiguous`` on the never-absent dummies only."""
    return _drop_contiguous(dummy) if dummy.is_optional else dummy


def relax_optional_contiguity(source: str) -> str:
    """Drop ``contiguous`` from every ``optional`` declaration of a kernel.

    The fallback receives the dispatcher's optional arrays, which are not
    ``contiguous``; forwarding an absent one to a ``contiguous`` dummy
    makes gfortran try to pack it and crash.  Every procedure of the text
    is rewritten (with ``work_array`` the internal body has its own
    declarations).

    Parameters
    ----------
    source : str
        Kernel text.

    Returns
    -------
    str
        The text without ``contiguous`` on the optional dummies.
    """
    lines = []
    for line in source.splitlines():
        if _DECL.match(line) and re.search(r"\boptional\b", line):
            line = re.sub(r",\s*contiguous\b", "", line)
        lines.append(line)
    return "\n".join(lines)


@dataclasses.dataclass(frozen=True)
class _ExplicitShape:
    """Editor that turns assumed-shape dummies into explicit-shape ones.

    Outputs always have extents ``nc``.  Inputs take theirs from
    ``extents``; without an entry they are ``nc`` too, unless the kernel
    has extra integer dimensions (then the shape is unknown and the dummy
    stays assumed shape, which is correct but slower).  With
    ``rename_nc`` the dummy ``nc`` is renamed to ``nc_in``.
    """

    iface: KernelInterface
    extents: Mapping[str, tuple[str, ...]]
    rename_nc: bool = False

    def __call__(self, dummy: Dummy) -> Dummy:
        """Return the new declaration of ``dummy``."""
        dummy = _drop_contiguous(dummy)
        if self.rename_nc and dummy.name == NC_NAME:
            return dataclasses.replace(dummy, name=NC_DUMMY_NAME)
        if dummy.rank == 0 or not self.iface.has_nc:
            return dummy
        dims = self._dims(dummy)
        if dims is None:
            return dummy
        if len(dims) != dummy.rank:
            raise ValueError(
                f"{self.iface.name}: extents of {dummy.name} have the "
                "wrong rank"
            )
        return dataclasses.replace(dummy, dims=list(dims))

    def _dims(self, dummy: Dummy) -> Sequence[str] | None:
        if dummy.is_output:
            return [NC_NAME] * dummy.rank
        if dummy.name.lower() in self.extents:
            return self.extents[dummy.name.lower()]
        if self.iface.has_extra_dimensions:
            return None
        return [NC_NAME] * dummy.rank


def specialise_kernel(
    iface: KernelInterface,
    *,
    name: str,
    extents: Mapping[str, Sequence[str]],
    fixed_nc: int | None = None,
) -> str:
    """Make a fixed-output kernel callable from the dispatcher.

    Array dummies become explicit shape, ``contiguous`` is dropped (it is
    illegal on explicit-shape dummies) and the no-inline directive is
    added.  With ``fixed_nc`` the dummy ``nc`` becomes the named constant
    and the signature keeps a dummy ``nc_in`` in its place.

    Parameters
    ----------
    iface : KernelInterface
        Parsed fixed-output kernel.
    name : str
        Name of the new subroutine.
    extents : mapping
        Extents of the input arrays (see :func:`workload_procedures`).
    fixed_nc : int, optional
        Compile-time number of components.

    Returns
    -------
    str
        The kernel text.
    """
    lowered = {key.lower(): tuple(value) for key, value in extents.items()}
    fixed = fixed_nc is not None and iface.has_nc
    extra = [NOINLINE_DIRECTIVE.format(name=name)]
    args = list(iface.args)
    if fixed:
        extra.append(f"integer, parameter :: {NC_NAME} = {fixed_nc}")
        args = [NC_DUMMY_NAME if arg == NC_NAME else arg for arg in args]
    return render_kernel(
        iface,
        name=name,
        args=args,
        edit=_ExplicitShape(iface, lowered, rename_nc=fixed),
        after_implicit=extra,
    )


@dataclasses.dataclass
class _Specialised:
    """A fixed-output kernel and the data of its dispatcher branch."""

    name: str
    iface: KernelInterface
    mask: int
    copies: dict[int, str]


def _call_lines(
    callee: str,
    iface: KernelInterface,
    indent: str,
    fixed_nc: bool = False,
) -> list[str]:
    """Keyword call of ``callee`` forwarding the dispatcher's dummies."""
    actuals = []
    for arg in iface.args:
        keyword = NC_DUMMY_NAME if fixed_nc and arg == NC_NAME else arg
        actuals.append(f"{keyword}={arg}")
    return _wrap_items(f"{indent}call {callee}(", actuals, ")")


def _dispatch_branch(
    kernel: _Specialised, fixed_nc: Sequence[int]
) -> list[str]:
    """Return the ``case`` of the dispatcher that handles ``kernel``."""
    outputs = ", ".join(kernel.iface.outputs)
    lines = [f"   case ({kernel.mask})  ! {outputs}"]
    inner = "      "
    if not (kernel.copies and kernel.iface.has_nc):
        return lines + _call_lines(kernel.name, kernel.iface, inner)
    for position, size in enumerate(fixed_nc):
        keyword = "if" if position == 0 else "else if"
        lines.append(f"{inner}{keyword} ({NC_NAME} == {size}) then")
        lines.extend(
            _call_lines(kernel.copies[size], kernel.iface, inner + "   ", True)
        )
    lines.append(f"{inner}else")
    lines.extend(_call_lines(kernel.name, kernel.iface, inner + "   "))
    lines.append(f"{inner}end if")
    return lines


def output_bits(fallback: KernelInterface) -> dict[str, int]:
    """Return the mask bit of every optional output of ``fallback``.

    Parameters
    ----------
    fallback : KernelInterface
        The ``optional_derivatives=True`` kernel.

    Returns
    -------
    dict
        Lower-case output name to its power of two, in argument order.

    Raises
    ------
    ValueError
        If there are more optional outputs than mask bits.
    """
    optional = [
        arg.lower()
        for arg in fallback.args
        if fallback.dummies[arg.lower()].is_optional
    ]
    if len(optional) > MAX_MASK_OUTPUTS:
        raise ValueError(f"too many optional outputs ({len(optional)})")
    return {arg: 1 << position for position, arg in enumerate(optional)}


def _dispatcher_head(name: str, fallback: KernelInterface) -> list[str]:
    """Header, specification part and mask computation of the dispatcher."""
    lines = _wrap_items(f"subroutine {name}(", fallback.args, ")")
    lines += [f"   {use}" for use in fallback.uses]
    lines.append("   implicit none")
    for arg in fallback.args:
        dummy = _drop_contiguous_if_optional(fallback.dummies[arg.lower()])
        lines.append(f"   {dummy.render()}")
    lines += [f"   integer :: {MASK_NAME}", "", f"   {MASK_NAME} = 0"]
    bits = output_bits(fallback)
    for arg in fallback.args:
        bit = bits.get(arg.lower())
        if bit:
            lines.append(
                f"   if (present({arg})) {MASK_NAME} = {MASK_NAME} + {bit}"
            )
    return lines


def build_dispatcher(
    name: str,
    fallback: KernelInterface,
    fallback_name: str,
    kernels: Sequence[_Specialised],
    fixed_nc: Sequence[int],
) -> str:
    """Build the dispatcher subroutine.

    Parameters
    ----------
    name : str
        Public name of the dispatcher.
    fallback : KernelInterface
        The ``optional_derivatives=True`` kernel; its signature is the
        public signature.
    fallback_name : str
        Name under which the fallback is emitted.
    kernels : sequence
        Fixed-output kernels.
    fixed_nc : sequence of int
        Fixed component counts of the copies.

    Returns
    -------
    str
        The dispatcher text.
    """
    lines = _dispatcher_head(name, fallback)
    lines += ["", f"   select case ({MASK_NAME})"]
    for kernel in kernels:
        lines.extend(_dispatch_branch(kernel, fixed_nc))
    lines.append("   case default")
    lines.extend(_call_lines(fallback_name, fallback, "      "))
    lines += ["   end select", f"end subroutine {name}"]
    return "\n".join(lines)


def _normalise_fixed_nc(fixed_nc: Iterable[int] | None) -> list[int]:
    """Return the fixed component counts as a list of unique integers."""
    sizes = [] if fixed_nc is None else [int(size) for size in fixed_nc]
    if any(size < 1 for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError(
            f"fixed_nc must hold distinct positive integers, got {sizes}"
        )
    return sizes


def _resolve_pair(
    preset: Preset | str | None,
    options: CodegenOptions | None,
    overrides: Mapping[str, Any],
) -> tuple[CodegenOptions, CodegenOptions]:
    """Return the options of the fixed-output kernels and the fallback."""
    for forbidden in ("optional_derivatives",):
        if overrides.get(forbidden) is not None:
            raise ValueError(
                f"{forbidden} is decided by the workload mode; do not set it"
            )
    given = {k: v for k, v in overrides.items() if k != "optional_derivatives"}
    fixed = resolve_codegen_options(
        preset, options, optional_derivatives=False, **given
    )
    with warnings.catch_warnings():
        # Options that are inert under optional outputs are expected here.
        warnings.simplefilter("ignore", UserWarning)
        fallback = resolve_codegen_options(
            preset, options, optional_derivatives=True, **given
        )
    return fixed, fallback


def _check_against_fallback(
    kernel: KernelInterface,
    fallback: KernelInterface,
    bits: Mapping[str, int],
) -> int:
    """Return the mask of ``kernel`` after checking it fits the fallback."""
    unknown = [
        arg for arg in kernel.args if arg.lower() not in fallback.dummies
    ]
    if unknown:
        raise ValueError(
            f"{kernel.name} needs {unknown}, which the fallback kernel "
            "does not have; request every derivative of the workloads "
            "through 'derivatives'"
        )
    missing = [out for out in kernel.outputs if out.lower() not in bits]
    if missing:
        raise ValueError(
            f"{kernel.name} computes {missing}, which are not optional "
            "outputs of the fallback kernel"
        )
    return sum(bits[out.lower()] for out in kernel.outputs)


@dataclasses.dataclass(frozen=True)
class _Plan:
    """Everything the kernel generation of one call shares."""

    diff: Any
    base: str
    fixed_options: CodegenOptions
    passed: Mapping[str, Any]
    extents: Mapping[str, Sequence[str]]
    sizes: Sequence[int]


def _fixed_kernel(
    plan: _Plan,
    workload: Any,
    position: int,
    fallback: KernelInterface,
    taken: Iterable[int],
) -> tuple[_Specialised, list[str]]:
    """Generate one fixed-output kernel, its copies and dispatcher data."""
    kernel_name = f"{plan.base}_w{position}"
    text = to_fortran(
        plan.diff,
        derivatives=workload,
        subroutine_name=kernel_name,
        options=plan.fixed_options,
        **plan.passed,
    )
    iface = parse_kernel(text)
    mask = _check_against_fallback(iface, fallback, output_bits(fallback))
    if mask in taken:
        raise ValueError(
            f"workload {position} has the same output set as an earlier one"
        )
    copies = {size: f"{kernel_name}_nc{size}" for size in plan.sizes}
    texts = [specialise_kernel(iface, name=kernel_name, extents=plan.extents)]
    if iface.has_nc:
        texts += [
            specialise_kernel(
                iface, name=copies[size], extents=plan.extents, fixed_nc=size
            )
            for size in plan.sizes
        ]
    return _Specialised(kernel_name, iface, mask, copies), texts


def workload_procedures(
    diff: Any,
    workloads: Sequence[Any],
    *,
    derivatives: Any = None,
    subroutine_name: str | None = None,
    fixed_nc: Iterable[int] | None = None,
    extents: Mapping[str, Sequence[str]] | None = None,
    preset: Preset | str | None = None,
    options: CodegenOptions | None = None,
    **kwargs: Any,
) -> str:
    """Generate the dispatcher, the fallback and the fixed-output kernels.

    Parameters
    ----------
    diff : DiffPlz
        The differentiated expression.
    workloads : sequence
        One derivative selection per fixed-output kernel, in the format of
        ``to_fortran(derivatives=...)`` (for example ``"dn,dndn"`` or
        ``["dT"]``).  The dispatcher matches the *output set* of each
        generated kernel (the function value included when ``to_fortran``
        emits it), exactly.
    derivatives : str or list of str, optional
        Derivatives of the fallback kernel and so of the public
        signature; it must contain every workload.  Default: every
        derivative of ``diff``.
    subroutine_name : str, optional
        Public name of the dispatcher.  The kernels are named
        ``<name>_fallback``, ``<name>_w<k>`` (``k`` from 1, in the order of
        ``workloads``) and ``<name>_w<k>_nc<size>``.  Default: the name
        of ``diff``.
    fixed_nc : iterable of int, optional
        Component counts for which every fixed-output kernel is also
        emitted with ``nc`` as a named constant; the dispatcher selects
        by ``nc`` and then by output set.
    extents : mapping, optional
        Extents of the array inputs by name (for example
        ``{"a": ("nc", "ng")}``).  Inputs without an entry are ``nc`` in
        every dimension, unless the kernel has extra integer dimensions;
        then they stay assumed shape.
    preset, options, **kwargs
        Code-generation options as in ``to_fortran``.  Other keywords
        (``integer_args``, ``module_symbols``, ``symmetric_params``,
        ``stats``, ``name``) are forwarded to every ``to_fortran`` call and
        every option name is accepted as an override.
        ``optional_derivatives`` is set by this function.

    Returns
    -------
    str
        The subroutines, without a module.

    Raises
    ------
    ValueError
        If two workloads have the same outputs, a workload is not
        covered by ``derivatives`` or ``fixed_nc`` is invalid.
    """
    to_fortran_keys = (
        "name",
        "integer_args",
        "module_symbols",
        "symmetric_params",
        "stats",
    )
    passed = {k: kwargs.pop(k) for k in to_fortran_keys if k in kwargs}
    fixed_options, fallback_options = _resolve_pair(preset, options, kwargs)
    base = subroutine_name or getattr(diff, "name", None) or "f"
    plan = _Plan(
        diff,
        base,
        fixed_options,
        passed,
        dict(extents or {}),
        _normalise_fixed_nc(fixed_nc),
    )
    fallback_name = f"{base}_fallback"
    fallback_text = to_fortran(
        diff,
        derivatives=derivatives,
        subroutine_name=fallback_name,
        options=fallback_options,
        **passed,
    )
    fallback = parse_kernel(fallback_text)
    fallback_text = relax_optional_contiguity(fallback_text)
    kernels: list[_Specialised] = []
    texts: list[str] = []
    for position, workload in enumerate(workloads, start=1):
        kernel, kernel_texts = _fixed_kernel(
            plan, workload, position, fallback, [k.mask for k in kernels]
        )
        kernels.append(kernel)
        texts += kernel_texts
    dispatcher = build_dispatcher(
        base, fallback, fallback_name, kernels, plan.sizes
    )
    return "\n\n".join([dispatcher, fallback_text, *texts]) + "\n"


def wrap_module(
    module_name: str,
    uses: Iterable[str],
    code: str,
    comments: Iterable[str] = (),
) -> str:
    """Put generated procedures in a module.

    Parameters
    ----------
    module_name : str
        Name of the module.
    uses : iterable of str
        Text after ``use`` for each statement of the module.
    code : str
        Procedures, indented by the caller's convention (0 here).
    comments : iterable of str
        Header comment lines (with their ``!``).

    Returns
    -------
    str
        The module source, indented by three spaces.
    """
    lines = [*comments, f"module {module_name}"]
    lines += [f"   use {use_text}" for use_text in uses]
    lines.extend(["   implicit none", "contains", ""])
    for line in code.splitlines():
        lines.append("   " + line if line.strip() else "")
    lines.extend([f"end module {module_name}", ""])
    return "\n".join(lines)


def workload_module(
    diff: Any,
    workloads: Sequence[Any],
    *,
    module_name: str | None = None,
    uses: Iterable[str] = (),
    **kwargs: Any,
) -> str:
    """Generate a Fortran module with a workload dispatcher.

    Parameters
    ----------
    diff : DiffPlz
        The differentiated expression.
    workloads : sequence
        Derivative selections, one fixed-output kernel each.
    module_name : str, optional
        Name of the module; default ``<subroutine name>_mod``.
    uses : iterable of str, optional
        Text after ``use`` for each module-level statement (for example
        ``"mykinds, only: dp"``).
    **kwargs
        See :func:`workload_procedures`.

    Returns
    -------
    str
        The complete module.  Its public procedure has the signature of
        the ``optional_derivatives=True`` kernel, so it is a drop-in
        replacement of that kernel.
    """
    code = workload_procedures(diff, workloads, **kwargs)
    base = kwargs.get("subroutine_name") or getattr(diff, "name", None) or "f"
    return wrap_module(
        module_name or f"{base}_mod",
        uses,
        code,
        [
            "! Generated by thermodiff (workload mode).",
            "! Do not edit by hand.",
        ],
    )
