"""Parser for the ``.model`` language.

The ``.model`` file is a small, line oriented text format.  It is divided
into sections introduced by the keywords ``model``, ``target``, ``use``,
``variables``, ``parameters``, ``expressions`` and ``function``.  A short
example looks like::

    model pr
    variables
        n   nc
        V
        T
    function
        R*T/V

This module turns the text into raw Python data structures (declarations
and expression strings).  The symbolic meaning is given later by
:mod:`thermodiff.model.ir`.
"""

import re
from dataclasses import dataclass, field

import sympy as sp

from thermodiff.model.errors import ModelError

__all__ = [
    "RawDecl",
    "RawDef",
    "RawModel",
    "read_sections",
    "ExpressionParser",
    "tokenize",
]


#: Keywords that start a section.
SECTION_KEYWORDS = (
    "model",
    "target",
    "use",
    "variables",
    "parameters",
    "expressions",
    "function",
)

#: Keywords that open a block of content.
_BLOCK_KEYWORDS = {"variables", "parameters", "expressions", "function"}

#: Built-in functions usable inside expressions.
FUNCTIONS = {
    "sqrt": sp.sqrt,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "abs": sp.Abs,
    "sign": sp.sign,
    "min": sp.Min,
    "max": sp.Max,
    "erf": sp.erf,
    "erfc": sp.erfc,
}

#: Built-in constants.
CONSTANTS = {
    "pi": sp.pi,
}

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_RE = re.compile(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][+-]?\d+)?")
_OPERATORS = set("^*/+-()[]:,")


@dataclass
class Token:
    """A single lexical token of an expression."""

    kind: str
    value: str
    pos: int


@dataclass
class RawDecl:
    """A raw declaration line, e.g. ``k  nc nc``."""

    name: str
    dims: tuple
    line: int


@dataclass
class RawDef:
    """A raw ``name = expression`` definition."""

    lhs: str
    rhs: str
    line: int


@dataclass
class RawModel:
    """The raw contents of a ``.model`` file."""

    model: str | None = None
    target: str | None = None
    uses: list = field(default_factory=list)
    variables: list = field(default_factory=list)
    parameters: list = field(default_factory=list)
    expressions: list = field(default_factory=list)
    function: RawDef | None = None
    filename: str | None = None


def _strip_comment(line):
    """Remove a ``#`` comment from ``line``."""
    marker = line.find("#")
    if marker >= 0:
        return line[:marker]
    return line


def _balanced(text):
    """Return whether all ``()`` and ``[]`` in ``text`` are balanced."""
    depth = 0
    for char in text:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def read_sections(text, filename=None):
    """Split the contents of a ``.model`` file into raw sections.

    Parameters
    ----------
    text : str
        Full contents of the ``.model`` file.
    filename : str, optional
        File name used to build error messages.

    Returns
    -------
    RawModel
        The raw, not yet interpreted, contents of the file.

    Raises
    ------
    ModelError
        If the file structure is invalid.
    """
    model = RawModel(filename=filename)
    section = None
    # ``function`` and the other blocks may span several physical lines.
    pending = []
    pending_line = None

    def flush():
        nonlocal pending, pending_line
        if pending:
            logical = " ".join(pending).strip()
            pending = []
            if logical:
                _consume(section, logical, pending_line)
            pending_line = None

    def _consume(current, logical, line_no):
        if current == "expressions":
            if "=" not in logical:
                raise ModelError(
                    "expression must have the form 'name = expr'",
                    line_no,
                    filename,
                )
            lhs, _, rhs = logical.partition("=")
            model.expressions.append(RawDef(lhs.strip(), rhs.strip(), line_no))
        elif current == "function":
            if "=" in logical:
                _, _, rhs = logical.partition("=")
                logical = rhs
            model.function = RawDef("f", logical.strip(), line_no)
        elif current in ("variables", "parameters"):
            decl = _parse_decl(logical, line_no, filename)
            getattr(model, current).append(decl)

    lines = text.splitlines()
    for number, raw in enumerate(lines, start=1):
        line = _strip_comment(raw).strip()
        if not line:
            continue
        first = line.split()[0]
        if first in SECTION_KEYWORDS and (
            first in _BLOCK_KEYWORDS or first in ("model", "target", "use")
        ):
            # A section header is a line that only contains the keyword,
            # except for 'model', 'target' and 'use' which take a value.
            keyword_length = len(first)
            rest = line[keyword_length:].strip()
            if first in _BLOCK_KEYWORDS:
                if rest:
                    # e.g. 'function R*T' on a single line.
                    flush()
                    section = first
                    _consume(first, rest, number)
                    section = None
                    continue
                flush()
                section = first
                continue
            if first == "model":
                flush()
                if not rest:
                    raise ModelError(
                        "'model' requires a name", number, filename
                    )
                model.model = rest.split()[0]
                section = None
                continue
            if first == "target":
                flush()
                if not rest:
                    raise ModelError(
                        "'target' requires a value", number, filename
                    )
                model.target = rest.split()[0]
                section = None
                continue
            if first == "use":
                flush()
                if not rest:
                    raise ModelError(
                        "'use' requires a module", number, filename
                    )
                model.uses.append((rest, number))
                section = None
                continue

        if section is None:
            raise ModelError(
                f"content outside of any section: {line!r}", number, filename
            )

        if section == "function":
            if pending_line is None:
                pending_line = number
            pending.append(line)
            if _balanced(line) and not line.endswith("\\"):
                flush()
            continue

        # variables / parameters / expressions
        if pending_line is None:
            pending_line = number
        pending.append(line)
        if _balanced(line) and not line.endswith("\\"):
            flush()

    flush()
    if model.function is None:
        raise ModelError("the 'function' section is mandatory", None, filename)
    return model


def _parse_decl(logical, line_no, filename):
    """Parse a single ``name [dim ...]`` declaration line."""
    cleaned = logical.replace(":", " ").replace(",", " ")
    parts = cleaned.split()
    if not parts:
        raise ModelError("empty declaration", line_no, filename)
    name = parts[0]
    if not _IDENT_RE.fullmatch(name):
        raise ModelError(
            f"invalid declaration name {name!r}", line_no, filename
        )
    dims = []
    for token in parts[1:]:
        if not _IDENT_RE.fullmatch(token):
            raise ModelError(
                f"invalid dimension {token!r} in declaration of {name!r}",
                line_no,
                filename,
            )
        dims.append(token)
    return RawDecl(name, tuple(dims), line_no)


def tokenize(text, line=None, filename=None):
    """Split an expression into tokens.

    Parameters
    ----------
    text : str
        Expression text.
    line : int, optional
        Line number used for error messages.
    filename : str, optional
        File name used for error messages.

    Returns
    -------
    list of Token
        The tokens of the expression, terminated by an ``eof`` token.
    """
    tokens = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if text.startswith("**", index):
            tokens.append(Token("op", "**", index))
            index += 2
            continue
        if char in _OPERATORS:
            tokens.append(Token("op", char, index))
            index += 1
            continue
        match = _NUMBER_RE.match(text, index)
        if match:
            tokens.append(Token("num", match.group(), index))
            index = match.end()
            continue
        match = _IDENT_RE.match(text, index)
        if match:
            tokens.append(Token("ident", match.group(), index))
            index = match.end()
            continue
        raise ModelError(
            f"unexpected character {char!r} in expression",
            line,
            filename,
        )
    tokens.append(Token("eof", "", length))
    return tokens


class ExpressionParser:
    """Recursive descent parser turning expression text into SymPy.

    Parameters
    ----------
    text : str
        Expression text.
    context : object
        Object providing name resolution.  It must expose ``resolve(name)``,
        ``index(letter)`` and the ``default_upper`` attribute.
    line : int, optional
        Line number used for error messages.
    filename : str, optional
        File name used for error messages.
    """

    def __init__(self, text, context, line=None, filename=None):
        self.text = text
        self.context = context
        self.line = line
        self.filename = filename
        self.tokens = tokenize(text, line, filename)
        self.pos = 0
        self._sum_stack = []

    # -- token helpers --------------------------------------------------
    def _peek(self, offset=0):
        return self.tokens[self.pos + offset]

    def _advance(self):
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _at_op(self, value):
        token = self._peek()
        return token.kind == "op" and token.value == value

    def _expect_op(self, value):
        if not self._at_op(value):
            self._fail(f"expected {value!r}")
        return self._advance()

    def _fail(self, message):
        token = self._peek()
        found = (
            "end of expression" if token.kind == "eof" else repr(token.value)
        )
        raise ModelError(f"{message}, found {found}", self.line, self.filename)

    # -- grammar --------------------------------------------------------
    def parse(self):
        """Parse the expression and return a SymPy object."""
        if self._peek().kind == "eof":
            raise ModelError("empty expression", self.line, self.filename)
        node = self._parse_expr()
        if self._peek().kind != "eof":
            self._fail("unexpected token")
        return node

    def _parse_expr(self):
        node = self._parse_term()
        while self._at_op("+") or self._at_op("-"):
            op = self._advance().value
            right = self._parse_term()
            node = node + right if op == "+" else node - right
        return node

    def _parse_term(self):
        node = self._parse_factor()
        while True:
            if self._at_op("*") or self._at_op("/"):
                op = self._advance().value
                right = self._parse_factor()
                node = node * right if op == "*" else node / right
            elif self._starts_implicit_product():
                right = self._parse_factor()
                node = node * right
            else:
                return node

    def _starts_implicit_product(self):
        token = self._peek()
        if token.kind in ("num", "ident"):
            return True
        return token.kind == "op" and token.value == "("

    def _parse_factor(self):
        token = self._peek()
        if token.kind == "op" and token.value in ("+", "-"):
            op = self._advance().value
            operand = self._parse_factor()
            return operand if op == "+" else -operand
        return self._parse_power()

    def _parse_power(self):
        base = self._parse_atom()
        if self._at_op("^") or self._at_op("**"):
            self._advance()
            exponent = self._parse_factor()
            return base**exponent
        return base

    def _parse_atom(self):
        token = self._advance()
        if token.kind == "num":
            return _to_number(token.value)
        if token.kind == "op" and token.value == "(":
            node = self._parse_expr()
            self._expect_op(")")
            return node
        if token.kind == "ident":
            name = token.value
            if name.startswith("sum_"):
                return self._parse_sum(name)
            if name in FUNCTIONS and self._at_op("("):
                return self._parse_call(name)
            if name in CONSTANTS:
                return CONSTANTS[name]
            return self._resolve(name, token)
        self._fail("unexpected token")

    def _resolve(self, name, token):
        try:
            return self.context.resolve(name)
        except ModelError as error:
            raise ModelError(error.message, self.line, self.filename)

    def _parse_call(self, name):
        self._expect_op("(")
        args = [self._parse_expr()]
        while self._at_op(","):
            self._advance()
            args.append(self._parse_expr())
        self._expect_op(")")
        return FUNCTIONS[name](*args)

    def _parse_sum(self, name):
        letter = name[4:]
        if len(letter) != 1 or not letter.isalpha():
            raise ModelError(
                f"invalid summation index in {name!r}; "
                "use a single letter, e.g. 'sum_i'",
                self.line,
                self.filename,
            )
        if letter in self._sum_stack:
            raise ModelError(
                f"summation index {letter!r} is reused in nested sums",
                self.line,
                self.filename,
            )
        low = sp.Integer(1)
        high = self.context.default_upper
        if self._at_op("["):
            self._advance()
            if self._at_op(":"):
                low = sp.Integer(1)
            else:
                low = self._parse_expr()
            self._expect_op(":")
            if self._at_op("]"):
                high = self.context.default_upper
            else:
                high = self._parse_expr()
            self._expect_op("]")
        index = self.context.index(letter)
        self._sum_stack.append(letter)
        body = self._parse_term()
        self._sum_stack.pop()
        return sp.Sum(body, (index, low, high))


def _to_number(token):
    """Convert a numeric token to a SymPy number."""
    text = token.replace("d", "e").replace("D", "e")
    if "." in text or "e" in text or "E" in text:
        return sp.Float(text)
    return sp.Integer(text)
