"""Exceptions raised while reading ``.model`` files.

All the errors produced by the front-end are :class:`ModelError` instances
that carry the offending file name and line number, if known.  They are
meant to be readable by non-programmers, since the ``.model`` file is the
main user interface of this feature.
"""

__all__ = ["ModelError"]


class ModelError(Exception):
    """Error found in a ``.model`` file.

    Parameters
    ----------
    message : str
        Human readable description of the problem.
    line : int, optional
        One based line number where the problem was found.
    filename : str, optional
        Name of the file being processed.

    Attributes
    ----------
    message : str
        The bare error message, without the location prefix.
    line : int or None
        Line number associated with the error.
    filename : str or None
        File name associated with the error.
    """

    def __init__(self, message, line=None, filename=None):
        self.message = message
        self.line = line
        self.filename = filename
        super().__init__(self._format())

    def _format(self):
        location = ""
        if self.filename:
            location += str(self.filename)
        if self.line is not None:
            location += f":{self.line}"
        if location:
            return f"{location}: {self.message}"
        return self.message
