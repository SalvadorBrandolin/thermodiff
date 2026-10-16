"""Allow ``python -m thermodiff.model``."""

import sys

from thermodiff.model.cli import main

if __name__ == "__main__":
    sys.exit(main())
