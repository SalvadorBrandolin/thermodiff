"""Consistency of the project history knowledge base (``docs/dev/history``).

Every entry must have valid front matter, a dated file name that matches its
``date`` field, and a row in the index (``README.md``) with the same type and
status. Every link in the index must point to an existing entry.
"""

import re
from pathlib import Path

import pytest

HISTORY_DIR = (
    Path(__file__).resolve().parent.parent / "docs" / "dev" / "history"
)
INDEX_FILE = HISTORY_DIR / "README.md"
TEMPLATE_FILE = HISTORY_DIR / "_template.md"

ENTRY_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2})-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")
FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
INDEX_ROW = re.compile(
    r"^\| (?P<date>[^|]+) \| \[[^\]]+\]\((?P<file>[^)]+)\) "
    r"\| (?P<type>[^|]+) \| (?P<status>[^|]+) \|",
    re.MULTILINE,
)
MARKDOWN_LINK = re.compile(r"\]\(([^)#]+\.md)\)")

REQUIRED_KEYS = ("title", "date", "type", "status", "area", "links")
ENTRY_TYPES = {
    "improvement",
    "bug",
    "measurement",
    "decision",
    "dead-end",
    "gotcha",
}
PLAIN_STATUSES = {"open", "resolved", "wontfix"}
SUPERSEDED_PREFIX = "superseded-by: "


def entry_files():
    """Return the entry files of the history directory, sorted by name."""
    return sorted(
        path
        for path in HISTORY_DIR.glob("*.md")
        if path not in (INDEX_FILE, TEMPLATE_FILE)
    )


def read_front_matter(path):
    """Return the front matter of an entry as a ``key -> value`` dict."""
    match = FRONT_MATTER.match(path.read_text(encoding="utf-8"))
    assert match, f"{path.name}: no front matter between '---' lines"
    fields = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(": ")
        assert separator, f"{path.name}: bad front matter line {line!r}"
        fields[key] = value.strip()
    return fields


def index_rows():
    """Return the rows of the index table, keyed by entry file name."""
    text = INDEX_FILE.read_text(encoding="utf-8")
    return {
        row["file"]: {
            key: row[key].strip() for key in ("date", "type", "status")
        }
        for row in INDEX_ROW.finditer(text)
    }


def test_history_has_entries():
    """The index, the template and at least one entry exist."""
    assert INDEX_FILE.is_file()
    assert TEMPLATE_FILE.is_file()
    assert entry_files()


@pytest.mark.parametrize("path", entry_files(), ids=lambda path: path.name)
def test_entry_front_matter(path):
    """An entry is dated, typed and has a known status."""
    name_match = ENTRY_NAME.match(path.name)
    assert name_match, f"{path.name}: name must be YYYY-MM-DD-slug.md"
    fields = read_front_matter(path)
    missing = [key for key in REQUIRED_KEYS if not fields.get(key)]
    assert not missing, f"{path.name}: missing {missing}"
    assert fields["date"] == name_match.group(1)
    assert fields["type"] in ENTRY_TYPES
    status = fields["status"]
    if status.startswith(SUPERSEDED_PREFIX):
        successor = status[len(SUPERSEDED_PREFIX) :]
        assert (HISTORY_DIR / successor).is_file(), status
    else:
        assert status in PLAIN_STATUSES
    for key in ("area", "links"):
        assert fields[key].startswith("[") and fields[key].endswith("]")


@pytest.mark.parametrize("path", entry_files(), ids=lambda path: path.name)
def test_entry_is_indexed(path):
    """Every entry has an index row with its date, type and status."""
    rows = index_rows()
    assert path.name in rows, f"{path.name} is not in README.md"
    fields = read_front_matter(path)
    expected = {key: fields[key] for key in ("date", "type", "status")}
    assert rows[path.name] == expected


def test_index_links_exist():
    """Every Markdown link of the index and the entries resolves."""
    for path in [INDEX_FILE, *entry_files()]:
        text = path.read_text(encoding="utf-8")
        for target in MARKDOWN_LINK.findall(text):
            if "://" in target:
                continue
            resolved = (path.parent / target).resolve()
            assert resolved.is_file(), f"{path.name}: broken link {target}"
