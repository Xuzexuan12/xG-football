"""Frontiers figure-generation entry point.

The plotting implementation is shared with the existing submission packaging
script so the Frontiers manifest can point to a journal-neutral file name.
"""

from jqas_figure_package import main


if __name__ == "__main__":
    main()
