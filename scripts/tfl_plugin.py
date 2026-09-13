#!/usr/bin/env python3
"""Run the bundled TFL CLI directly from an unpacked plugin checkout."""

from __future__ import annotations

import pathlib
import sys


PLUGIN_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLUGIN_ROOT))

from tfl.cli import main  # noqa: E402  (path is initialized above)


if __name__ == "__main__":
    raise SystemExit(main())
