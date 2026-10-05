"""Compatibility entry point for zils.models."""

import sys

if __name__ == "__main__":
    import runpy
    from pathlib import Path

    # Direct-file runners must not shadow the upstream jevk5 package.
    directory = Path(__file__).resolve().parent
    sys.path = [str(directory.parent), *[p for p in sys.path if Path(p).resolve() != directory]]
    runpy.run_module("zils.models", run_name="__main__")
else:
    import importlib

    sys.modules[__name__] = importlib.import_module("zils.models")
