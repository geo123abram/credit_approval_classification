# credit_approval_model/__init__.py

import pathlib

# … any existing imports …

# Read the version number from the VERSION file in the project root
try:
    # __file__ is .../credit_approval_model/__init__.py
    project_root = pathlib.Path(__file__).resolve().parent
    __version__ = (project_root / "VERSION").read_text().strip()
except Exception:
    __version__ = "0.0.0"  # fallback or raise if you prefer
