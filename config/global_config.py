"""Global defaults shared by all standalone scripts.

These values are intentionally conservative and can be overridden via CLI flags
in each script.
"""

DEFAULT_CHUNK_SIZE_MB = 25
DEFAULT_MAX_OUTPUT_ROWS = 500_000
DEFAULT_ENCODING = "utf-8"
DEFAULT_OUTPUT_DIR = "outputs"
