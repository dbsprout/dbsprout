"""DBSprout — realistic database seed data from your schema."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("dbsprout")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
