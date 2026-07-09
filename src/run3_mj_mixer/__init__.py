from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("run3-mj-mixer")
except PackageNotFoundError:
    __version__ = "unknown"
