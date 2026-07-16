from importlib.metadata import version, PackageNotFoundError

from run3_mj_mixer.library import (
    Hemisphere, HemisphereLibrary, directed_phi, query_direction,
)

try:
    __version__ = version("run3-mj-mixer")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["Hemisphere", "HemisphereLibrary", "directed_phi",
           "query_direction"]
