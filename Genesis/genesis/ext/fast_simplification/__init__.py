from ._version import __version__  # noqa: F401

# Lazy import to avoid circular import
def _get_replay_module():
    from . import _replay
    return _replay

# Import functions lazily
def _lazy_import_replay():
    from . import replay
    return replay._map_isolated_points, replay.replay_simplification

# Store lazy imports
_map_isolated_points, replay_simplification = _lazy_import_replay()
from .simplify import simplify, simplify_mesh  # noqa: F401
