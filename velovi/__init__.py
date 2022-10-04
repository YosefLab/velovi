"""velovi."""

import logging

from rich.console import Console
from rich.logging import RichHandler

from ._constants import REGISTRY_KEYS
from ._model import VELOVI, VELOVAE
from ._utils import get_permutation_scores, preprocess_data

# https://github.com/python-poetry/poetry/pull/2366#issuecomment-652418094
# https://github.com/python-poetry/poetry/issues/144#issuecomment-623927302
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "velovi"
__version__ = importlib_metadata.version(package_name)

logger = logging.getLogger(__name__)
# set the logging level
logger.setLevel(logging.INFO)

# nice logging outputs
console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("velovi: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# this prevents double outputs
logger.propagate = False

__all__ = [
    "VELOVI",
    "VELOVAE",
    "REGISTRY_KEYS",
    "get_permutation_scores",
    "preprocess_data",
]
