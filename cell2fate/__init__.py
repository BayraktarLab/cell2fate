"""cell2fate"""

import logging

from rich.console import Console
from rich.logging import RichHandler
from pyro.distributions import constraints
from pyro.distributions.transforms import SoftplusTransform
from torch.distributions import biject_to, transform_to

from ._cell2fate_DynamicalModel import \
Cell2fate_DynamicalModel

from ._cell2fate_DynamicalModel_ModuleSpecificTime_NonGaussianTimePrior import \
Cell2fate_DynamicalModel_ModuleSpecificTime_NonGaussianTimePrior

from ._cell2fate_DynamicalModel_PriorKnowledge import \
Cell2fate_DynamicalModel_PriorKnowledge

from ._cell2fate_DynamicalModel_ModuleSpecificTime import \
Cell2fate_DynamicalModel_ModuleSpecificTime

from ._cell2fate_DynamicalModel_FreeModules import \
Cell2fate_DynamicalModel_FreeModules

from ._cell2fate_DynamicalModelTimeAsParam import \
Cell2fate_DynamicalModelTimeAsParam

from ._cell2fate_DynamicalModel_SequentialModules import \
Cell2fate_DynamicalModel_SequentialModules

from ._cell2fate_DynamicalModel_TimeMixture import \
Cell2fate_DynamicalModel_TimeMixture

# from ._cell2fate_DynamicalModel_SequentialModules_DcdiDependent import \
# Cell2fate_DynamicalModel_SequentialModules_DcdiDependent

from ._cell2fate_DynamicalModel_SequentialModules_IdentityDependent import \
Cell2fate_DynamicalModel_SequentialModules_IdentityDependent

from ._cell2fate_DynamicalModel_SequentialModules_LinearDependent import \
Cell2fate_DynamicalModel_SequentialModules_LinearDependent

from ._cell2fate_DynamicalModel_PreprocessedCounts import \
Cell2fate_DynamicalModel_PreprocessedCounts

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "cell2fate"
# __version__ = importlib_metadata.version(package_name)

logger = logging.getLogger(__name__)
# set the logging level
logger.setLevel(logging.INFO)

# nice logging outputs
console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("cell2fate: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# this prevents double outputs
logger.propagate = False

__all__ = ["Cell2fate"]

@biject_to.register(constraints.positive)
@transform_to.register(constraints.positive)
def _transform_to_positive(constraint):
    return SoftplusTransform()
