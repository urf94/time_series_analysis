# __init__.py
from .train import train_prophet
from .inference import inference_prophet
from .ensemble import (
    voting,
    proba_depreceted as proba,
    NoChangePointDetectedError,
    proba_w_post,
    change_point_with_proba,
)


import logging
import warnings
from matplotlib import MatplotlibDeprecationWarning

logger = logging.getLogger("cmdstanpy")
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
