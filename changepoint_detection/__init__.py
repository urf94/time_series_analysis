# __init__.py
from .train import train_prophet, train_neuralprophet
from .inference import inference_prophet, inference_neuralprophet
from .ensemble import voting, proba, NoChangePointDetectedError


import logging
import warnings
from matplotlib import MatplotlibDeprecationWarning

logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

