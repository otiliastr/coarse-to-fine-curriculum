import logging.config
import os
import yaml

from . import data
from . import learning
from . import utils

__all__ = ['data', 'learning', 'utils']
__author__ = 'Otilia Stretcu'

__logging_config_path = os.path.join(os.path.dirname(__file__), 'logging.yaml')
if os.path.exists(__logging_config_path):
    with open(__logging_config_path, 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
else:
    logging.getLogger('').addHandler(logging.NullHandler())
