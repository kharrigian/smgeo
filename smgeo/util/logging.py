
########################
### Imports
########################

## Standard Libarary
import sys
import logging

########################
### Functions
########################

def initialize_logger(level=logging.INFO):
    """

    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger
