
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
    Create a logger object that can be used to output to
    standard out

    Args:
        level (int): Level of logging. Default is INFO
    
    Returns:
        logger (object): Logger object
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger
