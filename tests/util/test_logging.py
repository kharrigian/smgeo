
#####################
### Imports
#####################

## Standard Library
from logging import RootLogger

## External
import pytest

## Local
from smgeo.util.logging import initialize_logger

#####################
### Tests
#####################

def test_initialize_logger():
    """

    """
    ## Initialize a Logger
    try:
        logger = initialize_logger(level=20)
    except:
        assert False
    ## Check Object Type
    assert isinstance(logger, RootLogger)
    assert logger.level == 20


