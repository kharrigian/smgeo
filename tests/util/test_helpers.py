
#####################
### Imports
#####################

## External
import pytest

## Local
from smgeo.util import helpers

#####################
### Tests
#####################

def test_flatten():
    """

    """
    l = [["a","b","c"],[None], [1, 2, None]]
    l_flat = helpers.flatten(l)
    assert l_flat == ["a","b","c",None,1,2,None]

def test_chunks():
    """

    """
    l = list(range(14))
    l_chunk = list(helpers.chunks(l, 5))
    assert l_chunk == [[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13]]
    