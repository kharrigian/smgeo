
def flatten(l):
    """
    Flatten a list of lists by one level.

    Args:
        l (list of lists): List of lists
    
    Returns:
        flattened_list (list): Flattened list
    """
    flattened_list = [item for sublist in l for item in sublist]
    return flattened_list

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.

    Args:
        l (list): List of objects
        n (int): Chunksize
    
    Yields:
        Chunks
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]