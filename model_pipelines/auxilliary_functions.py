def find_common_elements(*arrays):
    """
    Finds the common subset of elements across an arbitrary number of arrays.
    """
    if not arrays:
        return []
    common_set = set(arrays[0]).intersection(*arrays[1:])
    return list(common_set)
