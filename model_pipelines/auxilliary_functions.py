import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
def find_common_elements(*arrays):
    """
    Finds the common subset of elements across an arbitrary number of arrays.
    """
    if not arrays:
        return []
    common_set = set(arrays[0]).intersection(*arrays[1:])
    return list(common_set)
def plot_correlation_matrix(m):
    plt.figure(figsize=(20, 15))
    sns.heatmap(m,
                annot=True,
                cmap='RdBu_r',  # Red-Blue is often more intuitive for +/- correlations
                center=0,  # Ensure 0 is the neutral color
                linewidths=.1,
                cbar_kws={"shrink": .8},
                vmin=-1, vmax=1)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
def get_subsets(arr):
    """
    Returns a list of all possible non-empty subsets of an array passed as input.
    It is intended for small arrays (max 10-15 elements), as it is exponentially complex.
    """
    result = [[]]
    for x in arr:
        result += [curr + [x] for curr in result]
    return result[1:]