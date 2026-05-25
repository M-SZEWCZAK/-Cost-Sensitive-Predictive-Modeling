import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
def find_common_elements(*arrays):
    """
    Finds the common subset of elements across an arbitrary number of arrays.
    """
    if not arrays:
        return []
    common_set = set(arrays[0]).intersection(*arrays[1:])
    return list(common_set)
def find_sum_of_arrays(*arrays):
    """
    Finds the set-theoretic sum of elements across an arbitrary number of arrays.
    """
    if not arrays:
        return []
    sum_set = set(arrays[0]).union(*arrays[1:])
    return list(sum_set)
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
def unpack_whole_feature_dict(dict_,extract='common'):
    """
    returns the features found by different methods as a list. if extract parameter is set to 'common'
    it will return the features found by all methods applied in the passed dict, if extract is set to 'all_found'
    it will return all the features found by at least one method, else it returns both sets with
    features found by at least one method as first element. all are return as arrays.
    """
    all_arrays = []
    for value in dict_.values():
        if isinstance(value, dict):
            all_arrays.extend(value.values())
        elif isinstance(value, list):
            all_arrays.append(value)
    # Pass the extracted lists to your function using * unpacking
    total_unique_features = find_sum_of_arrays(*all_arrays)
    total_common_features = find_common_elements(*all_arrays)
    if extract == 'common':
        return total_common_features
    elif extract == 'all_found':
        return total_unique_features
    else:
        return total_unique_features,total_common_features
def unpack_model_feature_dict(dict_,model_key,extract='common'):
    """
       returns the features found by a model method passed in argument model_key (ex. 'xgb')  as a list.
       if extract parameter is set to 'common' it will return the features found by all methods
        applied in the passed dict, if extract is set to 'all_found' it will return all the
        features found by at least one method, else it returns both sets with
       features found by at least one method as first element. all are return as arrays.
       """
    model_arrays = [
        arrays
        for key, sub_dict in dict_.items()
        if model_key in key and isinstance(sub_dict, dict)
        for arrays in sub_dict.values()
    ]
    model_unique_features = find_sum_of_arrays(*model_arrays)
    model_common_features = find_common_elements(*model_arrays)
    if extract == 'common':
        return model_common_features
    elif extract == 'all_found':
        return model_unique_features
    else:
        return model_unique_features,model_common_features
def add_interaction_features(df: pd.DataFrame, feature_subset: list) -> pd.DataFrame:
    """Generates pairwise interaction features (multiplication) for a specified
    subset of features in a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe.
    feature_subset : list
        List of column names to create interactions for (expected to be < 8).

    Returns:
    --------
    pd.DataFrame
        A new DataFrame containing the original columns plus the interaction
        features.
    """
    df_enhanced = df.copy()
    missing_features = [f for f in feature_subset if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"The following features were not found in the DataFrame: {missing_features}"
        )
    for feat1, feat2 in itertools.combinations(feature_subset, 2):
        feature_name = f"{feat1}_X_{feat2}"
        df_enhanced[feature_name] = df[feat1] * df[feat2]
    return df_enhanced