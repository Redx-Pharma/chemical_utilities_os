#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for providing generally useful utilities and helper functions
"""

import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import zscore

log = logging.getLogger(__name__)

random_seed = 15751


def get_pd_column_subset(
    data_df: pd.DataFrame,
    cols_to_keep: Optional[List[str]] = None,
    cols_to_drop: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Get a copy of a data frame with all columns apart from the ones you wish to keep dropped

    Args:
        data_df (pd.DataFrame): Raw input dataframe
        cols_to_keep (Optional[List[str]], optional): list of the columns you want to keep. Defaults to None.
        cols_to_drop (Optional[List[str]], optional): list of the columns you want to remove. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with the columns dropped as requested
    """

    if cols_to_drop is not None and cols_to_keep is not None:
        log.warning(
            "warning - please specify only the columns to keep or the columns to drop closing without dropping"
        )

    elif cols_to_keep is not None:
        # This should put df in the order from list cols_to_keep
        df = data_df.copy()
        try:
            df = df[cols_to_keep]
        except KeyError as kerr:
            log.error | ("ERROR - getting all columns to keep")
            raise kerr
        return df

    elif cols_to_drop is not None:
        df = data_df.copy()
        df = df.drop(cols_to_drop, axis=1)
        log.debug(f"Kept columns: {df.columns}")
        return df

    else:
        log.error(
            "ERROR - columns to keep and drop specification error one must be specified please check in the input"
        )


def check_lengths_same_two_lists(
    iterable0: Union[List, Tuple], iterable1: Union[List, Tuple]
) -> bool:
    """
    Check the length of two lists or tuples are the same

    Args:
        iterable0 (Union[List, Tuple]): The first list or tuple
        iterable1 (Union[List, Tuple]): The second list or tuple

    Returns:
        bool: Whether the two are the same size (True) or not (False)

    Doctest:
    >>> check_lengths_same_two_lists(['a', 'b', 'c', 'd'], ['e', 'f', 'g', 'h'])
    True
    >>> check_lengths_same_two_lists(['a', 'd'], ['e', 'f', 'g', 'h'])
    False
    >>> check_lengths_same_two_lists(('a', 'b', 'c', 'd'), ('e', 'f', 'g', 'h'))
    True
    >>> check_lengths_same_two_lists(('a', 'd'), ('e', 'f', 'g', 'h'))
    False
    >>> check_lengths_same_two_lists(('a', 'b', 'c', 'd'), ['e', 'f', 'g', 'h'])
    True
    >>> check_lengths_same_two_lists(('a', 'd'), ['e', 'f', 'g', 'h'])
    False
    """

    if len(iterable0) != len(iterable1):
        return False

    return True


def pandas_df_min_max_scale(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Function to min max normalize so all numerical columns are in the range [0, 1], but low values and high value maintain relative separation.
    This has poor performance if the min or max are outliers in the data.

    value_i' = (value_i - min(column I)) / (max(column I) - min(column I))

    Args:
        df0 (pd.DataFrame): Data frame to normalize

    Returns:
        pd.DataFrame: normalized dataframe

    Doctest:
    >>> df = pd.DataFrame({"A": [1.0, 1.25, 1.5, 2.0], "B": [0.0, 0.25, 0.5, 1.0]})
    >>> exp_df = pd.DataFrame({"A": [0.0, 0.25, 0.5, 1.0], "B": [0.0, 0.25, 0.5, 1.0]})
    >>> norm_df = pandas_df_min_max_scale(df)
    >>> exp_df.equals(norm_df)
    True
    """

    return (df - df.min()) / (df.max() - df.min())


def pandas_df_z_scale(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Function to run z score normalization so all numerical columns are in a normalized range i.e. the range of each column is unity but the scales for each column differ. This is better for
    when outliers are present in the data than min max scaling but lacks a common scale across columns.

    value_i' = value_i - mean(column I) / std(column I)

    Args:
        df0 (pd.DataFrame): Data frame to normalize

    Returns:
        pd.DataFrame: normalized dataframe

    Doctest:
    >>> df = pd.DataFrame({"A": [1.0, 1.25, 1.5, 2.0], "B": [0.0, 0.25, 0.5, 1.0]})
    >>> exp_df = pd.DataFrame({"A": [-1.183216, -0.507093, 0.169031, 1.521278], "B": [-1.183216, -0.507093, 0.169031, 1.521278]})
    >>> norm_df = pandas_df_z_scale(df)
    >>> exp_df.round(5).equals(norm_df.round(5))
    True
    """

    return df.apply(zscore)


def extract_and_remove_row_from_df(
    df: pd.DataFrame,
    standard_unique_identifer_column: Union[str, int],
    standard_unique_identifier: Union[str, int, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull out one row from a dataframe and stores it as a new dataframe (df_standard) then deletes this row from the origial dataframe (df).
    Returns both the original dataframe without the row and a new dataframe of just the row removed

    Args:
        df (pd.DataFrame): Dataframe to pull out one row from as new a dataframe and delete that row
        standard_unique_identifer_column (str): The column to use to identify the row to pull out and delete
        standard_unique_identifier (str): The value of the row in the standard_unique_identifer_column to define the row to pull out and delete

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: First the input dataframe with the row deleted and second the row pulled out as a dataframe

    >>> df = pd.DataFrame({"A": ["A1", 2, 3 ,4], "B": ["B1", 2, 3, 4]}).transpose()
    >>> df1, df2 = extract_and_remove_row_from_df(df, standard_unique_identifer_column=0, standard_unique_identifier="A1")
    >>> df1.values.tolist() # doctest: +NORMALIZE_WHITESPACE
    [['B1', 2, 3, 4]]
    >>> df2.values.tolist() # doctest: +NORMALIZE_WHITESPACE
    [['A1', 2, 3, 4]]
    """
    df = df.copy()
    standard_unique_identifer_index = df[
        df[standard_unique_identifer_column] == standard_unique_identifier
    ].index
    df_standard = pd.DataFrame([df.loc[standard_unique_identifer_index[0], :].copy()])
    df = df.drop(labels=[standard_unique_identifer_index[0]], axis=0)

    return df, df_standard


def check_dfs_have_the_same_number_of_columns(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df1_name: str = "df1",
    df2_name: str = "df2",
    raise_err: bool = True,
):
    """
    Check that the number of columns in two dataframes is the same

    Args:
        df1 (pd.DataFrame): The first dataframe to comppare the number of columns
        df2 (pd.DataFrame): The second dataframe to comppare the number of columns
        df1_name (str, optional): If you want to use a meaningful name for the error message for df1. Defaults to "df1".
        df2_name (str, optional): If you want to use a meaningful name for the error message for df2. Defaults to "df2".
        raise_err: (bool): whether to raise and error (True) or return False (False)

    Raises:
        AssertionError: The number of columns is not the same if raised

    Returns:
        bool: True if the number of columns is the same False or error raised if there are a different number
    """

    try:
        assert len(df1.columns) == len(df2.columns)
        return True
    except AssertionError:
        if raise_err is True:
            raise AssertionError(
                f"The number of objective columns for the {df1_name} and {df2_name} dataframes are different, they should be the same. ({df1_name} dataframe has {len(df1.columns)} columns. {df2_name} dataframe has {len(df2.columns)} columns)."
            )
        else:
            log.error(
                f"The number of objective columns for the {df1_name} and {df2_name} dataframes are different, they should be the same. ({df1_name} dataframe has {len(df1.columns)} columns. {df2_name} dataframe has {len(df2.columns)} columns)."
            )
            return False


def check_dfs_have_the_same_column_names(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df1_name: str = "df1",
    df2_name: str = "df2",
    raise_err: bool = True,
):
    """
    Check that the  columns in two dataframes are named the same

    Args:
        df1 (pd.DataFrame): The first dataframe to comppare the number of columns
        df2 (pd.DataFrame): The second dataframe to comppare the number of columns
        df1_name (str, optional): If you want to use a meaningful name for the error message for df1. Defaults to "df1".
        df2_name (str, optional): If you want to use a meaningful name for the error message for df2. Defaults to "df2".
        raise_err: (bool): whether to raise and error (True) or return False (False)

    Raises:
        AssertionError: The columns are not named the same if raised

    Returns:
        bool: True if the columns are named the same False or error raised if the columns are named differently
    """

    if not all(ent in df1.columns.to_list() for ent in df2.columns.to_list()):
        if raise_err is True:
            raise AssertionError(
                f"The objective columns in the {df1_name} and {df2_name} dataframes are not the same.{os.linesep}{df1_name} dataframe has {', '.join(df1.columns.to_list())} columns.{os.linesep}{df2_name} dataframe has {', '.join(df2.columns.to_list())} columns."
            )
        else:
            log.error(
                f"The objective columns in the {df1_name} and {df2_name} dataframes are not the same.{os.linesep}{df1_name} dataframe has {', '.join(df1.columns.to_list())} columns.{os.linesep}{df2_name} dataframe has {', '.join(df2.columns.to_list())} columns."
            )
            return False
    else:
        return True


def get_grid_layout(
    n_mols: int, mols_per_row: int = 5, return_map: bool = False
) -> Tuple[int, int]:
    """
    Define a sub plot grid layout based on the number of molecules to plot and the number to show per row

    Args:
        n_mols (int): The number of molecules
        mols_per_row (int): The number of molecules to display on a row
        return_map (bool): Whether to return a map dictionary of a continous index for each molecule to it row and column in the grid layout

    Returns:
        Tuple[int, int]: number of rows, number of columns

    >>> ret = get_grid_layout(4, 5)
    >>> ret[0]
    1
    >>> ret[1]
    4
    >>> ret = get_grid_layout(10, 5)
    >>> int(ret[0])
    2
    >>> ret[1]
    5
    >>> ret = get_grid_layout(12, 5)
    >>> ret[0]
    3
    >>> ret[1]
    5
    """
    # set the layout assuming 5 columns asa default
    if not isinstance(mols_per_row, int):
        mols_per_row = int(mols_per_row)

    if n_mols < mols_per_row:
        n_rows = 1
        m_columns = n_mols

    elif n_mols % mols_per_row == 0:
        n_rows = int(n_mols / mols_per_row)
        m_columns = mols_per_row

    else:
        n_rows = int(np.ceil(n_mols / mols_per_row))
        m_columns = mols_per_row

    if return_map is True:
        index_to_row_column_map = get_index_to_row_column_map(n_mols, mols_per_row)
        return n_rows, m_columns, index_to_row_column_map

    else:
        return int(n_rows), int(m_columns)


def get_index_to_row_column_map(n_mols: int, mols_per_row: int) -> dict:
    """
    Get a map of the index to the row and column in a grid layout

    Args:
        n_mols (int): The number of molecules
        mols_per_row (int): The number of molecules to display on a row

    Returns:
        dict: The index to row and column mapping

    >>> get_index_to_row_column_map(4, 2) # doctest: +NORMALIZE_WHITESPACE
    {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    """

    index_to_row_column_map = {}
    r, c = 0, 0
    for ith in range(n_mols):
        index_to_row_column_map[ith] = (r, c)
        if c < mols_per_row - 1:
            c += 1
        else:
            c = 0
            r += 1

    return index_to_row_column_map


def sort_list_using_another_list(
    list_to_sort: list, list_to_sort_by: list, no_internal_sort: bool = False
) -> list:
    """
    Function to sort a list using another list. We assume the two lists are the same length and in the same order. Assuming the list_to_sort_by
    is a list of unique values which can be ordered by sorted(). We then sort the list_to_sort_by using sorted() and then use the sorted order to
    map the list_to_sort to the sorted() order of list_to_sort_by.

    If you set no_internal_sort to True then the list_to_sort_by will not be sorted before the order is determined.

    Args:
        list_to_sort (list): The list to sort
        list_to_sort_by (list): The list to sort by. This will be passed to sorted() to get the order so should be a list or numbers or letters.
        no_internal_sort (bool): If you want to sort the list_to_sort_by before sorting the list_to_sort

    Returns:
        list: The list sorted by the order of the list_to_sort_by
    """

    if no_internal_sort is False:
        order = [list_to_sort_by.index(ent) for ent in sorted(list_to_sort_by)]
    else:
        order = [list_to_sort_by.index(ent) for ent in list_to_sort_by]

    return [list_to_sort[ith] for ith in order]


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
