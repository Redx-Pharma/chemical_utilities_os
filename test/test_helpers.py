import pandas as pd
import pytest
from chemutil import helpers


def test_get_pd_column_subset_keep():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    result = helpers.get_pd_column_subset(df, cols_to_keep=["A", "C"])
    expected = pd.DataFrame({"A": [1, 2, 3], "C": [7, 8, 9]})
    pd.testing.assert_frame_equal(result, expected)


def test_get_pd_column_subset_drop():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    result = helpers.get_pd_column_subset(df, cols_to_drop=["B"])
    expected = pd.DataFrame({"A": [1, 2, 3], "C": [7, 8, 9]})
    pd.testing.assert_frame_equal(result, expected)


def test_check_lengths_same_two_lists():
    assert (
        helpers.check_lengths_same_two_lists(["a", "b", "c", "d"], ["e", "f", "g", "h"])
        is True
    )
    assert (
        helpers.check_lengths_same_two_lists(["a", "d"], ["e", "f", "g", "h"]) is False
    )
    assert (
        helpers.check_lengths_same_two_lists(("a", "b", "c", "d"), ("e", "f", "g", "h"))
        is True
    )
    assert (
        helpers.check_lengths_same_two_lists(("a", "d"), ("e", "f", "g", "h")) is False
    )
    assert (
        helpers.check_lengths_same_two_lists(("a", "b", "c", "d"), ["e", "f", "g", "h"])
        is True
    )
    assert (
        helpers.check_lengths_same_two_lists(("a", "d"), ["e", "f", "g", "h"]) is False
    )


def test_pandas_df_min_max_scale():
    df = pd.DataFrame({"A": [1.0, 1.25, 1.5, 2.0], "B": [0.0, 0.25, 0.5, 1.0]})
    expected = pd.DataFrame({"A": [0.0, 0.25, 0.5, 1.0], "B": [0.0, 0.25, 0.5, 1.0]})
    result = helpers.pandas_df_min_max_scale(df)
    pd.testing.assert_frame_equal(result, expected)


def test_pandas_df_z_scale():
    df = pd.DataFrame({"A": [1.0, 1.25, 1.5, 2.0], "B": [0.0, 0.25, 0.5, 1.0]})
    expected = pd.DataFrame(
        {
            "A": [-1.183216, -0.507093, 0.169031, 1.521278],
            "B": [-1.183216, -0.507093, 0.169031, 1.521278],
        }
    )
    result = helpers.pandas_df_z_scale(df)
    pd.testing.assert_frame_equal(result.round(5), expected.round(5))


def test_extract_and_remove_row_from_df():
    df = pd.DataFrame({"A": ["A1", 2, 3, 4], "B": ["B1", 2, 3, 4]}).transpose()
    df1, df2 = helpers.extract_and_remove_row_from_df(
        df, standard_unique_identifer_column=0, standard_unique_identifier="A1"
    )
    assert df1.values.tolist() == [["B1", 2, 3, 4]]
    assert df2.values.tolist() == [["A1", 2, 3, 4]]


def test_check_dfs_have_the_same_number_of_columns():
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
    assert helpers.check_dfs_have_the_same_number_of_columns(df1, df2) is True

    df3 = pd.DataFrame({"A": [1, 2]})
    with pytest.raises(AssertionError):
        helpers.check_dfs_have_the_same_number_of_columns(df1, df3)


def test_check_dfs_have_the_same_column_names():
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
    assert helpers.check_dfs_have_the_same_column_names(df1, df2) is True

    df3 = pd.DataFrame({"A": [1, 2], "C": [3, 4]})
    with pytest.raises(AssertionError):
        helpers.check_dfs_have_the_same_column_names(df1, df3)


def test_get_grid_layout():
    assert helpers.get_grid_layout(4, 5) == (1, 4)
    assert helpers.get_grid_layout(10, 5) == (2, 5)
    assert helpers.get_grid_layout(12, 5) == (3, 5)


def test_get_index_to_row_column_map():
    expected = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    result = helpers.get_index_to_row_column_map(4, 2)
    assert result == expected


def test_sort_list_using_another_list():
    # Test case 1: Sorting integers
    list_to_sort = [10, 20, 30, 40]
    list_to_sort_by = [3, 1, 4, 2]
    expected = [20, 40, 10, 30]
    result = helpers.sort_list_using_another_list(list_to_sort, list_to_sort_by)
    assert result == expected

    # Test case 2: Sorting strings
    list_to_sort = ["apple", "banana", "cherry", "date"]
    list_to_sort_by = ["d", "b", "a", "c"]
    expected = ["cherry", "banana", "date", "apple"]
    result = helpers.sort_list_using_another_list(list_to_sort, list_to_sort_by)
    assert result == expected

    # Test case 3: Sorting mixed types
    list_to_sort = [1, "two", 3.0, "four"]
    list_to_sort_by = ["b", "a", "d", "c"]
    expected = ["two", 1, "four", 3.0]
    result = helpers.sort_list_using_another_list(list_to_sort, list_to_sort_by)
    assert result == expected

    # Test case 4: Sorting with already sorted list_to_sort_by
    list_to_sort = [100, 200, 300]
    list_to_sort_by = [1, 2, 3]
    expected = [100, 200, 300]
    result = helpers.sort_list_using_another_list(list_to_sort, list_to_sort_by)
    assert result == expected

    # Test case 5: Sorting with reversed list_to_sort_by
    list_to_sort = [100, 200, 300]
    list_to_sort_by = [3, 2, 1]
    expected = [300, 200, 100]
    result = helpers.sort_list_using_another_list(list_to_sort, list_to_sort_by)
    assert result == expected

    # Test case 6: Sorting without internal sort
    list_to_sort = [10, 20, 30, 40]
    list_to_sort_by = [3, 1, 4, 2]
    expected = [10, 20, 30, 40]
    result = helpers.sort_list_using_another_list(
        list_to_sort, list_to_sort_by, no_internal_sort=True
    )
    assert result == expected


# Run the tests
if __name__ == "__main__":
    pytest.main()
