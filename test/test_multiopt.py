#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module of unit test
"""

import logging

import pandas as pd
import pytest
from chemutil import multiopt

log = logging.getLogger(__name__)


@pytest.fixture
def pandas_dataframe() -> pd.DataFrame:
    """
    pandas_dataframe fixture holding the default input dataframe

    Returns:
        pd.DataFrame - pandas data frame
    """
    test_file = pd.DataFrame(
        [
            ["c1ccccc1", "benzene", 0.5, 1.0, 0.25, 1.2],
            ["C1CCCC1C(N)C", "1-cyclopentylethanamine", 0.9, 0.1, 1.2, 0.9],
            ["C1CCCC1C(=O)C", "1-cyclopentylethanone", 0.75, 0.05, 1.2, 0.9],
            ["C1CCCC1C(O)C", "1-cyclopentylethanol", 0.95, 0.12, 1.2, 0.9],
            ["C1CCCCC1C(N)C", "1-cyclohexylethanamine", 0.95, 0.15, 1.22, 0.95],
            ["C1CCCCC1C(=O)C", "1-cyclohexylethanone", 0.79, 0.02, 1.24, 0.97],
            ["C1CCCCC1C(O)C", "1-cyclohexylethanol", 1.1, 1.2, 1.4, 0.95],
            ["NCc1ccccc1", "benzylamine", 1.2, 0.02, 2.2, 0.75],
            ["C", "methane", -1.2, 0.01, 0.02, -10.0],
            ["CC", "ethane", -1.0, 0.2, 0.07, -10.2],
            ["CCC", "propane", -1.0, -0.4, 0.1, -10.7],
            ["CCCC", "butane", 0.75, 0.25, 0.4, 1.4],
            ["CCCCC", "pentane", -0.7, -0.9, 0.2, -11.0],
        ],
        columns=["smiles", "names", "bind_target_0", "bind_target_1", "tox", "sol"],
    )
    return test_file


@pytest.fixture
def pandas_dataframe_swapped_bind1_bind2_order() -> pd.DataFrame:
    """
    pandas_dataframe fixture holding the default input dataframe

    Returns:
        pd.DataFrame - pandas data frame
    """
    test_file = pd.DataFrame(
        [
            ["c1ccccc1", "benzene", 1.0, 0.5, 0.25, 1.2],
            ["C1CCCC1C(N)C", "1-cyclopentylethanamine", 0.1, 0.9, 1.2, 0.9],
            ["C1CCCC1C(=O)C", "1-cyclopentylethanone", 0.05, 0.75, 1.2, 0.9],
            ["C1CCCC1C(O)C", "1-cyclopentylethanol", 0.12, 0.95, 1.2, 0.9],
            ["C1CCCCC1C(N)C", "1-cyclohexylethanamine", 0.15, 0.95, 1.22, 0.95],
            ["C1CCCCC1C(=O)C", "1-cyclohexylethanone", 0.02, 0.79, 1.24, 0.97],
            ["C1CCCCC1C(O)C", "1-cyclohexylethanol", 1.2, 1.1, 1.4, 0.95],
            ["NCc1ccccc1", "benzylamine", 0.02, 1.2, 2.2, 0.75],
            ["C", "methane", 0.01, -1.2, 0.02, -10.0],
            ["CC", "ethane", 0.2, -1.0, 0.07, -10.2],
            ["CCC", "propane", -0.4, -1.0, 0.1, -10.7],
            ["CCCC", "butane", 0.25, 0.75, 0.4, 1.4],
            ["CCCCC", "pentane", -0.9, -0.7, 0.2, -11.0],
        ],
        columns=["smiles", "names", "bind_target_1", "bind_target_0", "tox", "sol"],
    )
    return test_file


@pytest.fixture
def exp_return_for_get_subplot_df() -> pd.DataFrame:
    """
    _summary_

    Returns:
        pd.DataFrame: _description_
    """

    data_df = pd.DataFrame.from_dict(
        {
            "ID": {0: "benzylamine", 1: "pentane"},
            "inchi": {0: "NCc1ccccc1", 1: "CCCCC"},
            "Mean_ineff_bind_target_0": {0: 0.3172727272727273, 1: 0.3172727272727273},
            "Mean_ineff_bind_target_1": {
                0: 0.24545454545454548,
                1: 0.24545454545454548,
            },
            "Mean_ineff_tox": {0: 0.7545454545454544, 1: 0.7545454545454544},
            "Mean_ineff_sol": {0: -2.0663636363636364, 1: -2.0663636363636364},
            "bind_target_0": {0: 1.2, 1: -0.7},
            "bind_target_1": {0: 0.02, 1: -0.9},
            "tox": {0: 2.2, 1: 0.2},
            "sol": {0: 0.75, 1: -11.0},
        }
    )
    return data_df


def test_filter_on_properties(pandas_dataframe: pd.DataFrame):
    """
    Function to test the filter on properties function

    Args:
        pandas_dataframe (pd.DataFrame): pandas dataframe of data to filter

    Raises:
        aerr: assertion error if the molecule names found are not expected or the number of molecules found is different than as expected
    """

    filtered_df = multiopt.filter_on_property_changes(
        pandas_dataframe,
        reference_properties=["bind_target_0"],
        experimental_properties=["bind_target_1"],
        change=["increase"],
        must_pass=[0],
    )

    filtered_names = filtered_df["names"].tolist()
    expected_names = ["benzene", "1-cyclohexylethanol", "methane", "ethane", "propane"]

    try:
        assert all(ent in expected_names for ent in filtered_names)
        assert len(filtered_names) == len(expected_names)
    except AssertionError as aerr:
        log.critical(f"Expected: {expected_names} but found {filtered_names}")
        raise aerr


def test_get_pareto_efficient_set(pandas_dataframe: pd.DataFrame):
    """
    Function to test finding a Pareto efficent set. Note that this misses some promising moleucles as the front is made up of twp points that
    stand out from the clusters.

    Args:
        pandas_dataframe (pd.DataFrame): pandas dataframe to find the efficent solutions of

    Raises:
        aerr: rasied if there are unexpected molecules found on the Pareto front or missing molecules in the Pareto front
    """

    efficent_set_df = multiopt.get_pareto_efficient_set(
        pandas_dataframe,
        minmax=["max", "min"],
        objective_columns=["bind_target_0", "bind_target_1"],
    )

    expected_efficent_set_nams = ["benzylamine", "pentane"]
    found_efficent_set_nams = efficent_set_df[
        efficent_set_df["pareto_efficent_set"] == 1
    ]["names"].tolist()

    try:
        assert all(ent in expected_efficent_set_nams for ent in found_efficent_set_nams)
        assert len(found_efficent_set_nams) == len(expected_efficent_set_nams)
    except AssertionError as aerr:
        log.critical(
            f"Expected: {expected_efficent_set_nams} but found {found_efficent_set_nams}"
        )
        raise aerr


def test_get_pareto_efficient_set_switch_column_order_leads_to_the_same_efficent_set(
    pandas_dataframe_swapped_bind1_bind2_order: pd.DataFrame,
):
    """
    Function to test if we pass in a dataframe with the columns in a different order we get the same efficent set. This is making sure that the data pulled out of the
    data frame for running the pareto efficent set analysis is consistently done to match the objective column list order so that the min max order is right.

    Args:
        pandas_dataframe (pd.DataFrame): pandas dataframe to find the efficent solutions of

    Raises:
        aerr: rasied if there are unexpected molecules found on the Pareto front or missing molecules in the Pareto front
    """

    efficent_set_df = multiopt.get_pareto_efficient_set(
        pandas_dataframe_swapped_bind1_bind2_order,
        minmax=["max", "min"],
        objective_columns=["bind_target_0", "bind_target_1"],
    )

    expected_efficent_set_nams = ["benzylamine", "pentane"]
    found_efficent_set_nams = efficent_set_df[
        efficent_set_df["pareto_efficent_set"] == 1
    ]["names"].tolist()

    try:
        assert all(ent in expected_efficent_set_nams for ent in found_efficent_set_nams)
        assert len(found_efficent_set_nams) == len(expected_efficent_set_nams)
    except AssertionError as aerr:
        log.critical(
            f"Expected: {expected_efficent_set_nams} but found {found_efficent_set_nams}"
        )
        raise aerr


def test_get_pareto_efficient_set_switch_column_order_but_not_minmax_leads_to_a_different_efficent_set(
    pandas_dataframe: pd.DataFrame,
):
    """
    Function to test that we find a different efficent set if the same data is used and the min max is swapped.

    Args:
        pandas_dataframe (pd.DataFrame): pandas dataframe to find the efficent solutions of

    Raises:
        aerr: rasied if there are unexpected molecules found on the Pareto front or missing molecules in the Pareto front
    """

    efficent_set_df = multiopt.get_pareto_efficient_set(
        pandas_dataframe,
        minmax=["min", "max"],
        objective_columns=["bind_target_0", "bind_target_1"],
    )

    expected_efficent_set_nams = ["benzene", "1-cyclohexylethanol", "methane", "ethane"]
    expected_efficent_set_nams_for_min_max_reversed = ["benzylamine", "pentane"]
    found_efficent_set_nams = efficent_set_df[
        efficent_set_df["pareto_efficent_set"] == 1
    ]["names"].tolist()

    try:
        assert len(found_efficent_set_nams) != len(
            expected_efficent_set_nams_for_min_max_reversed
        )
        assert any(
            ent != expected_efficent_set_nams_for_min_max_reversed[ith]
            for ith, ent in enumerate(sorted(found_efficent_set_nams))
        )
    except AssertionError as aerr:
        log.critical(
            f"Found a match to the efficent set with min max reversed. Expected: {expected_efficent_set_nams} but found {found_efficent_set_nams}"
        )
        raise aerr

    try:
        assert all(ent in expected_efficent_set_nams for ent in found_efficent_set_nams)
        assert len(found_efficent_set_nams) == len(expected_efficent_set_nams)

    except AssertionError as aerr:
        log.critical(
            f"Expected: {expected_efficent_set_nams} but found {found_efficent_set_nams}"
        )
        raise aerr


def test_get_pareto_ranking(pandas_dataframe: pd.DataFrame):
    """
    Function to test finding a Pareto ranking.

    Args:
        pandas_dataframe (pd.DataFrame): pandas dataframe to find the efficent solutions of

    Raises:
        aerr: rasied if there are unexpected molecules found on the Pareto front or missing molecules in the Pareto front
    """

    efficent_set_df = multiopt.get_pareto_ranking(
        pandas_dataframe,
        minmax=["max", "min"],
        objective_columns=["bind_target_0", "bind_target_1"],
    )

    expected_ranking = [5, 2, 3, 2, 3, 2, 2, 1, 3, 4, 2, 4, 1]
    found_ranking = efficent_set_df["pareto_rank"].tolist()

    try:
        assert all(
            ent == expected_ranking[ith] for ith, ent in enumerate(found_ranking)
        )
        assert len(found_ranking) == len(expected_ranking)
    except AssertionError as aerr:
        log.critical(f"Expected: {expected_ranking} but found {found_ranking}")
        raise aerr


def test_get_pareto_ranking_with_swapped_column_order_input_is_the_same(
    pandas_dataframe_swapped_bind1_bind2_order: pd.DataFrame,
):
    """
    Function to test finding a Pareto ranking finds the same ranking regardless of column order of the input data frame

    Args:
        pandas_dataframe (pd.DataFrame): pandas dataframe to find the efficent solutions of

    Raises:
        aerr: rasied if there are unexpected molecules found on the Pareto front or missing molecules in the Pareto front
    """

    efficent_set_df = multiopt.get_pareto_ranking(
        pandas_dataframe_swapped_bind1_bind2_order,
        minmax=["max", "min"],
        objective_columns=["bind_target_0", "bind_target_1"],
    )

    expected_ranking = [5, 2, 3, 2, 3, 2, 2, 1, 3, 4, 2, 4, 1]
    found_ranking = efficent_set_df["pareto_rank"].tolist()

    try:
        assert all(
            ent == expected_ranking[ith] for ith, ent in enumerate(found_ranking)
        )
        assert len(found_ranking) == len(expected_ranking)
    except AssertionError as aerr:
        log.critical(f"Expected: {expected_ranking} but found {found_ranking}")
        raise aerr


def test_get_pareto_ranking_with_swapped_min_max_leads_to_different_ranking(
    pandas_dataframe: pd.DataFrame,
):
    """
    Function to test finding a Pareto ranking finds a different ranking set if min and max are changed

    Args:
        pandas_dataframe (pd.DataFrame): pandas dataframe to find the efficent solutions of

    Raises:
        aerr: rasied if there are unexpected molecules found on the Pareto front or missing molecules in the Pareto front
    """

    efficent_set_df = multiopt.get_pareto_ranking(
        pandas_dataframe,
        minmax=["min", "max"],
        objective_columns=["bind_target_0", "bind_target_1"],
    )

    expected_ranking = [5, 2, 3, 2, 3, 2, 2, 1, 3, 4, 2, 4, 1]
    found_ranking = efficent_set_df["pareto_rank"].tolist()

    try:
        assert any(
            ent != expected_ranking[ith] for ith, ent in enumerate(found_ranking)
        )
        assert len(found_ranking) == len(expected_ranking)
    except AssertionError as aerr:
        log.critical(f"Expected: {expected_ranking} but found {found_ranking}")
        raise aerr


def test_get_radar_subplot_data(
    pandas_dataframe: pd.DataFrame, exp_return_for_get_subplot_df: pd.DataFrame
):
    df = multiopt.get_pareto_efficient_set(
        pandas_dataframe,
        minmax=["max", "min"],
        objective_columns=["bind_target_0", "bind_target_1"],
    )

    test_df = multiopt._get_radar_subplot_data(
        df,
        objective_columns=["bind_target_0", "bind_target_1", "tox", "sol"],
        label_col="names",
        pareto_eff_col=multiopt.par_eff_col_name,
        smiles_column="smiles",
        inchi_column=None,
    )

    pd.testing.assert_frame_equal(test_df, exp_return_for_get_subplot_df)
