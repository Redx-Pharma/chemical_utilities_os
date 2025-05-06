#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module of unit test
"""

import logging

import pandas as pd
import pytest
from chemutil import molecules

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
def series_dict() -> pd.DataFrame:
    """
    pandas_dataframe fixture holding the default input dataframe

    Returns:
        pd.DataFrame - pandas data frame
    """
    ser_dictionary = {"series a": "[#6]1[#6][#6][#6][#6][#6]1", "series b": "CC"}
    return ser_dictionary


@pytest.fixture
def subseries_dict_no_overlap() -> pd.DataFrame:
    """
    pandas_dataframe fixture holding the default input dataframe

    Returns:
        pd.DataFrame - pandas data frame
    """
    ser_dictionary = {
        "series a": ["11", "12", "13", "14", "15"],
        "series b": ["21", "22", "23", "24", "25"],
    }
    return ser_dictionary


@pytest.fixture
def subseries_dict_some_overlap() -> pd.DataFrame:
    """
    pandas_dataframe fixture holding the default input dataframe

    Returns:
        pd.DataFrame - pandas data frame
    """

    ser_dictionary = {
        "series a": ["11", "12", "13", "14", "15"],
        "series b": ["11", "22", "23", "14", "15"],
        "series c": ["11", "22", "23", "14", "15"],
        "series d": ["10", "21", "20", "04", "05"],
    }

    return ser_dictionary


def test_check_for_sub_series_overlap_non_overlapping_inp(subseries_dict_no_overlap):
    """
    Check that the sub series overlap check does nt see any overlaps when it should not
    """
    overlap, _ = molecules.check_for_sub_series_overlap(subseries_dict_no_overlap)
    assert overlap is False


def test_check_for_sub_series_overlap_with_overlapping_inp(subseries_dict_some_overlap):
    """
    Check that the sub series overlap check does see any overlaps when it should find them
    """
    overlap, overlaps = molecules.check_for_sub_series_overlap(
        subseries_dict_some_overlap
    )
    assert overlap is True
    assert overlaps["series a -- series b"] is True
    assert overlaps["series c -- series d"] is False


def test_get_sub_series(series_dict, pandas_dataframe):
    """
    Check that the sub series getter finds the correct sub series and ignores the other molecules
    """
    subseries_dict, subseries_dict_dfs = molecules.get_sub_series(
        series_dict, series_df=pandas_dataframe, id_column="names", smiles_col="smiles"
    )
    expect_seriesa = [
        "benzene",
        "1-cyclohexylethanamine",
        "1-cyclohexylethanone",
        "1-cyclohexylethanol",
        "benzylamine",
    ]
    expect_seriesb = [
        "1-cyclopentylethanamine",
        "1-cyclopentylethanone",
        "1-cyclopentylethanol",
        "1-cyclohexylethanamine",
        "1-cyclohexylethanone",
        "1-cyclohexylethanol",
        "ethane",
        "propane",
        "butane",
        "pentane",
    ]
    expect_not_in_any_series = ["methane"]
    assert all(ent in expect_seriesa for ent in subseries_dict["series a"])
    assert all(ent in expect_seriesb for ent in subseries_dict["series b"])
    assert all(
        ent not in expect_not_in_any_series
        for ent in subseries_dict["series a"] + subseries_dict["series b"]
    )
