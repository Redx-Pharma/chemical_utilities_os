#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module of unit test
"""

import logging

import numpy as np
import pandas as pd
import pytest
from chemutil import vis

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
            ["CCCC", "butane", -0.7, -0.9, 0.2, -11.0],
        ],
        columns=["smiles", "names", "bind_target_0", "bind_target_1", "tox", "sol"],
    )
    return test_file


@pytest.fixture
def tanimto_matrix_expected():
    cols = ["c1ccccc1", "c1ccccc1C", "c1ccccc1N", "c1ccccc1Cl"]
    mat = [
        [0.230769, 0.214286, 0.214286],
        [0.411765, 0.388889, 0.315789],
        [0.333333, 0.315789, 0.315789],
        [0.333333, 0.315789, 0.388889],
    ]
    df = pd.DataFrame(mat).transpose()
    df.columns = cols
    return df


def test_get_tanimoto_matrix(tanimto_matrix_expected):
    df1 = pd.DataFrame(
        [["c1ccccc1"], ["c1ccccc1C"], ["c1ccccc1N"], ["c1ccccc1Cl"]], columns=["smiles"]
    )
    df2 = pd.DataFrame(
        [["c1cccc1C"], ["c1ccccc1C(C)(C)C"], ["c1ccccc1NC"], ["c1ccccc1CCl"]],
        columns=["smiles"],
    )

    tmp = vis.get_tanimoto_matrix(df1, df2, "smiles", "smiles", "smiles")

    assert np.allclose(tmp.values, tanimto_matrix_expected.values)
