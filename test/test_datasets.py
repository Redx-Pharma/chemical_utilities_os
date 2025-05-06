#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module of unit test
"""

import logging

import deepchem as dc
import pandas as pd
import pytest
from chemutil import datasets, helpers

log = logging.getLogger(__name__)


number_of_defualt_models = 14
number_of_defualt_linear_models = 3
number_of_defualt_kernel_models = 4
number_of_defualt_bayesian_models = 2
number_of_defualt_ensemble_models = 4
number_of_defualt_neural_network_models = 1


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


def test_scaffold_split(pandas_dataframe):
    dataset = datasets.pandas_to_deepchem(
        pandas_dataframe,
        smiles_column="smiles",
        task_columns=["bind_target_0", "bind_target_1", "tox"],
        feature_columns=["sol"],
    )
    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, test_dataset = splitter.train_test_split(
        dataset, train=0.8, test=0.2, seed=helpers.random_seed
    )

    try:
        assert all(
            ent.strip()
            in [
                "C",
                "CC",
                "CCC",
                "CCCC",
                "C1CCCCC1C(N)C",
                "C1CCCCC1C(=O)C",
                "C1CCCCC1C(O)C",
                "c1ccccc1",
                "NCc1ccccc1",
            ]
            for ent in train_dataset.ids
        )
    except AssertionError as aerr:
        log.critical(
            f"Not all expected smiles are consistent with the split in the training data set expected: []'C', 'CC', 'CCC', 'CCCC', 'C1CCCCC1C(N)C', 'C1CCCCC1C(=O)C', 'C1CCCCC1C(O)C', 'c1ccccc1', 'NCc1ccccc1'] got: {train_dataset.ids}"
        )
        raise aerr

    try:
        assert all(
            ent.strip() in ["C1CCCC1C(N)C", "C1CCCC1C(=O)C", "C1CCCC1C(O)C"]
            for ent in test_dataset.ids
        )
    except AssertionError as aerr:
        log.critical(
            f"Not all expected smiles are consistent with the split in the testing data set expected: ['C1CCCC1C(N)C' 'C1CCCC1C(=O)C' 'C1CCCC1C(O)C'] got: {test_dataset.ids}"
        )
        raise aerr


def test_get_scaffold_split(pandas_dataframe):
    dataset = datasets.pandas_to_deepchem(
        pandas_dataframe,
        smiles_column="smiles",
        task_columns=["bind_target_0", "bind_target_1", "tox"],
        feature_columns=["sol"],
    )
    splitter = dc.splits.ScaffoldSplitter()
    expected_scaffold_group_indxs = splitter.generate_scaffolds(dataset)

    scaffold_maps = datasets.get_scaffolds_groups_from_scaffold_splitter(
        pandas_dataframe["smiles"].to_list()
    )

    for ent, k in zip(expected_scaffold_group_indxs, scaffold_maps.keys()):
        assert all(elt in ent for elt in scaffold_maps[k]["smiles_indexes"])
