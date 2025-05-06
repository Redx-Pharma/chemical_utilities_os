#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module of unit test
"""

import logging

import pandas as pd
from chemutil import smilesmod

log = logging.getLogger(__name__)


def test_get_valid_smiles_all_valid():
    """
    Test the codes can extract the smiles and finds all valid smiles to be valid
    """
    inp = ["c1ccccc1", "CCCCC"]
    ret = smilesmod.get_valid_smiles(inp)
    assert all(ent == ret[ith] for ith, ent in enumerate(inp))


def test_get_valid_smiles_some_invalid():
    """
    Test that invalid smiles can be remove automatically and the returned list is of the expected length
    """
    inp = ["c1ccccc1", "CCCCC", "ukn"]
    ret = smilesmod.get_valid_smiles(inp)
    assert len(ret) == 2
    assert ret[0] == inp[0]
    assert ret[1] == inp[1]


def test_get_valid_smiles_order_some_invalid():
    """
    Test that invalid smiles can be remove automatically and the returned list is of the expected length and order
    """
    inp = ["c1ccccc1", "CCCCC", "ukn", "c1ccoc1", "CCCNCC"]
    ret = smilesmod.get_valid_smiles(inp)
    assert len(ret) == 4
    assert ret[0] == inp[0]
    assert ret[1] == inp[1]
    # Test should compare index 2 to index 3 as the invalid smiles is removed
    assert ret[2] == inp[3]
    assert ret[3] == inp[4]


def test_get_valid_smiles_mask_all_valid():
    """
    Test that we get the expected all smiles valid vector boolean mask
    """
    inp = ["c1ccccc1", "CCCCC"]
    ret = smilesmod.get_valid_smiles_mask(inp)
    assert all(ent is True for ent in ret)


def test_get_valid_smiles_mask_some_invalid():
    """
    Test that we get the expected all but the last smiles valid vector boolean mask
    """
    inp = ["c1ccccc1", "CCCCC", "ukn"]
    ret = smilesmod.get_valid_smiles_mask(inp)
    assert len(ret) == 3
    assert ret[0] is True
    assert ret[1] is True
    assert ret[2] is False


def test_get_valid_smiles_mask_order_some_invalid():
    """
    Test that invalid smiles are listed as False in the vector boolean mask
    """
    inp = ["c1ccccc1", "CCCCC", "ukn", "c1cocc1", "CCCNCC"]
    ret = smilesmod.get_valid_smiles_mask(inp)
    assert len(ret) == 5
    assert ret[0] is True
    assert ret[1] is True
    assert ret[2] is False
    assert ret[3] is True
    assert ret[4] is True


def test_clean_and_validate_smiles():
    # Sample data
    data = {
        "smiles": ["CCO", "C1CCCCC1", "invalid_smiles", "CC(=O)O.OCC", "C1=CC=CC=C1"]
    }
    df = pd.DataFrame(data)

    # Call the function
    result = smilesmod.clean_and_validate_smiles(
        data_df=df,
        smiles_column="smiles",
        charge_neutralize=False,
        diconnect_metals=True,
        normalize=True,
        reionize=True,
        stereo=True,
        remove_salts=True,
        remove_fragmented=True,
        return_selfies=True,
        return_inchi=True,
        return_inchikey=True,
    )

    # Assertions
    assert "standardized_smiles" in result.columns
    assert "standardized_selfies" in result.columns
    assert "standardized_inchi" in result.columns
    assert "standardized_inchikey" in result.columns

    # Check that invalid smiles are removed
    assert "invalid_smiles" not in result["smiles"].values

    # Check that fragmented smiles are removed
    assert "CC(=O)O.OCC" not in result["smiles"].values

    # Check that valid smiles are processed
    assert "CCO" in result["standardized_smiles"].values
    assert "C1CCCCC1" in result["standardized_smiles"].values

    # Check that the additional representations are not None
    assert result["standardized_selfies"].notnull().all()
    assert result["standardized_inchi"].notnull().all()
    assert result["standardized_inchikey"].notnull().all()
