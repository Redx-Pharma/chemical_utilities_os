#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module of unit test
"""

import logging
from curses import resetty

import pandas as pd
from chemutil import inchimod

log = logging.getLogger(__name__)


def test_get_valid_inchi_all_valid():
    """
    Test the codes can extract the smiles and finds all valid smiles to be valid
    """
    inp = ["InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H", "InChI=1S/C4H10/c1-3-4-2/h3-4H2,1-2H3"]
    ret = inchimod.get_valid_inchi(inp)
    assert all(ent == ret[ith] for ith, ent in enumerate(inp))


def test_get_valid_inchi_some_invalid():
    """
    Test that invalid smiles can be remove automatically and the returned list is of the expected length
    """
    inp = [
        "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H",
        "InChI=1S/C4H10/c1-3-4-2/h3-4H2,1-2H3",
        "ukn",
    ]
    ret = inchimod.get_valid_inchi(inp)
    assert len(ret) == 2
    assert ret[0] == inp[0]
    assert ret[1] == inp[1]


def test_get_valid_inchi_order_some_invalid():
    """
    Test that invalid smiles can be remove automatically and the returned list is of the expected length and order
    """
    inp = [
        "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H",
        "InChI=1S/C4H10/c1-3-4-2/h3-4H2,1-2H3",
        "ukn",
        "InChI=1S/C4H4O/c1-2-4-5-3-1/h1-4H",
        "InChI=1S/C4H12N2/c5-3-1-2-4-6/h1-6H2",
    ]
    ret = inchimod.get_valid_inchi(inp)
    assert len(ret) == 4
    assert ret[0] == inp[0]
    assert ret[1] == inp[1]
    # Test should compare index 2 to index 3 as the invalid smiles is removed
    assert ret[2] == inp[3]
    assert ret[3] == inp[4]


def test_get_valid_inchi_mask_all_valid():
    """
    Test that we get the expected all smiles valid vector boolean mask
    """
    inp = ["InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H", "InChI=1S/C4H10/c1-3-4-2/h3-4H2,1-2H3"]
    ret = inchimod.get_valid_inchi_mask(inp)
    assert all(ent is True for ent in ret)


def test_get_valid_inchi_mask_some_invalid():
    """
    Test that we get the expected all but the last smiles valid vector boolean mask
    """
    inp = [
        "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H",
        "InChI=1S/C4H10/c1-3-4-2/h3-4H2,1-2H3",
        "ukn",
    ]
    ret = inchimod.get_valid_inchi_mask(inp)
    assert len(ret) == 3
    assert ret[0] is True
    assert ret[1] is True
    assert ret[2] is False


def test_get_valid_inchi_mask_order_some_invalid():
    """
    Test that invalid smiles are listed as False in the vector boolean mask
    """
    inp = [
        "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H",
        "InChI=1S/C4H10/c1-3-4-2/h3-4H2,1-2H3",
        "ukn",
        "InChI=1S/C4H4O/c1-2-4-5-3-1/h1-4H",
        "InChI=1S/C11H15ClN4O2/c1-3-15(11(13-2)8-16(17)18)7-9-4-5-10(12)14-6-9/h4-6,8,13H,3,7H2,1-2H3/b11-8+",
    ]
    ret = inchimod.get_valid_inchi_mask(inp)
    assert len(ret) == 5
    assert ret[0] is True
    assert ret[1] is True
    assert ret[2] is False
    assert ret[3] is True
    assert ret[4] is True
