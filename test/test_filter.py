#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module of unit tests for filtering
"""

import numpy as np
import pandas as pd
import pytest
from chemutil.filtering import (
    MoleculeFilter,
    conatains_pains,
    congreve_ro3,
    druglikeness,
    get_filter_properties,
    get_largest_ring_system,
    get_logp,
    get_molecular_mass,
    get_qed_common_versions,
    ghose_ro2,
    lipinski_ro5,
    qed_scores,
    qed_threshold,
    reos_ro7,
    synthetic_accessibility_score,
    tanimoto_bulk_similarity_metrics,
    tanimoto_similarity_metrics,
    tanimoto_single_similarity_metrics,
    veber_ro2,
)
from rdkit import Chem
from rdkit.Chem import Descriptors


def test_lipinski_ro5():
    mol = Chem.MolFromSmiles("CCO")
    assert lipinski_ro5(mol) == 1


def test_ghose():
    mol = Chem.MolFromSmiles("CCO")
    assert ghose_ro2(mol) == 0


def test_veber():
    mol = Chem.MolFromSmiles("CCO")
    assert veber_ro2(mol) == 1


def test_ro3():
    mol = Chem.MolFromSmiles("CCO")
    assert congreve_ro3(mol) == 1


def test_reos():
    mol = Chem.MolFromSmiles("CCO")
    assert reos_ro7(mol) == 0


def test_qed_threshold():
    mol = Chem.MolFromSmiles("CCO")
    assert qed_threshold(mol) == 0


def test_druglikeness():
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("C1CCCCC1")]
    expect = pd.DataFrame(
        {
            "ro5": {0: 1, 1: 1},
            "ghose": {0: 0, 1: 0},
            "veber": {0: 1, 1: 1},
            "ro3": {0: 1, 1: 1},
            "reos": {0: 0, 1: 0},
        }
    )
    result = druglikeness(mols)
    assert result.shape == (2, 5)
    pd.testing.assert_frame_equal(result, expect)


def test_synthetic_accessibility_score():
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("C1CCCCC1")]
    expect = np.array([1.98025704, 1.0])
    result = synthetic_accessibility_score(mols, return_numpy=True)
    assert result.shape == (2,)
    np.testing.assert_array_almost_equal(result, expect)


def test_get_qed_common_versions():
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("C1CCCCC1")]
    expect = pd.DataFrame(
        {
            "qed_max": {0: 0.43600348804972483, 1: 0.45780490695199694},
            "qed_mean": {0: 0.40680796565539457, 1: 0.42231618686094674},
            "qed_all_weights_one": {0: 0.39180198153728507, 1: 0.28704155492896427},
            "qed": {0: 0.40680796565539457, 1: 0.42231618686094674},
        }
    )
    result = get_qed_common_versions(mols)
    assert result.shape == (2, 4)
    pd.testing.assert_frame_equal(result, expect)


def test_conatains_pains():
    mols = [
        Chem.MolFromSmiles("CCO"),
        Chem.MolFromSmiles("C1CCCCC1"),
        Chem.MolFromSmiles("c1ccc(CN=NC)cc1"),
        Chem.MolFromSmiles("c1ccc(S2=CC=CC2(CN))cc1"),
    ]
    result = conatains_pains(mols, return_numpy=True)
    assert result.shape == (4,)
    np.testing.assert_array_equal(result, [False, False, True, True])


def test_qed_scores():
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("C1CCCCC1")]
    result = qed_scores(mols, return_numpy=True)
    assert result.shape == (2,)
    np.testing.assert_array_almost_equal(result, np.array([0.40680797, 0.42231619]))


def test_tanimoto_single_similarity_metrics():
    mols = [
        Chem.MolFromSmiles("CCO"),
        Chem.MolFromSmiles("C1CCCCC1"),
        Chem.MolFromSmiles("CCCCO"),
        Chem.MolFromSmiles("c1ccc(S2=CC=CC2(CN))cc1"),
    ]
    result = tanimoto_single_similarity_metrics(
        mols, ref_mol="CCCCO", return_numpy=True
    )
    assert result.shape == (4, 1)
    expect = np.array([[0.41666667], [0.0], [1.0], [0.02777778]])


def test_tanimoto_bulk_similarity_metrics():
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("C1CCCCC1")]
    similarity_set = pd.DataFrame({"smiles": ["CCO", "C1CCCCC1"]})
    expect = np.array(
        [[0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5]]
    )
    result = tanimoto_bulk_similarity_metrics(
        mols, similarity_set=similarity_set, return_numpy=True
    )
    assert result.shape == (2, 7)
    np.testing.assert_array_almost_equal(result, expect)


def test_get_largest_ring_system():
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("C1CCCCC1")]
    result = get_largest_ring_system(mols, return_numpy=True)
    assert result.shape == (2,)
    expect = np.array([0, 6])
    np.testing.assert_array_equal(result, expect)


def test_get_molecular_mass():
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("C1CCCCC1")]
    result = get_molecular_mass(mols, return_numpy=True)
    assert result.shape == (2,)


def test_get_logp():
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("C1CCCCC1")]
    result = get_logp(mols, return_numpy=True)
    assert result.shape == (2,)


def test_get_filter_properties():
    data = {"smiles": ["CCO", "C1CCCCC1"]}
    expect = pd.DataFrame(
        {
            "label": {0: "CCO", 1: "C1CCCCC1"},
            "synthetic_accessibility_score": {0: 1.9802570386349831, 1: 1.0},
            "ro5": {0: 1, 1: 1},
            "ghose": {0: 0, 1: 0},
            "veber": {0: 1, 1: 1},
            "ro3": {0: 1, 1: 1},
            "reos": {0: 0, 1: 0},
            "contains_pains": {0: False, 1: False},
            "molecular_mass": {0: 46.041864812, 1: 84.093900384},
            "largest_ring_system": {0: 0, 1: 6},
            "logp": {0: -0.0014000000000000123, 1: 2.3406000000000002},
            "qed_max": {0: 0.43600348804972483, 1: 0.45780490695199694},
            "qed_mean": {0: 0.40680796565539457, 1: 0.42231618686094674},
            "qed_all_weights_one": {0: 0.39180198153728507, 1: 0.28704155492896427},
            "qed": {0: 0.40680796565539457, 1: 0.42231618686094674},
        }
    )
    df = pd.DataFrame(data)
    result = get_filter_properties(df, representation_column="smiles")
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    np.testing.assert_array_almost_equal(
        result.select_dtypes(include=numerics).values,
        expect.select_dtypes(include=numerics).values,
    )


def test_molecule_filter():
    mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("C1CCCCC1")]
    expect = np.array(
        [
            [1.00000000e00, -1.40000000e-03, 4.60418648e01],
            [1.00000000e00, 2.34060000e00, 8.40939004e01],
        ]
    )
    funcs = [druglikeness, get_logp, get_molecular_mass]
    args = [{"ghose": False, "veber": False, "ro3": False, "reos": False}, {}, {}]
    mf = MoleculeFilter(funcs, mols, list_of_fx_arg_dicts=args)
    result = mf.filter_results
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    np.testing.assert_array_almost_equal(
        result.select_dtypes(include=numerics).values, expect
    )


# Run the tests
if __name__ == "__main__":
    pytest.main()
