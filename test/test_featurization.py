#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module of unit tests for featurization
"""

import logging

import numpy as np
import pandas as pd
import pytest
from chemutil import featurization
from rdkit.DataStructs import cDataStructs

log = logging.getLogger(__name__)


def test_list_of_bitvects_to_numpy_arrays():
    bitvect = cDataStructs.CreateFromBitString("1011")
    result = featurization.list_of_bitvects_to_numpy_arrays([bitvect])
    expected = np.array([[1, 0, 1, 1]], dtype=np.uint8)
    assert np.array_equal(result, expected)


def test_list_of_bitvects_to_list_of_lists():
    bitvect = cDataStructs.CreateFromBitString("1011")
    result = featurization.list_of_bitvects_to_list_of_lists([bitvect])
    expected = [[1, 0, 1, 1]]
    assert result == expected


def test_bitstring_to_bit_vect():
    bstring = "10101010001101"
    result = featurization.bitstring_to_bit_vect(bstring)
    assert isinstance(result, cDataStructs.ExplicitBitVect)


def test_df_rows_to_list_of_bit_vect():
    df = pd.DataFrame([[1, 0, 1, 0, 1, 1, 1, 1]])
    result = featurization.df_rows_to_list_of_bit_vect(df)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], cDataStructs.ExplicitBitVect)


def test_validate_smiles_and_get_ecfp_bitvect():
    result = featurization.validate_smiles_and_get_ecfp(
        smiles=["c1ccccc1C"], hash_length=1024
    )
    assert isinstance(result, list)
    assert isinstance(result[0], cDataStructs.ExplicitBitVect)


def test_validate_smiles_and_get_ecfp_numpy():
    result = featurization.validate_smiles_and_get_ecfp(
        smiles=["c1ccccc1C"], hash_length=1024, return_np=True
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1024)


def test_validate_smiles_and_get_ecfp_dataframe():
    out_df = featurization.validate_smiles_and_get_ecfp(
        smiles=["c1ccccc1C"], hash_length=1024, return_df=True
    )
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.columns[0] == "ecfp_bit_0"
    assert int(out_df.loc[0, "ecfp_bit_0"]) == 0
    assert int(out_df.loc[0, "ecfp_bit_175"]) == 1


def test_validate_smiles_and_get_ecfp_dataframe_with_input_df():
    data_df = pd.DataFrame([["toluene", "c1ccccc1C"]], columns=["id", "smiles"])
    out_df = featurization.validate_smiles_and_get_ecfp(
        data_df=data_df, smiles_column="smiles", hash_length=1024, return_df=True
    )
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.columns[0] == "id"
    assert int(out_df.loc[0, "ecfp_bit_0"]) == 0
    assert int(out_df.loc[0, "ecfp_bit_175"]) == 1


def test_get_ecfp_bitvect():
    result = featurization.get_ecfp(smiles=["c1ccccc1C"], hash_length=1024)
    assert isinstance(result, tuple)
    assert isinstance(result[0], cDataStructs.ExplicitBitVect)


def test_get_ecfp_numpy():
    result = featurization.get_ecfp(
        smiles=["c1ccccc1C"], hash_length=1024, return_np=True
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1024)


def test_get_ecfp_dataframe():
    out_df = featurization.get_ecfp(
        smiles=["c1ccccc1C"], hash_length=1024, return_df=True
    )
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.columns[0] == "ecfp_bit_0"
    assert int(out_df.loc[0, "ecfp_bit_0"]) == 0
    assert int(out_df.loc[0, "ecfp_bit_175"]) == 1


def test_get_ecfp_dataframe_with_input_df():
    data_df = pd.DataFrame([["toluene", "c1ccccc1C"]], columns=["id", "smiles"])
    out_df = featurization.get_ecfp(
        data_df=data_df, smiles_column="smiles", hash_length=1024, return_df=True
    )
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.columns[0] == "id"
    assert int(out_df.loc[0, "ecfp_bit_0"]) == 0
    assert int(out_df.loc[0, "ecfp_bit_175"]) == 1


def test_get_count_ecfp_bitvect():
    result = featurization.get_count_ecfp(smiles=["c1ccccc1C"], hash_length=1024)
    assert isinstance(result, tuple)
    assert isinstance(result[0], cDataStructs.UIntSparseIntVect)


def test_get_count_ecfp_numpy():
    result = featurization.get_count_ecfp(
        smiles=["c1ccccc1C"], hash_length=1024, return_np=True
    )
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1024)


def test_get_count_ecfp_dataframe():
    out_df = featurization.get_count_ecfp(
        smiles=["c1ccccc1C"], hash_length=1024, return_df=True
    )
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.columns[0] == "ecfp_count_bit_0"
    assert int(out_df.loc[0, "ecfp_count_bit_0"]) == 0
    assert int(out_df.loc[0, "ecfp_count_bit_175"]) == 2


def test_get_count_ecfp_dataframe_with_input_df():
    data_df = pd.DataFrame([["toluene", "c1ccccc1C"]], columns=["id", "smiles"])
    out_df = featurization.get_count_ecfp(
        data_df=data_df, smiles_column="smiles", hash_length=1024, return_df=True
    )
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.columns[0] == "id"
    assert int(out_df.loc[0, "ecfp_count_bit_0"]) == 0
    assert int(out_df.loc[0, "ecfp_count_bit_175"]) == 2


def test_get_maccs_bitvect():
    result = featurization.get_maccs(smiles=["c1ccccc1C"])
    assert isinstance(result, list)
    assert isinstance(result[0], cDataStructs.ExplicitBitVect)


def test_get_maccs_bitvect_on_bits():
    vecs = featurization.get_maccs(smiles=["c1ccccc1C"])  ##### CHECK THIS ONE
    assert tuple(vecs[0].GetOnBits()) == (160, 162, 163, 165)


def test_get_maccs_dataframe():
    out_df = featurization.get_maccs(smiles=["c1ccccc1C"], return_df=True)
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.columns[0] == "maccs_bit_0"
    assert int(out_df.loc[0, "maccs_bit_0"]) == 0
    assert int(out_df.loc[0, "maccs_bit_163"]) == 1


def test_get_maccs_dataframe_with_input_df():
    data_df = pd.DataFrame([["toluene", "c1ccccc1C"]], columns=["id", "smiles"])
    out_df = featurization.get_maccs(
        data_df=data_df, smiles_column="smiles", return_df=True
    )
    assert isinstance(out_df, pd.DataFrame)
    assert out_df.columns[0] == "id"
    assert int(out_df.loc[0, "maccs_bit_0"]) == 0
    assert int(out_df.loc[0, "maccs_bit_163"]) == 1


def test_clean_mordred_features():
    # Sample data with numeric and non-numeric columns
    data = {
        "numeric1": [1.0, 2.0, 3.0],
        "numeric2": [4.0, 5.0, 6.0],
        "non_numeric": ["a", "b", "c"],
    }
    df = pd.DataFrame(data)

    # Call the function
    result = featurization.clean_mordred_features(df)

    # Expected result
    expected = df[["numeric1", "numeric2"]]

    # Assertions
    pd.testing.assert_frame_equal(result, expected)


def test_get_mordred_descriptors():
    # Sample data
    data = {"smiles": ["CCO", "C1CCCCC1", "CC(=O)O"]}
    df = pd.DataFrame(data)

    expected_spabs_a = np.array([2.828427, 8.000000, 3.464102])

    # Call the function with DataFrame input
    result_df = featurization.get_mordred_descriptors(
        data_df=df, smiles_column="smiles", return_np=False
    )
    log.critical(result_df)
    assert np.allclose(result_df["SpAbs_A"].values, expected_spabs_a)
    assert isinstance(result_df, pd.DataFrame)
    assert not result_df.empty

    # Call the function with list of SMILES input
    result_np = featurization.get_mordred_descriptors(
        smiles=["CCO", "C1CCCCC1", "CC(=O)O"], return_np=True
    )
    assert isinstance(result_np, np.ndarray)
    assert result_np.shape[0] == 3

    # Check that the function raises an error when no input is provided
    with pytest.raises(RuntimeError):
        featurization.get_mordred_descriptors()

    data = {"smiles": ["CCO", "C1CCCCC1", "CC(=O)O", "not_a_smiles_string"]}
    df = pd.DataFrame(data)
    result_df = featurization.get_mordred_descriptors(
        data_df=df, smiles_column="smiles", return_np=False
    )
    assert isinstance(result_df, pd.DataFrame)
    assert not result_df.empty
    assert result_df.shape[0] == 3


def test_get_rdkit_descriptors_dict():
    des = featurization.get_rdkit_descriptors(smiles=["c1ccccc1C"])
    assert isinstance(des[0], dict)


def test_get_rdkit_descriptors_dataframe():
    # TODO: version of this with autocorrelation features
    out_df = featurization.get_rdkit_descriptors(
        smiles=["c1ccccc1C"], include_autocorrelation_features=False, return_df=True
    )
    log.error("\n".join(out_df.columns))
    assert len(out_df.columns) == 217
    assert round(float(out_df.loc[0, "rdkit_descriptor_Chi4n"]), 4) == 0.5344
    assert round(float(out_df.loc[0, "rdkit_descriptor_BCUT2D_MWLOW"]), 4) == 10.2446


def test_prepare_blm_selfies_input():
    # Sample data
    data = {
        "selfies": [
            "[C][C][=O]",
            "[C][C][C][C]",
            "[O][=C][C][C]",
            "[C][=C][C][=C][C][=C][Ring1][=Branch1][C][C][N][C][=Branch1][C][=O][C][C][C][Ring1][Ring1]",
        ]
    }
    df = pd.DataFrame(data)

    # Test case 1: Return list of SELFIES strings
    result = featurization.prepare_blm_selfies_input(df=df, selfies_column="selfies")
    expected = [
        "[C] [C] [=O]",
        "[C] [C] [C] [C]",
        "[O] [=C] [C] [C]",
        "[C] [=C] [C] [=C] [C] [=C] [Ring1] [=Branch1] [C] [C] [N] [C] [=Branch1] [C] [=O] [C] [C] [C] [Ring1] [Ring1]",
    ]
    assert result == expected

    # Test case 2: Write to file and return filename
    output_filename = "test_selfies.txt"
    result = featurization.prepare_blm_selfies_input(
        df=df, selfies_column="selfies", output_filename=output_filename
    )
    assert result == output_filename

    with open(output_filename, "r") as f:
        file_content = f.read().splitlines()
    assert file_content == expected

    # Clean up
    import os

    os.remove(output_filename)


def test_prepare_blm_smiles_input():
    # Sample data
    data = {
        "smiles": [
            "CCO",
            "C1CCCCC1",
            "CC(=O)O.OCC",
            "C1=CC=CC=C1Cl",
            "O[C@@H](N)Cc1ccccc1",
            "[C@H](O)(N)C.[Na+]",
        ]
    }
    df = pd.DataFrame(data)

    # Test case 1: Return list of SMILES strings
    result = featurization.prepare_blm_smiles_input(df=df, smiles_column="smiles")
    expect = [
        "C C O",
        "C 1 C C C C C 1",
        "C C ( = O ) O . O C C",
        "C 1 = C C = C C = C 1 Cl",
        "O [C@@H] ( N ) C c 1 c c c c c 1",
        "[C@H] ( O ) ( N ) C . [Na+]",
    ]

    assert result == expect

    # Test case 2: Write to file and return filename
    output_filename = "test_smiles.txt"
    result = featurization.prepare_blm_smiles_input(
        df=df, smiles_column="smiles", output_filename=output_filename
    )

    with open(output_filename, "r") as f:
        file_content = f.read().splitlines()
    assert file_content == expect

    # Clean up
    import os

    os.remove(output_filename)


# Run the tests
if __name__ == "__main__":
    pytest.main()
