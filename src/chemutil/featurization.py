#!/usr/bin.env python3
# -*- coding: utf-8 -*-

"""
Module for generating features such as chemical fingerprints and descriptors and the converison of common data types of those representations
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import deepchem as dc
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.Descriptors import CalcMolDescriptors, setupAUTOCorrDescriptors
from rdkit.DataStructs import cDataStructs
from sentence_transformers import SentenceTransformer

from chemutil import smilesmod

try:
    from chemutil.rdkit_descriptor_types import (
        categorical_rdkit_descriptor_names,
        continuous_rdkit_descriptor_names,
        semi_categorical_rdkit_descriptor_names,
    )
except ImportError:
    continuous_rdkit_descriptor_names = []
    categorical_rdkit_descriptor_names = []
    semi_categorical_rdkit_descriptor_names = []

import logging

# from simpletransformers.language_representation import RepresentationModel

log = logging.getLogger(__name__)

try:
    from mordred import Calculator, descriptors

    def clean_mordred_features(mordred_features: pd.DataFrame) -> pd.DataFrame:
        """
        Function to clean the Mordred features dataframe by removing columns that are not numeric

        Args:
            mordred_features (pd.DataFrame): The Mordred features dataframe
        Returns:
            pd.DataFrame: The cleaned Mordred features dataframe
        """
        numeric_mfeat = mordred_features.copy()
        del_cols = [
            col
            for col in mordred_features.columns
            if pd.to_numeric(mordred_features[col], errors="coerce").notnull().all()
            == 0
        ]
        numeric_mfeat = numeric_mfeat.drop(del_cols, axis=1)

        return numeric_mfeat

    def get_mordred_descriptors(
        data_df: Optional[pd.DataFrame] = None,
        smiles_column: Optional[str] = None,
        smiles: Optional[List[str]] = None,
        return_np: bool = False,
        threed: bool = True,
        return_unparsable: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Function to generate Mordred descriptors from smiles

        Args:
            data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.
            If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.
            smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.
            smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.
            return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defaults to False.

        Raises:
            RuntimeError: If incompatible inputs are given

        Returns:
            Union[pd.DataFrame, np.ndarray]: Depends on the return type asked for
        """

        if smiles is None:
            if all(ent is not None for ent in [data_df, smiles_column]):
                df = data_df.copy()
                smiles = df[smiles_column].to_list()
            else:
                raise RuntimeError(
                    "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"
                )
        # convert to molecules and warn if any are not parsable
        mols = [smilesmod.smiles_to_molecule(smi) for smi in smiles]
        tmp_df = pd.DataFrame(
            [[s, m] for s, m in zip(smiles, mols)], columns=["smiles", "molecules"]
        )
        log.info(
            f"The following smiles were not parsable: {f'{os.linesep}'.join(tmp_df[tmp_df['molecules'].isnull()]['smiles'].to_list())}"
        )
        if return_unparsable is True:
            unparsable = tmp_df[tmp_df["molecules"].isnull()].copy()

        # get the parsable molecules for descriptor calculations
        mols = tmp_df.dropna(axis=0, subset=["molecules"])["molecules"].to_list()

        # make the descriptors and clean so only the numeric ones are return using Mordred
        mord_calc = Calculator(descriptors, ignore_3D=threed)
        log.debug(f"Calculating Mordred descriptors for {len(smiles)} molecules")
        log.debug(f"Descriptor names: {mord_calc.descriptors}")
        mordred_features = mord_calc.pandas(mols)
        mordred_features = clean_mordred_features(mordred_features)

        # return the descriptors
        if return_np is True:
            if return_unparsable is True:
                return mordred_features.values, unparsable
            else:
                return mordred_features.values
        else:
            if return_unparsable is True:
                return mordred_features, unparsable
            else:
                return mordred_features

except ImportError:
    log.info("Mordred not installed")


def prepare_blm_selfies_input(
    df: Optional[pd.DataFrame] = None,
    selfies_column: Optional[str] = None,
    selfies: Optional[List[str]] = None,
    output_filename: Optional[str] = None,
    additional_pre_space_chars: Optional[List[str]] = None,
) -> Union[str, List[str]]:
    """
    Function to prepare the SELFIES for input to a blank language model
    Args:
        df (pd.DataFrame): The dataframe containing the SELFIES strings
        selfies_column (str): The column name in the dataframe containing the SELFIES strings
        output_filename (Optional[str], optional): The filename to write the SELFIES to. Defaults to None.
        additional_pre_space_chars (Optional[List[str]], optional): Additional characters to add a space afterwards. Defaults to None.
    Returns:
        Union[str, List[str]]: Either the filename or the list of SELFIES strings
    """

    # Get the selfies strings from the dataframe
    if selfies is None:
        if all(ent is not None for ent in [df, selfies_column]):
            selfies = df[selfies_column].to_list()
        else:
            raise RuntimeError(
                "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"
            )

    if additional_pre_space_chars is not None:
        pre_space_chars = ["]"] + additional_pre_space_chars
    else:
        pre_space_chars = ["]"]

    blm_selfies = []
    for ent in df[selfies_column].values:
        new = []
        for s in ent:
            if s in pre_space_chars:
                log.debug(s)
                new.append(f"{s} ")
            else:
                log.debug(s)
                new.append(s)
        blm_selfies.append("".join(new).strip())

    if output_filename is not None:
        with open(output_filename, "w") as fout:
            for ent in blm_selfies:
                fout.write(ent + "\n")
        return output_filename
    else:
        return blm_selfies


def prepare_blm_smiles_input(
    df: Optional[pd.DataFrame] = None,
    smiles_column: Optional[str] = None,
    smiles: Optional[List[str]] = None,
    output_filename: Optional[str] = None,
) -> Union[str, List[str]]:
    """
    Split a SMILES string so there is a single space between atoms, bonds, rings, and branch syntax.

    Args:
        smiles (str): The SMILES string to split.

    Returns:
        str: The split SMILES string.
    """

    # Get the smiles strings from the dataframe
    if smiles is None:
        if all(ent is not None for ent in [df, smiles_column]):
            smiles = df[smiles_column].to_list()
        else:
            raise RuntimeError(
                "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"
            )

    # Regular expression to match atoms, bonds, rings, and branch syntax
    pattern = (
        r"(\[.*?\]|Br|Cl|Na|Li|Be|Mg|Ca|Fe|Zn|[BCNOFPSIbcnops]|@{1,2}|=|#|\(|\)|\.|\d+)"
    )
    split_smiles = []
    # Split the SMILES string using the pattern and join with a single space
    for smile in smiles:
        split_smiles.append(" ".join(re.findall(pattern, smile)))

    if output_filename is not None:
        with open(output_filename, "w") as fout:
            for ent in split_smiles:
                fout.write(ent + "\n")
        return output_filename
    else:
        return split_smiles


def set_rdkit_descriptor_types() -> None:
    """
    Function to set the rdkit descriptor types (continuous, semi-continuous and categorical) for a standard data set (BACE).
    used in the rdkit descriptor definitions. This function is used to generate the rdkit_descriptor_types.py file which is stored
    and loaded by this library module for use in the featurization.

    Returns:
        None
    """

    # load a standard data set and make the rdkit descriptors
    _ = dc.molnet.load_bace_classification(seed=0)
    data = pd.read_csv(Path(dc.utils.data_utils.get_data_dir()).joinpath("bace.csv"))
    standard_set_for_rdkit_descriptor_definitions = get_rdkit_descriptors(
        smiles=data["mol"].values, return_df=True
    )

    assert (
        len(
            standard_set_for_rdkit_descriptor_definitions.select_dtypes(
                "object"
            ).columns
        )
        == 0
    ), "As expected there are object columns in the standard set for rdkit descriptor definitions "

    # get the categorical and continuous descriptors
    semi_categorical_rdkit_descriptor_names_for_file = []
    continuous_rdkit_descriptor_names_for_file = []
    categorical_rdkit_descriptor_names_for_file = [
        "_".join(ent.split("_")[2:])
        for ent in standard_set_for_rdkit_descriptor_definitions.select_dtypes(
            "int"
        ).columns
    ]

    floating_descriptors = standard_set_for_rdkit_descriptor_definitions.select_dtypes(
        "float"
    )
    for d in floating_descriptors.columns:
        if len(set(np.round(floating_descriptors[d], 4))) <= 0.75 * len(
            floating_descriptors[d]
        ):
            semi_categorical_rdkit_descriptor_names_for_file.append(
                "_".join(d.split("_")[2:])
            )
        else:
            continuous_rdkit_descriptor_names_for_file.append(
                "_".join(d.split("_")[2:])
            )

    log.info(
        f"There are {len(categorical_rdkit_descriptor_names_for_file)} categorical descriptors {len(semi_categorical_rdkit_descriptor_names_for_file)} semi-categorical descriptors and {len(continuous_rdkit_descriptor_names_for_file)} continuous descriptors"
    )
    p = Path(__file__).parent.resolve()
    log.info(
        f"writing rdkit descriptor types to file {p.joinpath('rdkit_descriptor_types.py')}"
    )
    with open(p.joinpath("rdkit_descriptor_types.py"), "w") as fout:
        fout.write("#!/usr/bin.env python3\n")
        fout.write("# -*- coding: utf-8 -*-\n")
        fout.write("'''\nA module to store descriptor names categorised by type\n'''\n")
        fout.write(
            f"continuous_rdkit_descriptor_names = {continuous_rdkit_descriptor_names_for_file}\n"
        )
        fout.write(
            f"categorical_rdkit_descriptor_names = {categorical_rdkit_descriptor_names_for_file}\n"
        )
        fout.write(
            f"semi_categorical_rdkit_descriptor_names = {semi_categorical_rdkit_descriptor_names_for_file}\n"
        )


def list_of_bitvects_to_numpy_arrays(
    bvs: List[cDataStructs.ExplicitBitVect],
) -> np.ndarray:
    """
    Function to convert list of explicit bitvectirs from RDKit to numpy arrays. Note that at the time of writing RDKit has functions to do this one at a time but not in batches.

    Args:
        bvs (List[cDataStructs.ExplicitBitVect]): List of bitvects from RDKit

    Returns:
        np.ndarray: Numpy array of the bit vector arrays

    Doctest:
    > list_of_bitvects_to_numpy_arrays([cDataStructs.CreateFromBitString("1011")]) # doctest: +NORMALIZE_WHITESPACE
    array([[1, 0, 1, 1]], dtype=uint8)
    """

    return np.array(
        [[int(ent) for ent in list(ent.ToBitString())] for ent in bvs]
    ).astype("uint8")


def list_of_bitvects_to_list_of_lists(
    bvs: List[cDataStructs.ExplicitBitVect],
) -> List[List[int]]:
    """
    Function to convert list of explicit bitvects from RDKit to list of lists. Note that at the time of writing RDKit has functions to do this one at a time but not in batches.

    Args:
        bvs (List[cDataStructs.ExplicitBitVect]): List of bitvects from RDKit

    Returns:
       List[List[int]]: list of lists of integer binary values

    Doctest:
    > list_of_bitvects_to_list_of_lists([cDataStructs.CreateFromBitString("1011")]) # doctest: +NORMALIZE_WHITESPACE
    [[1, 0, 1, 1]]
    """

    return [[int(ent) for ent in list(ent.ToBitString())] for ent in bvs]


def bitstring_to_bit_vect(bstring: str) -> cDataStructs.ExplicitBitVect:
    """
    Function to convert a bit string i.e. "100010101" to an RDKit explicit bit vector

    Args:
        bstring (str): bit string i.e. a string made up of 1 and 0 only

    Returns:
        cDataStructs.ExplicitBitVect: RDKit explicit bit vector

    Doctest:
    > bitstring_to_bit_vect('10101010001101') # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    <rdkit.DataStructs.cDataStructs.ExplicitBitVect object at ...>
    """
    return cDataStructs.CreateFromBitString(bstring)


def df_rows_to_list_of_bit_vect(df: pd.DataFrame) -> List[cDataStructs.ExplicitBitVect]:
    """
    _summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        List[cDataStructs.ExplicitBitVect]: _description_

    Doctest:
    > df_rows_to_list_of_bit_vect(pd.DataFrame([[1, 0, 1, 0, 1, 1, 1, 1]])) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    [<rdkit.DataStructs.cDataStructs.ExplicitBitVect object at ...>]
    """

    bitvectors = []
    for _, row in df.iterrows():
        log.debug(row)
        bs = "".join([str(ent) for ent in row.to_list()])
        log.debug(bs)
        bitvectors.append(bitstring_to_bit_vect(bs))

    return bitvectors


def validate_smiles_and_get_ecfp(
    data_df: Optional[pd.DataFrame] = None,
    smiles_column: Optional[str] = None,
    smiles: Optional[List[str]] = None,
    radius: int = 2,
    hash_length: int = 2048,
    return_df: bool = False,
    return_np: bool = False,
    **kwargs,
) -> Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]:
    """
    Function to generate ECFP representations from smiles

    Args:
        data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.
         If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.
        smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.
        smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.
        radius (int, optional): ECFP/Morgan radius, NOTE: ECFPX the X is the diameter i.e. radius*2 therefore ECFP4 means setting this value to 2. Defaults to 2.
        hash_length (int, optional): The length in number of vector elements of the fingerprint. Defaults to 2048.
        return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.
        return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defaults to False.

    Raises:
        RuntimeError: If incompatible inputs are given

    Returns:
        Union[List[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]: Depends on the return type asked for

    Example:
    ```python
    > validate_smiles_and_get_ecfp(smiles=["c1ccccc1C"], hash_length=1024) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    [<rdkit.DataStructs.cDataStructs.ExplicitBitVect object at ...>]

    > validate_smiles_and_get_ecfp(smiles=["c1ccccc1C"], hash_length=1024, return_np=True) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([[0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
    ```
    """

    if sum([return_df, return_np]) > 1:
        raise RuntimeError(
            f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."
        )

    if smiles is None:
        if all(ent is not None for ent in [data_df, smiles_column]):
            df = data_df.copy()
            input_n = len(df.index)
            smiles = smilesmod.validate_smiles(df[smiles_column].to_list())
            df["ECFP_smiles_standardized"] = smiles
            df = df.dropna(axis=0, subset=["ECFP_smiles_standardized"])
            smiles = df["ECFP_smiles_standardized"].to_list()
            log.info(
                f"{(len(smiles) / input_n) * 100.0}% ({len(smiles)} out of {input_n}) of the input smiles were successfully read"
            )
        else:
            raise RuntimeError(
                "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"
            )
    else:
        input_n = len(smiles)
        df = data_df
        smiles = smilesmod.validate_smiles(smiles, return_failed_as_None=False)
        log.info(
            f"{(len(smiles) / input_n) * 100.0}% of the input smiles were successfully read"
        )

    fp_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=hash_length, **kwargs)

    if return_np is True and return_df is False:
        fps_np = fps = np.array(
            [
                fp_gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi))
                if smi is not None
                else None
                for smi in smiles
            ]
        )

        return fps_np

    elif return_df is True and return_np is False:
        fps_df = pd.DataFrame(
            [
                fp_gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi))
                if smi is not None
                else None
                for smi in smiles
            ],
            columns=[f"ecfp_bit_{ith}" for ith in range(hash_length)],
        )

        if df is not None:
            return pd.concat([df, fps_df], axis=1)
        else:
            return fps_df
    else:
        fps = [
            fp_gen.GetFingerprint(Chem.MolFromSmiles(smi)) if smi is not None else None
            for smi in smiles
        ]
        return fps


def get_ecfp(
    data_df: Optional[pd.DataFrame] = None,
    smiles_column: Optional[str] = None,
    smiles: Optional[List[str]] = None,
    radius: int = 2,
    hash_length: int = 2048,
    return_df: bool = False,
    return_np: bool = False,
    n_threads: int = 8,
    **kwargs,
) -> Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]:
    """
    Function to generate ECFP representations from smiles

    Args:
        data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.
         If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.
        smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.
        smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.
        radius (int, optional): ECFP/Morgan radius, NOTE: ECFPX the X is the diameter i.e. radius*2 therefore ECFP4 means setting this value to 2. Defaults to 2.
        hash_length (int, optional): The length in number of vector elements of the fingerprint. Defaults to 2048.
        return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.
        return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defaults to False.

    Raises:
        RuntimeError: If incompatible inputs are given

    Returns:
        Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]: Depends on the return type asked for

    Examples:
    ```python
    > get_ecfp(smiles=["c1ccccc1C"], hash_length=1024) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    (<rdkit.DataStructs.cDataStructs.ExplicitBitVect object at ...>,)

    > get_ecfp(smiles=["c1ccccc1C"], hash_length=1024, return_np=True) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    array([[0, 0, 0, ..., 0, 0, 0]], dtype=uint8)
    ```
    """

    if sum([return_df, return_np]) > 1:
        raise RuntimeError(
            f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."
        )

    if smiles is None:
        if all(ent is not None for ent in [data_df, smiles_column]):
            df = data_df.copy()
            smiles = df[smiles_column].to_list()
        else:
            raise RuntimeError(
                "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"
            )
    else:
        df = data_df

    log.info(f"Making ECFP fingerprints for {len(smiles)} molecules")
    fp_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=hash_length, **kwargs)

    if return_np is True and return_df is False:
        fps = fp_gen.GetFingerprints(
            [Chem.MolFromSmiles(smi) for smi in smiles], numThreads=n_threads
        )
        fps_np = list_of_bitvects_to_numpy_arrays(fps)
        return fps_np

    elif return_df is True and return_np is False:
        fps = fp_gen.GetFingerprints(
            [Chem.MolFromSmiles(smi) for smi in smiles], numThreads=n_threads
        )
        fps_ll = list_of_bitvects_to_list_of_lists(fps)
        fps_df = pd.DataFrame(
            fps_ll,
            columns=[f"ecfp_bit_{ith}" for ith in range(hash_length)],
        )

        if df is not None:
            return pd.concat([df, fps_df], axis=1)
        else:
            return fps_df
    else:
        fps = fp_gen.GetFingerprints(
            [Chem.MolFromSmiles(smi) for smi in smiles], numThreads=n_threads
        )
        return fps


def get_count_ecfp(
    data_df: Optional[pd.DataFrame] = None,
    smiles_column: Optional[str] = None,
    smiles: Optional[List[str]] = None,
    radius: int = 2,
    hash_length: int = 2048,
    return_df: bool = False,
    return_np: bool = False,
    n_threads: int = 8,
    **kwargs,
) -> Union[Tuple[cDataStructs.UIntSparseIntVect], pd.DataFrame, np.ndarray]:
    """
     Function to generate count ECFP representations from smiles

     Args:
         data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.
          If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.
         smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.
         smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.
         radius (int, optional): ECFP/Morgan radius, NOTE: ECFPX the X is the diameter i.e. radius*2 therefore ECFP4 means setting this value to 2. Defaults to 2.
         hash_length (int, optional): The length in number of vector elements of the fingerprint. Defaults to 2048.
         return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.
         return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defaults to False.

     Raises:
         RuntimeError: If incompatible inputs are given

     Returns:
         Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]: Depends on the return type asked for

    Examples:
    ```python
     > get_count_ecfp(smiles=["c1ccccc1C"], hash_length=1024) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
     (<rdkit.DataStructs.cDataStructs.UIntSparseIntVect object at ...>,)

     > get_count_ecfp(smiles=["c1ccccc1C"], hash_length=1024, return_np=True) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
     array([[0, 0, 0, ..., 0, 0, 0]])
     ```
    """

    if sum([return_df, return_np]) > 1:
        raise RuntimeError(
            f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."
        )

    if smiles is None:
        if all(ent is not None for ent in [data_df, smiles_column]):
            df = data_df.copy()
            smiles = df[smiles_column].to_list()
        else:
            raise RuntimeError(
                "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"
            )
    else:
        df = data_df

    log.info(f"Making CECFP fingerprints for {len(smiles)} molecules")
    fp_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=hash_length, **kwargs)

    if return_np is True and return_df is False:
        fps = fp_gen.GetCountFingerprints(
            [Chem.MolFromSmiles(smi) for smi in smiles], numThreads=n_threads
        )
        fps_np = np.array([ent.ToList() for ent in fps])
        return fps_np

    elif return_df is True and return_np is False:
        fps = fp_gen.GetCountFingerprints(
            [Chem.MolFromSmiles(smi) for smi in smiles], numThreads=n_threads
        )
        fps_ll = [ent.ToList() for ent in fps]
        fps_df = pd.DataFrame(
            fps_ll,
            columns=[f"ecfp_count_bit_{ith}" for ith in range(hash_length)],
        )

        if df is not None:
            return pd.concat([df, fps_df], axis=1)
        else:
            return fps_df
    else:
        fps = fp_gen.GetCountFingerprints(
            [Chem.MolFromSmiles(smi) for smi in smiles], numThreads=n_threads
        )
        return fps


def get_maccs(
    data_df: Optional[pd.DataFrame] = None,
    smiles_column: Optional[str] = None,
    smiles: Optional[List[str]] = None,
    return_df: bool = False,
    return_np: bool = False,
    **kwargs,
) -> Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]:
    """
    Function to generate MACCS MDL keys representations from smiles

    Args:
        data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.
         If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.
        smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.
        smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.
        return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.
        return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defaults to False.

    Raises:
        RuntimeError: If incompatible inputs are given

    Returns:
        Union[Tuple[cDataStructs.ExplicitBitVect], pd.DataFrame, np.ndarray]: Depends on the return type asked for

    Doctest:
    ```python
    > get_maccs(smiles=["c1ccccc1C"]) # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    [<rdkit.DataStructs.cDataStructs.ExplicitBitVect object at ...>]

    > get_maccs(smiles=["c1ccccc1C"])[0].GetOnBits()
    (160, 162, 163, 165)
    ```
    """

    if sum([return_df, return_np]) > 1:
        raise RuntimeError(
            f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."
        )

    if smiles is None:
        if all(ent is not None for ent in [data_df, smiles_column]):
            df = data_df.copy()
            smiles = df[smiles_column].to_list()
        else:
            raise RuntimeError(
                "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"
            )
    else:
        df = data_df

    log.info(f"Making MACCS fingerprints for {len(smiles)} molecules")
    fps = [
        MACCSkeys.GenMACCSKeys(ent)
        for ent in [Chem.MolFromSmiles(smi) for smi in smiles]
    ]
    num_maccs_keys = len(fps[0])

    if return_np is True and return_df is False:
        fps_np = np.array([ent.ToList() for ent in fps])
        return fps_np

    elif return_df is True and return_np is False:
        fps_ll = [ent.ToList() for ent in fps]
        fps_df = pd.DataFrame(
            fps_ll,
            columns=[f"maccs_bit_{ith}" for ith in range(num_maccs_keys)],
        )

        if df is not None:
            return pd.concat([df, fps_df], axis=1)
        else:
            return fps_df
    else:
        return fps


def get_rdkit_descriptors(
    data_df: Optional[pd.DataFrame] = None,
    smiles_column: Optional[str] = None,
    smiles: Optional[List[str]] = None,
    return_df: bool = False,
    return_np: bool = False,
    include_autocorrelation_features: bool = False,
    **kwargs,
) -> Union[List[dict], pd.DataFrame, np.ndarray]:
    """
    Function to generate Rdkit descriptor representations from smiles

    Args:
        data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.
         If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.
        smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.
        smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.
        return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.
        return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defaults to False.

    Raises:
        RuntimeError: If incompatible inputs are given

    Returns:
        Union[List[dict], pd.DataFrame, np.ndarray]: Depends on the return type asked for

    Example:
    ```python
    > type(get_rdkit_descriptors(smiles=["c1ccccc1C"])[0])
    <class 'dict'>

    > len(get_rdkit_descriptors(smiles=["c1ccccc1C"], return_df=True).columns)
    210
    ```
    """

    if include_autocorrelation_features is True:
        setupAUTOCorrDescriptors()

    if sum([return_df, return_np]) > 1:
        raise RuntimeError(
            f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."
        )

    if smiles is None:
        if all(ent is not None for ent in [data_df, smiles_column]):
            df = data_df.copy()
            smiles = df[smiles_column].to_list()
        else:
            raise RuntimeError(
                "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"
            )
    else:
        df = data_df

    log.info(f"Making RDKit descriptors for {len(smiles)} molecules")

    fps = [
        CalcMolDescriptors(ent) for ent in [Chem.MolFromSmiles(smi) for smi in smiles]
    ]
    # make the order consistent sort on the keys in the dictionary important for the numpy return
    fps = [dict(sorted(ent.items())) for ent in fps]

    if return_np is True and return_df is False:
        fps_np = pd.DataFrame(fps).values
        return fps_np

    elif return_df is True and return_np is False:
        fps_df = pd.DataFrame(fps)
        fps_df.columns = [f"rdkit_descriptor_{name}" for name in fps_df.columns]

        if df is not None:
            return pd.concat([df, fps_df], axis=1)
        else:
            return fps_df
    else:
        return fps


def get_rdkit_descriptor_description(descriptor_name: str) -> str:
    """
    Function to get the description of an RDKit descriptor
    Args:
        descriptor_name (str): The name of the descriptor

    Returns:
        str: The description of the descriptor
    """

    # remove the rdkit_descriptor_ prefix if it is present
    if descriptor_name.startswith("rdkit_descriptor_"):
        descriptor_name = "_".join(descriptor_name.split("_")[2:])

    # get a list of the RDkit descriptors names in the order they are in the descriptor list
    descriptor_name_list = [ent[0] for ent in Descriptors._descList]

    # get the description of the descriptor from the __doc__ string for the descriptor
    descriptor_description = Descriptors._descList[
        descriptor_name_list.index(descriptor_name)
    ][1].__doc__

    return descriptor_description


def get_specific_set_of_rdkit_descriptors(
    mol: Chem.Mol, descriptor_names: List[str], missing=None
) -> dict:
    """
    Function to calculate a specific set of RDKit descriptors for a molecule. Based on the RDKit function licensed under BSD-3 clause
    https://github.com/rdkit/rdkit/blob/64061b6ca71121f7c3837393ff0a35f6261596a9/rdkit/Chem/Descriptors.py#L289

    Args:
        mol (Chem.Mol): The RDKit molecule object
        descriptor_names (List[str]): The list of descriptor names to calculate
        missing (Any, optional): The value to return if the descriptor calculation fails. Defaults to None.
    Returns:
        dict: A dictionary of the descriptor names and values
    """

    descriptors = {}
    name_to_index_list = [ent[0] for ent in Descriptors._descList]
    tmp_descList = [name_to_index_list.index(ent) for ent in descriptor_names]
    for name, fx in tmp_descList:
        try:
            descriptor_value = fx(mol)
        except Exception as e:
            log.error(
                f"Error calculating descriptor {name} for molecule {Chem.MolToSmiles(mol)}: {e}"
            )
            descriptor_value = missing
        descriptors[name] = descriptor_value

    return descriptors


def get_continuous_rdkit_descriptors(
    data_df: Optional[pd.DataFrame] = None,
    smiles_column: Optional[str] = None,
    smiles: Optional[List[str]] = None,
    return_df: bool = False,
    return_np: bool = False,
    include_autocorrelation_features: bool = False,
    **kwargs,
) -> Union[List[dict], pd.DataFrame, np.ndarray]:
    """
    Function to generate Rdkit descriptor representations from smiles

    Args:
        data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.
         If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.
        smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.
        smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.
        return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.
        return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defaults to False.

    Raises:
        RuntimeError: If incompatible inputs are given

    Returns:
        Union[List[dict], pd.DataFrame, np.ndarray]: Depends on the return type asked for

    Example:
    ```python
    > type(get_rdkit_descriptors(smiles=["c1ccccc1C"])[0])
    <class 'dict'>

    > len(get_rdkit_descriptors(smiles=["c1ccccc1C"], return_df=True).columns)
    210
    ```
    """

    if include_autocorrelation_features is True:
        setupAUTOCorrDescriptors()

    if sum([return_df, return_np]) > 1:
        raise RuntimeError(
            f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."
        )

    if smiles is None:
        if all(ent is not None for ent in [data_df, smiles_column]):
            df = data_df.copy()
            smiles = df[smiles_column].to_list()
        else:
            raise RuntimeError(
                "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"
            )
    else:
        df = data_df

    log.info(f"Making RDKit descriptors for {len(smiles)} molecules")

    fps = [
        get_specific_set_of_rdkit_descriptors(
            ent, descriptor_names=continuous_rdkit_descriptor_names
        )
        for ent in [Chem.MolFromSmiles(smi) for smi in smiles]
    ]
    # make the order consistent sort on the keys in the dictionary important for the numpy return
    fps = [dict(sorted(ent.items())) for ent in fps]

    if return_np is True and return_df is False:
        fps_np = pd.DataFrame(fps).values
        return fps_np

    elif return_df is True and return_np is False:
        fps_df = pd.DataFrame(fps)
        fps_df.columns = [f"rdkit_descriptor_{name}" for name in fps_df.columns]

        if df is not None:
            return pd.concat([df, fps_df], axis=1)
        else:
            return fps_df
    else:
        return fps


def get_categorical_rdkit_descriptors(
    data_df: Optional[pd.DataFrame] = None,
    smiles_column: Optional[str] = None,
    smiles: Optional[List[str]] = None,
    return_df: bool = False,
    return_np: bool = False,
    include_autocorrelation_features: bool = False,
    **kwargs,
) -> Union[List[dict], pd.DataFrame, np.ndarray]:
    """
    Function to generate Rdkit descriptor representations from smiles

    Args:
        data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.
         If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.
        smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.
        smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.
        return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.
        return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defaults to False.

    Raises:
        RuntimeError: If incompatible inputs are given

    Returns:
        Union[List[dict], pd.DataFrame, np.ndarray]: Depends on the return type asked for

    Example:
    ```python
    > type(get_rdkit_descriptors(smiles=["c1ccccc1C"])[0])
    <class 'dict'>

    > len(get_rdkit_descriptors(smiles=["c1ccccc1C"], return_df=True).columns)
    210
    ```
    """

    if include_autocorrelation_features is True:
        setupAUTOCorrDescriptors()

    if sum([return_df, return_np]) > 1:
        raise RuntimeError(
            f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."
        )

    if smiles is None:
        if all(ent is not None for ent in [data_df, smiles_column]):
            df = data_df.copy()
            smiles = df[smiles_column].to_list()
        else:
            raise RuntimeError(
                "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"
            )
    else:
        df = data_df

    log.info(f"Making RDKit descriptors for {len(smiles)} molecules")

    fps = [
        get_specific_set_of_rdkit_descriptors(
            ent, descriptor_names=categorical_rdkit_descriptor_names
        )
        for ent in [Chem.MolFromSmiles(smi) for smi in smiles]
    ]
    # make the order consistent sort on the keys in the dictionary important for the numpy return
    fps = [dict(sorted(ent.items())) for ent in fps]

    if return_np is True and return_df is False:
        fps_np = pd.DataFrame(fps).values
        return fps_np

    elif return_df is True and return_np is False:
        fps_df = pd.DataFrame(fps)
        fps_df.columns = [f"rdkit_descriptor_{name}" for name in fps_df.columns]

        if df is not None:
            return pd.concat([df, fps_df], axis=1)
        else:
            return fps_df
    else:
        return fps


def get_semi_categorical_rdkit_descriptors(
    data_df: Optional[pd.DataFrame] = None,
    smiles_column: Optional[str] = None,
    smiles: Optional[List[str]] = None,
    return_df: bool = False,
    return_np: bool = False,
    include_autocorrelation_features: bool = False,
    **kwargs,
) -> Union[List[dict], pd.DataFrame, np.ndarray]:
    """
    Function to generate Rdkit descriptor representations from smiles

    Args:
        data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.
         If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.
        smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.
        smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.
        return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.
        return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defaults to False.

    Raises:
        RuntimeError: If incompatible inputs are given

    Returns:
        Union[List[dict], pd.DataFrame, np.ndarray]: Depends on the return type asked for

    Example:
    ```python
    > type(get_rdkit_descriptors(smiles=["c1ccccc1C"])[0])
    <class 'dict'>

    > len(get_rdkit_descriptors(smiles=["c1ccccc1C"], return_df=True).columns)
    210
    ```
    """

    if include_autocorrelation_features is True:
        setupAUTOCorrDescriptors()

    if sum([return_df, return_np]) > 1:
        raise RuntimeError(
            f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."
        )

    if smiles is None:
        if all(ent is not None for ent in [data_df, smiles_column]):
            df = data_df.copy()
            smiles = df[smiles_column].to_list()
        else:
            raise RuntimeError(
                "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"
            )
    else:
        df = data_df

    log.info(f"Making RDKit descriptors for {len(smiles)} molecules")

    fps = [
        get_specific_set_of_rdkit_descriptors(
            ent, descriptor_names=semi_categorical_rdkit_descriptor_names
        )
        for ent in [Chem.MolFromSmiles(smi) for smi in smiles]
    ]
    # make the order consistent sort on the keys in the dictionary important for the numpy return
    fps = [dict(sorted(ent.items())) for ent in fps]

    if return_np is True and return_df is False:
        fps_np = pd.DataFrame(fps).values
        return fps_np

    elif return_df is True and return_np is False:
        fps_df = pd.DataFrame(fps)
        fps_df.columns = [f"rdkit_descriptor_{name}" for name in fps_df.columns]

        if df is not None:
            return pd.concat([df, fps_df], axis=1)
        else:
            return fps_df
    else:
        return fps


def get_nlp_smiles_rep(
    model_name: str = "Saideepthi55/sentencetransformer_ftmodel_on_chemical_dataset",
    data_df: Optional[pd.DataFrame] = None,
    smiles_column: Optional[str] = None,
    smiles: Optional[List[str]] = None,
    return_df: bool = False,
    return_np: bool = False,
    **kwargs,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Function to generate the NLP embedding representations from smiles using a transformer model

    Args:
        model_name (str): the model name to use default Saideepthi55/sentencetransformer_ftmodel_on_chemical_dataset apache 2.0 license accessed 30/10/24 https://huggingface.co/Saideepthi55/sentencetransformer_ftmodel_on_chemical_dataset,
        data_df (Optional[pd.DataFrame], optional): Dataframe containing at least the smiles strings to use.
         If this is passed and return_df is true the fingerprints are concatenated to a copy of the input dataframe and returned. Defaults to None.
        smiles_column (Optional[str], optional): Needed if data_df is given to define which column to find the smiles strings. Defaults to None.
        smiles (Optional[list[str]], optional): A list of smiles strings to generate fingerprints for. Defaults to None.
        return_df (bool): Whether to return a pandas dataframe rather than a list of bit vectors. Defaults to False.
        return_np (bool): Whether to return a numpy array rather than a list of bit vectors. Defaults to False.
        combine_strategy (Union[str, int]): How to combine word vectors (one of None, concat, mean or and int to get the embedding for a specific word) . Default is "mean"

    Raises:
        RuntimeError: If incompatible inputs are given

    Returns:
        Union[List[dict], pd.DataFrame, np.ndarray]: Depends on the return type asked for
    """

    if sum([return_df, return_np]) > 1:
        raise RuntimeError(
            f"WARNING - Multiple optional output formats set please set only one of return_df [set as {return_df}] and return_np [{return_np}] to True will return pandas dataframe only."
        )

    if smiles is None:
        if all(ent is not None for ent in [data_df, smiles_column]):
            df = data_df.copy()
            smiles = list(df[smiles_column])
        else:
            raise RuntimeError(
                "ERROR - neither smiles nor df together with smiles column were given. One of these must be given as input"
            )
    else:
        df = data_df

    input_n = len(smiles)

    valid_smiles = smilesmod.validate_smiles(smiles, return_failed_as_None=False)
    if len(valid_smiles) != input_n:
        raise RuntimeError(
            f"ERROR - only {(len(valid_smiles) / input_n) * 100.0}% of the input smiles were successfully read. Please correct the invalid smiles"
        )

    log.info(
        f"Making NLP embeddings using model {model_name} for {len(smiles)} molecules"
    )

    model = SentenceTransformer(model_name, **kwargs)
    embedding_rep = model.encode(smiles, **kwargs)

    if return_np is True and return_df is False:
        return embedding_rep

    elif return_df is True and return_np is False:
        fps_df = pd.DataFrame(embedding_rep)
        fps_df.columns = [
            f"embedding_{model_name.replace('/', '-')}_{ith}"
            for ith in range(fps_df.shape[1])
        ]

        if df is not None:
            return pd.concat([df, fps_df], axis=1)
        else:
            return fps_df
    else:
        return embedding_rep


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
