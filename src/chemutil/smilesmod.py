#!/usr/bin.env python3
# -*- coding: utf-8 -*-

"""
Module for processing smiles formats
"""

import logging
from typing import List, Union

import datamol as dm
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

log = logging.getLogger(__name__)


def clean_and_validate_smiles(
    data_df: pd.DataFrame,
    smiles_column: str,
    charge_neutralize: bool = False,
    diconnect_metals: bool = True,
    normalize: bool = True,
    reionize: bool = True,
    stereo: bool = True,
    remove_salts: bool = True,
    remove_fragmented: bool = True,
    return_selfies: bool = True,
    return_inchi: bool = True,
    return_inchikey: bool = True,
) -> pd.DataFrame:
    """
    Function to clean and validate smiles strings in a pandas dataframe
    Args:
        data_df (pd.DataFrame): pandas dataframe containing smiles strings
        smiles_column (str): column name containing smiles strings
        charge_neutralize (bool): whether to charge neutralize the molecule
        diconnect_metals (bool): whether to disconnect metals
        normalize (bool): whether to normalize the molecule
        reionize (bool): whether to reionize the molecule
        stereo (bool): whether to consider stereochemistry
        remove_salts (bool): whether to remove salts
        remove_fragmented (bool): whether to remove fragmented smiles
        return_selfies (bool): whether to return selfies representation
        return_inchi (bool): whether to return inchi representation
        return_inchikey (bool): whether to return inchikey representation

    Returns:
        pd.DataFrame: pandas dataframe with cleaned and validated smiles strings
    """

    df = data_df.copy()

    # Remove invalid smiles
    drop = []

    for s in df[smiles_column].values:
        if remove_fragmented is True and "." in s:
            drop.append(None)
        elif Chem.MolFromSmiles(s) is None:
            drop.append(None)
        else:
            drop.append(0)

    df["invalid_smiles"] = drop
    df = df.dropna(subset=["invalid_smiles"])
    # df = df.drop(columns=["invalid_smiles"])

    # Standardize smiles and clean
    if return_selfies is True:
        selfies = []

    if return_inchi is True:
        inchi = []

    if return_inchikey is True:
        inchikey = []

    standardized_smiles = []
    salt_remover = SaltRemover()
    for s in df[smiles_column].values:
        mol = dm.to_mol(
            s,
            ordered=True,
        )
        if remove_salts is True:
            mol = salt_remover.StripMol(mol)

        # Do not process fragmented smiles
        if remove_fragmented is True and len(Chem.rdmolops.GetMolFrags(mol)) > 1:
            log.error(f"Fragmented smiles {s} will be removed")
            continue

        mol = dm.fix_mol(mol)
        mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
        mol = dm.standardize_mol(
            mol,
            disconnect_metals=diconnect_metals,
            normalize=normalize,
            reionize=reionize,
            uncharge=charge_neutralize,
            stereo=stereo,
        )

        # Get other string representations
        standardized_smiles.append(dm.standardize_smiles(dm.to_smiles(mol)))

        if return_selfies is True:
            try:
                selfies.append(dm.to_selfies(mol))
            except Exception as e:
                log.error(f"Error in generating selfies for smiles {s} with error {e}")
                selfies.append(None)

        if return_inchi is True:
            try:
                inchi.append(dm.to_inchi(mol))
            except Exception as e:
                log.error(f"Error in generating inchi for smiles {s} with error {e}")
                inchi.append(None)

        if return_inchikey is True:
            try:
                inchikey.append(dm.to_inchikey(mol))
            except Exception as e:
                log.error(f"Error in generating inchikey for smiles {s} with error {e}")
                inchikey.append(None)

    df["standardized_smiles"] = standardized_smiles
    if return_selfies is True:
        df["standardized_selfies"] = selfies
    if return_inchi is True:
        df["standardized_inchi"] = inchi
    if return_inchikey is True:
        df["standardized_inchikey"] = inchikey

    return df


def get_valid_smiles(smis: List[str], canonicalize: bool = True) -> List[str]:
    """
    Function to extract only valid (RDKit parsable) smiles strings from a list of smiles strings

    Args:
        smis (List[str]): list of smiles strings to determine which are valid

    Returns:
        List[str]: valid (RDKit parsable) smiles strings
    """

    if canonicalize is False:
        log.info(
            f"Determining smiles validity using RDKit parsing for RDKit version {rdkit.__version__} without canonicalization"
        )
        return [s for s in smis if Chem.MolFromSmiles(s) is not None]

    else:
        log.info(
            f"Canonicalizing and determining smiles validity using RDKit parsing for RDKit version {rdkit.__version__}"
        )
        return [
            Chem.MolToSmiles(Chem.MolFromSmiles(s))
            for s in smis
            if Chem.MolFromSmiles(s) is not None
        ]


def get_valid_smiles_mask(smis: List[str]) -> List[str]:
    """
    Function to extract a boolean vector defining whether the smiles string in that row is valid (RDKit parsable) or not

    Args:
        smis (List[str]): list of smiles strings to determine which are valid

    Returns:
        List[str]: boolean vector defining if the smiles string of that row index is valid (RDKit parsable)
    """

    log.info(
        f"Determining smiles validity mask using RDKit parsing for RDKit version {rdkit.__version__}"
    )
    return [True if Chem.MolFromSmiles(s) is not None else False for s in smis]


def validate_smile(smile: str, canonicalize: bool = True) -> Union[None, str]:
    """
    Function to validate a single smiles string. This differs from get_valid_smiles as
    it operates on a single smiles string and garuntees a return as if the smiles is invalid it returns None.

    Args:
        smile (str): smiles string to check if it is valid
        canaonicalize (bool): whether to return the input smiles or a canonicalized version for a valid smiles string

    Returns:
        Union[None, str]: None for a invalid smiles string and a smiles string for a valid one
    """

    try:
        m = Chem.MolFromSmiles(smile)
    except Exception:
        log.error("Exception when converting smiles to RDKit molecule object")
        return None

    if m is None:
        log.error(f"SMILES string: {smile} is invalid in RDKit and will be skipped")
        return None

    if canonicalize is True:
        return Chem.MolToSmiles(m)
    else:
        return smile


def validate_smiles(
    smiles: List[str], return_failed_as_None: bool = True, canonicalize: bool = True
) -> List[Union[None, str]]:
    """
    Function to validate a list of m smiles strings. This differs from get_valid_smiles as
    it can guarantee a return, if return_failed_as_None is True (default), as if the smiles
    is invalid it returns None.

    Args:
        smile (str): smiles string to check if it is valid
        return_failed_as_None (bool): whether to return None if the smiles string is invalid
        canaonicalize (bool): whether to return the input smiles or a canonicalized version for a valid smiles string

    Returns:
        List[Union[None, str]]: None for a invalid smiles string and a smiles string for a valid one
    """

    if return_failed_as_None is True:
        return [validate_smile(smile, canonicalize=canonicalize) for smile in smiles]
    else:
        tmp = [validate_smile(smile, canonicalize=canonicalize) for smile in smiles]
        return [ent for ent in tmp if ent is not None]


def smiles_to_molecule(smi: str) -> Union[Chem.rdchem.Mol, None]:
    """
    Function to get an RDKit molecule object from a smiles string

    Args:
        smi (str): smiles representation of a molecule

    Returns:
        Chem.rdchem.Mol: RDKit molecule

    Doctest:
    >>> smiles_to_molecule("O")  # doctest: +ELLIPSIS
    <rdkit.Chem.rdchem.Mol object at ...>
    """

    try:
        m = Chem.MolFromSmiles(smi, sanitize=True)
    except Exception as eerr:
        log.error(
            f"ERROR - smiles string conversion for {smi} encountered an error {eerr}"
        )
        m = None

    if m is not None:
        return m
    else:
        log.error(
            f"ERROR - smiles {smi} could not be converted to molecule likely in valid"
        )

        return None


def smiles_to_inchi(smi: str) -> str:
    """
    Function to get inchi representation from a smiles string

    Args:
        smi (str): smiles representation of a molecule

    Returns:
        str: inchi representation

    Doctest:
    >>> smiles_to_inchi("O")
    'InChI=1S/H2O/h1H2'
    """

    s = Chem.CanonSmiles(smi, useChiral=1)

    return Chem.inchi.MolToInchi(Chem.MolFromSmiles(s))


def smiles_to_inchikey(smi: str) -> str:
    """
    Function to get inchikey representation from a smiles string

    Args:
        smi (str): smiles representation of a molecule

    Returns:
        str: inchikey representation of the molecule

    Doctest:
    >>> smiles_to_inchikey("O")
    'XLYOFNOQVPJJNP-UHFFFAOYSA-N'
    """

    s = Chem.CanonSmiles(smi, useChiral=1)

    return Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(s))


def sample_smiles(
    smi: str,
    n: int = 1,
    isomeric: bool = False,
    kekule: bool = True,
    randomseed: int = 15751,
) -> List[str]:
    """
    A function to sample smiles for the same molecule

    Args:
        smi (str): smiles representation of a molecule
        n (int, optional): integer the number of smiles for the same molecule to sample. Defaults to 1.
        isomeric: (bool, optional): use isomeric representation or not
        kekule: (bool, optional): kekulize smiles or not
        randomseed: (int, optional): random seed fixed to make sure the sample generated a consistent and repeatable each run

    Returns:
        List[str]: list of sampled smiles strings, will have length zero for an invalid problematic input smiles

    Doctest:
    There are only 2 version of valid smiles here hence n = 2
    >>> sample_smiles('O=C=O', 2) # doctest: +NORMALIZE_WHITESPACE
    ['C(=O)=O', 'O=C=O']
    """

    mol = smiles_to_molecule(smi)
    if mol is None:
        return []

    return Chem.MolToRandomSmilesVect(
        mol, n, isomericSmiles=isomeric, kekuleSmiles=kekule, randomSeed=randomseed
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
