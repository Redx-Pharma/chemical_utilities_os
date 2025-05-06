#!/usr/bin.env python3
# -*- coding: utf-8 -*-

"""
Module for processing inchi formats
"""

import logging
from typing import List

import datamol as dm
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

log = logging.getLogger(__name__)


def clean_and_validate_inchi(
    data_df: pd.DataFrame,
    inchi_column: str,
    charge_neutralize: bool = False,
    diconnect_metals: bool = True,
    normalize: bool = True,
    reionize: bool = True,
    stereo: bool = True,
    remove_salts: bool = True,
    remove_fragmented: bool = True,
    return_selfies: bool = True,
    return_smiles: bool = True,
    return_inchikey: bool = True,
) -> pd.DataFrame:
    """
    Function to clean and validate inchi strings in a pandas dataframe
    Args:
        data_df (pd.DataFrame): pandas dataframe containing smiles strings
        inchi_column (str): column name containing inchi strings
        charge_neutralize (bool): whether to charge neutralize the molecule
        diconnect_metals (bool): whether to disconnect metals
        normalize (bool): whether to normalize the molecule
        reionize (bool): whether to reionize the molecule
        stereo (bool): whether to consider stereochemistry
        remove_salts (bool): whether to remove salts
        remove_fragmented (bool): whether to remove fragmented smiles
        return_selfies (bool): whether to return selfies representation
        return_smiles (bool): whether to return inchi representation
        return_inchikey (bool): whether to return inchikey representation

    Returns:
        pd.DataFrame: pandas dataframe with cleaned and validated smiles strings
    """

    df = data_df.copy()

    # Remove invalid smiles
    drop = []

    df[inchi_column] = [
        f"inchi={ent}" if ent[:6] != "inchi=" else ent
        for ent in df[inchi_column].values
    ]
    for i in df[inchi_column].values:
        if Chem.MolFromInchi(i) is None:
            drop.append(None)
        else:
            drop.append(0)

    df["invalid_inchi"] = drop
    df = df.dropna(subset=["invalid_inchi"])

    # Standardize smiles and clean
    if return_selfies is True:
        selfies = []

    if return_smiles is True:
        smiles = []

    if return_inchikey is True:
        inchikey = []

    standardized_inchi = []
    salt_remover = SaltRemover()
    for i in df[inchi_column].values:
        mol = Chem.MolFromInchi(i)
        if remove_salts is True:
            mol = salt_remover.StripMol(mol)

        # Do not process fragmented smiles
        if remove_fragmented is True and len(Chem.rdmolops.GetMolFrags(mol)) > 1:
            log.error(f"Fragmented inchi {s} will be removed")
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
        standardized_inchi.append(dm.to_inchi(mol))

        if return_selfies is True:
            try:
                selfies.append(dm.to_selfies(mol))
            except Exception as e:
                log.error(f"Error in generating selfies for smiles {s} with error {e}")
                selfies.append(None)

        if return_smiles is True:
            try:
                smiles.append(dm.standardize_smiles(dm.to_smiles(mol)))
            except Exception as e:
                log.error(f"Error in generating inchi for smiles {s} with error {e}")
                smiles.append(None)

        if return_inchikey is True:
            try:
                inchikey.append(dm.to_inchikey(mol))
            except Exception as e:
                log.error(f"Error in generating inchikey for smiles {s} with error {e}")
                inchikey.append(None)

    df["standardized_inchi"] = standardized_inchi
    if return_selfies is True:
        df["standardized_selfies"] = selfies
    if return_smiles is True:
        df["standardized_smiles"] = smiles
    if return_inchikey is True:
        df["standardized_inchikey"] = inchikey

    return df


def get_valid_inchi(inchis: List[str]) -> List[str]:
    """
    Function to extract only valid (RDKit parsable) inchi strings from a list of inchi strings

    Args:
        inchi (List[str]): list of inchi strings to determine which are valid

    Returns:
        List[str]: valid (RDKit parsable) inchi strings
    """

    log.info(
        f"Determining inchi validity using RDKit parsing for RDKit version {rdkit.__version__}"
    )
    return [inc for inc in inchis if Chem.MolFromInchi(inc) is not None]


def get_valid_inchi_mask(inchis: List[str]) -> List[str]:
    """
    Function to extract a boolean vector defining whether the inchi string in that row is valid (RDKit parsable) or not

    Args:
        inchi (List[str]): list of inchi strings to determine which are valid

    Returns:
        List[str]: boolean vector defining if the inchi string of that row index is valid (RDKit parsable)
    """

    log.info(
        f"Determining inchi validity mask using RDKit parsing for RDKit version {rdkit.__version__}"
    )
    return [
        True if Chem.inchi.MolFromInchi(inc) is not None else False for inc in inchis
    ]


def inchi_to_molecule(inchi: str) -> Chem.rdchem.Mol:
    """
    Function to get an RDKit molecule object from a inchi string

    Args:
        inchi (str): inchi representation of a molecule

    Returns:
        Chem.rdchem.Mol: rdkit molecule

    Doctest:
    >>> inchi_to_molecule("InChI=1S/H2O/h1H2")  # doctest: +ELLIPSIS
    <rdkit.Chem.rdchem.Mol object at ...>
    """

    return Chem.inchi.MolFromInchi(inchi)


def inchi_to_smiles(inchi: str) -> str:
    """
    Function to get smiles representation from a inchi string

    Args:
        inchi (str): inchi representatio of a molecule

    Returns:
        str: smiles representation

    Doctest:
    >>> inchi_to_smiles("InChI=1S/H2O/h1H2")
    'O'
    """

    return Chem.MolToSmiles(Chem.inchi.MolFromInchi(inchi))


def inchi_to_inchikey(inchi: str) -> str:
    """
    Function to get inchikey representation from a inchi string

    Args:
        inchi (str): inchi representation of a molecule

    Returns:
        str: inchikey

    Doctest:
    >>> inchi_to_inchikey("InChI=1S/H2O/h1H2")
    'XLYOFNOQVPJJNP-UHFFFAOYSA-N'
    """

    return Chem.inchi.MolToInchiKey(Chem.inchi.MolFromInchi(inchi))


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
