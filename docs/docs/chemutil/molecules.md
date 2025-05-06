# Module chemutil.molecules

Module for dealing with molecule objects

??? example "View Source"
        #!/usr/bin.env python3

        # -*- coding: utf-8 -*-

        """

        Module for dealing with molecule objects

        """

        from typing import List, Union, Tuple, Optional

        import pandas as pd

        import rdkit

        from rdkit import Chem

        from rdkit.Chem import rdFMCS

        from rdkit.Chem.Scaffolds import MurckoScaffold

        import logging

        log = logging.getLogger(__name__)



        def get_maximum_common_substructure(

            mols: List[Chem.rdchem.Mol],

            return_smarts: bool = False,

            return_smiles: bool = False,

            return_mol: bool = False,

            **kwargs,

        ) -> Union[str, Chem.rdchem.Mol, rdkit.Chem.rdFMCS.MCSResult]:

            """

            Function to find the maximum common substructure across a list of molecules

            Args:

                mols (List[rdkit.Chem.rdchem.Mol]): List of RDKit molecule objects

            Returns:

                Based on user request:

                    * (Default): rdkit.Chem.MCS.MCSResult The raw MCS result

                    * return_smart is True: str Smarts string

                    * return_smiles is True: str Smiles string

                    * return_mol is True: Chem.rdchem.Mol Molecule object

            Doctest:

            >>> get_maximum_common_substructure([Chem.MolFromSmiles("Cc1ccccc1"), Chem.MolFromSmiles("Nc1ccccc1"), Chem.MolFromSmiles("Oc1ccccc1"), Chem.MolFromSmiles("CCc1ccccc1")], return_smiles=True)

            'C1:C:C:C:C:C:1'

            >>> get_maximum_common_substructure([Chem.MolFromSmiles("Cc1ccccc1"), Chem.MolFromSmiles("Nc1ccccc1"), Chem.MolFromSmiles("Oc1ccccc1"), Chem.MolFromSmiles("CCc1ccccc1")], return_smarts=True)

            '[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1'

            """

            mcs_res = rdFMCS.FindMCS(mols, **kwargs)

            if return_smarts:

                return mcs_res.smartsString

            elif return_smiles:

                return Chem.MolToSmiles(Chem.MolFromSmarts(mcs_res.smartsString))

            elif return_mol:

                return Chem.MolFromSmarts(mcs_res.smartsString)

            else:

                return mcs_res



        get_mcs = get_maximum_common_substructure



        def get_murcko_scaffold(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:

            """

            Function to get the Murcko scaffold of a RDKit molecule object

            Args:

                mol (Chem.rdchem.Mol): RDKit molecule to define a Murcko-Bemis scaffold of

            Returns:

                Chem.rdchem.Mol: RDKit molecule obeject of the Murcko-Bemis scaffold

            Doctest:

            >>> get_murcko_scaffold(Chem.MolFromSmiles("Cc1ccccc1")) # doctest: +ELLIPSIS

            <rdkit.Chem.rdchem.Mol object at ...>

            >>> m = get_murcko_scaffold(Chem.MolFromSmiles("Cc1ccccc1")) # doctest: +ELLIPSIS

            >>> s = Chem.MolToSmiles(m)

            >>> s == "c1ccccc1"

            True

            """

            mol_scaffold = MurckoScaffold.GetScaffoldForMol(mol)

            return mol_scaffold



        def check_for_sub_series_overlap(subseries_dict) -> Tuple[bool, dict]:

            """

            Check within a sub series dictionary if there is molecule choice overlap

            Args:

                subseries_dict (dict): Dictionary of the form {"series name": [series id 1, series id 2, series id 3, series id 4 .....]}

            Returns:

                Tuple[bool, dict]: Whether there is overlap or not and a dictionary defining the series name pairs where there is and is not overlap

            """

            subseries_overlap = False

            overlaps = {}

            for ith, (name1, list1) in enumerate(sorted(subseries_dict.items())):

                for jth, (name2, list2) in enumerate(sorted(subseries_dict.items())):

                    if jth <= ith:

                        continue

                    else:

                        log.info(f"Checking for any overlap {name1} {name2} .....")

                        if any(ent in list2 for ent in list1):

                            log.warning(f"Sub-series overlap {name1} {name2}")

                            subseries_overlap = True

                            overlaps[f"{name1} -- {name2}"] = True

                        else:

                            log.info("No overlap has been detected")

                            overlaps[f"{name1} -- {name2}"] = False

            return subseries_overlap, overlaps



        def get_sub_series(

            series_dict: dict,

            series_df: Optional[pd.DataFrame] = None,

            id_column: Optional[str] = "name_id",

            smiles_col: Optional[str] = "smiles",

            smiles: Optional[List[str]] = None,

        ) -> Union[Tuple[dict, dict], dict]:

            """

            Get a set of sub series from a series based on SMARTS defined substructures. This is a scffold defined sub-series search. The return depends on the input.

            If you input a smiles list you get back a dictionary with a mask and the smiles sub set in each sub series i.e.

            {

            "sub series name": {

                                "mask: [bool],

                                "smiles: [str]

                            }

            }

            If you input a dataframe with a smiles column you get back two dictionaries: the first as stated previously without the smiles key

            {

            "sub series name":  [bool]

            }

            The second is as above but is entire sub set of the input dataframe

            {

            "sub series name":  pd.DataFrame

            }

            Args:

                series_dict (dict): A dictionary of the form {"sub series name": "SMARTS or SMILES defining the sub series based on sub sturcture presence"}. There can be multiple keys and values for multiple sub-series.

                series_df (pd.Dataframe): Dataframe of containing at least the smiles column and the ID column

                id_column (str, optional): ID column for each molecule

                smiles_col (str, optional): The smiles column name in the series_df. Defaults to 'smiles'.

                smiles (Optional[List[str]], optional): List of smiles to put into sub series. Defaults to None.

            """

            # Check that one input has been provided

            if all(ent is None for ent in [series_df, smiles]):

                raise RuntimeError("Either series df or smiles list must be provided")

            # For a list of smiles we will return only a mask ans sub list

            if smiles is not None and series_df is None:

                # set up the outputs

                subseries_dict = {}

                mols = [Chem.MolFromSmiles(smi) for smi in smiles]

                # get the sub-series

                for name, smiles in series_dict.items():

                    sub_mol = Chem.MolFromSmarts(smiles)

                    mask = [mol.HasSubstructMatch(sub_mol) for mol in mols]

                    log.info(f"Sub-series {name} has {sum(mask)} molecules present")

                    subseries_dict[name] = {

                        "mask": mask,

                        "smiles": [s for s, m in zip(smiles, mask) if m is True],

                    }

                overlap_present, overlaps = check_for_sub_series_overlap(

                    subseries_dict=subseries_dict

                )

                if overlap_present is True:

                    log.info(overlaps)

                return subseries_dict

            elif series_df is not None:

                # set up the outputs

                subseries_dict = {}

                subseries_dict_dfs = {}

                if id_column is None:

                    id_column = smiles_col

                if smiles_col is None:

                    raise RuntimeError("Must define a smiles column when provding a dataframe")

                else:

                    try:

                        _ = series_df[smiles_col]

                    except KeyError:

                        raise KeyError(

                            f"ERROR - no smiles key {smiles_col} found in the dataframe columns"

                        )

                mols = [Chem.MolFromSmiles(smi) for smi in series_df[smiles_col].to_list()]

                for name, smiles in series_dict.items():

                    sub_mol = Chem.MolFromSmarts(smiles)

                    mask = [mol.HasSubstructMatch(sub_mol) for mol in mols]

                    log.info(f"Sub-series {name} has {sum(mask)} molecules present")

                    subseries_dict[name] = series_df.loc[mask][id_column].to_list()

                    subseries_dict_dfs[name] = series_df.loc[mask].copy()

                overlap_present, overlaps = check_for_sub_series_overlap(

                    subseries_dict=subseries_dict

                )

                if overlap_present is True:

                    log.info(overlaps)

                return subseries_dict, subseries_dict_dfs



        if __name__ == "__main__":

            import doctest

            doctest.testmod(verbose=True, optionflags=doctest.ELLIPSIS)

## Variables

```python3
log
```

## Functions


### check_for_sub_series_overlap

```python3
def check_for_sub_series_overlap(
    subseries_dict
) -> Tuple[bool, dict]
```

Check within a sub series dictionary if there is molecule choice overlap

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| subseries_dict | dict | Dictionary of the form {"series name": [series id 1, series id 2, series id 3, series id 4 .....]} | None |

**Returns:**

| Type | Description |
|---|---|
| Tuple[bool, dict] | Whether there is overlap or not and a dictionary defining the series name pairs where there is and is not overlap |

??? example "View Source"
        def check_for_sub_series_overlap(subseries_dict) -> Tuple[bool, dict]:

            """

            Check within a sub series dictionary if there is molecule choice overlap

            Args:

                subseries_dict (dict): Dictionary of the form {"series name": [series id 1, series id 2, series id 3, series id 4 .....]}

            Returns:

                Tuple[bool, dict]: Whether there is overlap or not and a dictionary defining the series name pairs where there is and is not overlap

            """

            subseries_overlap = False

            overlaps = {}

            for ith, (name1, list1) in enumerate(sorted(subseries_dict.items())):

                for jth, (name2, list2) in enumerate(sorted(subseries_dict.items())):

                    if jth <= ith:

                        continue

                    else:

                        log.info(f"Checking for any overlap {name1} {name2} .....")

                        if any(ent in list2 for ent in list1):

                            log.warning(f"Sub-series overlap {name1} {name2}")

                            subseries_overlap = True

                            overlaps[f"{name1} -- {name2}"] = True

                        else:

                            log.info("No overlap has been detected")

                            overlaps[f"{name1} -- {name2}"] = False

            return subseries_overlap, overlaps


### get_maximum_common_substructure

```python3
def get_maximum_common_substructure(
    mols: List[rdkit.Chem.rdchem.Mol],
    return_smarts: bool = False,
    return_smiles: bool = False,
    return_mol: bool = False,
    **kwargs
) -> Union[str, rdkit.Chem.rdchem.Mol, rdkit.Chem.rdFMCS.MCSResult]
```

Function to find the maximum common substructure across a list of molecules

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mols | List[rdkit.Chem.rdchem.Mol] | List of RDKit molecule objects | None |

**Returns:**

| Type | Description |
|---|---|
| None | Based on user request:<br>* (Default): rdkit.Chem.MCS.MCSResult The raw MCS result<br>* return_smart is True: str Smarts string<br>* return_smiles is True: str Smiles string<br>* return_mol is True: Chem.rdchem.Mol Molecule object |

??? example "View Source"
        def get_maximum_common_substructure(

            mols: List[Chem.rdchem.Mol],

            return_smarts: bool = False,

            return_smiles: bool = False,

            return_mol: bool = False,

            **kwargs,

        ) -> Union[str, Chem.rdchem.Mol, rdkit.Chem.rdFMCS.MCSResult]:

            """

            Function to find the maximum common substructure across a list of molecules

            Args:

                mols (List[rdkit.Chem.rdchem.Mol]): List of RDKit molecule objects

            Returns:

                Based on user request:

                    * (Default): rdkit.Chem.MCS.MCSResult The raw MCS result

                    * return_smart is True: str Smarts string

                    * return_smiles is True: str Smiles string

                    * return_mol is True: Chem.rdchem.Mol Molecule object

            Doctest:

            >>> get_maximum_common_substructure([Chem.MolFromSmiles("Cc1ccccc1"), Chem.MolFromSmiles("Nc1ccccc1"), Chem.MolFromSmiles("Oc1ccccc1"), Chem.MolFromSmiles("CCc1ccccc1")], return_smiles=True)

            'C1:C:C:C:C:C:1'

            >>> get_maximum_common_substructure([Chem.MolFromSmiles("Cc1ccccc1"), Chem.MolFromSmiles("Nc1ccccc1"), Chem.MolFromSmiles("Oc1ccccc1"), Chem.MolFromSmiles("CCc1ccccc1")], return_smarts=True)

            '[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1'

            """

            mcs_res = rdFMCS.FindMCS(mols, **kwargs)

            if return_smarts:

                return mcs_res.smartsString

            elif return_smiles:

                return Chem.MolToSmiles(Chem.MolFromSmarts(mcs_res.smartsString))

            elif return_mol:

                return Chem.MolFromSmarts(mcs_res.smartsString)

            else:

                return mcs_res


### get_mcs

```python3
def get_mcs(
    mols: List[rdkit.Chem.rdchem.Mol],
    return_smarts: bool = False,
    return_smiles: bool = False,
    return_mol: bool = False,
    **kwargs
) -> Union[str, rdkit.Chem.rdchem.Mol, rdkit.Chem.rdFMCS.MCSResult]
```

Function to find the maximum common substructure across a list of molecules

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mols | List[rdkit.Chem.rdchem.Mol] | List of RDKit molecule objects | None |

**Returns:**

| Type | Description |
|---|---|
| None | Based on user request:<br>* (Default): rdkit.Chem.MCS.MCSResult The raw MCS result<br>* return_smart is True: str Smarts string<br>* return_smiles is True: str Smiles string<br>* return_mol is True: Chem.rdchem.Mol Molecule object |

??? example "View Source"
        def get_maximum_common_substructure(

            mols: List[Chem.rdchem.Mol],

            return_smarts: bool = False,

            return_smiles: bool = False,

            return_mol: bool = False,

            **kwargs,

        ) -> Union[str, Chem.rdchem.Mol, rdkit.Chem.rdFMCS.MCSResult]:

            """

            Function to find the maximum common substructure across a list of molecules

            Args:

                mols (List[rdkit.Chem.rdchem.Mol]): List of RDKit molecule objects

            Returns:

                Based on user request:

                    * (Default): rdkit.Chem.MCS.MCSResult The raw MCS result

                    * return_smart is True: str Smarts string

                    * return_smiles is True: str Smiles string

                    * return_mol is True: Chem.rdchem.Mol Molecule object

            Doctest:

            >>> get_maximum_common_substructure([Chem.MolFromSmiles("Cc1ccccc1"), Chem.MolFromSmiles("Nc1ccccc1"), Chem.MolFromSmiles("Oc1ccccc1"), Chem.MolFromSmiles("CCc1ccccc1")], return_smiles=True)

            'C1:C:C:C:C:C:1'

            >>> get_maximum_common_substructure([Chem.MolFromSmiles("Cc1ccccc1"), Chem.MolFromSmiles("Nc1ccccc1"), Chem.MolFromSmiles("Oc1ccccc1"), Chem.MolFromSmiles("CCc1ccccc1")], return_smarts=True)

            '[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1'

            """

            mcs_res = rdFMCS.FindMCS(mols, **kwargs)

            if return_smarts:

                return mcs_res.smartsString

            elif return_smiles:

                return Chem.MolToSmiles(Chem.MolFromSmarts(mcs_res.smartsString))

            elif return_mol:

                return Chem.MolFromSmarts(mcs_res.smartsString)

            else:

                return mcs_res


### get_murcko_scaffold

```python3
def get_murcko_scaffold(
    mol: rdkit.Chem.rdchem.Mol
) -> rdkit.Chem.rdchem.Mol
```

Function to get the Murcko scaffold of a RDKit molecule object

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mol | Chem.rdchem.Mol | RDKit molecule to define a Murcko-Bemis scaffold of | None |

**Returns:**

| Type | Description |
|---|---|
| Chem.rdchem.Mol | RDKit molecule obeject of the Murcko-Bemis scaffold |

??? example "View Source"
        def get_murcko_scaffold(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:

            """

            Function to get the Murcko scaffold of a RDKit molecule object

            Args:

                mol (Chem.rdchem.Mol): RDKit molecule to define a Murcko-Bemis scaffold of

            Returns:

                Chem.rdchem.Mol: RDKit molecule obeject of the Murcko-Bemis scaffold

            Doctest:

            >>> get_murcko_scaffold(Chem.MolFromSmiles("Cc1ccccc1")) # doctest: +ELLIPSIS

            <rdkit.Chem.rdchem.Mol object at ...>

            >>> m = get_murcko_scaffold(Chem.MolFromSmiles("Cc1ccccc1")) # doctest: +ELLIPSIS

            >>> s = Chem.MolToSmiles(m)

            >>> s == "c1ccccc1"

            True

            """

            mol_scaffold = MurckoScaffold.GetScaffoldForMol(mol)

            return mol_scaffold


### get_sub_series

```python3
def get_sub_series(
    series_dict: dict,
    series_df: Optional[pandas.core.frame.DataFrame] = None,
    id_column: Optional[str] = 'name_id',
    smiles_col: Optional[str] = 'smiles',
    smiles: Optional[List[str]] = None
) -> Union[Tuple[dict, dict], dict]
```

Get a set of sub series from a series based on SMARTS defined substructures. This is a scffold defined sub-series search. The return depends on the input.

If you input a smiles list you get back a dictionary with a mask and the smiles sub set in each sub series i.e.
{
"sub series name": {
                    "mask: [bool],
                    "smiles: [str]
                }
}
If you input a dataframe with a smiles column you get back two dictionaries: the first as stated previously without the smiles key
{
"sub series name":  [bool]
}

The second is as above but is entire sub set of the input dataframe
{
"sub series name":  pd.DataFrame
}

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| series_dict | dict | A dictionary of the form {"sub series name": "SMARTS or SMILES defining the sub series based on sub sturcture presence"}. There can be multiple keys and values for multiple sub-series. | None |
| series_df | pd.Dataframe | Dataframe of containing at least the smiles column and the ID column | None |
| id_column | str | ID column for each molecule | None |
| smiles_col | str | The smiles column name in the series_df. Defaults to 'smiles'. | 'smiles' |
| smiles | Optional[List[str]] | List of smiles to put into sub series. Defaults to None. | None |

??? example "View Source"
        def get_sub_series(

            series_dict: dict,

            series_df: Optional[pd.DataFrame] = None,

            id_column: Optional[str] = "name_id",

            smiles_col: Optional[str] = "smiles",

            smiles: Optional[List[str]] = None,

        ) -> Union[Tuple[dict, dict], dict]:

            """

            Get a set of sub series from a series based on SMARTS defined substructures. This is a scffold defined sub-series search. The return depends on the input.

            If you input a smiles list you get back a dictionary with a mask and the smiles sub set in each sub series i.e.

            {

            "sub series name": {

                                "mask: [bool],

                                "smiles: [str]

                            }

            }

            If you input a dataframe with a smiles column you get back two dictionaries: the first as stated previously without the smiles key

            {

            "sub series name":  [bool]

            }

            The second is as above but is entire sub set of the input dataframe

            {

            "sub series name":  pd.DataFrame

            }

            Args:

                series_dict (dict): A dictionary of the form {"sub series name": "SMARTS or SMILES defining the sub series based on sub sturcture presence"}. There can be multiple keys and values for multiple sub-series.

                series_df (pd.Dataframe): Dataframe of containing at least the smiles column and the ID column

                id_column (str, optional): ID column for each molecule

                smiles_col (str, optional): The smiles column name in the series_df. Defaults to 'smiles'.

                smiles (Optional[List[str]], optional): List of smiles to put into sub series. Defaults to None.

            """

            # Check that one input has been provided

            if all(ent is None for ent in [series_df, smiles]):

                raise RuntimeError("Either series df or smiles list must be provided")

            # For a list of smiles we will return only a mask ans sub list

            if smiles is not None and series_df is None:

                # set up the outputs

                subseries_dict = {}

                mols = [Chem.MolFromSmiles(smi) for smi in smiles]

                # get the sub-series

                for name, smiles in series_dict.items():

                    sub_mol = Chem.MolFromSmarts(smiles)

                    mask = [mol.HasSubstructMatch(sub_mol) for mol in mols]

                    log.info(f"Sub-series {name} has {sum(mask)} molecules present")

                    subseries_dict[name] = {

                        "mask": mask,

                        "smiles": [s for s, m in zip(smiles, mask) if m is True],

                    }

                overlap_present, overlaps = check_for_sub_series_overlap(

                    subseries_dict=subseries_dict

                )

                if overlap_present is True:

                    log.info(overlaps)

                return subseries_dict

            elif series_df is not None:

                # set up the outputs

                subseries_dict = {}

                subseries_dict_dfs = {}

                if id_column is None:

                    id_column = smiles_col

                if smiles_col is None:

                    raise RuntimeError("Must define a smiles column when provding a dataframe")

                else:

                    try:

                        _ = series_df[smiles_col]

                    except KeyError:

                        raise KeyError(

                            f"ERROR - no smiles key {smiles_col} found in the dataframe columns"

                        )

                mols = [Chem.MolFromSmiles(smi) for smi in series_df[smiles_col].to_list()]

                for name, smiles in series_dict.items():

                    sub_mol = Chem.MolFromSmarts(smiles)

                    mask = [mol.HasSubstructMatch(sub_mol) for mol in mols]

                    log.info(f"Sub-series {name} has {sum(mask)} molecules present")

                    subseries_dict[name] = series_df.loc[mask][id_column].to_list()

                    subseries_dict_dfs[name] = series_df.loc[mask].copy()

                overlap_present, overlaps = check_for_sub_series_overlap(

                    subseries_dict=subseries_dict

                )

                if overlap_present is True:

                    log.info(overlaps)

                return subseries_dict, subseries_dict_dfs
