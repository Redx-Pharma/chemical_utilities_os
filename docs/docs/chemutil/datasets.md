# Module chemutil.datasets

Module for generating datasets and splitting them

??? example "View Source"
        #!/usr/bin.env python3

        # -*- coding: utf-8 -*-

        """

        Module for generating datasets and splitting them

        """

        from typing import List, Union, Tuple, Optional

        from collections import OrderedDict

        from rdkit import Chem

        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

        from rdkit.Chem.rdMolHash import MolHash

        from chemutil import helpers

        import pandas as pd

        import deepchem as dc

        import logging

        log = logging.getLogger(__name__)



        def find_columns_matching_regex(df: pd.DataFrame, regex: str) -> List[str]:

            """

            Function to get a set of columns based on a partial string match with the column name. For example columns = ["smiles", "feature_1", "feature_2", "target"] regex = "feature_" would return ["feature_1", "feature_2"].

            Args:

                df (pd.DataFrame): data frame whose columns we want to get a sub set of

                regex (str): partial or full string to find al column names that contain it

            Returns:

                List[str]: List of all of the column names that contain the regex

            """

            return [ent for ent in df.columns if regex in ent]



        def pandas_to_deepchem(

            df: pd.DataFrame,

            smiles_column: str,

            task_columns: List[str],

            feature_columns: Optional[List[str]] = None,

            feature_column_regex: Optional[str] = None,

            **kwargs,

        ) -> dc.data.datasets.NumpyDataset:

            """

            Function to build a deepchem data set from a pandas dataframe

            Args:

                df (pd.DataFrame): All of the data in one dataframe to split into a dataset for ML/AI

                smiles_column (str): The column name that contains smiles strings

                task_columns (List[str]): List of y task column names

                feature_columns (Optional[List[str]], optional): Must provide this or feature_column_regex. If this is provided it is a list of the column names that correspond to features i.e. the X mat. Defaults to None.

                fesature_column_regex (Optional[str], optional): Must provide this or feature_columns. If this is provided it is a partial or full string to identify all columns that contain it as feature columns. Defaults to None.

            Returns:

                dc.data.datasets.NumpyDataset: A deepchem numpy dataset

            """

            if feature_columns is None and feature_column_regex is not None:

                feature_columns = find_columns_matching_regex(df, feature_column_regex)

            return dc.data.NumpyDataset.from_dataframe(

                df, X=feature_columns, y=task_columns, ids=smiles_column

            )



        def dataset_int_to_fractions(ds: dc.data.datasets.NumpyDataset, n: int) -> float:

            """

            Function to convert a fixed numebr of data point to use from a dataset into a fraction

            Args:

                ds (dc.data.datasets.NumpyDataset): A deepchem dataset

                n (int): The number of points one wants from the dataset

            Returns:

                float: The fraction of the data set that n points corresponds to

            """

            m = ds.get_shape()[0][0]

            return round((n / m), 2)



        def molecular_dataset_split(

            ds: Optional[dc.data.datasets.NumpyDataset] = None,

            df: Optional[pd.DataFrame] = None,

            splitter_type: Union[str, dc.splits.Splitter] = "fp",

            train: Optional[Union[float, int]] = None,

            test: Optional[Union[float, int]] = None,

            holdout: Optional[Union[float, int]] = None,

            kfold: Optional[int] = None,

            butina_cutoff: float = 0.6,

            specified_validation_indexes: Optional[List[int]] = None,

            specified_test_indexes: Optional[List[int]] = None,

            groups: Optional[List[Union[int, str]]] = None,

            random_seed: int = helpers.random_seed,

            **kwargs,

        ) -> Tuple[dc.data.datasets.NumpyDataset, dc.splits.Splitter]:

            """

            Function to perform data set splitting

            Args:

                df (Optional[pd.DataFrame], optional): Dataframe to convert to a dataset and split. Please note the keyword args for this are passed through kwargs here, they correspond to the arguments for pandas_to_deepchem(). Defaults to None.

                ds (Optional[dc.data.datasets.NumpyDataset], optional): deepchem dataset to split. Defaults to None.

                train (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for training. Defaults to None.

                test (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for testing. Defaults to None.

                holdout (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for validation. Defaults to None.

                kfold (Optional[int], optional): The number of k fold splits. Defaults to None.

                butina_cutoff (float): The Butina cluster similarity cutoff only used with Butina splitter. Defaults as 0.6.

                specified_validation_indexes: (Optional[List[int]]): The validation indexes of the data set only used with the SpecifiedSplitter. Defaults as None.

                specified_test_indexes (Optional[List[int]]): The test indexes of the data set only used with the SpecifiedSplitter. Defaults as None.

                groups (Optional[List[Union[int, str]]]): Onlt used with RandomGroupSplitter to set which molecules are part of which group. Defaults as None.

                random_seed (int) The value of the random seed to seed a pseudo random number generator. Defaults as 15751.

                Kwargs are used to pass options to pandas_to_deepchem if a dataframe (df) is passed on here rather than a dataset ds.

            Returns:

                Tuple[dc.data.datasets.NumpyDataset, dc.splits.Splitter]: Tuple of data sets for training, testing, validation or kfold splits depending on the input arguments and the splitter

            """

            if df is not None and ds is None:

                ds = pandas_to_deepchem(df, **kwargs)

            elif all(ent is None for ent in [df, ds]):

                raise RuntimeError(

                    "ERROR - both ds (deepchem dataset) and df (pandas dataframe) are none one must not be."

                )

            log.debug(

                f"Training fraction {train} {type(train)} Testing fraction {test} {type(test)}validation fraction {holdout} {type(holdout)}"

            )

            if isinstance(train, int):

                train = dataset_int_to_fractions(ds, train)

                log.info(f"The trianing set will contain {train * 100.0}% of the molecles")

            if isinstance(test, int):

                test = dataset_int_to_fractions(ds, test)

                log.info(f"The test set will contain {test * 100.0}% of the molecles")

            if isinstance(holdout, int):

                holdout = dataset_int_to_fractions(ds, holdout)

                log.info(

                    f"The hold out validation set will contain {holdout * 100.0}% molecles"

                )

            known_splitters = sorted(

                [

                    "fingerprint",

                    "molecularweight",

                    "scaffold",

                    "minmax",

                    "butina",

                    "random",

                    "randomstratified",

                    "index",

                    "specified",

                    "random group",

                ]

            )

            if not isinstance(splitter_type, str):

                # instantiate a splitter class passed in

                splitter = splitter_type()

            else:

                splitter_type = splitter_type.strip().lower()

                if splitter_type == "fp" or splitter_type == "fingerprint":

                    splitter = dc.splits.FingerprintSplitter()

                elif splitter_type == "mw" or splitter_type == "molecularweight":

                    splitter = dc.splits.MolecularWeightSplitter()

                elif splitter_type == "sca" or splitter_type == "scaffold":

                    splitter = dc.splits.ScaffoldSplitter()

                elif splitter_type == "mm" or splitter_type == "minmax":

                    splitter = dc.splits.MaxMinSplitter()

                elif splitter_type == "but" or splitter_type == "butina":

                    splitter = dc.splits.ButinaSplitter(cutoff=butina_cutoff)

                elif splitter_type == "rad" or splitter_type == "random":

                    splitter = dc.splits.RandomSplitter()

                elif splitter_type == "rst" or splitter_type == "randomstratified":

                    splitter = dc.splits.RandomStratifiedSplitter()

                elif splitter_type == "rang" or splitter_type == "random_group":

                    splitter = dc.splits.RandomGroupSplitter(groups=groups)

                elif splitter_type == "inx" or splitter_type == "index":

                    log.info(

                        f"The index splitter assumes the data is ordered so that the first {train} data points are for training, the next {holdout} data points are for validation and the the remain {test} data points are for testing"

                    )

                    splitter = dc.splits.IndexSplitter()

                elif splitter_type == "spec" or splitter_type == "specified":

                    if any(

                        ent is None

                        for ent in [specified_validation_indexes, specified_test_indexes]

                    ):

                        raise RuntimeError(

                            f"Both the specified validation or test set indexes must be specified. Validation indexes: {specified_validation_indexes}, test indexes: {specified_test_indexes}."

                        )

                    splitter = dc.splits.SpecifiedSplitter(

                        valid_indices=specified_validation_indexes,

                        test_indices=specified_test_indexes,

                    )

                else:

                    raise RuntimeError(

                        f"ERROR - unrecognised splitter {splitter_type} requested. Avaliable types are {', '.join(known_splitters)}"

                    )

            log.info(f"Using splitter of type {str(splitter_type)}")

            # Internally deepchem splitter defines a self.split method this is called by each of these methods depending on the the splitters internally defne self.split method

            if kfold is not None:

                if holdout is None:

                    Xy = splitter.k_fold_split(ds, kfold, seed=random_seed)

                    log.info(

                        f"No holdout validation k fold Splitter: {splitter}, Kfolds: {kfold}"

                    )

                    return Xy, splitter

                if holdout is not None:

                    log.info(

                        f"Extracting holdout set before k fold split. This will be {holdout * 100.0}% of the data."

                    )

                    log.info(f"This is expected to be {ds.X.shape[0] * holdout} holdout points")

                    kf_train_set, holdout_set = splitter.train_test_split(

                        ds, frac_train=train, seed=random_seed

                    )

                    Xy = splitter.k_fold_split(kf_train_set, kfold, seed=random_seed)

                    log.info(

                        f"With holdout validation k fold Splitter: {splitter}, Kfolds: {kfold}"

                    )

                    return Xy, holdout_set, splitter

            if all(ent is not None for ent in [train, test]) and holdout is None:

                log.info(

                    f"Building train test splits of train: {train * 100.0}% and test: {test * 100.0}%"

                )

                log.info(

                    f"This is expected to be {ds.X.shape[0] * train} training points and {ds.X.shape[0] * test} test points"

                )

                Xy = splitter.train_test_split(ds, frac_train=train, seed=random_seed)

                log.info(

                    f"Splitter: {splitter}, train percentage {train * 100.0}% test percentage {test * 100.0}%"

                )

                return *Xy, splitter

            if all(ent is not None for ent in [train, holdout]) and test is None:

                log.info(

                    f"Building train holdout/validation splits of train: {train * 100.0}% and holdout/validation: {holdout * 100.0}%"

                )

                log.info(

                    f"This is expected to be {ds.X.shape[0] * train} training points and {ds.X.shape[0] * holdout} holdout points"

                )

                Xy = splitter.train_test_split(ds, frac_train=train, seed=random_seed)

                log.info(

                    f"Splitter: {splitter}, train percentage {train * 100.0}% test percentage {test * 100.0}%"

                )

                return *Xy, splitter

            if all(ent is not None for ent in [train, test, holdout]):

                Xy = splitter.train_valid_test_split(

                    ds,

                    frac_train=train,

                    frac_test=test,

                    frac_valid=holdout,

                    seed=random_seed,

                )

                log.info(

                    f"Splitter: {splitter} train percentage {train * 100.0}% test percentage {test * 100.0}% validation percentage {holdout * 100.0}%"

                )

                log.info(

                    f"This is expected to be {ds.X.shape[0] * train} training points, {ds.X.shape[0] * test} test points and {ds.X.shape[0] * holdout} holdout points"

                )

                return *Xy, splitter



        def get_scaffold(

            smiles: str, graph_only: bool = False, inc_chiral: bool = False

        ) -> str:

            """

            Function to generate scaffolds from smiles

            Args:

                smiles (str): smiles string fo a molecule

                graph_only (bool, optional): Wether to use the graph only i.e. make the scaffold indifferent to atom type and bond type. Defaults to False.

                inc_chiral (bool, optional): include chirality in the scaffold generation. Defaults to False.

            Returns:

                str: scaffold as a string SMILES/SMARTS

            """

            mol = Chem.MolFromSmiles(smiles)

            if mol is None:

                log.warning(f"WARNING - invalid smiles string {smiles} skipping")

                return None

            if graph_only is False:

                scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=inc_chiral)

                return scaffold

            else:

                scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=inc_chiral)

                skeleton_scaffold = MolHash(Chem.MolFromSmiles(scaffold, 1))

                return skeleton_scaffold



        def get_scaffolds_groups_from_scaffold_splitter(smiles: List[str]) -> dict:

            """

            Function to perform scaffold splitting as in DeepChem but returns a dictionary of scaffold smiles to smiles indexes and smiles strings directly

            Args:

                smiles (List[str]): list of smiles strings in the data set to scaffold split

            Returns:

                dict: dictionary of scaffold smiles to smiles indexes in the input smiles list and the smiles strings themselves

            """

            scaffold_dict = {}

            scaffolds = [get_scaffold(smi) for smi in smiles]

            for ith, s in enumerate(scaffolds):

                if s not in scaffold_dict:

                    scaffold_dict[s] = [ith]

                else:

                    scaffold_dict[s].append(ith)

            # Sorted as in deepchem but into a ordered dict with scaffold and the indexes of the smiles matching

            scaffolds = {key: sorted(value) for key, value in scaffold_dict.items()}

            scaffold_maps = OrderedDict(

                [

                    (scaffold, {"smiles_indexes": scaffold_set})

                    for scaffold, scaffold_set in sorted(

                        scaffolds.items(),

                        key=lambda ent: (len(ent[1]), ent[1][0]),

                        reverse=True,

                    )

                ]

            )

            for k in scaffold_maps.keys():

                scaffold_maps[k]["smiles_with_scaffold"] = [

                    smiles[jth] for jth in scaffold_maps[k]["smiles_indexes"]

                ]

            return scaffold_maps

## Variables

```python3
log
```

## Functions


### dataset_int_to_fractions

```python3
def dataset_int_to_fractions(
    ds: deepchem.data.datasets.NumpyDataset,
    n: int
) -> float
```

Function to convert a fixed numebr of data point to use from a dataset into a fraction

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| ds | dc.data.datasets.NumpyDataset | A deepchem dataset | None |
| n | int | The number of points one wants from the dataset | None |

**Returns:**

| Type | Description |
|---|---|
| float | The fraction of the data set that n points corresponds to |

??? example "View Source"
        def dataset_int_to_fractions(ds: dc.data.datasets.NumpyDataset, n: int) -> float:

            """

            Function to convert a fixed numebr of data point to use from a dataset into a fraction

            Args:

                ds (dc.data.datasets.NumpyDataset): A deepchem dataset

                n (int): The number of points one wants from the dataset

            Returns:

                float: The fraction of the data set that n points corresponds to

            """

            m = ds.get_shape()[0][0]

            return round((n / m), 2)


### find_columns_matching_regex

```python3
def find_columns_matching_regex(
    df: pandas.core.frame.DataFrame,
    regex: str
) -> List[str]
```

Function to get a set of columns based on a partial string match with the column name. For example columns = ["smiles", "feature_1", "feature_2", "target"] regex = "feature_" would return ["feature_1", "feature_2"].

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | pd.DataFrame | data frame whose columns we want to get a sub set of | None |
| regex | str | partial or full string to find al column names that contain it | None |

**Returns:**

| Type | Description |
|---|---|
| List[str] | List of all of the column names that contain the regex |

??? example "View Source"
        def find_columns_matching_regex(df: pd.DataFrame, regex: str) -> List[str]:

            """

            Function to get a set of columns based on a partial string match with the column name. For example columns = ["smiles", "feature_1", "feature_2", "target"] regex = "feature_" would return ["feature_1", "feature_2"].

            Args:

                df (pd.DataFrame): data frame whose columns we want to get a sub set of

                regex (str): partial or full string to find al column names that contain it

            Returns:

                List[str]: List of all of the column names that contain the regex

            """

            return [ent for ent in df.columns if regex in ent]


### get_scaffold

```python3
def get_scaffold(
    smiles: str,
    graph_only: bool = False,
    inc_chiral: bool = False
) -> str
```

Function to generate scaffolds from smiles

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| smiles | str | smiles string fo a molecule | None |
| graph_only | bool | Wether to use the graph only i.e. make the scaffold indifferent to atom type and bond type. Defaults to False. | False |
| inc_chiral | bool | include chirality in the scaffold generation. Defaults to False. | False |

**Returns:**

| Type | Description |
|---|---|
| str | scaffold as a string SMILES/SMARTS |

??? example "View Source"
        def get_scaffold(

            smiles: str, graph_only: bool = False, inc_chiral: bool = False

        ) -> str:

            """

            Function to generate scaffolds from smiles

            Args:

                smiles (str): smiles string fo a molecule

                graph_only (bool, optional): Wether to use the graph only i.e. make the scaffold indifferent to atom type and bond type. Defaults to False.

                inc_chiral (bool, optional): include chirality in the scaffold generation. Defaults to False.

            Returns:

                str: scaffold as a string SMILES/SMARTS

            """

            mol = Chem.MolFromSmiles(smiles)

            if mol is None:

                log.warning(f"WARNING - invalid smiles string {smiles} skipping")

                return None

            if graph_only is False:

                scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=inc_chiral)

                return scaffold

            else:

                scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=inc_chiral)

                skeleton_scaffold = MolHash(Chem.MolFromSmiles(scaffold, 1))

                return skeleton_scaffold


### get_scaffolds_groups_from_scaffold_splitter

```python3
def get_scaffolds_groups_from_scaffold_splitter(
    smiles: List[str]
) -> dict
```

Function to perform scaffold splitting as in DeepChem but returns a dictionary of scaffold smiles to smiles indexes and smiles strings directly

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| smiles | List[str] | list of smiles strings in the data set to scaffold split | None |

**Returns:**

| Type | Description |
|---|---|
| dict | dictionary of scaffold smiles to smiles indexes in the input smiles list and the smiles strings themselves |

??? example "View Source"
        def get_scaffolds_groups_from_scaffold_splitter(smiles: List[str]) -> dict:

            """

            Function to perform scaffold splitting as in DeepChem but returns a dictionary of scaffold smiles to smiles indexes and smiles strings directly

            Args:

                smiles (List[str]): list of smiles strings in the data set to scaffold split

            Returns:

                dict: dictionary of scaffold smiles to smiles indexes in the input smiles list and the smiles strings themselves

            """

            scaffold_dict = {}

            scaffolds = [get_scaffold(smi) for smi in smiles]

            for ith, s in enumerate(scaffolds):

                if s not in scaffold_dict:

                    scaffold_dict[s] = [ith]

                else:

                    scaffold_dict[s].append(ith)

            # Sorted as in deepchem but into a ordered dict with scaffold and the indexes of the smiles matching

            scaffolds = {key: sorted(value) for key, value in scaffold_dict.items()}

            scaffold_maps = OrderedDict(

                [

                    (scaffold, {"smiles_indexes": scaffold_set})

                    for scaffold, scaffold_set in sorted(

                        scaffolds.items(),

                        key=lambda ent: (len(ent[1]), ent[1][0]),

                        reverse=True,

                    )

                ]

            )

            for k in scaffold_maps.keys():

                scaffold_maps[k]["smiles_with_scaffold"] = [

                    smiles[jth] for jth in scaffold_maps[k]["smiles_indexes"]

                ]

            return scaffold_maps


### molecular_dataset_split

```python3
def molecular_dataset_split(
    ds: Optional[deepchem.data.datasets.NumpyDataset] = None,
    df: Optional[pandas.core.frame.DataFrame] = None,
    splitter_type: Union[str, deepchem.splits.splitters.Splitter] = 'fp',
    train: Union[float, int, NoneType] = None,
    test: Union[float, int, NoneType] = None,
    holdout: Union[float, int, NoneType] = None,
    kfold: Optional[int] = None,
    butina_cutoff: float = 0.6,
    specified_validation_indexes: Optional[List[int]] = None,
    specified_test_indexes: Optional[List[int]] = None,
    groups: Optional[List[Union[int, str]]] = None,
    random_seed: int = 15751,
    **kwargs
) -> Tuple[deepchem.data.datasets.NumpyDataset, deepchem.splits.splitters.Splitter]
```

Function to perform data set splitting

Args:
    df (Optional[pd.DataFrame], optional): Dataframe to convert to a dataset and split. Please note the keyword args for this are passed through kwargs here, they correspond to the arguments for pandas_to_deepchem(). Defaults to None.
    ds (Optional[dc.data.datasets.NumpyDataset], optional): deepchem dataset to split. Defaults to None.
    train (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for training. Defaults to None.
    test (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for testing. Defaults to None.
    holdout (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for validation. Defaults to None.
    kfold (Optional[int], optional): The number of k fold splits. Defaults to None.
    butina_cutoff (float): The Butina cluster similarity cutoff only used with Butina splitter. Defaults as 0.6.
    specified_validation_indexes: (Optional[List[int]]): The validation indexes of the data set only used with the SpecifiedSplitter. Defaults as None.
    specified_test_indexes (Optional[List[int]]): The test indexes of the data set only used with the SpecifiedSplitter. Defaults as None.
    groups (Optional[List[Union[int, str]]]): Onlt used with RandomGroupSplitter to set which molecules are part of which group. Defaults as None.
    random_seed (int) The value of the random seed to seed a pseudo random number generator. Defaults as 15751.
    Kwargs are used to pass options to pandas_to_deepchem if a dataframe (df) is passed on here rather than a dataset ds.

Returns:
    Tuple[dc.data.datasets.NumpyDataset, dc.splits.Splitter]: Tuple of data sets for training, testing, validation or kfold splits depending on the input arguments and the splitter

??? example "View Source"
        def molecular_dataset_split(

            ds: Optional[dc.data.datasets.NumpyDataset] = None,

            df: Optional[pd.DataFrame] = None,

            splitter_type: Union[str, dc.splits.Splitter] = "fp",

            train: Optional[Union[float, int]] = None,

            test: Optional[Union[float, int]] = None,

            holdout: Optional[Union[float, int]] = None,

            kfold: Optional[int] = None,

            butina_cutoff: float = 0.6,

            specified_validation_indexes: Optional[List[int]] = None,

            specified_test_indexes: Optional[List[int]] = None,

            groups: Optional[List[Union[int, str]]] = None,

            random_seed: int = helpers.random_seed,

            **kwargs,

        ) -> Tuple[dc.data.datasets.NumpyDataset, dc.splits.Splitter]:

            """

            Function to perform data set splitting

            Args:

                df (Optional[pd.DataFrame], optional): Dataframe to convert to a dataset and split. Please note the keyword args for this are passed through kwargs here, they correspond to the arguments for pandas_to_deepchem(). Defaults to None.

                ds (Optional[dc.data.datasets.NumpyDataset], optional): deepchem dataset to split. Defaults to None.

                train (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for training. Defaults to None.

                test (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for testing. Defaults to None.

                holdout (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for validation. Defaults to None.

                kfold (Optional[int], optional): The number of k fold splits. Defaults to None.

                butina_cutoff (float): The Butina cluster similarity cutoff only used with Butina splitter. Defaults as 0.6.

                specified_validation_indexes: (Optional[List[int]]): The validation indexes of the data set only used with the SpecifiedSplitter. Defaults as None.

                specified_test_indexes (Optional[List[int]]): The test indexes of the data set only used with the SpecifiedSplitter. Defaults as None.

                groups (Optional[List[Union[int, str]]]): Onlt used with RandomGroupSplitter to set which molecules are part of which group. Defaults as None.

                random_seed (int) The value of the random seed to seed a pseudo random number generator. Defaults as 15751.

                Kwargs are used to pass options to pandas_to_deepchem if a dataframe (df) is passed on here rather than a dataset ds.

            Returns:

                Tuple[dc.data.datasets.NumpyDataset, dc.splits.Splitter]: Tuple of data sets for training, testing, validation or kfold splits depending on the input arguments and the splitter

            """

            if df is not None and ds is None:

                ds = pandas_to_deepchem(df, **kwargs)

            elif all(ent is None for ent in [df, ds]):

                raise RuntimeError(

                    "ERROR - both ds (deepchem dataset) and df (pandas dataframe) are none one must not be."

                )

            log.debug(

                f"Training fraction {train} {type(train)} Testing fraction {test} {type(test)}validation fraction {holdout} {type(holdout)}"

            )

            if isinstance(train, int):

                train = dataset_int_to_fractions(ds, train)

                log.info(f"The trianing set will contain {train * 100.0}% of the molecles")

            if isinstance(test, int):

                test = dataset_int_to_fractions(ds, test)

                log.info(f"The test set will contain {test * 100.0}% of the molecles")

            if isinstance(holdout, int):

                holdout = dataset_int_to_fractions(ds, holdout)

                log.info(

                    f"The hold out validation set will contain {holdout * 100.0}% molecles"

                )

            known_splitters = sorted(

                [

                    "fingerprint",

                    "molecularweight",

                    "scaffold",

                    "minmax",

                    "butina",

                    "random",

                    "randomstratified",

                    "index",

                    "specified",

                    "random group",

                ]

            )

            if not isinstance(splitter_type, str):

                # instantiate a splitter class passed in

                splitter = splitter_type()

            else:

                splitter_type = splitter_type.strip().lower()

                if splitter_type == "fp" or splitter_type == "fingerprint":

                    splitter = dc.splits.FingerprintSplitter()

                elif splitter_type == "mw" or splitter_type == "molecularweight":

                    splitter = dc.splits.MolecularWeightSplitter()

                elif splitter_type == "sca" or splitter_type == "scaffold":

                    splitter = dc.splits.ScaffoldSplitter()

                elif splitter_type == "mm" or splitter_type == "minmax":

                    splitter = dc.splits.MaxMinSplitter()

                elif splitter_type == "but" or splitter_type == "butina":

                    splitter = dc.splits.ButinaSplitter(cutoff=butina_cutoff)

                elif splitter_type == "rad" or splitter_type == "random":

                    splitter = dc.splits.RandomSplitter()

                elif splitter_type == "rst" or splitter_type == "randomstratified":

                    splitter = dc.splits.RandomStratifiedSplitter()

                elif splitter_type == "rang" or splitter_type == "random_group":

                    splitter = dc.splits.RandomGroupSplitter(groups=groups)

                elif splitter_type == "inx" or splitter_type == "index":

                    log.info(

                        f"The index splitter assumes the data is ordered so that the first {train} data points are for training, the next {holdout} data points are for validation and the the remain {test} data points are for testing"

                    )

                    splitter = dc.splits.IndexSplitter()

                elif splitter_type == "spec" or splitter_type == "specified":

                    if any(

                        ent is None

                        for ent in [specified_validation_indexes, specified_test_indexes]

                    ):

                        raise RuntimeError(

                            f"Both the specified validation or test set indexes must be specified. Validation indexes: {specified_validation_indexes}, test indexes: {specified_test_indexes}."

                        )

                    splitter = dc.splits.SpecifiedSplitter(

                        valid_indices=specified_validation_indexes,

                        test_indices=specified_test_indexes,

                    )

                else:

                    raise RuntimeError(

                        f"ERROR - unrecognised splitter {splitter_type} requested. Avaliable types are {', '.join(known_splitters)}"

                    )

            log.info(f"Using splitter of type {str(splitter_type)}")

            # Internally deepchem splitter defines a self.split method this is called by each of these methods depending on the the splitters internally defne self.split method

            if kfold is not None:

                if holdout is None:

                    Xy = splitter.k_fold_split(ds, kfold, seed=random_seed)

                    log.info(

                        f"No holdout validation k fold Splitter: {splitter}, Kfolds: {kfold}"

                    )

                    return Xy, splitter

                if holdout is not None:

                    log.info(

                        f"Extracting holdout set before k fold split. This will be {holdout * 100.0}% of the data."

                    )

                    log.info(f"This is expected to be {ds.X.shape[0] * holdout} holdout points")

                    kf_train_set, holdout_set = splitter.train_test_split(

                        ds, frac_train=train, seed=random_seed

                    )

                    Xy = splitter.k_fold_split(kf_train_set, kfold, seed=random_seed)

                    log.info(

                        f"With holdout validation k fold Splitter: {splitter}, Kfolds: {kfold}"

                    )

                    return Xy, holdout_set, splitter

            if all(ent is not None for ent in [train, test]) and holdout is None:

                log.info(

                    f"Building train test splits of train: {train * 100.0}% and test: {test * 100.0}%"

                )

                log.info(

                    f"This is expected to be {ds.X.shape[0] * train} training points and {ds.X.shape[0] * test} test points"

                )

                Xy = splitter.train_test_split(ds, frac_train=train, seed=random_seed)

                log.info(

                    f"Splitter: {splitter}, train percentage {train * 100.0}% test percentage {test * 100.0}%"

                )

                return *Xy, splitter

            if all(ent is not None for ent in [train, holdout]) and test is None:

                log.info(

                    f"Building train holdout/validation splits of train: {train * 100.0}% and holdout/validation: {holdout * 100.0}%"

                )

                log.info(

                    f"This is expected to be {ds.X.shape[0] * train} training points and {ds.X.shape[0] * holdout} holdout points"

                )

                Xy = splitter.train_test_split(ds, frac_train=train, seed=random_seed)

                log.info(

                    f"Splitter: {splitter}, train percentage {train * 100.0}% test percentage {test * 100.0}%"

                )

                return *Xy, splitter

            if all(ent is not None for ent in [train, test, holdout]):

                Xy = splitter.train_valid_test_split(

                    ds,

                    frac_train=train,

                    frac_test=test,

                    frac_valid=holdout,

                    seed=random_seed,

                )

                log.info(

                    f"Splitter: {splitter} train percentage {train * 100.0}% test percentage {test * 100.0}% validation percentage {holdout * 100.0}%"

                )

                log.info(

                    f"This is expected to be {ds.X.shape[0] * train} training points, {ds.X.shape[0] * test} test points and {ds.X.shape[0] * holdout} holdout points"

                )

                return *Xy, splitter


### pandas_to_deepchem

```python3
def pandas_to_deepchem(
    df: pandas.core.frame.DataFrame,
    smiles_column: str,
    task_columns: List[str],
    feature_columns: Optional[List[str]] = None,
    feature_column_regex: Optional[str] = None,
    **kwargs
) -> deepchem.data.datasets.NumpyDataset
```

Function to build a deepchem data set from a pandas dataframe

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | pd.DataFrame | All of the data in one dataframe to split into a dataset for ML/AI | None |
| smiles_column | str | The column name that contains smiles strings | None |
| task_columns | List[str] | List of y task column names | None |
| feature_columns | Optional[List[str]] | Must provide this or feature_column_regex. If this is provided it is a list of the column names that correspond to features i.e. the X mat. Defaults to None. | None |
| fesature_column_regex | Optional[str] | Must provide this or feature_columns. If this is provided it is a partial or full string to identify all columns that contain it as feature columns. Defaults to None. | None |

**Returns:**

| Type | Description |
|---|---|
| dc.data.datasets.NumpyDataset | A deepchem numpy dataset |

??? example "View Source"
        def pandas_to_deepchem(

            df: pd.DataFrame,

            smiles_column: str,

            task_columns: List[str],

            feature_columns: Optional[List[str]] = None,

            feature_column_regex: Optional[str] = None,

            **kwargs,

        ) -> dc.data.datasets.NumpyDataset:

            """

            Function to build a deepchem data set from a pandas dataframe

            Args:

                df (pd.DataFrame): All of the data in one dataframe to split into a dataset for ML/AI

                smiles_column (str): The column name that contains smiles strings

                task_columns (List[str]): List of y task column names

                feature_columns (Optional[List[str]], optional): Must provide this or feature_column_regex. If this is provided it is a list of the column names that correspond to features i.e. the X mat. Defaults to None.

                fesature_column_regex (Optional[str], optional): Must provide this or feature_columns. If this is provided it is a partial or full string to identify all columns that contain it as feature columns. Defaults to None.

            Returns:

                dc.data.datasets.NumpyDataset: A deepchem numpy dataset

            """

            if feature_columns is None and feature_column_regex is not None:

                feature_columns = find_columns_matching_regex(df, feature_column_regex)

            return dc.data.NumpyDataset.from_dataframe(

                df, X=feature_columns, y=task_columns, ids=smiles_column

            )
