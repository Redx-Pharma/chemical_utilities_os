# Module chemutil.scaffold_analysis

module for methods to perform scaffold analysis

??? example "View Source"
        #!/usr/bin.env python3

        # -*- coding: utf-8 -*-

        """

        module for methods to perform scaffold analysis

        """

        import logging

        from typing import List, Optional, Tuple, Union

        import matplotlib.pyplot as plt

        import networkx as nx

        import numpy as np

        import pandas as pd

        import scaffoldgraph as sg

        from rdkit import Chem

        from rdkit.Chem import AllChem

        log = logging.getLogger(__name__)

        np.random.seed(15791)



        def get_scaffold_network_or_tree_from_dataframe(

            df: pd.DataFrame,

            tree_or_network: str = "network",

            smiles_column: str = "smiles",

            name_column: str = "labels",

            data_columns: Optional[List[str]] = None,

        ):  # -> Any | None:

            """

            Function to generate a scaffold network or tree from a dataframe

            Args:

                df (pd.DataFrame): The dataframe to generate the scaffold network or tree from

                tree_or_network (str): Whether to generate a scaffold network or tree

                smiles_column (str): The column in the dataframe containing the SMILES strings

                name_column (str): The column in the dataframe containing the names of the molecules

                data_columns (List[str]): The columns in the dataframe containing the data to be stored in the network or tree

            Returns:

                sg.ScaffoldNetwork | sg.ScaffoldTree: The scaffold network or tree

            """

            if tree_or_network == "network":

                network = sg.ScaffoldNetwork.from_dataframe(

                    df,

                    smiles_column=smiles_column,

                    name_column=name_column,

                    data_columns=data_columns,

                    progress=True,

                )

                log.info(

                    f"Generated scaffold network with {len(network.get_hierarchy_sizes())} hierarchy scaffold levels"

                )

                return network

            elif tree_or_network == "tree":

                tree = sg.ScaffoldTree.from_dataframe(

                    df,

                    smiles_column=smiles_column,

                    name_column=name_column,

                    data_columns=data_columns,

                    progress=True,

                )

                log.info(

                    f"Generated scaffold tree with {len(tree.get_hierarchy_sizes())} hierarchy scaffold levels"

                )

                return tree

            else:

                raise ValueError(f"Invalid value for tree_or_network: {tree_or_network}")



        def histogram_of_scaffold_levels(

            network: Union[sg.ScaffoldNetwork, sg.ScaffoldTree],

            title: str = "Scaffold Level Distribution",

            filename: str = "scaffold_level_hist.png",

        ) -> None:

            """

            Plot a histogram of the scaffold levels in the network

            Args:

                network (ScaffoldNetwork): The scaffold network to plot

                title (str): The title of the plot

                filename (str): The filename to save the plot to

            Returns:

                None

            """

            counts = network.get_hierarchy_sizes()

            lists = sorted(counts.items())

            x, y = zip(*lists)

            plt.figure(figsize=(8, 6))

            plt.bar(x, y)

            plt.xlabel("Hierarchy Scaffold Level")

            plt.ylabel("Scaffold Count")

            plt.title(title)

            plt.savefig(filename)



        def get_scaffolds_of_level(

            network: Union[sg.ScaffoldNetwork, sg.ScaffoldTree], level: int

        ) -> List[str]:

            """

            Function to get the scaffolds at a particular level in the network

            Args:

                network (ScaffoldNetwork): The scaffold network

                level (int): The level to get the scaffolds from

            Returns:

                List[str]: The scaffolds at the specified level

            """

            scaffold_list = list(network.get_scaffolds_in_hierarchy(level))

            log.info(f"There are {len(scaffold_list)} scaffolds for scaffold level {level}")

            return scaffold_list



        def get_molecules_of_scaffold(

            network: Union[sg.ScaffoldNetwork, sg.ScaffoldTree], scaffold: str

        ) -> List[str]:

            """

            Function to get the molecules that contain a scaffold

            Args:

                network (ScaffoldNetwork): The scaffold network

                scaffold (str): The scaffold to get the molecules for

            Returns:

                List[str]: The molecules that contain the scaffold

            """

            return list(network.get_molecules_for_scaffold(scaffold))



        if __name__ == "__main__":

            import doctest

            doctest.testmod(verbose=True)

## Variables

```python3
log
```

## Functions


### get_molecules_of_scaffold

```python3
def get_molecules_of_scaffold(
    network: Union[scaffoldgraph.network.ScaffoldNetwork, scaffoldgraph.tree.ScaffoldTree],
    scaffold: str
) -> List[str]
```

Function to get the molecules that contain a scaffold

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| network | ScaffoldNetwork | The scaffold network | None |
| scaffold | str | The scaffold to get the molecules for | None |

**Returns:**

| Type | Description |
|---|---|
| List[str] | The molecules that contain the scaffold |

??? example "View Source"
        def get_molecules_of_scaffold(

            network: Union[sg.ScaffoldNetwork, sg.ScaffoldTree], scaffold: str

        ) -> List[str]:

            """

            Function to get the molecules that contain a scaffold

            Args:

                network (ScaffoldNetwork): The scaffold network

                scaffold (str): The scaffold to get the molecules for

            Returns:

                List[str]: The molecules that contain the scaffold

            """

            return list(network.get_molecules_for_scaffold(scaffold))


### get_scaffold_network_or_tree_from_dataframe

```python3
def get_scaffold_network_or_tree_from_dataframe(
    df: pandas.core.frame.DataFrame,
    tree_or_network: str = 'network',
    smiles_column: str = 'smiles',
    name_column: str = 'labels',
    data_columns: Optional[List[str]] = None
)
```

Function to generate a scaffold network or tree from a dataframe

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | pd.DataFrame | The dataframe to generate the scaffold network or tree from | None |
| tree_or_network | str | Whether to generate a scaffold network or tree | None |
| smiles_column | str | The column in the dataframe containing the SMILES strings | None |
| name_column | str | The column in the dataframe containing the names of the molecules | None |
| data_columns | List[str] | The columns in the dataframe containing the data to be stored in the network or tree | None |

**Returns:**

| Type | Description |
|---|---|
| None | sg.ScaffoldNetwork | sg.ScaffoldTree: The scaffold network or tree |

??? example "View Source"
        def get_scaffold_network_or_tree_from_dataframe(

            df: pd.DataFrame,

            tree_or_network: str = "network",

            smiles_column: str = "smiles",

            name_column: str = "labels",

            data_columns: Optional[List[str]] = None,

        ):  # -> Any | None:

            """

            Function to generate a scaffold network or tree from a dataframe

            Args:

                df (pd.DataFrame): The dataframe to generate the scaffold network or tree from

                tree_or_network (str): Whether to generate a scaffold network or tree

                smiles_column (str): The column in the dataframe containing the SMILES strings

                name_column (str): The column in the dataframe containing the names of the molecules

                data_columns (List[str]): The columns in the dataframe containing the data to be stored in the network or tree

            Returns:

                sg.ScaffoldNetwork | sg.ScaffoldTree: The scaffold network or tree

            """

            if tree_or_network == "network":

                network = sg.ScaffoldNetwork.from_dataframe(

                    df,

                    smiles_column=smiles_column,

                    name_column=name_column,

                    data_columns=data_columns,

                    progress=True,

                )

                log.info(

                    f"Generated scaffold network with {len(network.get_hierarchy_sizes())} hierarchy scaffold levels"

                )

                return network

            elif tree_or_network == "tree":

                tree = sg.ScaffoldTree.from_dataframe(

                    df,

                    smiles_column=smiles_column,

                    name_column=name_column,

                    data_columns=data_columns,

                    progress=True,

                )

                log.info(

                    f"Generated scaffold tree with {len(tree.get_hierarchy_sizes())} hierarchy scaffold levels"

                )

                return tree

            else:

                raise ValueError(f"Invalid value for tree_or_network: {tree_or_network}")


### get_scaffolds_of_level

```python3
def get_scaffolds_of_level(
    network: Union[scaffoldgraph.network.ScaffoldNetwork, scaffoldgraph.tree.ScaffoldTree],
    level: int
) -> List[str]
```

Function to get the scaffolds at a particular level in the network

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| network | ScaffoldNetwork | The scaffold network | None |
| level | int | The level to get the scaffolds from | None |

**Returns:**

| Type | Description |
|---|---|
| List[str] | The scaffolds at the specified level |

??? example "View Source"
        def get_scaffolds_of_level(

            network: Union[sg.ScaffoldNetwork, sg.ScaffoldTree], level: int

        ) -> List[str]:

            """

            Function to get the scaffolds at a particular level in the network

            Args:

                network (ScaffoldNetwork): The scaffold network

                level (int): The level to get the scaffolds from

            Returns:

                List[str]: The scaffolds at the specified level

            """

            scaffold_list = list(network.get_scaffolds_in_hierarchy(level))

            log.info(f"There are {len(scaffold_list)} scaffolds for scaffold level {level}")

            return scaffold_list


### histogram_of_scaffold_levels

```python3
def histogram_of_scaffold_levels(
    network: Union[scaffoldgraph.network.ScaffoldNetwork, scaffoldgraph.tree.ScaffoldTree],
    title: str = 'Scaffold Level Distribution',
    filename: str = 'scaffold_level_hist.png'
) -> None
```

Plot a histogram of the scaffold levels in the network

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| network | ScaffoldNetwork | The scaffold network to plot | None |
| title | str | The title of the plot | None |
| filename | str | The filename to save the plot to | None |

**Returns:**

| Type | Description |
|---|---|
| None | None |

??? example "View Source"
        def histogram_of_scaffold_levels(

            network: Union[sg.ScaffoldNetwork, sg.ScaffoldTree],

            title: str = "Scaffold Level Distribution",

            filename: str = "scaffold_level_hist.png",

        ) -> None:

            """

            Plot a histogram of the scaffold levels in the network

            Args:

                network (ScaffoldNetwork): The scaffold network to plot

                title (str): The title of the plot

                filename (str): The filename to save the plot to

            Returns:

                None

            """

            counts = network.get_hierarchy_sizes()

            lists = sorted(counts.items())

            x, y = zip(*lists)

            plt.figure(figsize=(8, 6))

            plt.bar(x, y)

            plt.xlabel("Hierarchy Scaffold Level")

            plt.ylabel("Scaffold Count")

            plt.title(title)

            plt.savefig(filename)
