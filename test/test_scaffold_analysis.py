import pandas as pd
import pytest
import scaffoldgraph as sg
from chemutil.scaffold_analysis import (
    get_molecules_of_scaffold,
    get_scaffold_network_or_tree_from_dataframe,
    get_scaffolds_of_level,
    histogram_of_scaffold_levels,
)


@pytest.fixture
def df() -> pd.DataFrame:
    """
    pandas_dataframe fixture holding the default input dataframe

    Returns:
        pd.DataFrame - pandas data frame
    """
    data = {
        "smiles": ["c1ccccc1", "c1ccccc1C", "c1ccccc1CN", "CCCC(CN)CC(=O)O"],
        "labels": [
            "benzene",
            "toluene",
            "aminotoluene",
            "3-(aminomethyl)hexanoic acid",
        ],
        "fake_property": [1.0, 2.0, 4.2, 5.7],
    }
    dataf = pd.DataFrame(data)

    return dataf


def test_get_scaffold_network(df):
    network = get_scaffold_network_or_tree_from_dataframe(df, tree_or_network="network")
    assert isinstance(network, sg.ScaffoldNetwork)
    assert len(network.get_hierarchy_sizes()) > 0


def test_get_scaffold_tree(df):
    tree = get_scaffold_network_or_tree_from_dataframe(df, tree_or_network="tree")
    assert isinstance(tree, sg.ScaffoldTree)
    assert len(tree.get_hierarchy_sizes()) > 0


def test_histogram_of_scaffold_levels(tmp_path, df):
    network = get_scaffold_network_or_tree_from_dataframe(df, tree_or_network="network")
    filename = tmp_path / "scaffold_level_hist.png"
    histogram_of_scaffold_levels(network, filename=str(filename))
    assert filename.exists()


def test_get_scaffolds_of_level(df):
    network = get_scaffold_network_or_tree_from_dataframe(df, tree_or_network="network")
    scaffolds = get_scaffolds_of_level(network, level=1)
    assert isinstance(scaffolds, list)
    assert len(scaffolds) > 0


def test_get_molecules_of_scaffold(df):
    network = get_scaffold_network_or_tree_from_dataframe(df, tree_or_network="network")
    scaffolds = get_scaffolds_of_level(network, level=1)
    molecules = get_molecules_of_scaffold(network, scaffolds[0])
    assert isinstance(molecules, list)
    assert len(molecules) > 0


# Run the tests
if __name__ == "__main__":
    pytest.main()
