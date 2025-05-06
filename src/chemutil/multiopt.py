#!/usr/bin.env python3
# -*- coding: utf-8 -*-

"""
Module for dealing with multi objective problems. The main focus is to use Pareto front methods to identify the most promising candidates and rank.
"""

import logging
import os
import pathlib
import textwrap
from typing import List, Optional, Union

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paretoset import paretorank, paretoset

from chemutil import helpers, vis

log = logging.getLogger(__name__)

par_eff_col_name = "pareto_efficent_set"
par_rank_col_name = "pareto_rank"


def filter_on_property_changes(
    data_df: pd.DataFrame,
    reference_properties: List[str],
    experimental_properties: List[str],
    change: List[str],
    must_pass: List[int],
    number_passing: Optional[int] = None,
    equal_acceptable: bool = False,
) -> pd.DataFrame:
    """
    _summary_

    Args:
        df (pd.DataFrame): _description_
        reference_properties (List[str]): _description_
        experimental_properties (List[str]): _description_
        change (List[str]): _description_
        number_passing (Optional[int], optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """

    df = data_df.copy()

    log.info(
        "Note: If using normalized data ensure that the data is normalized together otherwise the results are invalid"
    )

    columns_add = []
    if equal_acceptable is False:
        for ref, exp, diff in zip(
            reference_properties, experimental_properties, change
        ):
            if diff.lower().strip() == "decrease":
                df[f"{exp} DECREASED compared to {ref}"] = (
                    df.loc[:, exp] < df.loc[:, ref]
                )
                columns_add.append(f"{exp} DECREASED compared to {ref}")
            elif diff.lower().strip() == "increase":
                df[f"{exp} INCREASED compared to {ref}"] = (
                    df.loc[:, exp] > df.loc[:, ref]
                )
                columns_add.append(f"{exp} INCREASED compared to {ref}")
    else:
        for ref, exp, diff in zip(
            reference_properties, experimental_properties, change
        ):
            if diff.lower().strip() == "decrease":
                df[f"{exp} DECREASED OR EQUAL compared to {ref}"] = (
                    df.loc[:, exp] <= df.loc[:, ref]
                )
                columns_add.append(f"{exp} DECREASED OR EQUAL compared to {ref}")
            elif diff.lower().strip() == "increase":
                df[f"{exp} INCREASED OR EQUAL compared to {ref}"] = (
                    df.loc[:, exp] >= df.loc[:, ref]
                )
                columns_add.append(f"{exp} INCREASED OR EQUAL compared to {ref}")

    if must_pass is not None:
        log.debug(
            f"Must pass sums {df[[columns_add[ith] for ith in must_pass]].sum(axis=1)}"
        )
        df["must pass"] = df[[columns_add[ith] for ith in must_pass]].sum(
            axis=1
        ) == len(must_pass)
        df = df[df["must pass"] == 1].copy()

    if number_passing is not None:
        log.debug(f"Pass sums {df[columns_add].sum(axis=1)}")
        df["passing"] = df[columns_add].sum(axis=1) >= number_passing
        df = df[df["passing"] == 1].copy()

    return df


def get_pareto_efficient_set(
    data_df: pd.DataFrame,
    minmax: List[str],
    ignore_duplicates: bool = True,
    objective_columns: Union[List[str], str] = "all",
    return_efficent_set_only: bool = False,
    _debug: bool = False,
    _debug_filename: str = "tmp_pareto_eff.csv",
    _verbose: bool = True,
) -> pd.DataFrame:
    """
    Function to derive the paraeto efficent set based on N objective column vectors and min or max optimization criteria

    Args:
        data_df (pd.DataFrame): Data frame that conatins at least the objective columns (can contain more)
        minmax (List[str]): list of optimal directions for the objective columns. Must be the same number of values as the number of objective columns
        ignore_duplicates (bool, optional): How to deal with duplicate rows, if True it keeps the first and ignores all others. Defaults to True.
        objective_columns (Union[List[str], str], optional): The column headers for the objetcive columns, if 'all' it uses all provided columns as the objective columns. Defaults to "all".
        return_efficent_set_only: (bool): whether to return only the pareto efficent set or not
        _debug (bool): Whether to save the dataframe that is used for the pareto analysis or not. Default False.
        _debug_filename (str): The debug file name to save to. Default is "tmp_pareto_eff.csv".
        _verbose (bool): Whether to log the optimal direction for each column or not

    Returns:
        pd.DataFrame: The same as the input data_df with a boolean column appended called `pareto_efficent_set`. If `return_efficent_set_only`
        True then only the rows of the input data_df which are in the pareto efficent set are retruned
    """

    # Get the objetcive function columns if they are not explicitly defined
    if objective_columns == "all":
        objective_columns = data_df.columns.values.tolist()

    # make sure the expected number of objetcive columns and minmax are there
    if helpers.check_lengths_same_two_lists(objective_columns, minmax) is False:
        raise RuntimeError(
            f"The number of entries in the objetcive columns ({len(objective_columns)}) and minmax ({len(minmax)}) are different. These must be the same."
        )

    if objective_columns != "all":
        df = helpers.get_pd_column_subset(data_df, cols_to_keep=objective_columns)
    else:
        df = data_df.copy()

    if _debug is True:
        df.to_csv(_debug_filename, index=False)

    if _verbose is True:
        log.info(
            f"{os.linesep}"
            + f"{os.linesep}".join(
                [f"{sense}: {nam}" for nam, sense in zip(df.columns, minmax)]
            )
        )

    if any(ent > 0 for ent in df.isna().sum()):
        raise UserWarning(
            "The data set has NaNs present. These cannot be automatically processed please infill, update or take out the NaNs"
        )

    mask = paretoset(df, sense=minmax, distinct=ignore_duplicates, use_numba=True)

    log.info(
        f"The Pareto efficent set contain {sum(mask)} data points out of {len(df.index)}."
    )

    if return_efficent_set_only is True:
        return data_df.loc[mask].copy()

    else:
        data_df[par_eff_col_name] = mask
        return data_df


def get_pareto_ranking(
    data_df: pd.DataFrame,
    minmax: List[str],
    ignore_duplicates: bool = True,
    objective_columns: Union[List[str], str] = "all",
    _debug: bool = False,
    _debug_filename: str = "tmp_pareto_eff.csv",
    _verbose: bool = True,
) -> pd.DataFrame:
    """
    Function to derive the paraeto ranking set based on N objective column vectors and min or max optimization criteria

    Args:
        data_df (pd.DataFrame): Data frame that conatins at least the objective columns (can contain more)
        minmax (List[str]): list of optimal directions for the objective columns. Must be the same number of values as the number of objective columns
        ignore_duplicates (bool, optional): How to deal with duplicate rows, if True it keeps the first and ignores all others. Defaults to True.
        objective_columns (Union[List[str], str], optional): The column headers for the objetcive columns, if 'all' it uses all provided columns as the objective columns. Defaults to "all".
        _debug (bool): Whether to save the dataframe that is used for the pareto analysis or not. Default False.
        _debug_filename (str): The debug file name to save to. Default is "tmp_pareto_eff.csv".
        _verbose (bool): Whether to log the optimal direction for each column or not

    Returns:
        pd.DataFrame: The same as the input data_df with a integer column appended called `pareto_ranks`.
    """

    # Get the objetcive function columns if they are not explicitly defined
    if objective_columns == "all":
        objective_columns = data_df.columns.values.tolist()

    # make sure the expected number of objetcive columns and minmax are there
    if helpers.check_lengths_same_two_lists(objective_columns, minmax) is False:
        raise RuntimeError(
            f"The number of entries in the objetcive columns ({len(objective_columns)}) and minmax ({len(minmax)}) are different. These must be the same."
        )

    if objective_columns != "all":
        df = helpers.get_pd_column_subset(data_df, cols_to_keep=objective_columns)
    else:
        df = data_df.copy()

    if _debug is True:
        df.to_csv(_debug_filename, index=False)

    if _verbose is True:
        log.info(
            f"{os.linesep}"
            + f"{os.linesep}".join(
                [f"{sense}: {nam}" for nam, sense in zip(df.columns, minmax)]
            )
        )

    ranks = paretorank(df, sense=minmax, distinct=ignore_duplicates, use_numba=True)

    log.info(
        f"The Pareto efficent set contain {len(np.where(ranks == 1)[0])} data points out of {len(df.index)}."
    )

    data_df[par_rank_col_name] = ranks

    return data_df


def plot_pareto_front(
    df: pd.DataFrame,
    minmax: List[str],
    objective_columns: Union[List[str], str] = "all",
    label_column: Optional[str] = None,
    filename: Optional[str] = "pareto_front.png",
    pareto_eff_col: Optional[str] = None,
    offset=0.02,
    summary_plot_type: Optional[str] = None,
    normalization: Optional[str] = "zscale",
    smiles_column: Optional[str] = None,
    inchi_column: Optional[str] = None,
    consistent_scale: bool = True,
    **kwargs,
):
    """
    Function to plot a Pareto front in ND (Scatter plots for 2D and 3D)

    Args:
        df (pd.DataFrame): Data frame that conatins at least the objective columns (can contain more). If the `label_column` or `pareto_eff_col` are set then it should also contain these.
        minmax (List[str]): list of optimal directions for the objective columns. Must be the same number of values as the number of objective columns
        objective_columns (Union[List[str], str], optional): The column headers for the objetcive columns, if 'all' it uses all provided columns as the objective columns. Defaults to "all".
        label_column (Optional[str], optional): column header for the column that conatins the molecule labels. If not given no label annotation will be on the plot. Defaults to None.
        filename (Optional[str], optional): The file to save the plot to. Defaults to "pareto_front.png".
        pareto_eff_col (Optional[str], optional): column header for the column that defines whether a molecule is in the Pareto efficent set (boolean column). Defaults to None.
        offset (float, optional): x and y offset in image units from the point for the labels and arrows. Defaults to 0.02.
        summary_plot_type (Optional[str], optional): mean, min, max or None to plot a summary of the Pareto efficent set vs the Pareto inefficent set for N D optimzation. Defaults to None.
        normalization (Optional[str], optional): The normalization to use for ND radar plots. The 'zscale' is better for data where there are outliers as each dimension is normalized to unit variance,
        but each dimension has its own scale. The 'minmax' scaling places each dimesion on a normalized [0, 1] scale but maintains the skew of outliers. None uses raw data which can be hard to
        interpret when dimesnions have very different scales.
        smiles_column (str, optional): the column that contains smiles strings
        inchi_column (str, optional): the column that contains inchi strings
        consistent_scale (bool): whether to use a consistent_scale for zscale axes on N dimensional radar plots, these will be the min and max values over the entire dataframe which may make some small changes hard to visualize. Defaults to True.

    Raises:
        RuntimeError: If a mismatched number of minmax and objective columns are provided
        UserWarning: If a user provides a singlar objective (i.e 1D) for which the Pareto solution is min(singular_objective_vect) or max(singular_objective_vect) and
        there is no need for using the Pareto method
    """

    # ensure not adding to the original
    df = df.copy()

    # Use/assume the defualt name if none given
    if pareto_eff_col is None:
        pareto_eff_col = par_eff_col_name

    # Get the objetcive function columns if they are not explicitly defined
    if objective_columns == "all":
        objective_columns = df.columns.values.tolist()

    # make sure the expected number of objetcive columns and minmax are there
    if helpers.check_lengths_same_two_lists(objective_columns, minmax) is False:
        raise RuntimeError(
            f"The number of entries in the objetcive columns ({len(objective_columns)}) and minmax ({len(minmax)}) are different. These must be the same."
        )

    # Make sure tehre is a pareto efficent column
    if par_eff_col_name not in df.columns:
        log.info(
            "Determining pareto efficent set as column not found in the currently passed in dataframe"
        )
        df = get_pareto_efficient_set(
            df,
            minmax,
            objective_columns=objective_columns,
            return_efficent_set_only=False,
        )

    # Pick the right plotting function
    if len(objective_columns) == 2:
        log.info("Plotting 2D Pareto front")
        plot_pareto_front_2d(
            df=df,
            objective_columns=objective_columns,
            label_col=label_column,
            filename=filename,
            pareto_eff_col=pareto_eff_col,
            offset=offset,
            smiles_column=smiles_column,
            inchi_column=inchi_column,
            **kwargs,
        )

    elif len(objective_columns) == 3:
        log.info("Plotting 3D Pareto front")
        plot_pareto_front_3d(
            df=df,
            objective_columns=objective_columns,
            label_col=label_column,
            filename=filename,
            pareto_eff_col=pareto_eff_col,
            offset=offset,
            smiles_column=smiles_column,
            inchi_column=inchi_column,
            **kwargs,
        )
    elif len(objective_columns) > 3:
        eff_eolutions_df = plot_pareto_front_nd(
            df=df,
            objective_columns=objective_columns,
            label_col=label_column,
            filename=filename,
            pareto_eff_col=pareto_eff_col,
            summary_plot_type=summary_plot_type,
            normalization=normalization,
            smiles_column=smiles_column,
            inchi_column=inchi_column,
            consistent_scale=consistent_scale,
            **kwargs,
        )

        return eff_eolutions_df
    else:
        raise UserWarning(
            "Number of objective columns is singular more than one objective should be used"
        )


def plot_pareto_front_2d(
    df: pd.DataFrame,
    objective_columns: List[str],
    label_col: Optional[str] = None,
    filename: Optional[str] = None,
    pareto_eff_col: Optional[str] = None,
    offset: float = 0.02,
    smiles_column: Optional[str] = None,
    inchi_column: Optional[str] = None,
    **kwargs,
):
    """
    Function to save a 2D scatter plot for the Pareto front

    Args:
        df (pd.DataFrame): Data frame that conatins at least the objective columns (can contain more). If the `label_column` or `pareto_eff_col` are set then it should also contain these.
        objective_columns (Union[List[str], str], optional): The column headers for the objetcive columns, if 'all' it uses all provided columns as the objective columns. Defaults to "all".
        label_col (Optional[str], optional): column header for the column that conatins the molecule labels. If not given no label annotation will be on the plot. Defaults to None.
        filename (Optional[str], optional): The file to save the plot to. Defaults to "pareto_front.png".
        pareto_eff_col (Optional[str], optional): column header for the column that defines whether a molecule is in the Pareto efficent set (boolean column). Defaults to None.
        offset (float, optional): x and y offset in image units from the point for the labels and arrows. Defaults to 0.02.
        smiles_column (str, optional): the column that contains smiles strings
        inchi_column (str, optional): the column that contains inchi strings

    Returns:
        None: Save a plot to file
    """

    if pareto_eff_col is None:
        pareto_eff_col = par_eff_col_name

    _ = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    par_eff_df = df[df[pareto_eff_col] == 1].copy()
    par_ineff_df = df[df[pareto_eff_col] == 0].copy()

    ax = par_ineff_df[objective_columns].plot.scatter(
        x=objective_columns[0],
        y=objective_columns[1],
        ax=ax,
        c="b",
        label="Pareto inefficent",
        s=40,
    )
    ax = par_eff_df[objective_columns].plot.scatter(
        x=objective_columns[0],
        y=objective_columns[1],
        ax=ax,
        c="r",
        label="Pareto efficent (optimal)",
        s=45,
        marker="^",
    )

    plt.xlabel(
        textwrap.fill(
            " ".join(objective_columns[0].lower().strip().split("_")),
            width=40,
            break_long_words=False,
        ),
        fontsize=25,
    )
    plt.ylabel(
        textwrap.fill(
            " ".join(objective_columns[1].lower().strip().split("_")),
            width=40,
            break_long_words=False,
        ),
        fontsize=25,
    )

    log.debug(f"Columns: {par_eff_df.columns}")

    if label_col is not None:
        x = par_eff_df[objective_columns[0]].values
        y = par_eff_df[objective_columns[1]].values
        log.info("Pareto optimal (efficent) set includes:")
        for ith, lab in enumerate(par_eff_df[label_col]):
            log.info(f"x: {x[ith]:.2f} y: {y[ith]:.2f} lab: {lab}")
            ax.annotate(
                lab,
                (x[ith], y[ith]),
                xytext=(x[ith] + offset, y[ith] + offset),
                arrowprops=dict(arrowstyle="->"),
            )

    plt.title("Pareto Front", fontsize=27)
    ax.tick_params(axis="both", which="major", labelsize=17)
    plt.legend(fontsize=17)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()

    if smiles_column is not None:
        vis.draw_aligned_mol_grid(
            smiles=par_eff_df[smiles_column].values.tolist(),
            filename="pareto_2d_efficent.png",
        )
        vis.draw_aligned_mol_grid(
            smiles=par_ineff_df[smiles_column].values.tolist(),
            filename="pareto_2d_inefficent.png",
        )
    elif inchi_column is not None:
        vis.draw_aligned_mol_grid(
            inchi=par_eff_df[inchi_column].values.tolist(),
            filename="pareto_2d_efficent.png",
        )
        vis.draw_aligned_mol_grid(
            smiles=par_ineff_df[inchi_column].values.tolist(),
            filename="pareto_2d_inefficent.png",
        )
    else:
        log.info("Not smiles or inchi column so cannot plot chemical structures")


def plot_pareto_front_3d(
    df: pd.DataFrame,
    objective_columns: List[str],
    label_col: Optional[str] = None,
    filename: Optional[str] = None,
    pareto_eff_col: Optional[str] = None,
    offset: float = 0.02,
    xpad: int = 35,
    ypad: int = 40,
    zpad: int = 35,
    boxpad: float = 1.5,
    xlabwidth: int = 27,
    ylabwidth: int = 25,
    zlabwidth: int = 27,
    smiles_column: Optional[str] = None,
    inchi_column: Optional[str] = None,
    **kwargs,
):
    """
    Function to save a 3D scatter plot for the Pareto front

    Args:
        df (pd.DataFrame): Data frame that conatins at least the objective columns (can contain more). If the `label_column` or `pareto_eff_col` are set then it should also contain these.
        objective_columns (Union[List[str], str], optional): The column headers for the objetcive columns, if 'all' it uses all provided columns as the objective columns. Defaults to "all".
        label_col (Optional[str], optional): column header for the column that conatins the molecule labels. If not given no label annotation will be on the plot. Defaults to None.
        filename (Optional[str], optional): The file to save the plot to. Defaults to "pareto_front.png".
        pareto_eff_col (Optional[str], optional): column header for the column that defines whether a molecule is in the Pareto efficent set (boolean column). Defaults to None.
        offset (float, optional): x and y offset in image units from the point for the labels and arrows. Defaults to 0.02.
        smiles_column (str, optional): the column that contains smiles strings
        inchi_column (str, optional): the column that contains inchi strings

    Returns:
        None: Save a plot to file
    """

    if pareto_eff_col is None:
        pareto_eff_col = par_eff_col_name

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    par_eff_df = df[df[pareto_eff_col] == 1].copy()
    par_ineff_df = df[df[pareto_eff_col] == 0].copy()

    ax.scatter(
        par_ineff_df[objective_columns[0]].values,
        par_ineff_df[objective_columns[1]].values,
        par_ineff_df[objective_columns[2]].values,
        marker=".",
        s=70,
        c="b",
        label="Pareto inefficent",
    )
    ax.scatter(
        par_eff_df[objective_columns[0]].values,
        par_eff_df[objective_columns[1]].values,
        par_eff_df[objective_columns[2]].values,
        marker="^",
        s=75,
        c="r",
        label="Pareto efficent (optimal)",
    )

    ax.set_xlabel(
        textwrap.fill(
            " ".join(objective_columns[0].lower().strip().split("_")),
            width=xlabwidth,
            break_long_words=False,
        ),
        fontsize=20,
        labelpad=xpad,
    )
    ax.set_ylabel(
        textwrap.fill(
            " ".join(objective_columns[1].lower().strip().split("_")),
            width=ylabwidth,
            break_long_words=False,
        ),
        fontsize=20,
        labelpad=ypad,
    )
    ax.set_zlabel(
        textwrap.fill(
            " ".join(objective_columns[2].lower().strip().split("_")),
            width=zlabwidth,
            break_long_words=False,
        ),
        fontsize=20,
        labelpad=zpad,
    )

    if label_col is not None:
        x = par_eff_df[objective_columns[0]].values
        y = par_eff_df[objective_columns[1]].values
        z = par_eff_df[objective_columns[2]].values
        log.info("Pareto optimal (efficent) set includes:")
        for ith, lab in enumerate(par_eff_df[label_col]):
            log.info(f"x: {x[ith]:.2f} y: {y[ith]:.2f} z: {z[ith]:.2f} lab: {lab}")
            ax.text(x[ith] + offset, y[ith] + offset, z[ith] + offset, lab)

    plt.title("Pareto Front", fontsize=27)
    ax.tick_params(axis="both", which="major", labelsize=17)
    plt.legend(fontsize=17)
    plt.savefig(filename, pad_inches=boxpad, bbox_inches="tight")
    plt.show()
    plt.close()

    if smiles_column is not None:
        vis.draw_aligned_mol_grid(
            smiles=par_eff_df[smiles_column].values.tolist(),
            filename="pareto_3d_efficent.png",
        )
        vis.draw_aligned_mol_grid(
            smiles=par_ineff_df[smiles_column].values.tolist(),
            filename="pareto_3d_inefficent.png",
        )
    elif inchi_column is not None:
        vis.draw_aligned_mol_grid(
            inchi=par_eff_df[inchi_column].values.tolist(),
            filename="pareto_3d_efficent.png",
        )
        vis.draw_aligned_mol_grid(
            smiles=par_ineff_df[inchi_column].values.tolist(),
            filename="pareto_3d_inefficent.png",
        )
    else:
        log.info("Not smiles or inchi column so cannot plot chemical structures")


def plot_pareto_front_nd(
    df: pd.DataFrame,
    objective_columns: List[str],
    label_col: Optional[str] = None,
    filename: Optional[str] = "pareto_nd.png",
    pareto_eff_col: Optional[str] = None,
    summary_plot_type: Union[str, None] = "mean",
    normalization: Optional[str] = "zscale",
    smiles_column: Optional[str] = None,
    inchi_column: Optional[str] = None,
    consistent_scale: bool = False,
    **kwargs,
) -> Union[None, pd.DataFrame]:
    """
    Function to plot an N dimensional pareto front on a radar plot. This is plotted by normalizeding all properties to a [0, 1] scale using min max normalization. Typically we summarize these by the mean, max or min of the inefficent set Vs the efficent set

    Args:
        df (pd.DataFrame): Data frame that conatins at least the objective columns (can contain more). If the `label_column` or `pareto_eff_col` are set then it should also contain these.
        objective_columns (Union[List[str], str], optional): The column headers for the objetcive columns, if 'all' it uses all provided columns as the objective columns. Defaults to "all".
        label_col (Optional[str], optional): column header for the column that conatins the molecule labels. If not given no label annotation will be on the plot. Defaults to None.
        filename (Optional[str], optional): The file to save the plot to. Defaults to "pareto_front.png".
        pareto_eff_col (Optional[str], optional): column header for the column that defines whether a molecule is in the Pareto efficent set (boolean column). Defaults to None.
        summay_plot_type (Union[str, None], optional): the summary type mean, min or max of each objective for each set (inefficent and efficent). Defaults to "mean".
        normalization (Optional[str]): string name of normalization method or None to use raw data. Defaults to "zscale".
        smiles_column (str, optional): the column that contains smiles strings
        inchi_column (str, optional): the column that contains inchi strings
        consistent_scale (bool): whether to use a consistent_scale for zscale axes, these will be the min and max values over the entire dataframe which may make some small changes hard to visualize. Defaults to False.

    Raises:
        UserWarning: if an unrecognized summary plot type is given
    """

    if pareto_eff_col is None:
        pareto_eff_col = par_eff_col_name

    if isinstance(summary_plot_type, str):
        summary_plot_df = radar_summary_plot(
            df,
            objective_columns=objective_columns,
            normalization=normalization,
            summary_type=summary_plot_type,
            filename=filename,
            pareto_eff_col=pareto_eff_col,
            **kwargs,
        )

        return summary_plot_df

    else:
        eff_eolutions_df = radar_sub_plots_pareto_front(
            df,
            objective_columns=objective_columns,
            filename=filename,
            label_col=label_col,
            pareto_eff_col=pareto_eff_col,
            normalization=normalization,
            smiles_column=smiles_column,
            inchi_column=inchi_column,
            consistent_scale=consistent_scale,
            **kwargs,
        )

        return eff_eolutions_df


def radar_sub_plots_pareto_front(
    df: pd.DataFrame,
    objective_columns: List[str],
    filename: Optional[str] = "subplot_pareto.png",
    label_col: Optional[str] = None,
    pareto_eff_col: Optional[str] = None,
    normalization: Union[str, None] = "zscale",
    legend_pos: str = "best",
    smiles_column: Optional[str] = None,
    inchi_column: Optional[str] = None,
    consistent_scale: bool = False,
    mols_per_row: int = 5,
    return_data: bool = True,
    y_ticks: Optional[Union[np.ndarray, list]] = None,
    show_y_ticks: bool = False,
) -> pd.DataFrame:
    """
    Function to plot a radar plot of the normalized (min max normalization) mean for each property over the inefficent and efficent set

    Args:
        ddf (pd.DataFrame): Data frame that conatins at least the objective columns (can contain more). If the `label_column` or `pareto_eff_col` are set then it should also contain these.
        objective_columns (Union[List[str], str], optional): The column headers for the objetcive columns, if 'all' it uses all provided columns as the objective columns. Defaults to "all".
        filename (Optional[str], optional): The file to save the plot to. Defaults to "pareto_front.png".
        label_col (Optional[str], optional): The column that defines the molecules labels.
        pareto_eff_col (Optional[str], optional): column header for the column that defines whether a molecule is in the Pareto efficent set (boolean column). Defaults to None.
        normalization (Union[str, None]): string defining teh normalization method
        legend_pos (str): legend position string for matplotlib
        smiles_column (str, optional): the column that contains smiles strings
        inchi_column (str, optional): the column that contains inchi strings
        consistent_scale (bool): whether to use a consistent_scale for zscale axes, these will be the min and max values over the entire dataframe which may make some small changes hard to visualize. Defaults to False.
        y_ticks (Optional[Union[np.ndarray, list]], optional): Set of y ticks to use to allow for comparison across sets. Defaults as None.

    Returns:
        None: saves plot to file
    """

    # Set the column containing the Pareto efficent labels as bools
    if pareto_eff_col is None:
        log.info(f"Assuming Pareto efficent bool column is {par_eff_col_name}")
        pareto_eff_col = par_eff_col_name

    log.info(f"Dataframe input has shape: {df.shape}")

    # Set the number of objectives and the convert to polar angles
    n_obj = len(objective_columns)
    theta = np.linspace(0, 2.0 * np.pi, n_obj, endpoint=False)
    theta = np.append(theta, theta[0])

    # Get the metric columns only and normalize as requested
    metric_assay_df = df[objective_columns].copy()
    if isinstance(normalization, str):
        # min max scale
        if normalization.lower().strip() == "minmax":
            log.info("Min max scaling")
            pareto_eff_set_in_nd_df_normed = helpers.pandas_df_min_max_scale(
                metric_assay_df
            )
            if y_ticks is None:
                y_ticks = np.linspace(0, 1, 5)
                log.debug(f"Min max scaled y ticks {y_ticks}")
            if show_y_ticks is False:
                y_labels = []
            else:
                y_labels = [f"{ent:.2f}" for ent in y_ticks]

        # z scale
        elif normalization.lower().strip() == "zscale":
            log.info("Z scaling")
            pareto_eff_set_in_nd_df_normed = helpers.pandas_df_z_scale(metric_assay_df)
            if consistent_scale is True:
                log.debug(f"Before scaling the data frame is {metric_assay_df}")
                log.debug(
                    f"After scaling the data frame is {pareto_eff_set_in_nd_df_normed}"
                )

                if y_ticks is None:
                    y_ticks = np.linspace(
                        pareto_eff_set_in_nd_df_normed.min(numeric_only=True).min(),
                        pareto_eff_set_in_nd_df_normed.max(numeric_only=True).max(),
                        5,
                    )
                    log.debug(f"Auto (z) scaled y ticks {y_ticks}")
                if show_y_ticks is False:
                    y_labels = []
                else:
                    y_labels = [f"{ent:.2f}" for ent in y_ticks]

    # No scaling - ill advised
    else:
        log.warning(
            "Will not scale data within this method. You should pre-scale your data."
        )
        pareto_eff_set_in_nd_df_normed = metric_assay_df.copy()

        if y_ticks is None:
            y_ticks = np.linspace(
                pareto_eff_set_in_nd_df_normed.min(numeric_only=True).min(),
                pareto_eff_set_in_nd_df_normed.max(numeric_only=True).max(),
                5,
            )
            log.debug(f"Unscaled y ticks {y_ticks}")
        if show_y_ticks is False:
            y_labels = []
        else:
            y_labels = [f"{ent:.2f}" for ent in y_ticks]

    log.info(f"Y ticks {y_ticks}")

    # get whether a molecule is an efficent solution
    efficent = df[pareto_eff_col].values
    log.info(f"There are {sum(efficent)} Pareto efficent solutions in the input data")

    # Get the mean of the Pareto inefficent solutions as a background
    plot_df = pd.DataFrame(
        pareto_eff_set_in_nd_df_normed.loc[~efficent].mean()
    ).transpose()

    # Get the normalized efficent solutions one per row
    chosen_df = pd.DataFrame(pareto_eff_set_in_nd_df_normed.loc[efficent])

    # set the layout assuming 5 columns asa default
    if not isinstance(mols_per_row, int):
        mols_per_row = int(mols_per_row)

    if len(chosen_df.index) < mols_per_row:
        n_rows = 1
        m_columns = len(chosen_df.index)

    elif len(chosen_df.index) % mols_per_row == 0:
        n_rows = len(chosen_df.index) / mols_per_row
        m_columns = mols_per_row

    else:
        n_rows = int(np.ceil(len(chosen_df.index) / mols_per_row))
        m_columns = mols_per_row

    # Set the overall figure
    _ = plt.figure(
        figsize=(m_columns * 7.0, n_rows * 7.0),
    )

    # loop over the rows and create a subplot per row using the mean of the inefficent set as a background for comparison
    for ith, idx in enumerate(chosen_df.index):
        log.info(
            f"Row {ith}: N rows: {int(n_rows)} M columns: {int(m_columns)} Plot ith: {int(ith+1)}"
        )
        ax = plt.subplot(int(n_rows), int(m_columns), int(ith + 1), projection="polar")
        ax.spines["polar"].set_color("lightgrey")

        if label_col is not None:
            lab = df.loc[idx, label_col]
        else:
            lab = str(ith)

        # plot mean of inefficent set as the background
        ax.plot(
            theta,
            plot_df.iloc[0, :].tolist() + [plot_df.iloc[0, :].tolist()[0]],
            linewidth=1.75,
            linestyle="solid",
            label="Inefficent set mean",
            marker="o",
            markersize=10,
            color="b",
        )
        ax.fill(
            theta,
            plot_df.iloc[0, :].tolist() + [plot_df.iloc[0, :].tolist()[0]],
            alpha=0.50,
            color="b",
        )

        # plot the efficent row example
        ax.plot(
            theta,
            chosen_df.iloc[ith, :].tolist() + [chosen_df.iloc[ith, :].tolist()[0]],
            linewidth=1.75,
            linestyle="solid",
            label=f"Efficent solution {lab}",
            marker="o",
            markersize=10,
            color="r",
        )
        ax.fill(
            theta,
            chosen_df.iloc[ith, :].tolist() + [chosen_df.iloc[ith, :].tolist()[0]],
            alpha=0.50,
            color="r",
        )
        plt.legend(fontsize=15, loc=legend_pos)
        plt.grid(True)
        plt.title(f"Mean Inefficent Vs.\n{lab}", fontsize=20)
        axes_labs = [
            textwrap.fill(
                " ".join(ent.lower().strip().split("_")),
                width=17,
                break_long_words=False,
            )
            for ent in plot_df.columns.values.tolist()
        ]
        plt.xticks(theta, axes_labs + [axes_labs[0]], color="black", size=17)
        plt.yticks(
            y_ticks,
            y_labels,
            color="black",
            size=15,
        )

    # save the plot and tidy up
    plt.tight_layout(h_pad=5, w_pad=5)
    if filename is not None:
        plt.savefig(filename)

        # for if structures are given as smiles or inchi
        efficent_filename = f"efficent_molecules_{n_obj}d_{filename}"
        inefficent_filename = f"inefficent_molecules_{n_obj}d_{filename}"
    plt.show()
    plt.close()

    if smiles_column is not None:
        log.debug(f"The data frame to get the smiles from is {df}")
        log.debug(f"The data frame to get the smiles from columns are {df.columns}")
        log.info(
            f"Efficent molecules images in file {efficent_filename} inefficent molecules images in file {inefficent_filename}"
        )
        if label_col is not None:
            vis.draw_aligned_mol_grid(
                smiles=df[df[pareto_eff_col] == 1][smiles_column].values.tolist(),
                labels=df[df[pareto_eff_col] == 1][label_col].values.tolist(),
                filename=efficent_filename,
                mols_per_row=mols_per_row,
            )
            vis.draw_aligned_mol_grid(
                smiles=df[df[pareto_eff_col] == 0][smiles_column].values.tolist(),
                labels=df[df[pareto_eff_col] == 0][label_col].values.tolist(),
                filename=inefficent_filename,
                mols_per_row=mols_per_row,
            )
        else:
            vis.draw_aligned_mol_grid(
                smiles=df[df[pareto_eff_col] == 1][smiles_column].values.tolist(),
                filename=efficent_filename,
                mols_per_row=mols_per_row,
            )
            vis.draw_aligned_mol_grid(
                smiles=df[df[pareto_eff_col] == 0][smiles_column].values.tolist(),
                filename=inefficent_filename,
                mols_per_row=mols_per_row,
            )
    elif inchi_column is not None:
        log.debug(f"The data frame to get the inchi from is {df}")
        log.debug(f"The data frame to get the inchi from columns are {df.columns}")
        log.info(
            f"Efficent molecules images in file {efficent_filename} inefficent molecules images in file {inefficent_filename}"
        )
        if label_col is not None:
            vis.draw_aligned_mol_grid(
                inchi=df[df[pareto_eff_col] == 1][inchi_column].values.tolist(),
                labels=df[df[pareto_eff_col] == 0][label_col].values.tolist(),
                filename=efficent_filename,
                mols_per_row=mols_per_row,
            )
            vis.draw_aligned_mol_grid(
                inchi=df[df[pareto_eff_col] == 0][inchi_column].values.tolist(),
                labels=df[df[pareto_eff_col] == 0][label_col].values.tolist(),
                filename=inefficent_filename,
                mols_per_row=mols_per_row,
            )
        else:
            vis.draw_aligned_mol_grid(
                inchi=df[df[pareto_eff_col] == 1][inchi_column].values.tolist(),
                filename=efficent_filename,
                mols_per_row=mols_per_row,
            )
            vis.draw_aligned_mol_grid(
                inchi=df[df[pareto_eff_col] == 0][inchi_column].values.tolist(),
                filename=inefficent_filename,
                mols_per_row=mols_per_row,
            )
    else:
        log.info("Not smiles or inchi column so cannot plot chemical structures")

    if return_data is True:
        eff_eolutions_df = _get_radar_subplot_data(
            df,
            objective_columns=objective_columns,
            label_col=label_col,
            pareto_eff_col=pareto_eff_col,
            smiles_column=smiles_column,
            inchi_column=inchi_column,
        )

        return eff_eolutions_df

    else:
        return None


def _get_radar_subplot_data(
    df: pd.DataFrame,
    objective_columns: List[str],
    label_col: Optional[str] = None,
    pareto_eff_col: Optional[str] = None,
    smiles_column: Optional[str] = None,
    inchi_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Function to get the raw data used for the radar plot of the Pareto efficent set for each property compared to the mean over the inefficent  set

    Args:
        df (pd.DataFrame): Data frame that conatins at least the objective columns (can contain more). If the `label_column` or `pareto_eff_col` are set then it should also contain these.
        objective_columns (Union[List[str], str], optional): The column headers for the objetcive columns, if 'all' it uses all provided columns as the objective columns. Defaults to "all".
        label_col (Optional[str], optional): The column that defines the molecules labels.
        pareto_eff_col (Optional[str], optional): column header for the column that defines whether a molecule is in the Pareto efficent set (boolean column). Defaults to None.
        smiles_column (str, optional): the column that contains smiles strings
        inchi_column (str, optional): the column that contains inchi strings

    Returns:
        pd.DataFrame: The data for the efficent set compared to the mean of the inefficent set
    """

    # Set the column containing the Pareto efficent labels as bools
    if pareto_eff_col is None:
        log.info(f"Assuming Pareto efficent bool column is {par_eff_col_name}")
        pareto_eff_col = par_eff_col_name

    log.info(f"Dataframe input has shape: {df.shape}")

    # Get the metric columns only but don't normalize here as we want to return teh raw data
    pareto_eff_set_in_nd_df_normed = df[objective_columns].copy()

    # get whether a molecule is an efficent solution
    efficent = df[pareto_eff_col].values
    log.info(f"There are {sum(efficent)} Pareto efficent solutions in the input data")

    # Get the mean of the Pareto inefficent solutions as a background
    plot_df = pd.DataFrame(
        pareto_eff_set_in_nd_df_normed.loc[~efficent].mean()
    ).transpose()

    # Get the normalized efficent solutions one per row
    chosen_df = pd.DataFrame(pareto_eff_set_in_nd_df_normed.loc[efficent])

    # Prepare for return dataframe with expected columns
    eff_solutions_lists = []
    if inchi_column is not None:
        struct_col = smiles_column
        eff_solutions_lists_header = (
            ["ID", "smiles"]
            + [f"Mean_ineff_{ent}" for ent in plot_df.iloc[0, :].index]
            + [ent for ent in chosen_df.iloc[0, :].index]
        )
    elif smiles_column is not None:
        struct_col = smiles_column
        eff_solutions_lists_header = (
            ["ID", "inchi"]
            + [f"Mean_ineff_{ent}" for ent in plot_df.iloc[0, :].index]
            + [ent for ent in chosen_df.iloc[0, :].index]
        )
    else:
        struct_col = None
        eff_solutions_lists_header = (
            ["ID"]
            + [f"Mean_ineff_{ent}" for ent in plot_df.iloc[0, :].index]
            + [ent for ent in chosen_df.iloc[0, :].index]
        )

    # loop over the rows and create a subplot per row using the mean of the inefficent set as a background for comparison
    log.info("Getting data as dataframe .....")
    for ith, idx in enumerate(chosen_df.index):
        if label_col is not None:
            lab = df.loc[idx, label_col]
        else:
            lab = str(ith)

        if struct_col is not None:
            str_struct = df.loc[idx, struct_col]

        # Gather the data together for a return dataframe
        if struct_col is not None:
            eff_solutions_lists.append(
                [lab, str_struct]
                + plot_df.iloc[0, :].tolist()
                + chosen_df.iloc[ith, :].tolist()
            )
        else:
            eff_solutions_lists.append(
                [lab] + plot_df.iloc[0, :].tolist() + chosen_df.iloc[ith, :].tolist()
            )

    log.debug(f"Efficent solution list of lists: {eff_solutions_lists}")
    log.debug(f"Efficent solution dataframe headers: {eff_solutions_lists_header}")

    eff_eolutions_df = pd.DataFrame(
        data=eff_solutions_lists, columns=eff_solutions_lists_header
    )
    log.debug(f"Efficent solution dataframe: {eff_eolutions_df}")

    return eff_eolutions_df


def radar_summary_plot(
    df: pd.DataFrame,
    objective_columns: List[str],
    normalization: Union[str, None] = "zscale",
    summary_type: str = "mean",
    consistent_scale: bool = True,
    filename: Optional[str] = None,
    pareto_eff_col: Optional[str] = None,
    y_ticks: Optional[Union[np.ndarray, List[float]]] = None,
    show_y_ticks: bool = False,
    legend_pos: Union[int, str] = 0,
):
    """
    Function to plot a radar plot of the normalized data using one of mean, median, max or min summary methods for each property over the inefficent and efficent set

    Args:
        df (pd.DataFrame): Data frame that conatins at least the objective columns (can contain more). If the `label_column` or `pareto_eff_col` are set then it should also contain these.
        objective_columns (Union[List[str], str], optional): The column headers for the objetcive columns, if 'all' it uses all provided columns as the objective columns. Defaults to "all".
        normalization (Union[str, None]): The normalization type to use or None to use the raw data. Default is zscale.
        summary_type (str): the type of summary statistic to use out of mean. median, max and min. Default is mean.
        filename (Optional[str], optional): The file to save the plot to. Defaults to "pareto_front.png".
        pareto_eff_col (Optional[str], optional): column header for the column that defines whether a molecule is in the Pareto efficent set (boolean column). Defaults to None.
        y_ticks (Optional[Union[np.ndarray, List[float]]], optional): Defined set of y tickmarks values to use
        show_y_ticks (bool): Show z scaled y values. Default is False
        legend_pos: (Union[int, str]): where to put the legend see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html. Defaults to 0 best.

    Returns:
        None: saves plot to file
    """

    if pareto_eff_col is None:
        pareto_eff_col = par_eff_col_name

    n_obj = len(objective_columns)
    theta = np.linspace(0, 2.0 * np.pi, n_obj, endpoint=False)
    theta = np.append(theta, theta[0])

    # Get the metric columns only and normalize as requested
    metric_assay_df = df[objective_columns].copy()
    if isinstance(normalization, str):
        # min max scale
        if normalization.lower().strip() == "minmax":
            log.info("Min max scaling")
            pareto_eff_set_in_nd_df_normed = helpers.pandas_df_min_max_scale(
                metric_assay_df
            )
            if y_ticks is None:
                y_ticks = np.linspace(0, 1, 5)
                log.debug(f"Min max scaled y ticks {y_ticks}")
            if show_y_ticks is False:
                y_labels = []
            else:
                y_labels = [f"{ent:.2f}" for ent in y_ticks]
        # z scale
        elif normalization.lower().strip() == "zscale":
            log.info("Z scaling")
            pareto_eff_set_in_nd_df_normed = helpers.pandas_df_z_scale(metric_assay_df)
            if consistent_scale is True:
                log.debug(f"Before scaling the data frame is {metric_assay_df}")
                log.debug(
                    f"After scaling the data frame is {pareto_eff_set_in_nd_df_normed}"
                )

                if y_ticks is None:
                    y_ticks = np.linspace(
                        pareto_eff_set_in_nd_df_normed.min(numeric_only=True).min(),
                        pareto_eff_set_in_nd_df_normed.max(numeric_only=True).max(),
                        5,
                    )
                    log.debug(f"Auto (z) scaled y ticks {y_ticks}")
                if show_y_ticks is False:
                    y_labels = []
                else:
                    y_labels = [f"{ent:.2f}" for ent in y_ticks]

    # No scaling - ill advised
    else:
        log.warning(
            "Will not perform scaling within the function. You should pre-scale your data."
        )
        pareto_eff_set_in_nd_df_normed = metric_assay_df.copy()

        if y_ticks is None:
            y_ticks = np.linspace(
                min(pareto_eff_set_in_nd_df_normed),
                max(pareto_eff_set_in_nd_df_normed),
                5,
            )
            log.debug(f"Unscaled y ticks {y_ticks}")
        if show_y_ticks is False:
            y_labels = []
        else:
            y_labels = [f"{ent:.2f}" for ent in y_ticks]

    log.info(f"Y ticks {y_ticks}")

    efficent = df[pareto_eff_col].values

    if summary_type.lower().strip() == "mean":
        background_df = pd.DataFrame(
            pareto_eff_set_in_nd_df_normed.loc[~efficent].mean()
        ).transpose()
        chosen_df = pd.DataFrame(
            pareto_eff_set_in_nd_df_normed.loc[efficent].mean()
        ).transpose()
        plot_df = pd.concat([background_df, chosen_df])
    elif summary_type.lower().strip() == "median":
        background_df = pd.DataFrame(
            pareto_eff_set_in_nd_df_normed.loc[~efficent].median()
        ).transpose()
        chosen_df = pd.DataFrame(
            pareto_eff_set_in_nd_df_normed.loc[efficent].median()
        ).transpose()
        plot_df = pd.concat([background_df, chosen_df])
    elif summary_type.lower().strip() == "max":
        background_df = pd.DataFrame(
            pareto_eff_set_in_nd_df_normed.loc[~efficent].max()
        ).transpose()
        chosen_df = pd.DataFrame(
            pareto_eff_set_in_nd_df_normed.loc[efficent].max()
        ).transpose()
        plot_df = pd.concat([background_df, chosen_df])

    elif summary_type.lower().strip() == "min":
        background_df = pd.DataFrame(
            pareto_eff_set_in_nd_df_normed.loc[~efficent].min()
        ).transpose()
        chosen_df = pd.DataFrame(
            pareto_eff_set_in_nd_df_normed.loc[efficent].min()
        ).transpose()
        plot_df = pd.concat([background_df, chosen_df])

    else:
        raise RuntimeError(
            f"Unknown summary type '{summary_type}' (type is {type(summary_type)}), please give one of: 'mean', 'median', 'min' or 'max'"
        )

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(10, 10))
    ax.spines["polar"].set_color("lightgrey")

    ax.plot(
        theta,
        plot_df.iloc[0, :].tolist() + [plot_df.iloc[0, :].tolist()[0]],
        linewidth=1.75,
        linestyle="solid",
        label=f"Inefficent set {summary_type}",
        marker="o",
        markersize=10,
        color="b",
    )
    ax.fill(
        theta,
        plot_df.iloc[0, :].tolist() + [plot_df.iloc[0, :].tolist()[0]],
        alpha=0.50,
        color="b",
    )
    ax.plot(
        theta,
        plot_df.iloc[1, :].tolist() + [plot_df.iloc[1, :].tolist()[0]],
        linewidth=1.75,
        linestyle="solid",
        label=f"Efficent set {summary_type}",
        marker="o",
        markersize=10,
        color="r",
    )
    ax.fill(
        theta,
        plot_df.iloc[1, :].tolist() + [plot_df.iloc[1, :].tolist()[0]],
        alpha=0.50,
        color="r",
    )
    plt.legend(loc=legend_pos, fontsize=17)
    plt.grid(True)
    plt.title(
        f"{summary_type.title()} Inefficent Set Vs. {summary_type.title()} Efficent Set",
        fontsize=27,
    )
    axes_labs = [
        textwrap.fill(
            " ".join(ent.lower().strip().split("_")), width=25, break_long_words=False
        )
        for ent in plot_df.columns.values.tolist()
    ]
    plt.xticks(theta, axes_labs + [axes_labs[0]], color="black", size=17)

    plt.yticks(
        y_ticks,
        y_labels,
        color="black",
        size=17,
    )

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()

    return plot_df


######################################################################################################################################################################################
######################################################################################################################################################################################
################################################################## PARETO RANKING PLOTTING FUNCTIONS #################################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################


def plot_pareto_rank(
    df: pd.DataFrame,
    minmax: List[str],
    objective_columns: Union[List[str], str] = "all",
    label_column: Optional[str] = None,
    filename: Optional[str] = "pareto_ranks.png",
    pareto_rank_col: Optional[str] = None,
    offset=0.02,
    colour_map="bwr_r",
    summary_plot_type: Optional[str] = None,
    normalization: Optional[str] = "zscale",
    smiles_column: Optional[str] = None,
    inchi_column: Optional[str] = None,
    consistent_scale: bool = True,
    **kwargs,
):
    """
    Function to plot a Pareto front in ND (Scatter plots for 2D and 3D)

    Args:
        df (pd.DataFrame): Data frame that conatins at least the objective columns (can contain more). If the `label_column` or `pareto_eff_col` are set then it should also contain these.
        minmax (List[str]): list of optimal directions for the objective columns. Must be the same number of values as the number of objective columns
        objective_columns (Union[List[str], str], optional): The column headers for the objetcive columns, if 'all' it uses all provided columns as the objective columns. Defaults to "all".
        label_column (Optional[str], optional): column header for the column that conatins the molecule labels. If not given no label annotation will be on the plot. Defaults to None.
        filename (Optional[str], optional): The file to save the plot to. Defaults to "pareto_front.png".
        pareto_rank_col (Optional[str], optional): column header for the column that defines whether a molecule is in the Pareto efficent set (boolean column). Defaults to None.
        offset (float, optional): x and y offset in image units from the point for the labels and arrows. Defaults to 0.02.
        summary_plot_type (Optional[str], optional): mean, min, max or None to plot a summary of the Pareto efficent set vs the Pareto inefficent set for N D optimzation. Defaults to None.
        normalization (Optional[str], optional): The normalization to use for ND radar plots. The 'zscale' is better for data where there are outliers as each dimension is normalized to unit variance,
        but each dimension has its own scale. The 'minmax' scaling places each dimesion on a normalized [0, 1] scale but maintains the skew of outliers. None uses raw data which can be hard to
        interpret when dimesnions have very different scales.
        smiles_column (str, optional): the column that contains smiles strings
        inchi_column (str, optional): the column that contains inchi strings
        consistent_scale (bool): whether to use a consistent_scale for zscale axes on N dimensional radar plots, these will be the min and max values over the entire dataframe which may make some small changes hard to visualize. Defaults to True.

    Raises:
        RuntimeError: If a mismatched number of minmax and objective columns are provided
        UserWarning: If a user provides a singlar objective (i.e 1D) for which the Pareto solution is min(singular_objective_vect) or max(singular_objective_vect) and
        there is no need for using the Pareto method
    """

    # ensure not adding to the original
    df = df.copy()

    # Use/assume the defualt name if none given
    if pareto_rank_col is None:
        pareto_rank_col = par_rank_col_name

    # Get the objetcive function columns if they are not explicitly defined
    if objective_columns == "all":
        objective_columns = df.columns.values.tolist()

    # make sure the expected number of objetcive columns and minmax are there
    if helpers.check_lengths_same_two_lists(objective_columns, minmax) is False:
        raise RuntimeError(
            f"The number of entries in the objetcive columns ({len(objective_columns)}) and minmax ({len(minmax)}) are different. These must be the same."
        )

    # Make sure tehre is a pareto efficent column
    if par_eff_col_name not in df.columns:
        log.info(
            "Determining pareto ranks set as column not found in the currently passed in dataframe"
        )
        df = get_pareto_ranking(
            df,
            minmax,
            objective_columns=objective_columns,
        )

    # Pick the right plotting function
    if len(objective_columns) == 2:
        log.info("Plotting 2D Pareto ranks")
        plot_pareto_rank_2d(
            df=df,
            objective_columns=objective_columns,
            label_col=label_column,
            filename=filename,
            colour_map=colour_map,
            pareto_rank_col=pareto_rank_col,
            offset=offset,
            smiles_column=smiles_column,
            inchi_column=inchi_column,
            **kwargs,
        )

    elif len(objective_columns) == 3:
        log.info("Plotting 3D Pareto ranks")
        plot_pareto_rank_3d(
            df=df,
            objective_columns=objective_columns,
            label_col=label_column,
            filename=filename,
            colour_map=colour_map,
            pareto_rank_col=pareto_rank_col,
            offset=offset,
            smiles_column=smiles_column,
            inchi_column=inchi_column,
            **kwargs,
        )
    elif len(objective_columns) > 3:
        plot_pareto_rank_nd(
            df=df,
            objective_columns=objective_columns,
            label_col=label_column,
            filename=filename,
            pareto_rank_col=pareto_rank_col,
            summary_plot_type=summary_plot_type,
            normalization=normalization,
            smiles_column=smiles_column,
            inchi_column=inchi_column,
            consistent_scale=consistent_scale,
            **kwargs,
        )
    else:
        raise UserWarning(
            "Number of objective columns is singular more than one objective should be used"
        )


def plot_pareto_rank_2d(
    df: pd.DataFrame,
    objective_columns: List[str],
    label_col: Optional[str] = None,
    filename: Optional[str] = None,
    colour_map: Optional[str] = "bwr_r",
    pareto_rank_col: Optional[str] = None,
    smiles_column: Optional[str] = None,
    inchi_column: Optional[str] = None,
    **kwargs,
):
    """
    Function to save a 2D scatter plot for the Pareto front

    Args:
        df (pd.DataFrame): Data frame that conatins at least the objective columns (can contain more). If the `label_column` or `pareto_eff_col` are set then it should also contain these.
        objective_columns (Union[List[str], str], optional): The column headers for the objetcive columns, if 'all' it uses all provided columns as the objective columns. Defaults to "all".
        label_col (Optional[str], optional): column header for the column that conatins the molecule labels. If not given no label annotation will be on the plot. Defaults to None.
        filename (Optional[str], optional): The file to save the plot to. Defaults to "pareto_front.png".
        pareto_rank_col (Optional[str], optional): column header for the column that defines whether a molecule is in the Pareto efficent set (boolean column). Defaults to None.
        offset (float, optional): x and y offset in image units from the point for the labels and arrows. Defaults to 0.02.
        smiles_column (str, optional): the column that contains smiles strings
        inchi_column (str, optional): the column that contains inchi strings

    Returns:
        None: Save a plot to file
    """

    if pareto_rank_col is None:
        pareto_rank_col = par_rank_col_name

    _ = plt.figure(figsize=(10, 10))
    ax = plt.gca()

    ax = df.plot.scatter(
        x=objective_columns[0],
        y=objective_columns[1],
        ax=ax,
        c=pareto_rank_col,
        colormap=colour_map,
        label="Pareto Ranking",
        s=40,
    )

    plt.xlabel(
        textwrap.fill(
            " ".join(objective_columns[0].lower().strip().split("_")),
            width=40,
            break_long_words=False,
        ),
        fontsize=25,
    )
    plt.ylabel(
        textwrap.fill(
            " ".join(objective_columns[1].lower().strip().split("_")),
            width=40,
            break_long_words=False,
        ),
        fontsize=25,
    )

    log.debug(f"Columns: {df.columns}")

    plt.title("Pareto Ranking", fontsize=27)
    ax.tick_params(axis="both", which="major", labelsize=17)
    plt.legend(fontsize=17)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()

    if smiles_column is not None:
        if label_col is None:
            for rank in df[pareto_rank_col].unqiue():
                vis.draw_aligned_mol_grid(
                    smiles=df[df[pareto_rank_col] == rank][
                        smiles_column
                    ].values.tolist(),
                    filename=f"pareto_rank_{rank}_2d.png",
                )
        elif label_col is not None:
            for rank in df[pareto_rank_col].unqiue():
                vis.draw_aligned_mol_grid(
                    smiles=df[df[pareto_rank_col] == rank][
                        smiles_column
                    ].values.tolist(),
                    filename=f"pareto_rank_{rank}_2d.png",
                    labels=df[df[pareto_rank_col] == rank][label_col].values.tolist(),
                )
    elif inchi_column is not None:
        if label_col is None:
            for rank in df[pareto_rank_col].unqiue():
                vis.draw_aligned_mol_grid(
                    inchi=df[df[pareto_rank_col] == rank][inchi_column].values.tolist(),
                    filename=f"pareto_rank_{rank}_2d.png",
                )
        elif label_col is not None:
            for rank in df[pareto_rank_col].unqiue():
                vis.draw_aligned_mol_grid(
                    inchi=df[df[pareto_rank_col] == rank][inchi_column].values.tolist(),
                    filename=f"pareto_rank_{rank}_2d.png",
                    labels=df[df[pareto_rank_col] == rank][label_col].values.tolist(),
                )
    else:
        log.info("Not smiles or inchi column so cannot plot chemical structures")


def plot_pareto_rank_3d(
    df: pd.DataFrame,
    objective_columns: List[str],
    label_col: Optional[str] = None,
    filename: Optional[str] = None,
    pareto_rank_col: Optional[str] = None,
    colour_map: Optional[str] = "bwr_r",
    xpad: int = 35,
    ypad: int = 40,
    zpad: int = 35,
    boxpad: float = 1.5,
    xlabwidth: int = 27,
    ylabwidth: int = 25,
    zlabwidth: int = 27,
    smiles_column: Optional[str] = None,
    inchi_column: Optional[str] = None,
    **kwargs,
):
    """
    Function to save a 3D scatter plot for the Pareto front

    Args:
        df (pd.DataFrame): Data frame that conatins at least the objective columns (can contain more). If the `label_column` or `pareto_eff_col` are set then it should also contain these.
        objective_columns (Union[List[str], str], optional): The column headers for the objetcive columns, if 'all' it uses all provided columns as the objective columns. Defaults to "all".
        label_col (Optional[str], optional): column header for the column that conatins the molecule labels. If not given no label annotation will be on the plot. Defaults to None.
        filename (Optional[str], optional): The file to save the plot to. Defaults to "pareto_front.png".
        pareto_eff_col (Optional[str], optional): column header for the column that defines whether a molecule is in the Pareto efficent set (boolean column). Defaults to None.
        offset (float, optional): x and y offset in image units from the point for the labels and arrows. Defaults to 0.02.
        smiles_column (str, optional): the column that contains smiles strings
        inchi_column (str, optional): the column that contains inchi strings

    Returns:
        None: Save a plot to file
    """

    if pareto_rank_col is None:
        pareto_rank_col = par_rank_col_name

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    cmap = plt.get_cmap(colour_map)
    cmap_norm = colors.Normalize(
        vmin=df[pareto_rank_col].min(), vmax=df[pareto_rank_col].max()
    )
    scalable_cmap = cm.ScalarMappable(norm=cmap_norm, cmap=cmap)

    ax.scatter(
        df[objective_columns[0]].values,
        df[objective_columns[1]].values,
        df[objective_columns[2]].values,
        marker=".",
        s=70,
        c=[scalable_cmap.to_rgba(ent) for ent in df[pareto_rank_col].values],
        label="Pareto ranking",
    )
    ax.set_xlabel(
        textwrap.fill(
            " ".join(objective_columns[0].lower().strip().split("_")),
            width=xlabwidth,
            break_long_words=False,
        ),
        fontsize=20,
        labelpad=xpad,
    )
    ax.set_ylabel(
        textwrap.fill(
            " ".join(objective_columns[1].lower().strip().split("_")),
            width=ylabwidth,
            break_long_words=False,
        ),
        fontsize=20,
        labelpad=ypad,
    )
    ax.set_zlabel(
        textwrap.fill(
            " ".join(objective_columns[2].lower().strip().split("_")),
            width=zlabwidth,
            break_long_words=False,
        ),
        fontsize=20,
        labelpad=zpad,
    )

    plt.title("Pareto Ranking", fontsize=27)
    ax.tick_params(axis="both", which="major", labelsize=17)
    plt.legend(fontsize=17)
    plt.savefig(filename, pad_inches=boxpad, bbox_inches="tight")
    plt.show()
    plt.close()

    if smiles_column is not None:
        if label_col is None:
            for rank in df[pareto_rank_col].unqiue():
                vis.draw_aligned_mol_grid(
                    smiles=df[df[pareto_rank_col] == rank][
                        smiles_column
                    ].values.tolist(),
                    filename=f"pareto_rank_{rank}_3d.png",
                )
        elif label_col is not None:
            for rank in df[pareto_rank_col].unqiue():
                vis.draw_aligned_mol_grid(
                    smiles=df[df[pareto_rank_col] == rank][
                        smiles_column
                    ].values.tolist(),
                    filename=f"pareto_rank_{rank}_3d.png",
                    labels=df[df[pareto_rank_col] == rank][label_col].values.tolist(),
                )
    elif inchi_column is not None:
        if label_col is None:
            for rank in df[pareto_rank_col].unqiue():
                vis.draw_aligned_mol_grid(
                    inchi=df[df[pareto_rank_col] == rank][inchi_column].values.tolist(),
                    filename=f"pareto_rank_{rank}_3d.png",
                )
        elif label_col is not None:
            for rank in df[pareto_rank_col].unqiue():
                vis.draw_aligned_mol_grid(
                    inchi=df[df[pareto_rank_col] == rank][inchi_column].values.tolist(),
                    filename=f"pareto_rank_{rank}_3d.png",
                    labels=df[df[pareto_rank_col] == rank][label_col].values.tolist(),
                )
    else:
        log.info("Not smiles or inchi column so cannot plot chemical structures")


def plot_pareto_rank_nd(
    df: pd.DataFrame,
    objective_columns: List[str],
    label_col: Optional[str] = None,
    filename: Optional[str] = "pareto_nd.png",
    pareto_rank_col: Optional[str] = None,
    summary_plot_type: Union[str, None] = "mean",
    normalization: Optional[str] = "zscale",
    smiles_column: Optional[str] = None,
    inchi_column: Optional[str] = None,
    consistent_scale: bool = False,
    **kwargs,
):
    """
    Function to plot an N dimensional pareto front on a radar plot. This is plotted by normalizeding all properties to a [0, 1] scale using min max normalization. Typically we summarize these by the mean, max or min of the inefficent set Vs the efficent set

    Args:
        df (pd.DataFrame): Data frame that conatins at least the objective columns (can contain more). If the `label_column` or `pareto_eff_col` are set then it should also contain these.
        objective_columns (Union[List[str], str], optional): The column headers for the objetcive columns, if 'all' it uses all provided columns as the objective columns. Defaults to "all".
        label_col (Optional[str], optional): column header for the column that conatins the molecule labels. If not given no label annotation will be on the plot. Defaults to None.
        filename (Optional[str], optional): The file to save the plot to. Defaults to "pareto_front.png".
        pareto_eff_col (Optional[str], optional): column header for the column that defines whether a molecule is in the Pareto efficent set (boolean column). Defaults to None.
        summay_plot_type (Union[str, None], optional): the summary type mean, min or max of each objective for each set (inefficent and efficent). Defaults to "mean".
        normalization (Optional[str]): string name of normalization method or None to use raw data. Defaults to "zscale".
        smiles_column (str, optional): the column that contains smiles strings
        inchi_column (str, optional): the column that contains inchi strings
        consistent_scale (bool): whether to use a consistent_scale for zscale axes, these will be the min and max values over the entire dataframe which may make some small changes hard to visualize. Defaults to False.

    Raises:
        UserWarning: if an unrecognized summary plot type is given
    """

    if pareto_rank_col is None:
        pareto_rank_col = par_rank_col_name

    radar_sub_plots_pareto_ranking(
        df,
        objective_columns=objective_columns,
        filename=filename,
        label_col=label_col,
        pareto_rank_col=pareto_rank_col,
        normalization=normalization,
        smiles_column=smiles_column,
        inchi_column=inchi_column,
        consistent_scale=consistent_scale,
        **kwargs,
    )


def radar_sub_plots_pareto_ranking(
    df: pd.DataFrame,
    objective_columns: List[str],
    filename: Optional[str] = "subplot_pareto_ranks.png",
    label_col: Optional[str] = None,
    pareto_rank_col: Optional[str] = None,
    normalization: Union[str, None] = "zscale",
    legend_pos: str = "best",
    smiles_column: Optional[str] = None,
    inchi_column: Optional[str] = None,
    consistent_scale: bool = False,
    mols_per_row: int = 5,
):
    """
    Function to plot a radar plot of the normalized (zscale) mean for each property over the inefficent and the values for each molecule

    Args:
        ddf (pd.DataFrame): Data frame that conatins at least the objective columns (can contain more). If the `label_column` or `pareto_eff_col` are set then it should also contain these.
        objective_columns (Union[List[str], str], optional): The column headers for the objetcive columns, if 'all' it uses all provided columns as the objective columns. Defaults to "all".
        filename (Optional[str], optional): The file to save the plot to. Defaults to "subplot_pareto_ranks.png".
        label_col (Optional[str], optional): The column that defines the molecules labels.
        pareto_rank_col (Optional[str], optional): column header for the column that defines a molecules Pareto rank (int column). Defaults to None.
        normalization (Union[str, None]): string defining teh normalization method
        legend_pos (str): legend position string for matplotlib
        smiles_column (str, optional): the column that contains smiles strings
        inchi_column (str, optional): the column that contains inchi strings
        consistent_scale (bool): whether to use a consistent_scale for zscale axes, these will be the min and max values over the entire dataframe which may make some small changes hard to visualize. Defaults to False.

    Returns:
        None: saves plot to file
    """

    # Set teh column containing the Pareto efficent labels as bools
    if pareto_rank_col is None:
        log.info(f"Assuming Pareto rank int column is {par_rank_col_name}")
        pareto_rank_col = par_rank_col_name

    log.info(f"Dataframe input has shape: {df.shape}")

    # Set the number of objectives and the convert to polar angles
    n_obj = len(objective_columns)
    theta = np.linspace(0, 2.0 * np.pi, n_obj, endpoint=False)
    theta = np.append(theta, theta[0])

    # Get the metric columns only and normalize as requested
    metric_assay_df = df[objective_columns].copy()
    if isinstance(normalization, str):
        # min max scale
        if normalization.lower().strip() == "minmax":
            log.info("Min max scaling")
            pareto_rank_df = helpers.pandas_df_min_max_scale(metric_assay_df)
            y_ticks = np.linspace(0, 1, 5)
            y_labels = [f"{ent:.2f}" for ent in y_ticks]
        # z scale
        elif normalization.lower().strip() == "zscale":
            log.info("Z scaling")
            pareto_rank_df = helpers.pandas_df_z_scale(metric_assay_df)
            if consistent_scale is True:
                log.debug(f"Before scaling the data frame is {metric_assay_df}")
                log.debug(f"After scaling the data frame is {pareto_rank_df}")
            y_ticks = np.linspace(
                pareto_rank_df.min(numeric_only=True).min(),
                pareto_rank_df.max(numeric_only=True).max(),
                5,
            )
            log.debug(f"scaled y ticks {y_ticks}")
            y_labels = []  # [f"{ent:.2f}" for ent in y_ticks]

    # No scaling - ill advised
    else:
        log.warning("Using un-normalized property values")
        pareto_rank_df = metric_assay_df.copy()
        y_ticks = np.linspace(min(pareto_rank_df), max(pareto_rank_df), 5)
        y_labels = [f"{ent:.2f}" for ent in y_ticks]

    # At the end of this:
    #   df is the raw input dataframe
    #   metric_assay_df is a copy of the same dataframe but a subset of the columns i.e. only the objective columns
    #   pareto_rank_df is a copy of the metric_assay dataframe but scaled values

    # Get the mean of the Pareto inefficent solutions (those with a rank over 1) as a background
    # log.info(df.dtypes)
    # log.info(df)
    plot_df = pd.DataFrame(
        pareto_rank_df[df[pareto_rank_col] > 1][objective_columns].mean()
    ).transpose()

    # loop over the unique classes of the Pareto ranking an make a figure for each
    if not isinstance(mols_per_row, int):
        mols_per_row = int(mols_per_row)

    for pr in df[pareto_rank_col].unique():
        chosen_df = pd.DataFrame(pareto_rank_df[df[pareto_rank_col] == pr].copy())
        log.info(f"Length of chosen df: {len(chosen_df)}")

        # set the layout assuming 5 columns
        if len(chosen_df.index) < mols_per_row:
            log.info(f"There're less than {mols_per_row}")
            n_rows = 1
            m_columns = len(chosen_df.index)

        elif len(chosen_df.index) % mols_per_row == 0:
            log.info(f"There're are a multiple of {mols_per_row}")
            n_rows = len(chosen_df.index) / mols_per_row
            m_columns = 5

        else:
            log.info(f"There're are not a multiple of {mols_per_row}")
            n_rows = int(np.ceil(len(chosen_df.index) / mols_per_row))
            m_columns = mols_per_row

        # Set the overall figure per group
        _ = plt.figure(
            figsize=(m_columns * 7.0, n_rows * 7.0),
        )

        log.info(
            f"for Pareto ranking {pr} there are n rows: {n_rows} m columns: {m_columns}"
        )
        assert len(pareto_rank_df.index) == len(df.index)

        # log.info(chosen_df)

        # loop over the rows and create a subplot per row using the mean of the inefficent set as a background for comparison
        for ith, idx in enumerate(chosen_df.index):
            log.info(
                f"Row {ith}: N rows: {int(n_rows)} M columns: {int(m_columns)} Plot ith: {int(ith+1)}"
            )
            ax = plt.subplot(
                int(n_rows), int(m_columns), int(ith + 1), projection="polar"
            )
            ax.spines["polar"].set_color("lightgrey")

            if label_col is not None:
                lab = df.loc[idx, label_col]
            else:
                lab = str(ith)

            # plot mean of inefficent set as the background
            ax.plot(
                theta,
                plot_df.iloc[0, :].tolist() + [plot_df.iloc[0, :].tolist()[0]],
                linewidth=1.75,
                linestyle="solid",
                label="Inefficent set mean",
                marker="o",
                markersize=10,
                color="b",
            )
            ax.fill(
                theta,
                plot_df.iloc[0, :].tolist() + [plot_df.iloc[0, :].tolist()[0]],
                alpha=0.50,
                color="b",
            )

            # plot the efficent row example
            ax.plot(
                theta,
                chosen_df.iloc[ith, :].tolist() + [chosen_df.iloc[ith, :].tolist()[0]],
                linewidth=1.75,
                linestyle="solid",
                label=f"Rank {pr} solution {lab}",
                marker="o",
                markersize=10,
                color="r",
            )
            ax.fill(
                theta,
                chosen_df.iloc[ith, :].tolist() + [chosen_df.iloc[ith, :].tolist()[0]],
                alpha=0.50,
                color="r",
            )
            plt.legend(fontsize=12, loc=legend_pos)
            plt.grid(True)
            plt.title(f"Mean Inefficent Vs.\nRank {pr} {lab}", fontsize=20)
            axes_labs = [
                textwrap.fill(
                    " ".join(ent.lower().strip().split("_")),
                    width=17,
                    break_long_words=False,
                )
                for ent in plot_df.columns.values.tolist()
            ]
            plt.xticks(theta, axes_labs + [axes_labs[0]], color="black", size=12)
            plt.yticks(
                y_ticks,
                y_labels,
                color="black",
                size=12,
            )

        # save the plot and tidy up
        plt.tight_layout(h_pad=5, w_pad=5)
        if filename is not None:
            savep = pathlib.Path(filename)
            base = savep.stem
            ext = savep.suffix
            plt.savefig(f"{base}_rank_{pr}.{ext}")
        plt.show()
        plt.close()

        if smiles_column is not None:
            if label_col is None:
                vis.draw_aligned_mol_grid(
                    smiles=df[df[pareto_rank_col] == pr][smiles_column].values.tolist(),
                    filename=f"pareto_rank_{pr}_{n_obj}d.png",
                    mols_per_row=mols_per_row,
                )
            elif label_col is not None:
                vis.draw_aligned_mol_grid(
                    smiles=df[df[pareto_rank_col] == pr][smiles_column].values.tolist(),
                    filename=f"pareto_rank_{pr}_{n_obj}d.png",
                    labels=df[df[pareto_rank_col] == pr][label_col].values.tolist(),
                    mols_per_row=mols_per_row,
                )
        elif inchi_column is not None:
            if label_col is None:
                vis.draw_aligned_mol_grid(
                    inchi=df[df[pareto_rank_col] == pr][inchi_column].values.tolist(),
                    filename=f"pareto_rank_{pr}_{n_obj}d.png",
                    mols_per_row=mols_per_row,
                )
            elif label_col is not None:
                vis.draw_aligned_mol_grid(
                    inchi=df[df[pareto_rank_col] == pr][inchi_column].values.tolist(),
                    filename=f"pareto_rank_{pr}_{n_obj}d.png",
                    labels=df[df[pareto_rank_col] == pr][label_col].values.tolist(),
                    mols_per_row=mols_per_row,
                )
        else:
            log.info("Not smiles or inchi column so cannot plot chemical structures")


######################################################################################################################################################################################
######################################################################################################################################################################################
################################################################## COMPARISON RADAR PLOTS AND ONE OFFS ###############################################################################
######################################################################################################################################################################################
######################################################################################################################################################################################


def plot_single_radar_without_standard_data(
    plot_df: pd.DataFrame,
    y_ticks: List[float],
    y_labels: Optional[List[float]] = None,
    filename: str = "single_radar_without_standard.png",
    experimental_label: str = "Experimental",
    legend_pos: Union[int, str] = "best",
    experimental_colour: str = "r",
) -> None:
    """
    Function to plot Radar plots directly on the provided data. This funtion will make a single plot for a single experimental
    datapoint i.e. will only plot the first row of the experimental dataframes

    Args:
        plot_df (pd.DataFrame): Dataframe to use for plotting query/experimental results
        y_ticks (Optional[list], optional): The y tick marks to use if you want the radial scales to match other plots. Defaults to None.
        y_labels (Optional[List[float]], optional): List of y (radial) tick labels to display. Defualts to None.
        filename (str, optional): The filename to save the images to. Defaults to "radar_plot.png".
        experimental_label (Optional[str], optional): Fixed label for the experimental. Defaults to "Experimental".
        legend_pos (Union[int, str], optional): position of the legend on the plot. Defaults to "best".
        experimental_colour (str, optional): The colour to use for the experimental data. Defaults to "r".

    Returns:
        None
    """

    # Set the number of objectives and the convert to polar angles
    n_obj = len(plot_df.columns)
    theta = np.linspace(0, 2.0 * np.pi, n_obj, endpoint=False)
    theta = np.append(theta, theta[0])

    fig, ax = plt.subplots(figsize=(7.0, 7.0), subplot_kw={"projection": "polar"})

    ax.plot(
        theta,
        plot_df.iloc[0, :].tolist() + [plot_df.iloc[0, :].tolist()[0]],
        linewidth=1.75,
        linestyle="solid",
        label=experimental_label,
        marker="o",
        markersize=10,
        color=experimental_colour,
    )
    ax.fill(
        theta,
        plot_df.iloc[0, :].tolist() + [plot_df.iloc[0, :].tolist()[0]],
        alpha=0.50,
        color=experimental_colour,
    )

    plt.legend(fontsize=15, loc=legend_pos)
    plt.grid(True)
    plt.title(f"{experimental_label} Radar Plot", fontsize=20)
    axes_labs = [
        textwrap.fill(
            " ".join(ent.lower().strip().split("_")),
            width=17,
            break_long_words=False,
        )
        for ent in plot_df.columns.values.tolist()
    ]
    plt.xticks(theta, axes_labs + [axes_labs[0]], color="black", size=17)
    plt.yticks(
        y_ticks,
        y_labels,
        color="black",
        size=15,
    )

    # save the plot and tidy up
    plt.tight_layout(h_pad=5, w_pad=5)
    plt.savefig(filename)
    plt.close()


def plot_single_radar_with_standard_data(
    plot_df: pd.DataFrame,
    standard_plot_df: pd.DataFrame,
    y_ticks: Optional[List[float]] = None,
    y_labels: Optional[List[float]] = None,
    filename: str = "single_radar_with_standard.png",
    standard_label: str = "Standard",
    experimental_label: str = "Experimental",
    legend_pos: Union[int, str] = "best",
    standard_colour: str = "b",
    experimental_colour: str = "r",
) -> None:
    """
    Function to plot Radar plots directly on the provided data. This funtion will make a single plot for a single experimental and standard
    datapoint i.e. will only plot the first row of the experimental and standard dataframes

    Args:
        plot_df (pd.DataFrame): Dataframe to use for plotting query/experimental results
        standard_plot_df (Optional[pd.DataFrame], optional): Dataframe to use as standard data for comparing the experimenatl data to. Defaults to None.
        y_ticks (Optional[list], optional): The y tick marks to use if you want the radial scales to match other plots. Defaults to None.
        y_label (Optional[List[float]], optional): List of y (radial) tick labels to display. Defualts to None.
        filename (str, optional): The filename to save the images to. Defaults to "radar_plot.png".
        standard_label (Optional[str], optional): Fixed label for the standard. Defaults to "Standard".
        experimental_label (Optional[str], optional): Fixed label for the experimental. Defaults to "Experimental".
        legend_pos (Union[int, str], optional): position of the legend on the plot. Defaults to "best".
        standard_colour (str, optional): The colour to use for the standard. Defaults to "b".
        experimental_colour (str, optional): The colour to use for the experimental data. Defaults to "r".

    Returns:
        None
    """
    # base check that the columns are teh same length
    _ = helpers.check_dfs_have_the_same_number_of_columns(
        plot_df, standard_plot_df, df1_name="Experimental", df2_name="Standard"
    )

    # base check that the columns have the same names
    _ = helpers.check_dfs_have_the_same_column_names(
        plot_df, standard_plot_df, df1_name="Experimental", df2_name="Standard"
    )

    # Set the number of objectives and the convert to polar angles
    n_obj = len(plot_df.columns)
    theta = np.linspace(0, 2.0 * np.pi, n_obj, endpoint=False)
    theta = np.append(theta, theta[0])

    fig, ax = plt.subplots(figsize=(7.0, 7.0), subplot_kw={"projection": "polar"})
    ax.plot(
        theta,
        standard_plot_df.iloc[0, :].tolist()
        + [standard_plot_df.iloc[0, :].tolist()[0]],
        linewidth=1.75,
        linestyle="solid",
        label=standard_label,
        marker="o",
        markersize=10,
        color=standard_colour,
    )
    ax.fill(
        theta,
        standard_plot_df.iloc[0, :].tolist()
        + [standard_plot_df.iloc[0, :].tolist()[0]],
        alpha=0.50,
        color=standard_colour,
    )

    ax.plot(
        theta,
        plot_df.iloc[0, :].tolist() + [plot_df.iloc[0, :].tolist()[0]],
        linewidth=1.75,
        linestyle="solid",
        label=experimental_label,
        marker="o",
        markersize=10,
        color=experimental_colour,
    )
    ax.fill(
        theta,
        plot_df.iloc[0, :].tolist() + [plot_df.iloc[0, :].tolist()[0]],
        alpha=0.50,
        color=experimental_colour,
    )

    plt.legend(fontsize=15, loc=legend_pos)
    plt.grid(True)
    plt.title(f"{standard_label} Vs.\n{experimental_label} Radar Plot", fontsize=20)
    axes_labs = [
        textwrap.fill(
            " ".join(ent.lower().strip().split("_")),
            width=17,
            break_long_words=False,
        )
        for ent in plot_df.columns.values.tolist()
    ]
    plt.xticks(theta, axes_labs + [axes_labs[0]], color="black", size=17)
    plt.yticks(
        y_ticks,
        y_labels,
        color="black",
        size=15,
    )

    # save the plot and tidy up
    plt.tight_layout(h_pad=5, w_pad=5)
    plt.savefig(filename)
    plt.close()


def plot_multi_radar_with_standard_data(
    plot_df: pd.DataFrame,
    standard_plot_df: pd.DataFrame,
    y_ticks: List[float],
    y_labels: Optional[List[float]] = None,
    filename: str = "multi_radar_with_standard.png",
    standard_label: Optional[str] = None,
    experimental_label: Optional[str] = None,
    legend_pos: Union[int, str] = "best",
    standard_colour: str = "b",
    experimental_colour: str = "r",
    mols_per_row: int = 5,
    experimental_labels: Union[List[str], None] = None,
    standard_labels: Union[List[str], None] = None,
) -> None:
    """
    Function to plot Radar plots directly on the provided data. This funtion will make a single plot of subplots one for each experimental molecules
    (row in df dataframe) and will plot the standards dataframe, if it is avaliable, row(s) on each  of the subplots as a coparison point that is consistent
    over the sub plots.

    Args:
        plot_df (pd.DataFrame): Dataframe to use for plotting query/experimental results
        standard_plot_df (Optional[pd.DataFrame], optional): Dataframe to use as standard data for comparing the experimenatl data to. Defaults to None.
        y_ticks (Optional[list], optional): The y tick marks to use if you want the radial scales to match other plots. Defaults to None.
        y_label (Optional[List[float]], optional): List of y (radial) tick labels to display. Defualts to None.
        filename (str, optional): The filename to save the images to. Defaults to "radar_plot.png".
        standard_label (Optional[str], optional): Fixed label for the standard. Defaults to "Standard".
        experimental_label (Optional[str], optional): Fixed label for the experimental. Defaults to "Experimental".
        legend_pos (Union[int, str], optional): position of the legend on the plot. Defaults to "best".
        standard_colour (str, optional): The colour to use for the standard. Defaults to "b".
        experimental_colour (str, optional): The colour to use for the experimental data. Defaults to "r".
        mols_per_row: int: The number of molecule to plot in a sub plot row. Defaults to 5.
        experimental_label (Optional[str], optional): List of labels for each row (moleucle) of the experimantal dataframe. Defaults to None.
        standard_labels (Optional[str], optional): List of labels for each row (moleucle) of the standard dataframe. Defaults to None.

    Returns:
        None
    """
    # base check that the columns are teh same length
    _ = helpers.check_dfs_have_the_same_number_of_columns(
        plot_df, standard_plot_df, df1_name="Experimental", df2_name="Standard"
    )

    # base check that the columns have the same names
    _ = helpers.check_dfs_have_the_same_column_names(
        plot_df, standard_plot_df, df1_name="Experimental", df2_name="Standard"
    )

    # Set the number of objectives and the convert to polar angles
    n_obj = len(plot_df.columns)
    theta = np.linspace(0, 2.0 * np.pi, n_obj, endpoint=False)
    theta = np.append(theta, theta[0])

    n_rows, m_columns = helpers.get_grid_layout(
        len(plot_df.index), mols_per_row=mols_per_row
    )

    # Set the overall figure
    _ = plt.figure(
        figsize=(m_columns * 7.0, n_rows * 7.0),
    )

    # loop over the rows and create a subplot per row using the mean of the inefficent set as a background for comparison
    for ith, idx in enumerate(plot_df.index):
        log.debug(
            f"Row {ith}: N rows: {int(n_rows)} M columns: {int(m_columns)} Plot ith: {int(ith+1)}"
        )

        ax = plt.subplot(int(n_rows), int(m_columns), int(ith + 1), projection="polar")
        ax.spines["polar"].set_color("lightgrey")

        #### STANDARD PLOTTING
        # normalize item number values to colormap
        norm = colors.Normalize(vmin=0, vmax=len(standard_plot_df.index))
        # Loop over the standards if there are multiple
        for jth, standard_idx in enumerate(standard_plot_df.index):
            log.debug(standard_idx)
            if standard_labels is not None:
                try:
                    standard_label = standard_labels[jth]
                except KeyError:
                    pass

            # colour map that should not overlap with Experimental red default
            if len(standard_plot_df.index) == 1:
                rgba_color = standard_colour
            else:
                rgba_color = cm.terrain(norm(jth))

            ax.plot(
                theta,
                standard_plot_df.iloc[jth, :].tolist()
                + [standard_plot_df.iloc[jth, :].tolist()[0]],
                linewidth=1.75,
                linestyle="solid",
                label=standard_label,
                marker="o",
                markersize=10,
                color=rgba_color,
            )
            ax.fill(
                theta,
                standard_plot_df.iloc[jth, :].tolist()
                + [standard_plot_df.iloc[jth, :].tolist()[0]],
                alpha=0.50,
                color=rgba_color,
            )

        # For the title of the plot
        if len(standard_plot_df.index) > 1:
            standard_label = "Standards"

        ##### EXPERIMENTAL PLOTTING
        if experimental_labels is not None:
            try:
                experimental_label = experimental_labels[ith]
                log.info(experimental_label)
            except KeyError:
                log.error(plot_df.columns)

        # Now plot the Experimental molecule on one per sub plot
        ax.plot(
            theta,
            plot_df.iloc[ith, :].tolist() + [plot_df.iloc[ith, :].tolist()[0]],
            linewidth=1.75,
            linestyle="solid",
            label=experimental_label,
            marker="o",
            markersize=10,
            color=experimental_colour,
        )
        ax.fill(
            theta,
            plot_df.iloc[ith, :].tolist() + [plot_df.iloc[ith, :].tolist()[0]],
            alpha=0.50,
            color=experimental_colour,
        )

        plt.legend(fontsize=15, loc=legend_pos)
        plt.grid(True)
        plt.title(f"{standard_label} Vs.\n{experimental_label} Radar Plot", fontsize=20)
        axes_labs = [
            textwrap.fill(
                " ".join(ent.lower().strip().split("_")),
                width=17,
                break_long_words=False,
            )
            for ent in plot_df.columns.values.tolist()
        ]
        plt.xticks(theta, axes_labs + [axes_labs[0]], color="black", size=17)
        plt.yticks(
            y_ticks,
            y_labels,
            color="black",
            size=15,
        )

    # save the plot and tidy up
    plt.tight_layout(h_pad=5, w_pad=5)
    plt.savefig(filename)
    plt.close()


def plot_multi_radar_without_standard_data(
    plot_df: pd.DataFrame,
    y_ticks: List[float],
    y_labels: Optional[List[float]] = None,
    filename: str = "multi_radar_with_standard.png",
    experimental_label: Optional[str] = None,
    legend_pos: Union[int, str] = "best",
    experimental_colour: str = "r",
    mols_per_row: int = 5,
    experimental_labels: Union[List[str], None] = None,
) -> None:
    """
    Function to plot Radar plots directly on the provided data. This funtion will make a single plot of subplots one for each experimental molecules
    (row in df dataframe).

    Args:
        plot_df (pd.DataFrame): Dataframe to use for plotting query/experimental results
        y_ticks (Optional[list], optional): The y tick marks to use if you want the radial scales to match other plots. Defaults to None.
        y_labels (Optional[List[float]], optional): List of radial (y) ticks label values. Defaults to None.
        filename (str, optional): The filename to save the images to. Defaults to "radar_plot.png".
        experimental_label (Optional[str], optional): Fixed label for the experimental. Defaults to "Experimental".
        legend_pos (Union[int, str], optional): position of the legend on the plot. Defaults to "best".
        experimental_colour (str, optional): The colour to use for the experimental data. Defaults to "r".
        mols_per_row (int): The number of molecules to plot per row. Defaults to 5.
        experimental_label (Optional[str], optional): List of labels for each row (moleucle). Defaults to None.

    Returns:
        None
    """

    # Set the number of objectives and the convert to polar angles
    n_obj = len(plot_df.columns)
    theta = np.linspace(0, 2.0 * np.pi, n_obj, endpoint=False)
    theta = np.append(theta, theta[0])

    n_rows, m_columns = helpers.get_grid_layout(
        len(plot_df.index), mols_per_row=mols_per_row
    )

    # Set the overall figure
    _ = plt.figure(
        figsize=(m_columns * 7.0, n_rows * 7.0),
    )

    # loop over the rows and create a subplot per row using the mean of the inefficent set as a background for comparison
    for ith, idx in enumerate(plot_df.index):
        log.debug(
            f"Row {ith}: N rows: {int(n_rows)} M columns: {int(m_columns)} Plot ith: {int(ith+1)}"
        )

        ax = plt.subplot(int(n_rows), int(m_columns), int(ith + 1), projection="polar")
        ax.spines["polar"].set_color("lightgrey")

        ##### EXPERIMENTAL PLOTTING
        if experimental_labels is not None:
            try:
                experimental_label = experimental_labels[ith]
                log.info(experimental_label)
            except KeyError:
                log.error(plot_df.columns)

        # Now plot the Experimental molecule on one per sub plot
        ax.plot(
            theta,
            plot_df.iloc[ith, :].tolist() + [plot_df.iloc[ith, :].tolist()[0]],
            linewidth=1.75,
            linestyle="solid",
            label=experimental_label,
            marker="o",
            markersize=10,
            color=experimental_colour,
        )
        ax.fill(
            theta,
            plot_df.iloc[ith, :].tolist() + [plot_df.iloc[ith, :].tolist()[0]],
            alpha=0.50,
            color=experimental_colour,
        )

        plt.legend(fontsize=15, loc=legend_pos)
        plt.grid(True)
        plt.title(f"{experimental_label} Radar Plot", fontsize=20)
        axes_labs = [
            textwrap.fill(
                " ".join(ent.lower().strip().split("_")),
                width=17,
                break_long_words=False,
            )
            for ent in plot_df.columns.values.tolist()
        ]
        plt.xticks(theta, axes_labs + [axes_labs[0]], color="black", size=17)
        plt.yticks(
            y_ticks,
            y_labels,
            color="black",
            size=15,
        )

    # save the plot and tidy up
    plt.tight_layout(h_pad=5, w_pad=5)
    plt.savefig(filename)
    plt.close()


def plot_multi_radar_overlay(
    plot_df: pd.DataFrame,
    standard_plot_df: pd.DataFrame,
    y_ticks: List[float],
    y_labels: Optional[List[float]] = None,
    filename: str = "multi_radar_with_standard.png",
    standard_label: Optional[str] = None,
    experimental_label: Optional[str] = None,
    legend_pos: Union[int, str] = "best",
    standard_colour: str = "b",
    experimental_colour: str = "r",
    mols_per_row: int = 5,
    experimental_labels: Union[List[str], None] = None,
    standard_labels: Union[List[str], None] = None,
) -> None:
    """
    Function to plot Radar plots directly on the provided data. This funtion will make a single plot of subplots one for each experimental molecules
    (row in df dataframe) and will plot the standards dataframe, if it is avaliable, row(s) on each  of the subplots as a coparison point that is consistent
    over the sub plots.

    Args:
        plot_df (pd.DataFrame): Dataframe to use for plotting query/experimental results
        standard_plot_df (Optional[pd.DataFrame], optional): Dataframe to use as standard data for comparing the experimenatl data to. Defaults to None.
        y_ticks (Optional[list], optional): The y tick marks to use if you want the radial scales to match other plots. Defaults to None.
        y_label (Optional[List[float]], optional): List of y (radial) tick labels to display. Defualts to None.
        filename (str, optional): The filename to save the images to. Defaults to "radar_plot.png".
        standard_label (Optional[str], optional): Fixed label for the standard. Defaults to "Standard".
        experimental_label (Optional[str], optional): Fixed label for the experimental. Defaults to "Experimental".
        legend_pos (Union[int, str], optional): position of the legend on the plot. Defaults to "best".
        standard_colour (str, optional): The colour to use for the standard. Defaults to "b".
        experimental_colour (str, optional): The colour to use for the experimental data. Defaults to "r".
        mols_per_row: int: The number of molecule to plot in a sub plot row. Defaults to 5.
        experimental_label (Optional[str], optional): List of labels for each row (moleucle) of the experimantal dataframe. Defaults to None.
        standard_labels (Optional[str], optional): List of labels for each row (moleucle) of the standard dataframe. Defaults to None.

    Returns:
        None
    """
    # base check that the columns are teh same length
    _ = helpers.check_dfs_have_the_same_number_of_columns(
        plot_df, standard_plot_df, df1_name="Experimental", df2_name="Standard"
    )

    # base check that the columns have the same names
    _ = helpers.check_dfs_have_the_same_column_names(
        plot_df, standard_plot_df, df1_name="Experimental", df2_name="Standard"
    )

    # Set the number of objectives and the convert to polar angles
    n_obj = len(plot_df.columns)
    theta = np.linspace(0, 2.0 * np.pi, n_obj, endpoint=False)
    theta = np.append(theta, theta[0])

    n_rows, m_columns = helpers.get_grid_layout(
        len(plot_df.index), mols_per_row=mols_per_row
    )

    # Set the overall figure
    _, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    # loop over the rows and create a subplot per row using the mean of the inefficent set as a background for comparison
    for ith, idx in enumerate(plot_df.index):
        log.debug(
            f"Row {ith}: N rows: {int(n_rows)} M columns: {int(m_columns)} Plot ith: {int(ith+1)}"
        )

        ax.spines["polar"].set_color("lightgrey")

        #### STANDARD PLOTTING
        # normalize item number values to colormap
        norm = colors.Normalize(vmin=0, vmax=len(standard_plot_df.index))
        # Loop over the standards if there are multiple
        for jth, standard_idx in enumerate(standard_plot_df.index):
            log.debug(standard_idx)
            if standard_labels is not None:
                try:
                    standard_label = standard_labels[jth]
                except KeyError:
                    pass

            # colour map that should not overlap with Experimental red default
            if len(standard_plot_df.index) == 1:
                rgba_color = standard_colour
            else:
                rgba_color = cm.terrain(norm(jth))

            ax.plot(
                theta,
                standard_plot_df.iloc[jth, :].tolist()
                + [standard_plot_df.iloc[jth, :].tolist()[0]],
                linewidth=1.75,
                linestyle="solid",
                label=standard_label,
                marker="o",
                markersize=10,
                color=rgba_color,
            )
            ax.fill(
                theta,
                standard_plot_df.iloc[jth, :].tolist()
                + [standard_plot_df.iloc[jth, :].tolist()[0]],
                alpha=0.50,
                color=rgba_color,
            )

        # For the title of the plot
        if len(standard_plot_df.index) > 1:
            standard_label = "Standards"

        ##### EXPERIMENTAL PLOTTING
        norm = colors.Normalize(vmin=0, vmax=len(plot_df.index))

        # colour map that should not overlap with Experimental red default
        if len(plot_df.index) == 1:
            rgba_color = experimental_colour
        else:
            rgba_color = cm.autumn(norm(ith))

        if experimental_labels is not None:
            try:
                experimental_label = experimental_labels[ith]
                log.info(experimental_label)
            except KeyError:
                log.error(plot_df.columns)

        # Now plot the Experimental molecule on one per sub plot
        ax.plot(
            theta,
            plot_df.iloc[ith, :].tolist() + [plot_df.iloc[ith, :].tolist()[0]],
            linewidth=1.75,
            linestyle="solid",
            label=experimental_label,
            marker="o",
            markersize=10,
            color=rgba_color,
        )
        ax.fill(
            theta,
            plot_df.iloc[ith, :].tolist() + [plot_df.iloc[ith, :].tolist()[0]],
            alpha=0.50,
            color=rgba_color,
        )

        plt.legend(fontsize=15, loc=legend_pos)
        plt.grid(True)
        plt.title(f"{standard_label} Vs.\n{experimental_label} Radar Plot", fontsize=20)
        axes_labs = [
            textwrap.fill(
                " ".join(ent.lower().strip().split("_")),
                width=17,
                break_long_words=False,
            )
            for ent in plot_df.columns.values.tolist()
        ]
        plt.xticks(theta, axes_labs + [axes_labs[0]], color="black", size=17)
        plt.yticks(
            y_ticks,
            y_labels,
            color="black",
            size=15,
        )

    # save the plot and tidy up
    plt.tight_layout(h_pad=5, w_pad=5)
    plt.savefig(filename)
    plt.close()


def radar_plot(
    df: pd.DataFrame,
    objective_columns: Optional[List[Union[str, int]]] = None,
    df_standard: Optional[pd.DataFrame] = None,
    standard_unique_identifier: Optional[str] = None,
    standard_unique_identifer_column: Optional[str] = None,
    y_ticks: Optional[list] = None,
    filename: str = "radar_plot.png",
    label_column: Optional[Union[str, int]] = None,
    standard_label: Optional[str] = "Standard",
    experimental_label: Optional[str] = "Experimental",
    legend_pos: Union[int, str] = "best",
    show_y_ticks: bool = False,
    standard_colour: str = "b",
    experimental_colour: str = "r",
    **kwargs,
):
    """
    Function to plot Radar plots directly on the provided data. There is no automated scaling in this function as in the analysis based functions above
    this is just to plot the data. Having this separated allows for flexibility in plotting for example MMP series. This funtion will make a single plot
    of subplots one for each experimental molecules (row in df dataframe) and will plot the standards dataframe, if it is avaliable, row(s) on each
    of the subplots as a coparison point that is consistent over the sub plots.

    Args:
        df (pd.DataFrame): Dataframe to use for plotting query/experimental results
        objective_columns (Optional[List[str, int]], optional): The columns to use as the objectives if not given assumes all columns are objective columns. Defaults to None.
        df_standard (Optional[pd.DataFrame], optional): Dataframe to use as standard data for comparing the experimenatl data to. Defaults to None.
        standard_unique_identifier (Optional[str], optional): Column name in standard dataframe to identify a molecule. Defaults to None.
        standard_unique_identifer_column (Optional[str], optional): Column name in experimental dataframe to identify a molecule. Defaults to None.
        y_ticks (Optional[list], optional): The y tick marks to use if you want the radial scales to match other plots. Defaults to None.
        filename (str, optional): The filename to save the images to. Defaults to "radar_plot.png".
        label_column (Optional[Union[str, int]], optional): Column name in experimental and standard  dataframe to give a label a molecule. Defaults to None.
        standard_label (Optional[str], optional): Fixed label for the standard. Defaults to "Standard".
        experimental_label (Optional[str], optional): Fixed label for the experimental. Defaults to "Experimental".
        legend_pos (Union[int, str], optional): position of the legend on the plot. Defaults to "best".
        show_y_ticks (bool, optional): Bool True show the radial y_tick values False don't show them. Defaults to False.
        standard_colour (str, optional): The colour to use for the standard. Defaults to "b".
        experimental_colour (str, optional): The colour to use for the experimental data. Defaults to "r".

    Returns:
        None
    """

    # If objective columns is None not given assume all query/experimental columns are objective columns
    if objective_columns is None:
        objective_columns = df.columns.tolist()

    # Get the standard data if there is standard data and extract the objective columns for plotting
    # if there is a stanard df provoided
    if df_standard is not None:
        log.info(f"Using standard data frame{os.linesep}{df_standard}")
        standard_plot_df = df_standard[objective_columns]

    # if the standard data need to be taken out of a dataframe
    elif df_standard is None and standard_unique_identifer_column is not None:
        # If the standard data is part of the main data frame copy it and delete it
        df, df_standard = helpers.extract_and_remove_row_from_df(
            df,
            standard_unique_identifer_column=standard_unique_identifer_column,
            standard_unique_identifier=standard_unique_identifier,
        )
        log.info(f"Using row from main df as the standard {os.linesep}{df_standard}")
        standard_plot_df = df_standard[objective_columns]
    # if there is no standard data
    else:
        log.info("No standard provided will plot just the objective functions")
        standard_plot_df = None

    # Get the plot data for the experimental points
    plot_df = df[objective_columns]

    if label_column is not None:
        try:
            experimental_label = df[label_column].to_list()
        except KeyError:
            experimental_label = None

        try:
            standard_label = df_standard[label_column].to_list()
        except KeyError:
            standard_label = None

    # set the y ticks if they are not given
    if y_ticks is None:
        y_ticks = np.linspace(
            min(
                df.min(numeric_only=True).min(),
                standard_plot_df.min(numeric_only=True).min(),
            ),
            max(
                df.max(numeric_only=True).max(),
                standard_plot_df.max(numeric_only=True).max(),
            ),
            5,
        )
        log.debug(f"Y ticks {y_ticks}")

    if show_y_ticks is False:
        y_labels = []
    else:
        y_labels = [f"{ent:.2f}" for ent in y_ticks]

    if df.shape[0] == 1:
        if standard_label is None:
            standard_label = "Standard"
        if experimental_label is None:
            experimental_label = "Experimental"

        if df_standard is not None:
            plot_single_radar_with_standard_data(
                plot_df,
                standard_plot_df,
                y_ticks=y_ticks,
                y_labels=y_labels,
                filename=filename,
                standard_label=standard_label,
                experimental_label=experimental_label,
                legend_pos=legend_pos,
                standard_colour=standard_colour,
                experimental_colour=experimental_colour,
                **kwargs,
            )
        else:
            plot_single_radar_without_standard_data(
                plot_df,
                y_ticks=y_ticks,
                y_labels=y_labels,
                filename=filename,
                experimental_label=experimental_label,
                legend_pos=legend_pos,
                experimental_colour=experimental_colour,
                **kwargs,
            )
    else:
        if standard_label is None:
            standard_label = "Standard"
        if experimental_label is None:
            experimental_label = "Experimental"

        if df_standard is not None:
            plot_multi_radar_with_standard_data(
                plot_df,
                standard_plot_df,
                y_ticks=y_ticks,
                y_labels=y_labels,
                filename=filename,
                standard_label=standard_label,
                experimental_label=experimental_label,
                legend_pos=legend_pos,
                standard_colour=standard_colour,
                experimental_colour=experimental_colour,
                experimental_labels=experimental_label,
                standard_labels=standard_label,
                **kwargs,
            )
        else:
            plot_multi_radar_without_standard_data(
                plot_df,
                y_ticks=y_ticks,
                y_labels=y_labels,
                filename=filename,
                experimental_label=experimental_label,
                legend_pos=legend_pos,
                experimental_colour=experimental_colour,
                **kwargs,
            )


def comparison_grid_radar_plots(
    df,
    objective_columns: List[Union[str, int]],
    df_standard: Optional[
        pd.DataFrame
    ] = None,  # if not given and standard idenitifer not given then only plot the objective columns
    standard_unique_identifier: Optional[str] = None,
    standard_unique_identifer_column: Optional[str] = None,
):
    # TODO
    pass


def plot_idealized_radar_plot(
    min_value: float = 0.1,
    max_value: float = 1.0,
    objectives: Optional[List[str]] = None,
    legend_labels=("Best", "Worst"),
    filename: str = "illustrtaion_idelaized_pareto_front_radar_plot.png",
):
    """
    Plot an idelaized radar plot

    Args:
        min_value (float, optional): The minimum value for the plot. Defaults to 0.1.
        max_value (float, optional): The maximum value for the plot. Defaults to 1.0.
        objectives (Optional[List[str]]): List of objetcives to use
        filename (str, optional): file name to save to. Defaults to "illustrtaion_idelaized_pareto_front_radar_plot.png".
    """

    if objectives is None:
        objectives = [
            "Objective 1",
            "Objective 2",
            "Objective 3",
            "Objective 4",
            "Objective 5",
            "Objective 6",
        ]

    plot_df = pd.DataFrame(
        [
            [max_value, max_value, max_value, min_value, min_value, min_value],
            [min_value, min_value, min_value, max_value, max_value, max_value],
        ],
        columns=objectives,
    )
    _, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    theta = np.linspace(0, 2.0 * np.pi, len(objectives), endpoint=False)
    theta = np.append(theta, theta[0])
    ax.set_theta_zero_location("N", offset=330)
    ax.set_theta_direction(-1)
    ax.spines["polar"].set_zorder(1)

    # Now plot the Experimental molecule on one per sub plot
    ax.plot(
        theta,
        plot_df.iloc[0, :].tolist() + [plot_df.iloc[0, :].tolist()[0]],
        linewidth=1.75,
        linestyle="solid",
        label=legend_labels[0],
        marker="o",
        markersize=10,
        color="r",
    )
    ax.fill(
        theta,
        plot_df.iloc[0, :].tolist() + [plot_df.iloc[0, :].tolist()[0]],
        alpha=0.50,
        color="r",
    )

    ax.plot(
        theta,
        plot_df.iloc[1, :].tolist() + [plot_df.iloc[1, :].tolist()[0]],
        linewidth=1.75,
        linestyle="solid",
        label=legend_labels[1],
        marker="o",
        markersize=10,
        color="b",
    )
    ax.fill(
        theta,
        plot_df.iloc[1, :].tolist() + [plot_df.iloc[1, :].tolist()[0]],
        alpha=0.50,
        color="b",
    )

    plt.legend(fontsize=15, loc="center")
    plt.grid(True)
    plt.title("Illustrative Radar Pareto Front Plot\n", fontsize=20)
    axes_labs = [
        textwrap.fill(
            " ".join(ent.strip().split("_")),
            width=17,
            break_long_words=False,
        )
        for ent in plot_df.columns.values.tolist()
    ]
    plt.xticks(theta, axes_labs + [axes_labs[0]], color="black", size=17)
    plt.yticks(
        [0.0, 0.5, 1.0, max_value + (min_value * 0.2)],
        [None, None, None, None],
        color="black",
        size=15,
    )

    # save the plot and tidy up
    plt.tight_layout(h_pad=5, w_pad=5)
    plt.savefig(filename)
    plt.close()
