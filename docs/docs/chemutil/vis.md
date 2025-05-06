# Module chemutil.vis

Module for providing generally useful visulaization functions

??? example "View Source"
        #!/usr/bin.env python3

        # -*- coding: utf-8 -*-

        """

        Module for providing generally useful visulaization functions

        """

        from typing import List, Optional, Callable, Union, Tuple

        from PIL.PngImagePlugin import PngImageFile

        from PIL import Image

        import pandas as pd

        import numpy as np

        from rdkit import Chem

        from rdkit.Chem import Draw, AllChem, rdFMCS

        from rdkit import DataStructs

        from rdkit.Chem.Draw import SimilarityMaps

        from rdkit.Chem.rdMolDescriptors import CalcMolFormula

        import matplotlib.pyplot as plt

        from matplotlib.axes import Axes

        from chemutil import featurization, smilesmod, inchimod

        import textwrap

        import seaborn as sns

        import logging

        log = logging.getLogger(__name__)



        def draw_molecule_to_file(

            mol: Optional[Chem.rdchem.Mol] = None,

            smiles: Optional[str] = None,

            inchi: Optional[str] = None,

            mcs_smarts: Optional[str] = None,

            labels: Optional[List[str]] = None,

            image_size: Tuple[int, int] = (400, 400),

            filename: Optional[str] = None,

        ) -> PngImageFile:

            """

            Draw a molecule image and write to file if requested. Return the raw image object.

            Args:

                mols (Optional[List[rdkit.rdMol.Mol]], optional): List of RDKit molecule objects. Defaults to None.

                smiles (Optional[List[str]], optional): List of smiles strings. Defaults to None.

                inchi (Optional[List[str]], optional): List of InChi strings. Defaults to None.

                mcs_smarts (Optional[List[str]], optional): Pre computed maximum common substructure smarts string. Defaults to None.

                labels (Optional[List[str]], optional): List of string labels for the molecules. Defaults to None.

                sub_image_size (Tuple[int, int], optional): the size in pixels of each molecule image. Defaults to (400, 400).

                filename (Optional[str], optional): What to save the image as if None it is not saved. Defaults to None.

            Raises:

                UserWarning: If None of the molecules formats are provided by the use

            Returns:

                PIL.PngImagePlugin.PngImageFile: Raw image object (.save(file) to save the image)

            """

            # check the user has input one of the required inputs

            if all(ent is None for ent in [mol, smiles, inchi]):

                raise UserWarning("One of mols, smiles or inchi must be given for input")

            # Make sure there is a list of molecules

            if mol is None:

                # smiles is the lowest priority

                if inchi is not None:

                    mol = inchimod.inchi_to_molecule(inchi=inchi)

                elif smiles is not None:

                    mol = smilesmod.smiles_to_molecule(smi=smiles)

            # one moleucle image code aligned to the mcs smarts from a user or not when not avaliable

            if mcs_smarts is None:

                img = Draw.MolToImage(

                    mol, size=image_size, kekulize=True, wedgeBonds=True, legends=labels

                )

            else:

                mcs_mol = Chem.MolFromSmarts(mcs_smarts)

                AllChem.Compute2DCoords(mcs_mol)

                _ = AllChem.GenerateDepictionMatching2DStructure(mol, mcs_mol)

                img = Draw.MolToImage(

                    mol, size=image_size, kekulize=True, wedgeBonds=True, legends=labels

                )

            if filename is not None:

                img.save(filename)

            return img



        def draw_aligned_mol_grid(

            mols: Optional[List[Chem.rdchem.Mol]] = None,

            smiles: Optional[List[str]] = None,

            inchi: Optional[List[str]] = None,

            mcs_smarts: Optional[str] = None,

            labels: Optional[List[str]] = None,

            mols_per_row: int = 5,

            sub_image_size: Tuple[int, int] = (400, 400),

            filename: Optional[str] = None,

            max_mols: Optional[int] = None,

            return_image: bool = True,

            legend_fontsize: int = 20,

            legend_fraction: float = 0.1,

            highlight_mcs: bool = False,

        ) -> PngImageFile:

            """

            Draw a grid of molecule images and write to file if requested. Return the raw image object.

            Args:

                mols (Optional[List[rdkit.rdMol.Mol]], optional): List of RDKit molecule objects. Defaults to None.

                smiles (Optional[List[str]], optional): List of smiles strings. Defaults to None.

                inchi (Optional[List[str]], optional): List of InChi strings. Defaults to None.

                mcs_smarts (Optional[List[str]], optional): Pre computed maximum common substructure smarts string. Defaults to None.

                labels (Optional[List[str]], optional): List of string labels for the molecules. Defaults to None.

                mols_per_row (int, optional): number of molecules to show in each row. Defaults to 5.

                sub_image_size (Tuple[int, int], optional): the size in pixels of each molecule image. Defaults to (400, 400).

                filename (Optional[str], optional): What to save the image as if None it is not saved. Defaults to None.

                max_mols (Optional[int], optional): The max mols to show in a grid plot None plot all. Defaults to None.

            Raises:

                UserWarning: If None of the molecules formats are provided by the use

            Returns:

                PIL.PngImagePlugin.PngImageFile: Raw image object (.save(file) to save the image)

            """

            # check the user has input one of the required inputs

            if all(ent is None for ent in [mols, smiles, inchi]):

                raise UserWarning("One of mols, smiles or inchi must be given for input")

            # Make sure there is a list of molecules

            if mols is None:

                # smiles is the lowest priority

                if inchi is not None:

                    mols = [inchimod.inchi_to_molecule(inchi=inch) for inch in inchi]

                elif smiles is not None:

                    mols = [smilesmod.smiles_to_molecule(smi=smil) for smil in smiles]

            if max_mols is None:

                max_mols = len(mols)

            opts = Draw.MolDrawOptions()

            opts.legendFraction = legend_fraction

            opts.legendFontSize = legend_fontsize

            # Save one molecule image using the grid image function

            if len(mols) == 1 and mcs_smarts is None:

                log.warning("Plotting only one molecule")

                img = Draw.MolsToGridImage(

                    mols,

                    molsPerRow=1,

                    subImgSize=sub_image_size,

                    legends=labels,

                    useSVG=False,

                    returnPNG=False,

                    maxMols=max_mols,

                    drawOptions=opts,

                )

            else:

                # Get maximum common substructure across list of mols to align to

                if mcs_smarts is None:

                    mcs_res = rdFMCS.FindMCS(

                        mols,

                        completeRingsOnly=True,

                        matchChiralTag=True,

                        ringMatchesRingOnly=True,

                        timeout=30,

                    )

                    mcs_smarts = mcs_res.smartsString

                    log.debug(f"MCS found: {mcs_smarts}")

                mcs_mol = Chem.MolFromSmarts(mcs_smarts)

                AllChem.Compute2DCoords(mcs_mol)

                # Align all molecules using the maximum common substructure

                for m in mols:

                    _ = AllChem.GenerateDepictionMatching2DStructure(m, mcs_mol)

                if highlight_mcs is True:

                    highlight_atom_list = [mol.GetSubstructMatch(mcs_mol) for mol in mols]

                else:

                    highlight_atom_list = None

                if len(mols) < mols_per_row:

                    mols_per_row = len(mols)

                # Draw the grid image

                img = Draw.MolsToGridImage(

                    mols,

                    molsPerRow=mols_per_row,

                    subImgSize=sub_image_size,

                    legends=labels,

                    useSVG=False,

                    returnPNG=False,

                    maxMols=max_mols,

                    highlightAtomLists=highlight_atom_list,

                    drawOptions=opts,

                )

            # If there is a filename save the image to file

            if filename is not None:

                img.save(filename)

            if return_image is True:

                return img

            else:

                img.close()



        def list_of_png_files_to_single_pdf(pngs: List[str], pdf_filename: str = "joined.pdf"):

            """

            Function to make one PDF file from a list of image files and paths

            Args:

                pngs (list[str]): list of image files and paths

                pdf_filename (str, optional): The filename to save the PDF to. Defaults to "joined.pdf".

            """

            images = [Image.open(fin) for fin in pngs]

            images[0].save(

                pdf_filename, "PDF", resolution=100.0, save_all=True, append_images=images[1:]

            )



        def molecule_similarity_maps(

            exp_smiles: str,

            ref_smiles: Optional[str] = None,

            ref_id: Optional[str] = None,

            exp_id: Optional[str] = None,

            filename: Optional[str] = None,

            weights: Optional[List[float]] = None,

            normalized: bool = False,

            radius: int = 2,

            fp: str = "morgan",

            fp_type: str = "bv",

            metric: Callable = DataStructs.TanimotoSimilarity,

            **kwargs,

        ):

            """

            Function to plot the similarity between two molecules as a coloured contour. See https://www.rdkit.org/docs/GettingStartedInPython.html#generating-similarity-maps-using-fingerprints for more information.

             Args:

                 ref_smiles (str): Reference molecule smiles string

                 exp_smiles (str): Experimental molecule smiles string

                 ref_id (Optional[str], optional): A name or id for the reference molecule. Defaults to None.

                 exp_id (Optional[str], optional): A name or id for the experimetal molecule. Defaults to None.

                 filename (Optional[str], optional): Filename to save the image to. Defaults to None.

                 weights (Optional[List[float]]): list of atomic contributions to use as weighting. Default to None.

                 normalized (bool): whether to minmax normalize the weights or use the raw values. Defaults to False.

                 radius (int, optional): radius of teh fingerprint if using morgan fingerprint. Defaults to 2.

                 fp (str, optional): The fingerprint to use for a molecule rep. Options are: 'morgan', 'atompair', 'torsion' or 'custom' (use custom if passing in weights). Defaults to "morgan".

                 fp_type (str, optional): The type of fingerprint if using morgan. Use 'bv' for binary and 'count' for the number of occurence. Defaults to "bv".

                 metric (Callable, optional): Function to calculate similarity between the fingerprints. Defaults to DataStructs.TanimotoSimilarity.

             Raises:

                 terr: If the wrong type is input for the fingerprint 'fp' option

            """

            if ref_smiles is not None:

                ref_mol = Chem.MolFromSmiles(ref_smiles)

            exp_mol = Chem.MolFromSmiles(exp_smiles)

            if filename is None:

                if ref_smiles is not None:

                    if all(ent is not None for ent in [ref_id, exp_id]):

                        filename = f"similarity_map_{ref_id}_{exp_id}.png"

                    else:

                        filename = f"similarity_map_{CalcMolFormula(ref_mol)}_{CalcMolFormula(exp_mol)}.png"

                else:

                    if exp_id is not None:

                        filename = f"similarity_map_{exp_id}.png"

                    else:

                        filename = f"similarity_map_{CalcMolFormula(exp_mol)}.png"

            try:

                fp = fp.lower().strip()

            except TypeError as terr:

                raise terr(

                    "fp type should be string one of 'morgan', 'atompair', 'torsion' or 'custom'"

                )

            if fp == "morgan":

                if normalized is False:

                    weights = SimilarityMaps.GetAtomicWeightsForFingerprint(

                        ref_mol,

                        exp_mol,

                        lambda m, inx: SimilarityMaps.GetMorganFingerprint(

                            m, atomId=inx, radius=radius, fpType=fp_type

                        ),

                        metric=metric,

                    )

                    image = SimilarityMaps.GetSimilarityMapFromWeights(

                        exp_mol,

                        weights,

                    )

                    image.savefig(filename, bbox_inches="tight")

                else:

                    image, maxweight = SimilarityMaps.GetSimilarityMapForFingerprint(

                        ref_mol,

                        exp_mol,

                        lambda m, inx: SimilarityMaps.GetMorganFingerprint(

                            m, atomId=inx, radius=radius, fpType=fp_type

                        ),

                        metric=metric,

                    )

                    log.info(f"Max weight before normalization is {maxweight}")

                    image.savefig(filename, bbox_inches="tight")

            elif fp == "atompair":

                if normalized is False:

                    weights = SimilarityMaps.GetAtomicWeightsForFingerprint(

                        ref_mol, exp_mol, SimilarityMaps.GetAPFingerprint, metric=metric

                    )

                    image = SimilarityMaps.GetSimilarityMapFromWeights(

                        exp_mol,

                        weights,

                    )

                    image.savefig(filename, bbox_inches="tight")

                else:

                    image, maxweight = SimilarityMaps.GetSimilarityMapForFingerprint(

                        ref_mol, exp_mol, SimilarityMaps.GetAPFingerprint, metric=metric

                    )

                    log.info(f"Max weight before normalization is {maxweight}")

                    image.savefig(filename, bbox_inches="tight")

            elif fp == "torsion":

                if normalized is False:

                    weights = SimilarityMaps.GetAtomicWeightsForFingerprint(

                        ref_mol, exp_mol, SimilarityMaps.GetTTFingerprint, metric=metric

                    )

                    image = SimilarityMaps.GetSimilarityMapFromWeights(

                        exp_mol,

                        weights,

                    )

                    image.savefig(filename, bbox_inches="tight")

                else:

                    image, maxweight = SimilarityMaps.GetSimilarityMapForFingerprint(

                        ref_mol, exp_mol, SimilarityMaps.GetTTFingerprint, metric=metric

                    )

                    log.info(f"Max weight before normalization is {maxweight}")

                    image.savefig(filename, bbox_inches="tight")

            elif weights is not None:

                if normalized is False:

                    image = SimilarityMaps.GetSimilarityMapFromWeights(exp_mol, weights)

                    image.savefig(filename, bbox_inches="tight")

                else:

                    weights_normed = [

                        (float(elt) - min(weights)) / (max(weights) - min(weights))

                        for elt in weights

                    ]

                    image = SimilarityMaps.GetSimilarityMapFromWeights(

                        exp_mol, weights_normed, kwargs

                    )

                    log.info(

                        f"Min weight before normalization is {min(weights)}. Max weight before normalization is {max(weights)}"

                    )

                    image.savefig(filename, bbox_inches="tight")

            else:

                raise RuntimeError(

                    f"Unknown options given fp should be one of 'morgan', 'atompair' or 'torsion' user has given {fp} or weights must not be None user has given {weights}."

                )



        def get_tanimoto_matrix(

            df1: Optional[pd.DataFrame] = None,

            df2: Optional[pd.DataFrame] = None,

            labels_column_1: Optional[str] = None,

            smiles_column_1: Optional[str] = None,

            smiles_column_2: Optional[str] = None,

            fps1: Optional[list] = None,

            fps2: Optional[list] = None,

            dist: bool = False,

        ) -> pd.DataFrame:

            """

            Function to calculate the symmetric Tanimoto similarity matrix

            Args:

                df1 (Optional[pd.DataFrame], optional): Dataframe of the first set of smiles to compare against. These are used as the reference set and will provide labels to the columns if the label column is defined. Defaults to None.

                df2 (Optional[pd.DataFrame], optional): Dataframe of the first set of smiles to compare to. These for the test cases to make the distrbution i.e. how similar is each molecule in this set to each one at a time from df1. Defaults to None.

                labels_column_1 (Optional[str], optional): The column containing labels from data set 1. Defaults to None.

                smiles_column_1 (Optional[str], optional): The column containing smiles for the molecules. Defaults to None.

                smiles_column_2 (Optional[str], optional): The column containing smiles for the molecules. Defaults to None.

                fps1 (Optional[list], optional): Precomputed fingerprints for df1. Defaults to None.

                fps2 (Optional[list], optional): Precomputed fingerprints for df2. Defaults to None.

                dist (bool, optional): Calculate simialrity if this False (default) or distance is this is True. Defaults to False.

            Returns:

                pd.DataFrame: Dataframe of the Tanimoto simialrity/distance matrix

            """

            if fps1 is None or fps2 is None:

                # Remove invalid smiles

                valid_smiles_1_mask = smilesmod.get_valid_smiles_mask(

                    df1[smiles_column_1].to_list()

                )

                log.info(f"Dropping {len(df1) - sum(valid_smiles_1_mask)} rows from df1")

                df1 = df1[valid_smiles_1_mask].copy()

                log.debug(df1)

                valid_smiles_2_mask = smilesmod.get_valid_smiles_mask(

                    df2[smiles_column_2].to_list()

                )

                log.info(f"Dropping {len(df2) - sum(valid_smiles_2_mask)} rows from df2")

                df2 = df2[valid_smiles_2_mask].copy()

                log.debug(df2)

                # Get ECFP4 fingerprints

                fps1 = featurization.get_ecfp(

                    df1, smiles_column=smiles_column_1, hash_length=1024, radius=2

                )

                fps2 = featurization.get_ecfp(

                    df2, smiles_column=smiles_column_2, hash_length=1024, radius=2

                )

            # Calculate Tanimoto simialrity matrix

            tanimoto_measures = np.array(

                [

                    DataStructs.BulkTanimotoSimilarity(fps1[ith], fps2)

                    for ith in range(len(fps1))

                ]

            ).T

            if dist is True:

                tanimoto_measures = tanimoto_measures - 1.0

            tanimoto_measures = pd.DataFrame(tanimoto_measures)

            tanimoto_measures.columns = df1[labels_column_1].to_list()

            return tanimoto_measures



        def tanimoto_box_plot(

            df1: pd.DataFrame,

            df2: pd.DataFrame,

            labels_column_1: str,

            smiles_column_1: str,

            smiles_column_2: str,

            filename: str = "tanimoto_box_plot.png",

            fps1: Optional[list] = None,

            fps2: Optional[list] = None,

            dist: bool = False,

        ):

            """

            Function to calculate the symmetric Tanimoto similarity matrix

            Args:

                df1 (pd.DataFrame): Dataframe of the first set of smiles to compare against. These are used as the reference set and will provide labels to the columns if the label column is defined. Defaults to None.

                df2 (pd.DataFrame): Dataframe of the first set of smiles to compare to. These for the test cases to make the distrbution i.e. how similar is each molecule in this set to each one at a time from df1. Defaults to None.

                labels_column_1 (str): The column containing labels from data set 1. Defaults to None.

                smiles_column_1 (str): The column containing smiles for the molecules. Defaults to None.

                smiles_column_2 (str): The column containing smiles for the molecules. Defaults to None.

                filename (str): the file to save the plot to

                fps1 (Optional[list], optional): Precomputed fingerprints for df1. Defaults to None.

                fps2 (Optional[list], optional): Precomputed fingerprints for df2. Defaults to None.

                dist (bool, optional): Calculate simialrity if this False (default) or distance is this is True. Defaults to False.

            """

            tanimoto_measures = get_tanimoto_matrix(

                df1,

                df2,

                labels_column_1,

                smiles_column_1,

                smiles_column_2,

                dist=dist,

                fps1=fps1,

                fps2=fps2,

            )

            # plot box plot

            plt.figure(figsize=(2 * len(tanimoto_measures.columns), 10))

            tanimoto_measures.plot.box(ax=plt.gca(), rot=90)

            plt.xlabel("Molecule Labels", fontsize=20)

            plt.xticks(fontsize=17)

            if dist is False:

                plt.ylabel("Tanimoto Smiliarlity", fontsize=20)

                plt.yticks(fontsize=17)

            else:

                plt.ylabel("Tanimoto Distance", fontsize=20)

                plt.yticks(fontsize=17)

            plt.tight_layout()

            plt.savefig(filename, bbox_inches="tight")



        def tanimoto_distrbution_plots(

            df1: pd.DataFrame,

            df2: pd.DataFrame,

            labels_column_1: str,

            smiles_column_1: str,

            smiles_column_2: str,

            filename: str = "tanimoto_distrbution_plot.png",

            fps1=None,

            fps2=None,

            dist=False,

        ):

            """

            Function to calculate the symmetric Tanimoto similarity matrix

            Args:

                df1 (pd.DataFrame): Dataframe of the first set of smiles to compare against. These are used as the reference set and will provide labels to the columns if the label column is defined. Defaults to None.

                df2 (pd.DataFrame): Dataframe of the first set of smiles to compare to. These for the test cases to make the distrbution i.e. how similar is each molecule in this set to each one at a time from df1. Defaults to None.

                labels_column_1 (str): The column containing labels from data set 1. Defaults to None.

                smiles_column_1 (str): The column containing smiles for the molecules. Defaults to None.

                smiles_column_2 (str): The column containing smiles for the molecules. Defaults to None.

                filename (str): the file to save the plot to

                fps1 (Optional[list], optional): Precomputed fingerprints for df1. Defaults to None.

                fps2 (Optional[list], optional): Precomputed fingerprints for df2. Defaults to None.

                dist (bool, optional): Calculate simialrity if this False (default) or distance is this is True. Defaults to False.

            """

            tm_mat = get_tanimoto_matrix(

                df1,

                df2,

                labels_column_1=labels_column_1,

                smiles_column_1=smiles_column_1,

                smiles_column_2=smiles_column_2,

                fps1=fps1,

                fps2=fps2,

                dist=dist,

            )

            medians = pd.DataFrame(tm_mat.median(), columns=["median"])

            means = pd.DataFrame(tm_mat.mean(), columns=["mean"])

            for indx_med, indx_mean in zip(medians.index.to_list(), means.index.to_list()):

                if indx_med != indx_mean:

                    raise RuntimeError(

                        "Indexes are different between the mean and median series!"

                    )

            averages = pd.DataFrame([medians["median"], means["mean"]]).transpose()

            plt.figure(figsize=(10, 10))

            averages.plot.kde(alpha=0.75, ax=plt.gca(), colormap="bwr")

            if dist is False:

                plt.xlabel("Average Tanimoto Similarity Distrbution", fontsize=20)

            elif dist is True:

                plt.xlabel("Average Tanimoto Distance Distrbution", fontsize=20)

            plt.xticks(fontsize=17)

            plt.ylabel("Frequency", fontsize=20)

            plt.yticks(fontsize=17)

            plt.legend(fontsize=15)

            plt.savefig(filename)



        def get_tanimoto_intra_dataset_matrix(

            df1: Optional[pd.DataFrame] = None,

            labels_column_1: Optional[str] = None,

            smiles_column_1: Optional[str] = None,

            fps1: Optional[list] = None,

            dist: bool = False,

        ) -> pd.DataFrame:

            """

            Function to calculate the symmetric Tanimoto similarity matrix of all molecules in a data set to all other molecules in that dataset

            Args:

                df1 (Optional[pd.DataFrame], optional): Dataframe of the first set of smiles to compare against. These are used as the reference set and will provide labels to the columns if the label column is defined. Defaults to None.

                labels_column_1 (Optional[str], optional): The column containing labels from data set 1. Defaults to None.

                smiles_column_1 (Optional[str], optional): The column containing smiles for the molecules. Defaults to None.

                fps1 (Optional[list], optional): Precomputed fingerprints for df1. Defaults to None.

                dist (bool, optional): Calculate simialrity if this False (default) or distance is this is True. Defaults to False.

            Returns:

                pd.DataFrame: Dataframe of the Tanimoto simialrity/distance matrix

            """

            if fps1 is None:

                # Remove invalid smiles

                valid_smiles_1_mask = smilesmod.get_valid_smiles_mask(

                    df1[smiles_column_1].to_list()

                )

                log.info(f"Dropping {len(df1) - sum(valid_smiles_1_mask)} rows from df1")

                df1 = df1[valid_smiles_1_mask].copy()

                log.debug(df1)

                # Get ECFP4 fingerprints

                fps1 = featurization.get_ecfp(

                    df1, smiles_column=smiles_column_1, hash_length=1024, radius=2

                )

            # Calculate Tanimoto simialrity matrix

            tanimoto_measures = np.array(

                [

                    DataStructs.BulkTanimotoSimilarity(fps1[ith], fps1[:ith] + fps1[ith + 1 :])

                    for ith in range(len(fps1))

                ]

            ).T

            if dist is True:

                tanimoto_measures = tanimoto_measures - 1.0

            tanimoto_measures = pd.DataFrame(tanimoto_measures)

            tanimoto_measures.columns = df1[labels_column_1].to_list()

            return tanimoto_measures



        def tanimoto_intra_dataset_box_plot(

            df1: pd.DataFrame,

            labels_column_1: str,

            smiles_column_1: str,

            filename: str = "tanimoto_box_plot.png",

            fps1: Optional[list] = None,

            dist: bool = False,

        ):

            """

            Function to calculate the symmetric Tanimoto similarity matrix

            Args:

                df1 (pd.DataFrame): Dataframe of the first set of smiles to compare against. These are used as the reference set and will provide labels to the columns if the label column is defined. Defaults to None.

                labels_column_1 (str): The column containing labels from data set 1. Defaults to None.

                smiles_column_1 (str): The column containing smiles for the molecules. Defaults to None.

                filename (str): the file to save the plot to

                fps1 (Optional[list], optional): Precomputed fingerprints for df1. Defaults to None.

                dist (bool, optional): Calculate simialrity if this False (default) or distance is this is True. Defaults to False.

            """

            tanimoto_measures = get_tanimoto_intra_dataset_matrix(

                df1,

                labels_column_1,

                smiles_column_1,

                dist=dist,

                fps1=fps1,

            )

            log.info(tanimoto_measures)

            # plot box plot

            plt.figure(figsize=(2 * len(tanimoto_measures.columns), 10))

            tanimoto_measures.plot.box(ax=plt.gca(), rot=90)

            plt.xlabel("Molecule Labels", fontsize=20)

            plt.xticks(fontsize=17)

            if dist is False:

                plt.ylabel("Tanimoto Smiliarlity", fontsize=20)

                plt.yticks(fontsize=17)

            else:

                plt.ylabel("Tanimoto Distance", fontsize=20)

                plt.yticks(fontsize=17)

            plt.tight_layout()

            plt.savefig(filename, bbox_inches="tight")



        def tanimoto_distrbution_plots_intra_dataset(

            df1: pd.DataFrame,

            labels_column_1: str,

            smiles_column_1: str,

            filename: str = "tanimoto_distrbution_plot.png",

            fps1=None,

            dist=False,

        ):

            """

            Function to calculate the symmetric Tanimoto similarity matrix

            Args:

                df1 (pd.DataFrame): Dataframe of the first set of smiles to compare against. These are used as the reference set and will provide labels to the columns if the label column is defined. Defaults to None.

                labels_column_1 (str): The column containing labels from data set 1. Defaults to None.

                smiles_column_1 (str): The column containing smiles for the molecules. Defaults to None.

                filename (str): the file to save the plot to

                fps1 (Optional[list], optional): Precomputed fingerprints for df1. Defaults to None.

                dist (bool, optional): Calculate simialrity if this False (default) or distance is this is True. Defaults to False.

            """

            tm_mat = get_tanimoto_intra_dataset_matrix(

                df1,

                labels_column_1=labels_column_1,

                smiles_column_1=smiles_column_1,

                fps1=fps1,

                dist=dist,

            )

            medians = pd.DataFrame(tm_mat.median(), columns=["median"])

            means = pd.DataFrame(tm_mat.mean(), columns=["mean"])

            for indx_med, indx_mean in zip(medians.index.to_list(), means.index.to_list()):

                if indx_med != indx_mean:

                    raise RuntimeError(

                        "Indexes are different between the mean and median series!"

                    )

            averages = pd.DataFrame([medians["median"], means["mean"]]).transpose()

            plt.figure(figsize=(10, 10))

            averages.plot.kde(alpha=0.75, ax=plt.gca(), colormap="bwr")

            if dist is False:

                plt.xlabel("Average Tanimoto Similarity Distrbution", fontsize=20)

            elif dist is True:

                plt.xlabel("Average Tanimoto Distance Distrbution", fontsize=20)

            plt.xticks(fontsize=17)

            plt.ylabel("Frequency", fontsize=20)

            plt.yticks(fontsize=17)

            plt.legend(fontsize=15)

            plt.savefig(filename)



        def seaborn_pair_plot(

            df: pd.DataFrame,

            subset_colummns: Optional[List[str]] = None,

            width: int = 15,

            filename: str = "pair_plot.png",

            row_column_scaler: float = 2.0,

            kind: str = "reg",

            diag_kind: str = "kde",

            title: Optional[str] = None,

            fontsize: int = 20,

            **kwargs,

        ) -> Axes:

            """

            Function to plot a seaborn pair plot of a dataframe

            Args:

                df (pd.DataFrame): The dataframe to plot

                subset_colummns (Optional[List[str]], optional): The columns to subset the dataframe to. Defaults to None.

                width (int, optional): The width of the column names. Defaults to 15.

                filename (str, optional): Filename to save the plot to. Defaults to "pair_plot.png".

                row_column_scaler (float, optional): The scaler to multiply the row and column size by. Defaults to 2.0.

                kind (str, optional): The kind of plot to use. Defaults to "reg".

                diag_kind (str, optional): The kind of plot to use on the diagonal. Defaults to "kde".

                title (Optional[str], optional): The title of the plot. Defaults to None.

                fontsize (int, optional): The fontsize of the title. Defaults to 20.

            Returns:

                Axes: The axis of the plot

            """

            if subset_colummns is not None:

                df = df[subset_colummns].copy()

            df.columns = [textwrap.fill(col, width=width) for col in df.columns]

            plt.figure(

                figsize=(

                    row_column_scaler * len(df.columns),

                    row_column_scaler * len(df.columns),

                )

            )

            sns.pairplot(df, kind=kind, diag_kind=diag_kind, **kwargs)

            if title is not None:

                plt.title(title, fontsize=fontsize)

            plt.tight_layout()

            plt.savefig(filename)

            return plt.gca()



        def seaborn_correlation_heat_map(

            df: pd.DataFrame,

            subset_colummns: Optional[List[str]] = None,

            width: int = 15,

            filename: str = "heat_map.png",

            row_column_scaler: float = 2.0,

            square: bool = True,

            vmin: float = -1.0,

            vmax: float = 1.0,

            method: Union[str, Callable] = "pearson",

            title: Optional[str] = None,

            xlabel: Optional[str] = None,

            ylabel: Optional[str] = None,

            fontsize: int = 20,

            **kwargs,

        ) -> Axes:

            """

            Function to plot a seaborn correlation heat map of a dataframe

            Args:

                df (pd.DataFrame): The dataframe to plot

                subset_colummns (Optional[List[str]], optional): The columns to subset the dataframe to. Defaults to None.

                width (int, optional): The width of the column names. Defaults to 15.

                filename (str, optional): Filename to save the plot to. Defaults to "heat_map.png".

                row_column_scaler (float, optional): The scaler to multiply the row and column size by. Defaults to 2.0.

                square (bool, optional): Whether to make the plot square. Defaults to True.

                vmin (float, optional): The minimum value of the colour map. Defaults to -1.0.

                vmax (float, optional): The maximum value of the colour map. Defaults to 1.0.

                method (Union[str, Callable], optional): The method to calculate the correlation. Defaults to "pearson".

                title (Optional[str], optional): The title of the plot. Defaults to None.

                xlabel (Optional[str], optional): The x axis label. Defaults to None.

                ylabel (Optional[str], optional): The y axis label. Defaults to None.

                fontsize (int, optional): The fontsize of the title. Defaults to 20.

            Returns:

                Axes: The axis of the plot

            """

            if subset_colummns is not None:

                df = df[subset_colummns].copy()

            df.columns = [textwrap.fill(col, width=width) for col in df.columns]

            plt.figure(

                figsize=(

                    row_column_scaler * len(df.columns),

                    row_column_scaler * len(df.columns),

                )

            )

            sns.heatmap(

                df.corr(method=method),

                annot=True,

                fmt=".2f",

                cmap="coolwarm",

                cbar=True,

                square=square,

                vmin=vmin,

                vmax=vmax,

            )

            if title is not None:

                plt.title(title, fontsize=fontsize)

            if xlabel is not None:

                plt.xlabel(xlabel, fontsize=fontsize)

            if ylabel is not None:

                plt.ylabel(ylabel, fontsize=fontsize)

            plt.tight_layout()

            plt.savefig(filename)

            return plt.gca()

## Variables

```python3
log
```

## Functions


### draw_aligned_mol_grid

```python3
def draw_aligned_mol_grid(
    mols: Optional[List[rdkit.Chem.rdchem.Mol]] = None,
    smiles: Optional[List[str]] = None,
    inchi: Optional[List[str]] = None,
    mcs_smarts: Optional[str] = None,
    labels: Optional[List[str]] = None,
    mols_per_row: int = 5,
    sub_image_size: Tuple[int, int] = (400, 400),
    filename: Optional[str] = None,
    max_mols: Optional[int] = None,
    return_image: bool = True,
    legend_fontsize: int = 20,
    legend_fraction: float = 0.1,
    highlight_mcs: bool = False
) -> PIL.PngImagePlugin.PngImageFile
```

Draw a grid of molecule images and write to file if requested. Return the raw image object.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mols | Optional[List[rdkit.rdMol.Mol]] | List of RDKit molecule objects. Defaults to None. | None |
| smiles | Optional[List[str]] | List of smiles strings. Defaults to None. | None |
| inchi | Optional[List[str]] | List of InChi strings. Defaults to None. | None |
| mcs_smarts | Optional[List[str]] | Pre computed maximum common substructure smarts string. Defaults to None. | None |
| labels | Optional[List[str]] | List of string labels for the molecules. Defaults to None. | None |
| mols_per_row | int | number of molecules to show in each row. Defaults to 5. | 5 |
| sub_image_size | Tuple[int, int] | the size in pixels of each molecule image. Defaults to (400, 400). | (400, 400) |
| filename | Optional[str] | What to save the image as if None it is not saved. Defaults to None. | None |
| max_mols | Optional[int] | The max mols to show in a grid plot None plot all. Defaults to None. | None |

**Returns:**

| Type | Description |
|---|---|
| PIL.PngImagePlugin.PngImageFile | Raw image object (.save(file) to save the image) |

**Raises:**

| Type | Description |
|---|---|
| UserWarning | If None of the molecules formats are provided by the use |

??? example "View Source"
        def draw_aligned_mol_grid(

            mols: Optional[List[Chem.rdchem.Mol]] = None,

            smiles: Optional[List[str]] = None,

            inchi: Optional[List[str]] = None,

            mcs_smarts: Optional[str] = None,

            labels: Optional[List[str]] = None,

            mols_per_row: int = 5,

            sub_image_size: Tuple[int, int] = (400, 400),

            filename: Optional[str] = None,

            max_mols: Optional[int] = None,

            return_image: bool = True,

            legend_fontsize: int = 20,

            legend_fraction: float = 0.1,

            highlight_mcs: bool = False,

        ) -> PngImageFile:

            """

            Draw a grid of molecule images and write to file if requested. Return the raw image object.

            Args:

                mols (Optional[List[rdkit.rdMol.Mol]], optional): List of RDKit molecule objects. Defaults to None.

                smiles (Optional[List[str]], optional): List of smiles strings. Defaults to None.

                inchi (Optional[List[str]], optional): List of InChi strings. Defaults to None.

                mcs_smarts (Optional[List[str]], optional): Pre computed maximum common substructure smarts string. Defaults to None.

                labels (Optional[List[str]], optional): List of string labels for the molecules. Defaults to None.

                mols_per_row (int, optional): number of molecules to show in each row. Defaults to 5.

                sub_image_size (Tuple[int, int], optional): the size in pixels of each molecule image. Defaults to (400, 400).

                filename (Optional[str], optional): What to save the image as if None it is not saved. Defaults to None.

                max_mols (Optional[int], optional): The max mols to show in a grid plot None plot all. Defaults to None.

            Raises:

                UserWarning: If None of the molecules formats are provided by the use

            Returns:

                PIL.PngImagePlugin.PngImageFile: Raw image object (.save(file) to save the image)

            """

            # check the user has input one of the required inputs

            if all(ent is None for ent in [mols, smiles, inchi]):

                raise UserWarning("One of mols, smiles or inchi must be given for input")

            # Make sure there is a list of molecules

            if mols is None:

                # smiles is the lowest priority

                if inchi is not None:

                    mols = [inchimod.inchi_to_molecule(inchi=inch) for inch in inchi]

                elif smiles is not None:

                    mols = [smilesmod.smiles_to_molecule(smi=smil) for smil in smiles]

            if max_mols is None:

                max_mols = len(mols)

            opts = Draw.MolDrawOptions()

            opts.legendFraction = legend_fraction

            opts.legendFontSize = legend_fontsize

            # Save one molecule image using the grid image function

            if len(mols) == 1 and mcs_smarts is None:

                log.warning("Plotting only one molecule")

                img = Draw.MolsToGridImage(

                    mols,

                    molsPerRow=1,

                    subImgSize=sub_image_size,

                    legends=labels,

                    useSVG=False,

                    returnPNG=False,

                    maxMols=max_mols,

                    drawOptions=opts,

                )

            else:

                # Get maximum common substructure across list of mols to align to

                if mcs_smarts is None:

                    mcs_res = rdFMCS.FindMCS(

                        mols,

                        completeRingsOnly=True,

                        matchChiralTag=True,

                        ringMatchesRingOnly=True,

                        timeout=30,

                    )

                    mcs_smarts = mcs_res.smartsString

                    log.debug(f"MCS found: {mcs_smarts}")

                mcs_mol = Chem.MolFromSmarts(mcs_smarts)

                AllChem.Compute2DCoords(mcs_mol)

                # Align all molecules using the maximum common substructure

                for m in mols:

                    _ = AllChem.GenerateDepictionMatching2DStructure(m, mcs_mol)

                if highlight_mcs is True:

                    highlight_atom_list = [mol.GetSubstructMatch(mcs_mol) for mol in mols]

                else:

                    highlight_atom_list = None

                if len(mols) < mols_per_row:

                    mols_per_row = len(mols)

                # Draw the grid image

                img = Draw.MolsToGridImage(

                    mols,

                    molsPerRow=mols_per_row,

                    subImgSize=sub_image_size,

                    legends=labels,

                    useSVG=False,

                    returnPNG=False,

                    maxMols=max_mols,

                    highlightAtomLists=highlight_atom_list,

                    drawOptions=opts,

                )

            # If there is a filename save the image to file

            if filename is not None:

                img.save(filename)

            if return_image is True:

                return img

            else:

                img.close()


### draw_molecule_to_file

```python3
def draw_molecule_to_file(
    mol: Optional[rdkit.Chem.rdchem.Mol] = None,
    smiles: Optional[str] = None,
    inchi: Optional[str] = None,
    mcs_smarts: Optional[str] = None,
    labels: Optional[List[str]] = None,
    image_size: Tuple[int, int] = (400, 400),
    filename: Optional[str] = None
) -> PIL.PngImagePlugin.PngImageFile
```

Draw a molecule image and write to file if requested. Return the raw image object.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mols | Optional[List[rdkit.rdMol.Mol]] | List of RDKit molecule objects. Defaults to None. | None |
| smiles | Optional[List[str]] | List of smiles strings. Defaults to None. | None |
| inchi | Optional[List[str]] | List of InChi strings. Defaults to None. | None |
| mcs_smarts | Optional[List[str]] | Pre computed maximum common substructure smarts string. Defaults to None. | None |
| labels | Optional[List[str]] | List of string labels for the molecules. Defaults to None. | None |
| sub_image_size | Tuple[int, int] | the size in pixels of each molecule image. Defaults to (400, 400). | (400, 400) |
| filename | Optional[str] | What to save the image as if None it is not saved. Defaults to None. | None |

**Returns:**

| Type | Description |
|---|---|
| PIL.PngImagePlugin.PngImageFile | Raw image object (.save(file) to save the image) |

**Raises:**

| Type | Description |
|---|---|
| UserWarning | If None of the molecules formats are provided by the use |

??? example "View Source"
        def draw_molecule_to_file(

            mol: Optional[Chem.rdchem.Mol] = None,

            smiles: Optional[str] = None,

            inchi: Optional[str] = None,

            mcs_smarts: Optional[str] = None,

            labels: Optional[List[str]] = None,

            image_size: Tuple[int, int] = (400, 400),

            filename: Optional[str] = None,

        ) -> PngImageFile:

            """

            Draw a molecule image and write to file if requested. Return the raw image object.

            Args:

                mols (Optional[List[rdkit.rdMol.Mol]], optional): List of RDKit molecule objects. Defaults to None.

                smiles (Optional[List[str]], optional): List of smiles strings. Defaults to None.

                inchi (Optional[List[str]], optional): List of InChi strings. Defaults to None.

                mcs_smarts (Optional[List[str]], optional): Pre computed maximum common substructure smarts string. Defaults to None.

                labels (Optional[List[str]], optional): List of string labels for the molecules. Defaults to None.

                sub_image_size (Tuple[int, int], optional): the size in pixels of each molecule image. Defaults to (400, 400).

                filename (Optional[str], optional): What to save the image as if None it is not saved. Defaults to None.

            Raises:

                UserWarning: If None of the molecules formats are provided by the use

            Returns:

                PIL.PngImagePlugin.PngImageFile: Raw image object (.save(file) to save the image)

            """

            # check the user has input one of the required inputs

            if all(ent is None for ent in [mol, smiles, inchi]):

                raise UserWarning("One of mols, smiles or inchi must be given for input")

            # Make sure there is a list of molecules

            if mol is None:

                # smiles is the lowest priority

                if inchi is not None:

                    mol = inchimod.inchi_to_molecule(inchi=inchi)

                elif smiles is not None:

                    mol = smilesmod.smiles_to_molecule(smi=smiles)

            # one moleucle image code aligned to the mcs smarts from a user or not when not avaliable

            if mcs_smarts is None:

                img = Draw.MolToImage(

                    mol, size=image_size, kekulize=True, wedgeBonds=True, legends=labels

                )

            else:

                mcs_mol = Chem.MolFromSmarts(mcs_smarts)

                AllChem.Compute2DCoords(mcs_mol)

                _ = AllChem.GenerateDepictionMatching2DStructure(mol, mcs_mol)

                img = Draw.MolToImage(

                    mol, size=image_size, kekulize=True, wedgeBonds=True, legends=labels

                )

            if filename is not None:

                img.save(filename)

            return img


### get_tanimoto_intra_dataset_matrix

```python3
def get_tanimoto_intra_dataset_matrix(
    df1: Optional[pandas.core.frame.DataFrame] = None,
    labels_column_1: Optional[str] = None,
    smiles_column_1: Optional[str] = None,
    fps1: Optional[list] = None,
    dist: bool = False
) -> pandas.core.frame.DataFrame
```

Function to calculate the symmetric Tanimoto similarity matrix of all molecules in a data set to all other molecules in that dataset

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df1 | Optional[pd.DataFrame] | Dataframe of the first set of smiles to compare against. These are used as the reference set and will provide labels to the columns if the label column is defined. Defaults to None. | None |
| labels_column_1 | Optional[str] | The column containing labels from data set 1. Defaults to None. | None |
| smiles_column_1 | Optional[str] | The column containing smiles for the molecules. Defaults to None. | None |
| fps1 | Optional[list] | Precomputed fingerprints for df1. Defaults to None. | None |
| dist | bool | Calculate simialrity if this False (default) or distance is this is True. Defaults to False. | False |

**Returns:**

| Type | Description |
|---|---|
| pd.DataFrame | Dataframe of the Tanimoto simialrity/distance matrix |

??? example "View Source"
        def get_tanimoto_intra_dataset_matrix(

            df1: Optional[pd.DataFrame] = None,

            labels_column_1: Optional[str] = None,

            smiles_column_1: Optional[str] = None,

            fps1: Optional[list] = None,

            dist: bool = False,

        ) -> pd.DataFrame:

            """

            Function to calculate the symmetric Tanimoto similarity matrix of all molecules in a data set to all other molecules in that dataset

            Args:

                df1 (Optional[pd.DataFrame], optional): Dataframe of the first set of smiles to compare against. These are used as the reference set and will provide labels to the columns if the label column is defined. Defaults to None.

                labels_column_1 (Optional[str], optional): The column containing labels from data set 1. Defaults to None.

                smiles_column_1 (Optional[str], optional): The column containing smiles for the molecules. Defaults to None.

                fps1 (Optional[list], optional): Precomputed fingerprints for df1. Defaults to None.

                dist (bool, optional): Calculate simialrity if this False (default) or distance is this is True. Defaults to False.

            Returns:

                pd.DataFrame: Dataframe of the Tanimoto simialrity/distance matrix

            """

            if fps1 is None:

                # Remove invalid smiles

                valid_smiles_1_mask = smilesmod.get_valid_smiles_mask(

                    df1[smiles_column_1].to_list()

                )

                log.info(f"Dropping {len(df1) - sum(valid_smiles_1_mask)} rows from df1")

                df1 = df1[valid_smiles_1_mask].copy()

                log.debug(df1)

                # Get ECFP4 fingerprints

                fps1 = featurization.get_ecfp(

                    df1, smiles_column=smiles_column_1, hash_length=1024, radius=2

                )

            # Calculate Tanimoto simialrity matrix

            tanimoto_measures = np.array(

                [

                    DataStructs.BulkTanimotoSimilarity(fps1[ith], fps1[:ith] + fps1[ith + 1 :])

                    for ith in range(len(fps1))

                ]

            ).T

            if dist is True:

                tanimoto_measures = tanimoto_measures - 1.0

            tanimoto_measures = pd.DataFrame(tanimoto_measures)

            tanimoto_measures.columns = df1[labels_column_1].to_list()

            return tanimoto_measures


### get_tanimoto_matrix

```python3
def get_tanimoto_matrix(
    df1: Optional[pandas.core.frame.DataFrame] = None,
    df2: Optional[pandas.core.frame.DataFrame] = None,
    labels_column_1: Optional[str] = None,
    smiles_column_1: Optional[str] = None,
    smiles_column_2: Optional[str] = None,
    fps1: Optional[list] = None,
    fps2: Optional[list] = None,
    dist: bool = False
) -> pandas.core.frame.DataFrame
```

Function to calculate the symmetric Tanimoto similarity matrix

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df1 | Optional[pd.DataFrame] | Dataframe of the first set of smiles to compare against. These are used as the reference set and will provide labels to the columns if the label column is defined. Defaults to None. | None |
| df2 | Optional[pd.DataFrame] | Dataframe of the first set of smiles to compare to. These for the test cases to make the distrbution i.e. how similar is each molecule in this set to each one at a time from df1. Defaults to None. | None |
| labels_column_1 | Optional[str] | The column containing labels from data set 1. Defaults to None. | None |
| smiles_column_1 | Optional[str] | The column containing smiles for the molecules. Defaults to None. | None |
| smiles_column_2 | Optional[str] | The column containing smiles for the molecules. Defaults to None. | None |
| fps1 | Optional[list] | Precomputed fingerprints for df1. Defaults to None. | None |
| fps2 | Optional[list] | Precomputed fingerprints for df2. Defaults to None. | None |
| dist | bool | Calculate simialrity if this False (default) or distance is this is True. Defaults to False. | False |

**Returns:**

| Type | Description |
|---|---|
| pd.DataFrame | Dataframe of the Tanimoto simialrity/distance matrix |

??? example "View Source"
        def get_tanimoto_matrix(

            df1: Optional[pd.DataFrame] = None,

            df2: Optional[pd.DataFrame] = None,

            labels_column_1: Optional[str] = None,

            smiles_column_1: Optional[str] = None,

            smiles_column_2: Optional[str] = None,

            fps1: Optional[list] = None,

            fps2: Optional[list] = None,

            dist: bool = False,

        ) -> pd.DataFrame:

            """

            Function to calculate the symmetric Tanimoto similarity matrix

            Args:

                df1 (Optional[pd.DataFrame], optional): Dataframe of the first set of smiles to compare against. These are used as the reference set and will provide labels to the columns if the label column is defined. Defaults to None.

                df2 (Optional[pd.DataFrame], optional): Dataframe of the first set of smiles to compare to. These for the test cases to make the distrbution i.e. how similar is each molecule in this set to each one at a time from df1. Defaults to None.

                labels_column_1 (Optional[str], optional): The column containing labels from data set 1. Defaults to None.

                smiles_column_1 (Optional[str], optional): The column containing smiles for the molecules. Defaults to None.

                smiles_column_2 (Optional[str], optional): The column containing smiles for the molecules. Defaults to None.

                fps1 (Optional[list], optional): Precomputed fingerprints for df1. Defaults to None.

                fps2 (Optional[list], optional): Precomputed fingerprints for df2. Defaults to None.

                dist (bool, optional): Calculate simialrity if this False (default) or distance is this is True. Defaults to False.

            Returns:

                pd.DataFrame: Dataframe of the Tanimoto simialrity/distance matrix

            """

            if fps1 is None or fps2 is None:

                # Remove invalid smiles

                valid_smiles_1_mask = smilesmod.get_valid_smiles_mask(

                    df1[smiles_column_1].to_list()

                )

                log.info(f"Dropping {len(df1) - sum(valid_smiles_1_mask)} rows from df1")

                df1 = df1[valid_smiles_1_mask].copy()

                log.debug(df1)

                valid_smiles_2_mask = smilesmod.get_valid_smiles_mask(

                    df2[smiles_column_2].to_list()

                )

                log.info(f"Dropping {len(df2) - sum(valid_smiles_2_mask)} rows from df2")

                df2 = df2[valid_smiles_2_mask].copy()

                log.debug(df2)

                # Get ECFP4 fingerprints

                fps1 = featurization.get_ecfp(

                    df1, smiles_column=smiles_column_1, hash_length=1024, radius=2

                )

                fps2 = featurization.get_ecfp(

                    df2, smiles_column=smiles_column_2, hash_length=1024, radius=2

                )

            # Calculate Tanimoto simialrity matrix

            tanimoto_measures = np.array(

                [

                    DataStructs.BulkTanimotoSimilarity(fps1[ith], fps2)

                    for ith in range(len(fps1))

                ]

            ).T

            if dist is True:

                tanimoto_measures = tanimoto_measures - 1.0

            tanimoto_measures = pd.DataFrame(tanimoto_measures)

            tanimoto_measures.columns = df1[labels_column_1].to_list()

            return tanimoto_measures


### list_of_png_files_to_single_pdf

```python3
def list_of_png_files_to_single_pdf(
    pngs: List[str],
    pdf_filename: str = 'joined.pdf'
)
```

Function to make one PDF file from a list of image files and paths

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| pngs | list[str] | list of image files and paths | None |
| pdf_filename | str | The filename to save the PDF to. Defaults to "joined.pdf". | "joined.pdf" |

??? example "View Source"
        def list_of_png_files_to_single_pdf(pngs: List[str], pdf_filename: str = "joined.pdf"):

            """

            Function to make one PDF file from a list of image files and paths

            Args:

                pngs (list[str]): list of image files and paths

                pdf_filename (str, optional): The filename to save the PDF to. Defaults to "joined.pdf".

            """

            images = [Image.open(fin) for fin in pngs]

            images[0].save(

                pdf_filename, "PDF", resolution=100.0, save_all=True, append_images=images[1:]

            )


### molecule_similarity_maps

```python3
def molecule_similarity_maps(
    exp_smiles: str,
    ref_smiles: Optional[str] = None,
    ref_id: Optional[str] = None,
    exp_id: Optional[str] = None,
    filename: Optional[str] = None,
    weights: Optional[List[float]] = None,
    normalized: bool = False,
    radius: int = 2,
    fp: str = 'morgan',
    fp_type: str = 'bv',
    metric: Callable = <Boost.Python.function object at 0x60000399ae40>,
    **kwargs
)
```

Function to plot the similarity between two molecules as a coloured contour. See https://www.rdkit.org/docs/GettingStartedInPython.html#generating-similarity-maps-using-fingerprints for more information.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| ref_smiles | str | Reference molecule smiles string | None |
| exp_smiles | str | Experimental molecule smiles string | None |
| ref_id | Optional[str] | A name or id for the reference molecule. Defaults to None. | None |
| exp_id | Optional[str] | A name or id for the experimetal molecule. Defaults to None. | None |
| filename | Optional[str] | Filename to save the image to. Defaults to None. | None |
| weights | Optional[List[float]] | list of atomic contributions to use as weighting. Default to None. | None |
| normalized | bool | whether to minmax normalize the weights or use the raw values. Defaults to False. | False |
| radius | int | radius of teh fingerprint if using morgan fingerprint. Defaults to 2. | 2 |
| fp | str | The fingerprint to use for a molecule rep. Options are: 'morgan', 'atompair', 'torsion' or 'custom' (use custom if passing in weights). Defaults to "morgan". | "morgan" |
| fp_type | str | The type of fingerprint if using morgan. Use 'bv' for binary and 'count' for the number of occurence. Defaults to "bv". | "bv" |
| metric | Callable | Function to calculate similarity between the fingerprints. Defaults to DataStructs.TanimotoSimilarity. | DataStructs.TanimotoSimilarity |

**Raises:**

| Type | Description |
|---|---|
| terr | If the wrong type is input for the fingerprint 'fp' option |

??? example "View Source"
        def molecule_similarity_maps(

            exp_smiles: str,

            ref_smiles: Optional[str] = None,

            ref_id: Optional[str] = None,

            exp_id: Optional[str] = None,

            filename: Optional[str] = None,

            weights: Optional[List[float]] = None,

            normalized: bool = False,

            radius: int = 2,

            fp: str = "morgan",

            fp_type: str = "bv",

            metric: Callable = DataStructs.TanimotoSimilarity,

            **kwargs,

        ):

            """

            Function to plot the similarity between two molecules as a coloured contour. See https://www.rdkit.org/docs/GettingStartedInPython.html#generating-similarity-maps-using-fingerprints for more information.

             Args:

                 ref_smiles (str): Reference molecule smiles string

                 exp_smiles (str): Experimental molecule smiles string

                 ref_id (Optional[str], optional): A name or id for the reference molecule. Defaults to None.

                 exp_id (Optional[str], optional): A name or id for the experimetal molecule. Defaults to None.

                 filename (Optional[str], optional): Filename to save the image to. Defaults to None.

                 weights (Optional[List[float]]): list of atomic contributions to use as weighting. Default to None.

                 normalized (bool): whether to minmax normalize the weights or use the raw values. Defaults to False.

                 radius (int, optional): radius of teh fingerprint if using morgan fingerprint. Defaults to 2.

                 fp (str, optional): The fingerprint to use for a molecule rep. Options are: 'morgan', 'atompair', 'torsion' or 'custom' (use custom if passing in weights). Defaults to "morgan".

                 fp_type (str, optional): The type of fingerprint if using morgan. Use 'bv' for binary and 'count' for the number of occurence. Defaults to "bv".

                 metric (Callable, optional): Function to calculate similarity between the fingerprints. Defaults to DataStructs.TanimotoSimilarity.

             Raises:

                 terr: If the wrong type is input for the fingerprint 'fp' option

            """

            if ref_smiles is not None:

                ref_mol = Chem.MolFromSmiles(ref_smiles)

            exp_mol = Chem.MolFromSmiles(exp_smiles)

            if filename is None:

                if ref_smiles is not None:

                    if all(ent is not None for ent in [ref_id, exp_id]):

                        filename = f"similarity_map_{ref_id}_{exp_id}.png"

                    else:

                        filename = f"similarity_map_{CalcMolFormula(ref_mol)}_{CalcMolFormula(exp_mol)}.png"

                else:

                    if exp_id is not None:

                        filename = f"similarity_map_{exp_id}.png"

                    else:

                        filename = f"similarity_map_{CalcMolFormula(exp_mol)}.png"

            try:

                fp = fp.lower().strip()

            except TypeError as terr:

                raise terr(

                    "fp type should be string one of 'morgan', 'atompair', 'torsion' or 'custom'"

                )

            if fp == "morgan":

                if normalized is False:

                    weights = SimilarityMaps.GetAtomicWeightsForFingerprint(

                        ref_mol,

                        exp_mol,

                        lambda m, inx: SimilarityMaps.GetMorganFingerprint(

                            m, atomId=inx, radius=radius, fpType=fp_type

                        ),

                        metric=metric,

                    )

                    image = SimilarityMaps.GetSimilarityMapFromWeights(

                        exp_mol,

                        weights,

                    )

                    image.savefig(filename, bbox_inches="tight")

                else:

                    image, maxweight = SimilarityMaps.GetSimilarityMapForFingerprint(

                        ref_mol,

                        exp_mol,

                        lambda m, inx: SimilarityMaps.GetMorganFingerprint(

                            m, atomId=inx, radius=radius, fpType=fp_type

                        ),

                        metric=metric,

                    )

                    log.info(f"Max weight before normalization is {maxweight}")

                    image.savefig(filename, bbox_inches="tight")

            elif fp == "atompair":

                if normalized is False:

                    weights = SimilarityMaps.GetAtomicWeightsForFingerprint(

                        ref_mol, exp_mol, SimilarityMaps.GetAPFingerprint, metric=metric

                    )

                    image = SimilarityMaps.GetSimilarityMapFromWeights(

                        exp_mol,

                        weights,

                    )

                    image.savefig(filename, bbox_inches="tight")

                else:

                    image, maxweight = SimilarityMaps.GetSimilarityMapForFingerprint(

                        ref_mol, exp_mol, SimilarityMaps.GetAPFingerprint, metric=metric

                    )

                    log.info(f"Max weight before normalization is {maxweight}")

                    image.savefig(filename, bbox_inches="tight")

            elif fp == "torsion":

                if normalized is False:

                    weights = SimilarityMaps.GetAtomicWeightsForFingerprint(

                        ref_mol, exp_mol, SimilarityMaps.GetTTFingerprint, metric=metric

                    )

                    image = SimilarityMaps.GetSimilarityMapFromWeights(

                        exp_mol,

                        weights,

                    )

                    image.savefig(filename, bbox_inches="tight")

                else:

                    image, maxweight = SimilarityMaps.GetSimilarityMapForFingerprint(

                        ref_mol, exp_mol, SimilarityMaps.GetTTFingerprint, metric=metric

                    )

                    log.info(f"Max weight before normalization is {maxweight}")

                    image.savefig(filename, bbox_inches="tight")

            elif weights is not None:

                if normalized is False:

                    image = SimilarityMaps.GetSimilarityMapFromWeights(exp_mol, weights)

                    image.savefig(filename, bbox_inches="tight")

                else:

                    weights_normed = [

                        (float(elt) - min(weights)) / (max(weights) - min(weights))

                        for elt in weights

                    ]

                    image = SimilarityMaps.GetSimilarityMapFromWeights(

                        exp_mol, weights_normed, kwargs

                    )

                    log.info(

                        f"Min weight before normalization is {min(weights)}. Max weight before normalization is {max(weights)}"

                    )

                    image.savefig(filename, bbox_inches="tight")

            else:

                raise RuntimeError(

                    f"Unknown options given fp should be one of 'morgan', 'atompair' or 'torsion' user has given {fp} or weights must not be None user has given {weights}."

                )


### seaborn_correlation_heat_map

```python3
def seaborn_correlation_heat_map(
    df: pandas.core.frame.DataFrame,
    subset_colummns: Optional[List[str]] = None,
    width: int = 15,
    filename: str = 'heat_map.png',
    row_column_scaler: float = 2.0,
    square: bool = True,
    vmin: float = -1.0,
    vmax: float = 1.0,
    method: Union[str, Callable] = 'pearson',
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    fontsize: int = 20,
    **kwargs
) -> matplotlib.axes._axes.Axes
```

Function to plot a seaborn correlation heat map of a dataframe

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | pd.DataFrame | The dataframe to plot | None |
| subset_colummns | Optional[List[str]] | The columns to subset the dataframe to. Defaults to None. | None |
| width | int | The width of the column names. Defaults to 15. | 15 |
| filename | str | Filename to save the plot to. Defaults to "heat_map.png". | "heat_map.png" |
| row_column_scaler | float | The scaler to multiply the row and column size by. Defaults to 2.0. | 2.0 |
| square | bool | Whether to make the plot square. Defaults to True. | True |
| vmin | float | The minimum value of the colour map. Defaults to -1.0. | -1.0 |
| vmax | float | The maximum value of the colour map. Defaults to 1.0. | 1.0 |
| method | Union[str, Callable] | The method to calculate the correlation. Defaults to "pearson". | "pearson" |
| title | Optional[str] | The title of the plot. Defaults to None. | None |
| xlabel | Optional[str] | The x axis label. Defaults to None. | None |
| ylabel | Optional[str] | The y axis label. Defaults to None. | None |
| fontsize | int | The fontsize of the title. Defaults to 20. | 20 |

**Returns:**

| Type | Description |
|---|---|
| Axes | The axis of the plot |

??? example "View Source"
        def seaborn_correlation_heat_map(

            df: pd.DataFrame,

            subset_colummns: Optional[List[str]] = None,

            width: int = 15,

            filename: str = "heat_map.png",

            row_column_scaler: float = 2.0,

            square: bool = True,

            vmin: float = -1.0,

            vmax: float = 1.0,

            method: Union[str, Callable] = "pearson",

            title: Optional[str] = None,

            xlabel: Optional[str] = None,

            ylabel: Optional[str] = None,

            fontsize: int = 20,

            **kwargs,

        ) -> Axes:

            """

            Function to plot a seaborn correlation heat map of a dataframe

            Args:

                df (pd.DataFrame): The dataframe to plot

                subset_colummns (Optional[List[str]], optional): The columns to subset the dataframe to. Defaults to None.

                width (int, optional): The width of the column names. Defaults to 15.

                filename (str, optional): Filename to save the plot to. Defaults to "heat_map.png".

                row_column_scaler (float, optional): The scaler to multiply the row and column size by. Defaults to 2.0.

                square (bool, optional): Whether to make the plot square. Defaults to True.

                vmin (float, optional): The minimum value of the colour map. Defaults to -1.0.

                vmax (float, optional): The maximum value of the colour map. Defaults to 1.0.

                method (Union[str, Callable], optional): The method to calculate the correlation. Defaults to "pearson".

                title (Optional[str], optional): The title of the plot. Defaults to None.

                xlabel (Optional[str], optional): The x axis label. Defaults to None.

                ylabel (Optional[str], optional): The y axis label. Defaults to None.

                fontsize (int, optional): The fontsize of the title. Defaults to 20.

            Returns:

                Axes: The axis of the plot

            """

            if subset_colummns is not None:

                df = df[subset_colummns].copy()

            df.columns = [textwrap.fill(col, width=width) for col in df.columns]

            plt.figure(

                figsize=(

                    row_column_scaler * len(df.columns),

                    row_column_scaler * len(df.columns),

                )

            )

            sns.heatmap(

                df.corr(method=method),

                annot=True,

                fmt=".2f",

                cmap="coolwarm",

                cbar=True,

                square=square,

                vmin=vmin,

                vmax=vmax,

            )

            if title is not None:

                plt.title(title, fontsize=fontsize)

            if xlabel is not None:

                plt.xlabel(xlabel, fontsize=fontsize)

            if ylabel is not None:

                plt.ylabel(ylabel, fontsize=fontsize)

            plt.tight_layout()

            plt.savefig(filename)

            return plt.gca()


### seaborn_pair_plot

```python3
def seaborn_pair_plot(
    df: pandas.core.frame.DataFrame,
    subset_colummns: Optional[List[str]] = None,
    width: int = 15,
    filename: str = 'pair_plot.png',
    row_column_scaler: float = 2.0,
    kind: str = 'reg',
    diag_kind: str = 'kde',
    title: Optional[str] = None,
    fontsize: int = 20,
    **kwargs
) -> matplotlib.axes._axes.Axes
```

Function to plot a seaborn pair plot of a dataframe

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | pd.DataFrame | The dataframe to plot | None |
| subset_colummns | Optional[List[str]] | The columns to subset the dataframe to. Defaults to None. | None |
| width | int | The width of the column names. Defaults to 15. | 15 |
| filename | str | Filename to save the plot to. Defaults to "pair_plot.png". | "pair_plot.png" |
| row_column_scaler | float | The scaler to multiply the row and column size by. Defaults to 2.0. | 2.0 |
| kind | str | The kind of plot to use. Defaults to "reg". | "reg" |
| diag_kind | str | The kind of plot to use on the diagonal. Defaults to "kde". | "kde" |
| title | Optional[str] | The title of the plot. Defaults to None. | None |
| fontsize | int | The fontsize of the title. Defaults to 20. | 20 |

**Returns:**

| Type | Description |
|---|---|
| Axes | The axis of the plot |

??? example "View Source"
        def seaborn_pair_plot(

            df: pd.DataFrame,

            subset_colummns: Optional[List[str]] = None,

            width: int = 15,

            filename: str = "pair_plot.png",

            row_column_scaler: float = 2.0,

            kind: str = "reg",

            diag_kind: str = "kde",

            title: Optional[str] = None,

            fontsize: int = 20,

            **kwargs,

        ) -> Axes:

            """

            Function to plot a seaborn pair plot of a dataframe

            Args:

                df (pd.DataFrame): The dataframe to plot

                subset_colummns (Optional[List[str]], optional): The columns to subset the dataframe to. Defaults to None.

                width (int, optional): The width of the column names. Defaults to 15.

                filename (str, optional): Filename to save the plot to. Defaults to "pair_plot.png".

                row_column_scaler (float, optional): The scaler to multiply the row and column size by. Defaults to 2.0.

                kind (str, optional): The kind of plot to use. Defaults to "reg".

                diag_kind (str, optional): The kind of plot to use on the diagonal. Defaults to "kde".

                title (Optional[str], optional): The title of the plot. Defaults to None.

                fontsize (int, optional): The fontsize of the title. Defaults to 20.

            Returns:

                Axes: The axis of the plot

            """

            if subset_colummns is not None:

                df = df[subset_colummns].copy()

            df.columns = [textwrap.fill(col, width=width) for col in df.columns]

            plt.figure(

                figsize=(

                    row_column_scaler * len(df.columns),

                    row_column_scaler * len(df.columns),

                )

            )

            sns.pairplot(df, kind=kind, diag_kind=diag_kind, **kwargs)

            if title is not None:

                plt.title(title, fontsize=fontsize)

            plt.tight_layout()

            plt.savefig(filename)

            return plt.gca()


### tanimoto_box_plot

```python3
def tanimoto_box_plot(
    df1: pandas.core.frame.DataFrame,
    df2: pandas.core.frame.DataFrame,
    labels_column_1: str,
    smiles_column_1: str,
    smiles_column_2: str,
    filename: str = 'tanimoto_box_plot.png',
    fps1: Optional[list] = None,
    fps2: Optional[list] = None,
    dist: bool = False
)
```

Function to calculate the symmetric Tanimoto similarity matrix

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df1 | pd.DataFrame | Dataframe of the first set of smiles to compare against. These are used as the reference set and will provide labels to the columns if the label column is defined. Defaults to None. | None |
| df2 | pd.DataFrame | Dataframe of the first set of smiles to compare to. These for the test cases to make the distrbution i.e. how similar is each molecule in this set to each one at a time from df1. Defaults to None. | None |
| labels_column_1 | str | The column containing labels from data set 1. Defaults to None. | None |
| smiles_column_1 | str | The column containing smiles for the molecules. Defaults to None. | None |
| smiles_column_2 | str | The column containing smiles for the molecules. Defaults to None. | None |
| filename | str | the file to save the plot to | None |
| fps1 | Optional[list] | Precomputed fingerprints for df1. Defaults to None. | None |
| fps2 | Optional[list] | Precomputed fingerprints for df2. Defaults to None. | None |
| dist | bool | Calculate simialrity if this False (default) or distance is this is True. Defaults to False. | False |

??? example "View Source"
        def tanimoto_box_plot(

            df1: pd.DataFrame,

            df2: pd.DataFrame,

            labels_column_1: str,

            smiles_column_1: str,

            smiles_column_2: str,

            filename: str = "tanimoto_box_plot.png",

            fps1: Optional[list] = None,

            fps2: Optional[list] = None,

            dist: bool = False,

        ):

            """

            Function to calculate the symmetric Tanimoto similarity matrix

            Args:

                df1 (pd.DataFrame): Dataframe of the first set of smiles to compare against. These are used as the reference set and will provide labels to the columns if the label column is defined. Defaults to None.

                df2 (pd.DataFrame): Dataframe of the first set of smiles to compare to. These for the test cases to make the distrbution i.e. how similar is each molecule in this set to each one at a time from df1. Defaults to None.

                labels_column_1 (str): The column containing labels from data set 1. Defaults to None.

                smiles_column_1 (str): The column containing smiles for the molecules. Defaults to None.

                smiles_column_2 (str): The column containing smiles for the molecules. Defaults to None.

                filename (str): the file to save the plot to

                fps1 (Optional[list], optional): Precomputed fingerprints for df1. Defaults to None.

                fps2 (Optional[list], optional): Precomputed fingerprints for df2. Defaults to None.

                dist (bool, optional): Calculate simialrity if this False (default) or distance is this is True. Defaults to False.

            """

            tanimoto_measures = get_tanimoto_matrix(

                df1,

                df2,

                labels_column_1,

                smiles_column_1,

                smiles_column_2,

                dist=dist,

                fps1=fps1,

                fps2=fps2,

            )

            # plot box plot

            plt.figure(figsize=(2 * len(tanimoto_measures.columns), 10))

            tanimoto_measures.plot.box(ax=plt.gca(), rot=90)

            plt.xlabel("Molecule Labels", fontsize=20)

            plt.xticks(fontsize=17)

            if dist is False:

                plt.ylabel("Tanimoto Smiliarlity", fontsize=20)

                plt.yticks(fontsize=17)

            else:

                plt.ylabel("Tanimoto Distance", fontsize=20)

                plt.yticks(fontsize=17)

            plt.tight_layout()

            plt.savefig(filename, bbox_inches="tight")


### tanimoto_distrbution_plots

```python3
def tanimoto_distrbution_plots(
    df1: pandas.core.frame.DataFrame,
    df2: pandas.core.frame.DataFrame,
    labels_column_1: str,
    smiles_column_1: str,
    smiles_column_2: str,
    filename: str = 'tanimoto_distrbution_plot.png',
    fps1=None,
    fps2=None,
    dist=False
)
```

Function to calculate the symmetric Tanimoto similarity matrix

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df1 | pd.DataFrame | Dataframe of the first set of smiles to compare against. These are used as the reference set and will provide labels to the columns if the label column is defined. Defaults to None. | None |
| df2 | pd.DataFrame | Dataframe of the first set of smiles to compare to. These for the test cases to make the distrbution i.e. how similar is each molecule in this set to each one at a time from df1. Defaults to None. | None |
| labels_column_1 | str | The column containing labels from data set 1. Defaults to None. | None |
| smiles_column_1 | str | The column containing smiles for the molecules. Defaults to None. | None |
| smiles_column_2 | str | The column containing smiles for the molecules. Defaults to None. | None |
| filename | str | the file to save the plot to | None |
| fps1 | Optional[list] | Precomputed fingerprints for df1. Defaults to None. | None |
| fps2 | Optional[list] | Precomputed fingerprints for df2. Defaults to None. | None |
| dist | bool | Calculate simialrity if this False (default) or distance is this is True. Defaults to False. | False |

??? example "View Source"
        def tanimoto_distrbution_plots(

            df1: pd.DataFrame,

            df2: pd.DataFrame,

            labels_column_1: str,

            smiles_column_1: str,

            smiles_column_2: str,

            filename: str = "tanimoto_distrbution_plot.png",

            fps1=None,

            fps2=None,

            dist=False,

        ):

            """

            Function to calculate the symmetric Tanimoto similarity matrix

            Args:

                df1 (pd.DataFrame): Dataframe of the first set of smiles to compare against. These are used as the reference set and will provide labels to the columns if the label column is defined. Defaults to None.

                df2 (pd.DataFrame): Dataframe of the first set of smiles to compare to. These for the test cases to make the distrbution i.e. how similar is each molecule in this set to each one at a time from df1. Defaults to None.

                labels_column_1 (str): The column containing labels from data set 1. Defaults to None.

                smiles_column_1 (str): The column containing smiles for the molecules. Defaults to None.

                smiles_column_2 (str): The column containing smiles for the molecules. Defaults to None.

                filename (str): the file to save the plot to

                fps1 (Optional[list], optional): Precomputed fingerprints for df1. Defaults to None.

                fps2 (Optional[list], optional): Precomputed fingerprints for df2. Defaults to None.

                dist (bool, optional): Calculate simialrity if this False (default) or distance is this is True. Defaults to False.

            """

            tm_mat = get_tanimoto_matrix(

                df1,

                df2,

                labels_column_1=labels_column_1,

                smiles_column_1=smiles_column_1,

                smiles_column_2=smiles_column_2,

                fps1=fps1,

                fps2=fps2,

                dist=dist,

            )

            medians = pd.DataFrame(tm_mat.median(), columns=["median"])

            means = pd.DataFrame(tm_mat.mean(), columns=["mean"])

            for indx_med, indx_mean in zip(medians.index.to_list(), means.index.to_list()):

                if indx_med != indx_mean:

                    raise RuntimeError(

                        "Indexes are different between the mean and median series!"

                    )

            averages = pd.DataFrame([medians["median"], means["mean"]]).transpose()

            plt.figure(figsize=(10, 10))

            averages.plot.kde(alpha=0.75, ax=plt.gca(), colormap="bwr")

            if dist is False:

                plt.xlabel("Average Tanimoto Similarity Distrbution", fontsize=20)

            elif dist is True:

                plt.xlabel("Average Tanimoto Distance Distrbution", fontsize=20)

            plt.xticks(fontsize=17)

            plt.ylabel("Frequency", fontsize=20)

            plt.yticks(fontsize=17)

            plt.legend(fontsize=15)

            plt.savefig(filename)


### tanimoto_distrbution_plots_intra_dataset

```python3
def tanimoto_distrbution_plots_intra_dataset(
    df1: pandas.core.frame.DataFrame,
    labels_column_1: str,
    smiles_column_1: str,
    filename: str = 'tanimoto_distrbution_plot.png',
    fps1=None,
    dist=False
)
```

Function to calculate the symmetric Tanimoto similarity matrix

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df1 | pd.DataFrame | Dataframe of the first set of smiles to compare against. These are used as the reference set and will provide labels to the columns if the label column is defined. Defaults to None. | None |
| labels_column_1 | str | The column containing labels from data set 1. Defaults to None. | None |
| smiles_column_1 | str | The column containing smiles for the molecules. Defaults to None. | None |
| filename | str | the file to save the plot to | None |
| fps1 | Optional[list] | Precomputed fingerprints for df1. Defaults to None. | None |
| dist | bool | Calculate simialrity if this False (default) or distance is this is True. Defaults to False. | False |

??? example "View Source"
        def tanimoto_distrbution_plots_intra_dataset(

            df1: pd.DataFrame,

            labels_column_1: str,

            smiles_column_1: str,

            filename: str = "tanimoto_distrbution_plot.png",

            fps1=None,

            dist=False,

        ):

            """

            Function to calculate the symmetric Tanimoto similarity matrix

            Args:

                df1 (pd.DataFrame): Dataframe of the first set of smiles to compare against. These are used as the reference set and will provide labels to the columns if the label column is defined. Defaults to None.

                labels_column_1 (str): The column containing labels from data set 1. Defaults to None.

                smiles_column_1 (str): The column containing smiles for the molecules. Defaults to None.

                filename (str): the file to save the plot to

                fps1 (Optional[list], optional): Precomputed fingerprints for df1. Defaults to None.

                dist (bool, optional): Calculate simialrity if this False (default) or distance is this is True. Defaults to False.

            """

            tm_mat = get_tanimoto_intra_dataset_matrix(

                df1,

                labels_column_1=labels_column_1,

                smiles_column_1=smiles_column_1,

                fps1=fps1,

                dist=dist,

            )

            medians = pd.DataFrame(tm_mat.median(), columns=["median"])

            means = pd.DataFrame(tm_mat.mean(), columns=["mean"])

            for indx_med, indx_mean in zip(medians.index.to_list(), means.index.to_list()):

                if indx_med != indx_mean:

                    raise RuntimeError(

                        "Indexes are different between the mean and median series!"

                    )

            averages = pd.DataFrame([medians["median"], means["mean"]]).transpose()

            plt.figure(figsize=(10, 10))

            averages.plot.kde(alpha=0.75, ax=plt.gca(), colormap="bwr")

            if dist is False:

                plt.xlabel("Average Tanimoto Similarity Distrbution", fontsize=20)

            elif dist is True:

                plt.xlabel("Average Tanimoto Distance Distrbution", fontsize=20)

            plt.xticks(fontsize=17)

            plt.ylabel("Frequency", fontsize=20)

            plt.yticks(fontsize=17)

            plt.legend(fontsize=15)

            plt.savefig(filename)


### tanimoto_intra_dataset_box_plot

```python3
def tanimoto_intra_dataset_box_plot(
    df1: pandas.core.frame.DataFrame,
    labels_column_1: str,
    smiles_column_1: str,
    filename: str = 'tanimoto_box_plot.png',
    fps1: Optional[list] = None,
    dist: bool = False
)
```

Function to calculate the symmetric Tanimoto similarity matrix

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df1 | pd.DataFrame | Dataframe of the first set of smiles to compare against. These are used as the reference set and will provide labels to the columns if the label column is defined. Defaults to None. | None |
| labels_column_1 | str | The column containing labels from data set 1. Defaults to None. | None |
| smiles_column_1 | str | The column containing smiles for the molecules. Defaults to None. | None |
| filename | str | the file to save the plot to | None |
| fps1 | Optional[list] | Precomputed fingerprints for df1. Defaults to None. | None |
| dist | bool | Calculate simialrity if this False (default) or distance is this is True. Defaults to False. | False |

??? example "View Source"
        def tanimoto_intra_dataset_box_plot(

            df1: pd.DataFrame,

            labels_column_1: str,

            smiles_column_1: str,

            filename: str = "tanimoto_box_plot.png",

            fps1: Optional[list] = None,

            dist: bool = False,

        ):

            """

            Function to calculate the symmetric Tanimoto similarity matrix

            Args:

                df1 (pd.DataFrame): Dataframe of the first set of smiles to compare against. These are used as the reference set and will provide labels to the columns if the label column is defined. Defaults to None.

                labels_column_1 (str): The column containing labels from data set 1. Defaults to None.

                smiles_column_1 (str): The column containing smiles for the molecules. Defaults to None.

                filename (str): the file to save the plot to

                fps1 (Optional[list], optional): Precomputed fingerprints for df1. Defaults to None.

                dist (bool, optional): Calculate simialrity if this False (default) or distance is this is True. Defaults to False.

            """

            tanimoto_measures = get_tanimoto_intra_dataset_matrix(

                df1,

                labels_column_1,

                smiles_column_1,

                dist=dist,

                fps1=fps1,

            )

            log.info(tanimoto_measures)

            # plot box plot

            plt.figure(figsize=(2 * len(tanimoto_measures.columns), 10))

            tanimoto_measures.plot.box(ax=plt.gca(), rot=90)

            plt.xlabel("Molecule Labels", fontsize=20)

            plt.xticks(fontsize=17)

            if dist is False:

                plt.ylabel("Tanimoto Smiliarlity", fontsize=20)

                plt.yticks(fontsize=17)

            else:

                plt.ylabel("Tanimoto Distance", fontsize=20)

                plt.yticks(fontsize=17)

            plt.tight_layout()

            plt.savefig(filename, bbox_inches="tight")
