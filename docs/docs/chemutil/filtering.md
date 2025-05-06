# Module chemutil.filtering

Module for generating features such as chemical fingerprints and descriptors and the converison of common data types of those representations

??? example "View Source"
        #!/usr/bin.env python3

        # -*- coding: utf-8 -*-

        """

        Module for generating features such as chemical fingerprints and descriptors and the converison of common data types of those representations

        """

        import os

        import sys

        from collections import namedtuple

        from typing import Callable, List, Optional, Tuple, Union

        import numpy as np

        import pandas as pd

        from rdkit.Chem import QED, AllChem, Descriptors, PandasTools, RDConfig

        from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

        from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity, TanimotoSimilarity

        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

        import logging

        import datamol as dm

        import sascorer

        from rdkit import Chem

        from rdkit.Chem import AllChem

        from chemutil import inchimod, multiopt, smilesmod

        log = logging.getLogger(__name__)

        dm.disable_rdkit_log()



        def lipinski_ro5(mol: Chem.rdchem.Mol, ro5_n_exceptions: int = 0) -> int:

            """

            Lipinski Rule of 5

            Args:

                mol (Chem.rdchem.Mol): RDKit molecule object

                ro5_n_exceptions (int): Number of exceptions to the Lipinski Rule of 5

            Returns:

                bool: True if molecule passes the Lipinski Rule of 5, False otherwise

            """

            molecular_weight = Descriptors.ExactMolWt(mol)

            logp = Descriptors.MolLogP(mol)

            h_bond_donor = Descriptors.NumHDonors(mol)

            h_bond_acceptors = Descriptors.NumHAcceptors(mol)

            rotatable_bonds = Descriptors.NumRotatableBonds(mol)

            ro5 = [

                molecular_weight <= 500,

                logp <= 5,

                h_bond_donor <= 5,

                h_bond_acceptors <= 10,

                rotatable_bonds <= 5,

            ]

            if sum(ro5) >= len(ro5) - ro5_n_exceptions:

                return 1

            else:

                return 0



        def ghose_ro2(mol: Chem.rdchem.Mol, ghose_n_exceptions: int = 0) -> int:

            """

            Ghose Filter

            Args:

                mol (Chem.rdchem.Mol): RDKit molecule object

                ghose_n_exceptions (int): Number of exceptions to the Ghose Filter

            Returns:

                bool: True if molecule passes the Ghose Filter, False otherwise

            """

            molecular_weight = Descriptors.ExactMolWt(mol)

            logp = Descriptors.MolLogP(mol)

            number_of_atoms = Chem.rdchem.Mol.GetNumAtoms(mol)

            molar_refractivity = Chem.Crippen.MolMR(mol)

            ghose = [

                molecular_weight >= 160,

                molecular_weight <= 480,

                logp >= -0.4,

                logp <= 5.6,

                number_of_atoms >= 20,

                number_of_atoms <= 70,

                molar_refractivity >= 40,

                molar_refractivity <= 130,

            ]

            if sum(ghose) >= len(ghose) - ghose_n_exceptions:

                return 1

            else:

                return 0



        def veber_ro2(mol: Chem.rdchem.Mol, veber_n_exceptions: int = 0) -> int:

            """

            Veber Filter

            Args:

                mol (Chem.rdchem.Mol): RDKit molecule object

                veber_n_exceptions (int): Number of exceptions to the Veber Filter

            Returns:

                bool: True if molecule passes the Veber Filter, False otherwise

            """

            rotatable_bonds = Descriptors.NumRotatableBonds(mol)

            topological_surface_area_mapping = Chem.QED.properties(mol).PSA

            veber = [rotatable_bonds <= 10, topological_surface_area_mapping <= 140]

            if sum(veber) >= len(veber) - veber_n_exceptions:

                return 1

            else:

                return 0



        def congreve_ro3(mol: Chem.rdchem.Mol, ro3_n_exceptions: int = 0) -> int:

            """

            Rule of 3

            Args:

                mol (Chem.rdchem.Mol): RDKit molecule object

                ro3_n_exceptions (int): Number of exceptions to the Rule of 3

            Returns:

                bool: True if molecule passes the Rule of 3, False otherwise

            """

            molecular_weight = Descriptors.ExactMolWt(mol)

            logp = Descriptors.MolLogP(mol)

            h_bond_donor = Descriptors.NumHDonors(mol)

            h_bond_acceptors = Descriptors.NumHAcceptors(mol)

            rotatable_bonds = Descriptors.NumRotatableBonds(mol)

            ro3 = [

                molecular_weight <= 300,

                logp <= 3,

                h_bond_donor <= 3,

                h_bond_acceptors <= 3,

                rotatable_bonds <= 3,

            ]

            if sum(ro3) >= len(ro3) - ro3_n_exceptions:

                return 1

            else:

                return 0



        def reos_ro7(mol: Chem.rdchem.Mol, reos_n_exceptions: int = 0) -> int:

            """

            REOS Filter

            Args:

                mol (Chem.rdchem.Mol): RDKit molecule object

                reos_n_exceptions (int): Number of exceptions to the REOS Filter

            Returns:

                bool: True if molecule passes the REOS Filter, False otherwise

            """

            molecular_weight = Descriptors.ExactMolWt(mol)

            logp = Descriptors.MolLogP(mol)

            h_bond_donor = Descriptors.NumHDonors(mol)

            h_bond_acceptors = Descriptors.NumHAcceptors(mol)

            formal_charge = Chem.rdmolops.GetFormalCharge(mol)

            rotatable_bonds = Descriptors.NumRotatableBonds(mol)

            heavy_atoms = Chem.rdchem.Mol.GetNumHeavyAtoms(mol)

            reos = [

                200 <= molecular_weight <= 500,

                -5 <= logp <= 5,

                0 <= h_bond_donor <= 5,

                0 <= h_bond_acceptors <= 10,

                -2 <= formal_charge <= 2,

                0 <= rotatable_bonds <= 8,

                15 <= heavy_atoms <= 50,

            ]

            if sum(reos) >= len(reos) - reos_n_exceptions:

                return 1

            else:

                return 0



        def qed_threshold(mol: Chem.rdchem.Mol, qed_threshold: float = 0.5) -> float:

            """

            Quantitative Estimate of Drug-likeness (QED)

            Args:

                mol (Chem.rdchem.Mol): RDKit molecule object

                qed_threshold (float): QED threshold

            Returns:

                float: QED score

            """

            qed_score = QED.qed(mol)

            if qed_score >= qed_threshold:

                return 1

            else:

                return 0



        def druglikeness(

            mols,

            ro5: bool = True,

            ro5_n_exceptions: int = 0,

            ghose: bool = True,

            ghose_n_exceptions: int = 0,

            veber: bool = True,

            veber_n_exceptions: int = 0,

            ro3: bool = True,

            ro3_n_exceptions: int = 0,

            reos: bool = True,

            reos_n_exceptions: int = 0,

            qed: bool = False,

            qed_thresh: float = 0.5,

            return_numpy: bool = False,

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate druglikeness properties for a list of molecules. The following properties are calculated:

            - Lipinski Rule of 5

            - Ghose Filter

            - Veber Filter

            - Rule of 3

            - REOS Filter

            - QED (with a threshold)

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objects

                ro5 (bool): Apply Lipinski Rule of 5

                ro5_n_exceptions (int): Number of exceptions to the Lipinski Rule of 5

                ghose (bool): Apply Ghose Filter

                ghose_n_exceptions (int): Number of exceptions to the Ghose Filter

                veber (bool): Apply Veber Filter

                veber_n_exceptions (int): Number of exceptions to the Veber Filter

                ro3 (bool): Apply Rule of 3

                ro3_n_exceptions (int): Number of exceptions to the Rule of 3

                reos (bool): Apply REOS Filter

                reos_n_exceptions (int): Number of exceptions to the REOS Filter

                qed (bool): Apply QED

                qed_threshold (float): QED threshold

                return_numpy (bool): Return a numpy array

            Returns:

                np.ndarray: Druglikeness properties for the list of molecules

            """

            ret = np.zeros([len(mols), sum([ro5, ghose, veber, ro3, reos])], dtype=int)

            cols = []

            for ith, mol in enumerate(mols):

                column_index = 0

                if ro5 is True:

                    log.debug(f"Calculating Lipinski Rule of 5 for molecule {ith}")

                    ret[ith, column_index] = lipinski_ro5(mol, ro5_n_exceptions)

                    column_index += 1

                    if ith == 0:

                        cols.append("ro5")

                if ghose is True:

                    log.debug(f"Calculating Ghose Filter for molecule {ith}")

                    ret[ith, column_index] = ghose_ro2(mol, ghose_n_exceptions)

                    column_index += 1

                    if ith == 0:

                        cols.append("ghose")

                if veber is True:

                    log.debug(f"Calculating Veber Filter for molecule {ith}")

                    ret[ith, column_index] = veber_ro2(mol, veber_n_exceptions)

                    column_index += 1

                    if ith == 0:

                        cols.append("veber")

                if ro3 is True:

                    log.debug(f"Calculating Rule of 3 for molecule {ith}")

                    ret[ith, column_index] = congreve_ro3(mol, ro3_n_exceptions)

                    column_index += 1

                    if ith == 0:

                        cols.append("ro3")

                if reos is True:

                    log.debug(f"Calculating REOS Filter for molecule {ith}")

                    ret[ith, column_index] = reos_ro7(mol, reos_n_exceptions)

                    column_index += 1

                    if ith == 0:

                        cols.append("reos")

                if qed is True:

                    log.debug(f"Calculating QED for molecule {ith}")

                    ret[ith, column_index] = qed_threshold(mol, qed_thresh)

                    column_index += 1

                    if ith == 0:

                        cols.append("qed")

            log.debug(f"The raw return is: {ret}")

            if return_numpy is True:

                return ret

            else:

                return pd.DataFrame(ret, columns=cols)



        def synthetic_accessibility_score(

            mols: List[Chem.rdchem.Mol], invalid_value=np.nan, return_numpy: bool = False

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate the synthetic accessibility score for a list of molecules

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objects

                invalid_value (float): Value to return if the synthetic accessibility score cannot be calculated

                return_numpy (bool): Return a numpy array

            Returns:

                np.ndarray: Synthetic accessibility score for the list of molecules

            """

            syn_acc_score = []

            for mol in mols:

                try:

                    syn_acc_score.append(sascorer.calculateScore(mol))

                except ZeroDivisionError:

                    syn_acc_score.append(invalid_value)

            return (

                pd.DataFrame(

                    np.array([[ent] for ent in syn_acc_score]),

                    columns=["synthetic_accessibility_score"],

                )

                if return_numpy is False

                else np.array(syn_acc_score)

            )



        def get_qed_common_versions(

            mols: List[Chem.rdchem.Mol], invalid_value=np.nan, return_numpy: bool = False

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate QED scores for a list of molecules using the different versions of QED:

            - QED weights max (i.e. the maximum weights used)

            - QED weights mean (i.e. the original QED score)

            - QED weights none (i.e. all weights are set to 1.0)

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objects

                invalid_value (float): Value to return if the QED score cannot be calculated

                return_numpy (bool): Return a numpy array

            Returns:

                pd.DataFrame: QED scores for the list of molecules

            """

            qed_scores = np.zeros([len(mols), 4], dtype=float)

            for ith, m in enumerate(mols):

                try:

                    qed_scores[ith, 0] = QED.weights_max(m)

                except Exception as e:

                    qed_scores[ith, 0] = invalid_value

                # This looks like it is the same as the original QED score QED.qed(m)

                try:

                    qed_scores[ith, 1] = QED.weights_mean(m)

                except Exception as e:

                    qed_scores[ith, 1] = invalid_value

                try:

                    qed_scores[ith, 2] = QED.weights_none(m)

                except Exception as e:

                    qed_scores[ith, 2] = invalid_value

                try:

                    qed_scores[ith, 3] = QED.qed(m)

                except Exception as e:

                    qed_scores[ith, 3] = invalid_value

            return (

                pd.DataFrame(

                    qed_scores, columns=["qed_max", "qed_mean", "qed_all_weights_one", "qed"]

                )

                if return_numpy is False

                else qed_scores

            )



        def conatains_pains(

            mols: List[Chem.rdchem.Mol],

            invalid_value=np.nan,

            default_pains_only: bool = True,

            all_pains: bool = False,

            catalogue: Optional[Union[FilterCatalog, FilterCatalogParams]] = None,

            return_numpy: bool = False,

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Check if a list of molecules contains PAINS (Pan Assay INterference compoundS) compounds or other substructures that are known to be a problem

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objects

                invalid_value (float): Value to return if the PAINS check cannot be calculated

                default_pains_only (bool): Use the default PAINS catalogue (PAINS_A, PAINS_B, PAINS_C)

                all_pains (bool): Use all PAINS catalogues

                catalogue (Union[FilterCatalog, FilterCatalogParams]): PAINS catalogue to use

                return_numpy (bool): Return a numpy array

            Returns:

                np.ndarray: Boolean vector defining if the molecule contains PAINS or not

            """

            contains_pains = []

            if catalogue is not None:

                if isinstance(catalogue, FilterCatalogParams):

                    log.info(f"Using provided PAINS catalogue {str(catalogue)}")

                    catalogue = FilterCatalog(catalogue)

                elif isinstance(catalogue, FilterCatalog):

                    log.info("Using provided PAINS catalogue")

                else:

                    raise ValueError(

                        f"Catalogue type not recognised {type(catalogue)} should be one of FilterCatalog or FilterCatalogParams"

                    )

            elif default_pains_only is True:

                log.info("Using default PAINS catalogue (PAINS_A, PAINS_B, PAINS_C)")

                params = FilterCatalogParams()

                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)

                catalogue = FilterCatalog(params)

            elif all_pains is True:

                log.info("Using all PAINS catalogues")

                params_all = FilterCatalogParams()

                params_all.AddCatalog(FilterCatalogParams.FilterCatalogs.ALL)

                catalogue = FilterCatalog(params_all)

            for mol in mols:

                try:

                    contains_pains.append(catalogue.HasMatch(mol))

                except Exception as e:

                    contains_pains.append(invalid_value)

            return (

                pd.DataFrame(

                    np.array([[ent] for ent in contains_pains]), columns=["contains_pains"]

                )

                if return_numpy is False

                else np.array(contains_pains)

            )



        def qed_scores(

            mols: List[Chem.rdchem.Mol],

            invalid_value=np.nan,

            weights: Tuple[float] = (0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95),

            qed_properties: Optional[namedtuple] = None,

            return_all_common_varients: bool = False,

            return_numpy: bool = False,

            append_weights_to_qed_column_names: bool = False,

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate QED scores for a list of molecules. See https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/QED.py

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objects

                invalid_value (float): Value to return if the QED score cannot be calculated

                weights (Tuple[float]): Weights for the QED properties. The default weights are (0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95) which

                  are from the RDKit documentation https://www.rdkit.org/docs/source/rdkit.Chem.QED.html#:~:text=%3DNone)-,%C2%B6,-Calculate%20the%20weighted accessed 10/12/24

                qed_properties (namedtuple): QED properties

                return_all_common_varients (bool): Return all common varients of QED

                return_numpy (bool): Return a numpy array

                append_weights_to_qed_column_names (bool): Append the weights to the QED column names

            Returns:

                Union[np.ndarray, pd.DataFrame]: QED scores for the list of molecules

            """

            if return_all_common_varients is True:

                qed_versions_df = get_qed_common_versions(mols, invalid_value)

                if return_numpy is True:

                    return qed_versions_df.values

                else:

                    return qed_versions_df

            qed_scores = np.zeros([len(mols)], dtype=float)

            for ith, m in enumerate(mols):

                try:

                    qed_scores[ith] = QED.qed(m, weights, qed_properties)

                except Exception as e:

                    qed_scores[ith] = invalid_value

            if return_numpy is True:

                return qed_scores

            else:

                if append_weights_to_qed_column_names is True:

                    return pd.DataFrame(

                        qed_scores,

                        columns=[

                            f"qed_weights_{'_'.join([str(weight) for weight in weights])}"

                        ],

                    )

                return pd.DataFrame(qed_scores, columns=["qed"])



        def tanimoto_single_similarity_metrics(

            mols: List[Chem.rdchem.Mol],

            ref_mol: Optional[Union[str, Chem.rdchem.Mol]] = None,

            return_numpy: bool = False,

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate Tanimoto similarity metrics for a list of molecules

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objectsd

                ref_mol (Union[str, Chem.rdchem.Mol]): Reference molecule to compare the molecules to

                return_numpy (bool): Return a numpy array

            Returns:

                Union[np.ndarray, pd.DataFrame]: Tanimoto similarity metrics for the list of molecules

            """

            fpgen = AllChem.GetMorganGenerator(radius=2)

            log.info("Calculating Tanimoto similarity metric against one molecule")

            if isinstance(ref_mol, str):

                if not ref_mol.startswith("InChI="):

                    ref_mol = Chem.MolFromSmiles(ref_mol)

                elif ref_mol.startswith("InChI="):

                    ref_mol = Chem.MolFromInchi(ref_mol)

                else:

                    raise ValueError(

                        f"Reference molecule type not recognised {type(ref_mol)} should be one of str (SMILES or InChI) or Chem.rdchem.Mol"

                    )

            elif isinstance(ref_mol, Chem.rdchem.Mol):

                log.info("Using provided reference molecule")

            reference_fingerprints = fpgen.GetFingerprint(ref_mol)

            scores = np.zeros((len(mols), 1))

            for ith, gen_mol in enumerate(mols):

                fp_test = fpgen.GetFingerprint(gen_mol)

                scores[ith] = TanimotoSimilarity(fp_test, reference_fingerprints)

            if return_numpy is True:

                return scores

            else:

                return pd.DataFrame(scores, columns=["tanimoto_sim"])



        def tanimoto_bulk_similarity_metrics(

            mols: List[Chem.rdchem.Mol],

            similarity_set: Optional[Union[str, pd.DataFrame]] = None,

            smiles_col: Optional[str] = "smiles",

            return_numpy: bool = False,

            clean_and_validate: bool = True,

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate Tanimoto similarity metrics for a list of molecules

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objectsd

                similarity_set (Union[str, pd.DataFrame]): Similarity set to compare the molecules to

                smiles_col (str): Column name of the SMILES in the similarity set

                return_numpy (bool): Return a numpy array

            Returns:

                Union[np.ndarray, pd.DataFrame]: Tanimoto similarity metrics for the list of molecules

            """

            fpgen = AllChem.GetMorganGenerator(radius=2)

            std_smiles_col_name = "standardized_smiles"

            log.info("Calculating Tanimoto similarity metrics stats over set of molecules")

            log.info("prepare the similarity set for comparison ")

            if isinstance(similarity_set, str):

                try:

                    similarity_set = pd.read_csv(similarity_set)

                    if clean_and_validate is True:

                        log.info("Cleaning and validating the similarity set")

                        similarity_set = smilesmod.clean_and_validate_smiles(

                            similarity_set,

                            smiles_column=smiles_col,

                            return_selfies=False,

                            return_inchi=False,

                            return_inchikey=False,

                        )

                    else:

                        log.info(

                            "Using provided similarity set without cleaning and validation"

                        )

                        std_smiles_col_name = smiles_col

                    rmols = [

                        smilesmod.smiles_to_molecule(s)

                        for s in similarity_set[std_smiles_col_name].values

                    ]

                except Exception as e:

                    raise ValueError(f"Similarity set file not found {similarity_set}")

            elif isinstance(similarity_set, pd.DataFrame):

                log.info("Using provided similarity set")

                if clean_and_validate is True:

                    log.info("Cleaning and validating the similarity set")

                    similarity_set = smilesmod.clean_and_validate_smiles(

                        similarity_set,

                        smiles_column=smiles_col,

                        return_selfies=False,

                        return_inchi=False,

                        return_inchikey=False,

                    )

                else:

                    log.info("Using provided similarity set without cleaning and validation")

                    std_smiles_col_name = smiles_col

                rmols = [

                    smilesmod.smiles_to_molecule(s)

                    for s in similarity_set[std_smiles_col_name].values

                ]

            else:

                raise ValueError(

                    f"Similarity set type not recognised {type(similarity_set)} should be one of str (file path) or pd.DataFrame"

                )

            log.info("Calculating Tanimoto similarity metrics")

            reference_set_fingerprints = [fpgen.GetFingerprint(m) for m in rmols]

            scores = np.zeros((len(mols), 7))

            for ith, gen_mol in enumerate(mols):

                fp_test = fpgen.GetFingerprint(gen_mol)

                score = BulkTanimotoSimilarity(fp_test, reference_set_fingerprints)

                scores[ith][0] = ith

                scores[ith][1] = np.min(score)

                scores[ith][2] = np.max(score)

                scores[ith][3] = np.mean(score)

                scores[ith][4] = np.std(score)

                scores[ith][5] = np.median(score)

                quart75, quart25 = np.percentile(score, [75, 25])

                scores[ith][6] = quart75 - quart25

            if return_numpy is True:

                return scores

            else:

                return pd.DataFrame(

                    scores,

                    columns=[

                        "id",

                        "tanimoto_sim_min",

                        "tanimoto_sim_max",

                        "tanimoto_sim_mean",

                        "tanimoto_sim_std",

                        "tanimoto_sim_median",

                        "tanimoto_sim_iqr",

                    ],

                )



        def tanimoto_similarity_metrics(

            mols: List[Chem.rdchem.Mol],

            similarity_set: Optional[Union[str, pd.DataFrame]] = None,

            smiles_col: Optional[str] = "smiles",

            ref_mol: Optional[Union[str, Chem.rdchem.Mol]] = None,

            return_numpy: bool = False,

            clean_and_validate: bool = True,

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate Tanimoto similarity metrics for a list of molecules

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objectsd

                similarity_set (Union[str, pd.DataFrame]): Similarity set to compare the molecules to

                smiles_col (str): Column name of the SMILES in the similarity set

                ref_mol (Union[str, Chem.rdchem.Mol]): Reference molecule to compare the molecules to

                return_numpy (bool): Return a numpy array

                clean_and_validate (bool): Clean and validate the similarity set

            Returns:

                Union[np.ndarray, pd.DataFrame]: Tanimoto similarity metrics for the list of molecules

            """

            if ref_mol is not None:

                return tanimoto_single_similarity_metrics(mols, ref_mol, return_numpy)

            if similarity_set is not None:

                return tanimoto_bulk_similarity_metrics(

                    mols, similarity_set, smiles_col, return_numpy, clean_and_validate

                )



        def get_largest_ring_system(

            mols: List[Chem.rdchem.Mol], return_numpy: bool = False

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate the largest ring system for a list of molecules

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objects

                return_numpy (bool): Return a numpy array

            Returns:

                np.ndarray: Largest ring system for the list of molecules

            """

            largest_ring_n_atoms = []

            for ith, mol in enumerate(mols):

                log.debug(f"Calculating largest ring system for molecule {ith} of {len(mols)}")

                try:

                    largest_ring_n_atoms.append(

                        max([len(a) for a in mol.GetRingInfo().AtomRings()])

                    )

                except ValueError:

                    largest_ring_n_atoms.append(0)

            log.debug(largest_ring_n_atoms)

            log.debug(len(largest_ring_n_atoms))

            if return_numpy is True:

                return np.array(largest_ring_n_atoms)

            else:

                return pd.DataFrame(

                    np.array([[ent] for ent in largest_ring_n_atoms]),

                    columns=["largest_ring_system"],

                )



        def get_molecular_mass(

            mols: List[Chem.rdchem.Mol], invalid_value=np.nan, return_numpy: bool = False

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate the molecular mass for a list of molecules

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objects

                invalid_value (float): Value to return if the molecular mass cannot be calculated

                return_numpy (bool): Return a numpy array

            Returns:

                Union[np.ndarray, pd.DataFrame]: Molecular mass for the list of molecules

            """

            molecular_mass = []

            for mol in mols:

                try:

                    molecular_mass.append(Descriptors.ExactMolWt(mol))

                except Exception as e:

                    molecular_mass.append(invalid_value)

            if return_numpy is True:

                return np.array(molecular_mass)

            else:

                return pd.DataFrame(

                    np.array([[ent] for ent in molecular_mass]), columns=["molecular_mass"]

                )



        def get_logp(

            mols: List[Chem.rdchem.Mol], invalid_value=np.nan, return_numpy: bool = False

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate the logP for a list of molecules

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objects

                invalid_value (float): Value to return if the logP cannot be calculated

                return_numpy (bool): Return a numpy array

            Returns:

                Union[np.ndarray, pd.DataFrame]: logP for the list of molecules

            """

            logp = []

            for mol in mols:

                try:

                    logp.append(Descriptors.MolLogP(mol))

                except Exception as e:

                    logp.append(invalid_value)

            if return_numpy is True:

                return np.array(logp)

            else:

                return pd.DataFrame(np.array([[ent] for ent in logp]), columns=["logp"])



        def get_filter_properties(

            data_df: pd.DataFrame,

            representation_column: str = "smiles",

            representation_type: str = "smiles",

            label_column: Optional[str] = None,

            training_data_csv_or_df: Optional[Union[str, pd.DataFrame]] = None,

            tanimoto_similarity_set: Optional[Union[str, pd.DataFrame]] = None,

            synthetic_accessibility: bool = True,

            druglike: bool = True,

            pains: bool = True,

            tanimoto_similarity: bool = True,

            molecular_weight: bool = True,

            largest_ring_system: bool = True,

            logp: bool = True,

            qed_score: bool = True,

        ) -> pd.DataFrame:

            """

            Get the filter properties for a DataFrame of molecules. The filter properties are:

            - Synthetic accessibility score

            - Druglikeness

            - PAINS

            - Tanimoto similarity to the training set

            - Molecular weight

            - Largest ring system

            - LogP

            - QED

            Args:

                data_df (pd.DataFrame): DataFrame containing the data

                representation_column (str): Column name of the representation

                representation_type (str): Representation type (smiles or inchi)

                label_column (Optional[str]): Column name of the labels

                training_data_csv_or_df (Optional[Union[str, pd.DataFrame]]): Training data to compare the Tanimoto similarity to

                synthetic_accessibility (bool): Apply synthetic accessibility score

                druglike (bool): Apply druglikeness filter

                pains (bool): Apply PAINS filter

                tanimoto_similarity (bool): Apply Tanimoto similarity filter

                molecular_weight (bool): Apply molecular weight filter

                largest_ring_system (bool): Apply largest ring system filter

                logp (bool): Apply logP filter

                qed_score (bool): Apply QED score filter

            Returns:

                pd.DataFrame: DataFrame containing the filter properties for the list of molecules

            Raises:

                ValueError: If the representation type is not recognised

            """

            df = data_df.copy()

            log.info(f"Cleaning and validating the {representation_type} representation")

            if representation_type == "smiles":

                cleaned_df = smilesmod.clean_and_validate_smiles(

                    df, smiles_column=representation_column

                )

                mols = [

                    Chem.MolFromSmiles(smi) for smi in cleaned_df[f"standardized_smiles"].values

                ]

            elif representation_type == "inchi":

                cleaned_df = inchimod.clean_and_validate_inchi(

                    df, inchi_column=representation_column

                )

                mols = [

                    Chem.MolFromSmiles(smi) for smi in cleaned_df[f"standardized_inchi"].values

                ]

            else:

                raise ValueError(

                    f"Representation type not recognised {representation_type}. Currently this should be on of the following: 'smiles' or 'inchi'"

                )

            if label_column is not None:

                labels = df[label_column].values

            else:

                labels = None

            funcs = []

            args = []

            if synthetic_accessibility is True:

                funcs.append(synthetic_accessibility_score)

                args.append({})

            if druglike is True:

                funcs.append(druglikeness)

                args.append({"qed": False})

            if pains is True:

                funcs.append(conatains_pains)

                catalogue_all = FilterCatalogParams()

                catalogue_all.AddCatalog(FilterCatalogParams.FilterCatalogs.ALL)

                args.append({"catalogue": FilterCatalog(catalogue_all)})

            if tanimoto_similarity is True and tanimoto_similarity_set is not None:

                funcs.append(tanimoto_similarity_metrics)

                args.append(

                    {"similarity_set": tanimoto_similarity_set, "clean_and_validate": True}

                )

            if molecular_weight is True:

                funcs.append(get_molecular_mass)

                args.append({})

            if largest_ring_system is True:

                funcs.append(get_largest_ring_system)

                args.append({})

            if logp is True:

                funcs.append(get_logp)

                args.append({})

            if qed_score is True:

                funcs.append(get_qed_common_versions)

                args.append({})

            mf = MoleculeFilter(funcs, mols, list_of_fx_arg_dicts=args, labels=labels)

            return mf.filter_results



        class MoleculeFilter(object):

            """

            Class to apply a series of filters to a list of molecules. The filters are applied in order and the results are stored in a pandas DataFrame. Additional arguments can be passed to the filter functions using the list_of_fx_arg_dicts argument.

            This argument should be a list of dictionaries with each dictionary containing the arguments for the corresponding filter function in the filter_fx list. Each filter function must accept a list of RDKit molecule objects as the first

            argument and the additional arguments as keyword arguments.

            """

            def __init__(

                self,

                filter_fx: List[Callable],

                molecules: List[Chem.rdchem.Mol],

                list_of_fx_arg_dicts: Optional[List[dict]] = None,

                labels: Optional[List[str]] = None,

            ) -> None:

                """

                Initialization function for the MoleculeFilter class

                Args:

                    filter_fx (List[Callable]): List of filter functions

                    molecules (List[Chem.rdchem.Mol]): List of RDKit molecule objects

                    list_of_fx_arg_dicts (Optional[List[dict]]): List of dictionaries containing the arguments for the filter functions

                    labels (Optional[List[str]]): List of labels for the

                Returns:

                    None

                """

                self.filter_fx = filter_fx

                self.list_of_fx_arg_dicts = list_of_fx_arg_dicts

                if self.list_of_fx_arg_dicts is not None:

                    if len(self.list_of_fx_arg_dicts) != len(self.filter_fx):

                        raise ValueError(

                            f"Length of list_of_fx_arg_dicts {len(self.list_of_fx_arg_dicts)} does not match the length of filter_fx {len(self.filter_fx)}. They should be the same length with a blank dict for a function with no new arguments to update."

                        )

                self.molecules = molecules

                if labels is None:

                    self.labels = [Chem.MolToSmiles(mol) for mol in molecules]

                else:

                    self.labels = labels

                self.filter_results = self._apply_filters()

            def _apply_filters(self) -> pd.DataFrame:

                """

                Apply the filters to the list of molecules

                Args:

                    None

                Returns:

                    pd.DataFrame: DataFrame containing the filter properties for the list of molecules

                """

                rets = pd.DataFrame([[ent] for ent in self.labels], columns=["label"])

                for fx, args in zip(self.filter_fx, self.list_of_fx_arg_dicts):

                    log.info(

                        f"{os.linesep}-----{os.linesep}Applying filter {fx.__name__}{os.linesep}-----"

                    )

                    tmp_df = fx(self.molecules, **args)

                    for col in tmp_df.columns:

                        rets[col] = tmp_df[col].copy()

                return rets

## Variables

```python3
log
```

## Functions


### conatains_pains

```python3
def conatains_pains(
    mols: List[rdkit.Chem.rdchem.Mol],
    invalid_value=nan,
    default_pains_only: bool = True,
    all_pains: bool = False,
    catalogue: Union[rdkit.Chem.rdfiltercatalog.FilterCatalog, rdkit.Chem.rdfiltercatalog.FilterCatalogParams, NoneType] = None,
    return_numpy: bool = False
) -> Union[numpy.ndarray, pandas.core.frame.DataFrame]
```

Check if a list of molecules contains PAINS (Pan Assay INterference compoundS) compounds or other substructures that are known to be a problem

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mols | List[Chem.rdchem.Mol] | List of RDKit molecule objects | None |
| invalid_value | float | Value to return if the PAINS check cannot be calculated | None |
| default_pains_only | bool | Use the default PAINS catalogue (PAINS_A, PAINS_B, PAINS_C) | None |
| all_pains | bool | Use all PAINS catalogues | None |
| catalogue | Union[FilterCatalog, FilterCatalogParams] | PAINS catalogue to use | None |
| return_numpy | bool | Return a numpy array | None |

**Returns:**

| Type | Description |
|---|---|
| np.ndarray | Boolean vector defining if the molecule contains PAINS or not |

??? example "View Source"
        def conatains_pains(

            mols: List[Chem.rdchem.Mol],

            invalid_value=np.nan,

            default_pains_only: bool = True,

            all_pains: bool = False,

            catalogue: Optional[Union[FilterCatalog, FilterCatalogParams]] = None,

            return_numpy: bool = False,

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Check if a list of molecules contains PAINS (Pan Assay INterference compoundS) compounds or other substructures that are known to be a problem

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objects

                invalid_value (float): Value to return if the PAINS check cannot be calculated

                default_pains_only (bool): Use the default PAINS catalogue (PAINS_A, PAINS_B, PAINS_C)

                all_pains (bool): Use all PAINS catalogues

                catalogue (Union[FilterCatalog, FilterCatalogParams]): PAINS catalogue to use

                return_numpy (bool): Return a numpy array

            Returns:

                np.ndarray: Boolean vector defining if the molecule contains PAINS or not

            """

            contains_pains = []

            if catalogue is not None:

                if isinstance(catalogue, FilterCatalogParams):

                    log.info(f"Using provided PAINS catalogue {str(catalogue)}")

                    catalogue = FilterCatalog(catalogue)

                elif isinstance(catalogue, FilterCatalog):

                    log.info("Using provided PAINS catalogue")

                else:

                    raise ValueError(

                        f"Catalogue type not recognised {type(catalogue)} should be one of FilterCatalog or FilterCatalogParams"

                    )

            elif default_pains_only is True:

                log.info("Using default PAINS catalogue (PAINS_A, PAINS_B, PAINS_C)")

                params = FilterCatalogParams()

                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)

                catalogue = FilterCatalog(params)

            elif all_pains is True:

                log.info("Using all PAINS catalogues")

                params_all = FilterCatalogParams()

                params_all.AddCatalog(FilterCatalogParams.FilterCatalogs.ALL)

                catalogue = FilterCatalog(params_all)

            for mol in mols:

                try:

                    contains_pains.append(catalogue.HasMatch(mol))

                except Exception as e:

                    contains_pains.append(invalid_value)

            return (

                pd.DataFrame(

                    np.array([[ent] for ent in contains_pains]), columns=["contains_pains"]

                )

                if return_numpy is False

                else np.array(contains_pains)

            )


### congreve_ro3

```python3
def congreve_ro3(
    mol: rdkit.Chem.rdchem.Mol,
    ro3_n_exceptions: int = 0
) -> int
```

Rule of 3

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mol | Chem.rdchem.Mol | RDKit molecule object | None |
| ro3_n_exceptions | int | Number of exceptions to the Rule of 3 | None |

**Returns:**

| Type | Description |
|---|---|
| bool | True if molecule passes the Rule of 3, False otherwise |

??? example "View Source"
        def congreve_ro3(mol: Chem.rdchem.Mol, ro3_n_exceptions: int = 0) -> int:

            """

            Rule of 3

            Args:

                mol (Chem.rdchem.Mol): RDKit molecule object

                ro3_n_exceptions (int): Number of exceptions to the Rule of 3

            Returns:

                bool: True if molecule passes the Rule of 3, False otherwise

            """

            molecular_weight = Descriptors.ExactMolWt(mol)

            logp = Descriptors.MolLogP(mol)

            h_bond_donor = Descriptors.NumHDonors(mol)

            h_bond_acceptors = Descriptors.NumHAcceptors(mol)

            rotatable_bonds = Descriptors.NumRotatableBonds(mol)

            ro3 = [

                molecular_weight <= 300,

                logp <= 3,

                h_bond_donor <= 3,

                h_bond_acceptors <= 3,

                rotatable_bonds <= 3,

            ]

            if sum(ro3) >= len(ro3) - ro3_n_exceptions:

                return 1

            else:

                return 0


### druglikeness

```python3
def druglikeness(
    mols,
    ro5: bool = True,
    ro5_n_exceptions: int = 0,
    ghose: bool = True,
    ghose_n_exceptions: int = 0,
    veber: bool = True,
    veber_n_exceptions: int = 0,
    ro3: bool = True,
    ro3_n_exceptions: int = 0,
    reos: bool = True,
    reos_n_exceptions: int = 0,
    qed: bool = False,
    qed_thresh: float = 0.5,
    return_numpy: bool = False
) -> Union[numpy.ndarray, pandas.core.frame.DataFrame]
```

Calculate druglikeness properties for a list of molecules. The following properties are calculated:

- Lipinski Rule of 5
- Ghose Filter
- Veber Filter
- Rule of 3
- REOS Filter
- QED (with a threshold)

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mols | List[Chem.rdchem.Mol] | List of RDKit molecule objects | None |
| ro5 | bool | Apply Lipinski Rule of 5 | None |
| ro5_n_exceptions | int | Number of exceptions to the Lipinski Rule of 5 | None |
| ghose | bool | Apply Ghose Filter | None |
| ghose_n_exceptions | int | Number of exceptions to the Ghose Filter | None |
| veber | bool | Apply Veber Filter | None |
| veber_n_exceptions | int | Number of exceptions to the Veber Filter | None |
| ro3 | bool | Apply Rule of 3 | None |
| ro3_n_exceptions | int | Number of exceptions to the Rule of 3 | None |
| reos | bool | Apply REOS Filter | None |
| reos_n_exceptions | int | Number of exceptions to the REOS Filter | None |
| qed | bool | Apply QED | None |
| qed_threshold | float | QED threshold | None |
| return_numpy | bool | Return a numpy array | None |

**Returns:**

| Type | Description |
|---|---|
| np.ndarray | Druglikeness properties for the list of molecules |

??? example "View Source"
        def druglikeness(

            mols,

            ro5: bool = True,

            ro5_n_exceptions: int = 0,

            ghose: bool = True,

            ghose_n_exceptions: int = 0,

            veber: bool = True,

            veber_n_exceptions: int = 0,

            ro3: bool = True,

            ro3_n_exceptions: int = 0,

            reos: bool = True,

            reos_n_exceptions: int = 0,

            qed: bool = False,

            qed_thresh: float = 0.5,

            return_numpy: bool = False,

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate druglikeness properties for a list of molecules. The following properties are calculated:

            - Lipinski Rule of 5

            - Ghose Filter

            - Veber Filter

            - Rule of 3

            - REOS Filter

            - QED (with a threshold)

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objects

                ro5 (bool): Apply Lipinski Rule of 5

                ro5_n_exceptions (int): Number of exceptions to the Lipinski Rule of 5

                ghose (bool): Apply Ghose Filter

                ghose_n_exceptions (int): Number of exceptions to the Ghose Filter

                veber (bool): Apply Veber Filter

                veber_n_exceptions (int): Number of exceptions to the Veber Filter

                ro3 (bool): Apply Rule of 3

                ro3_n_exceptions (int): Number of exceptions to the Rule of 3

                reos (bool): Apply REOS Filter

                reos_n_exceptions (int): Number of exceptions to the REOS Filter

                qed (bool): Apply QED

                qed_threshold (float): QED threshold

                return_numpy (bool): Return a numpy array

            Returns:

                np.ndarray: Druglikeness properties for the list of molecules

            """

            ret = np.zeros([len(mols), sum([ro5, ghose, veber, ro3, reos])], dtype=int)

            cols = []

            for ith, mol in enumerate(mols):

                column_index = 0

                if ro5 is True:

                    log.debug(f"Calculating Lipinski Rule of 5 for molecule {ith}")

                    ret[ith, column_index] = lipinski_ro5(mol, ro5_n_exceptions)

                    column_index += 1

                    if ith == 0:

                        cols.append("ro5")

                if ghose is True:

                    log.debug(f"Calculating Ghose Filter for molecule {ith}")

                    ret[ith, column_index] = ghose_ro2(mol, ghose_n_exceptions)

                    column_index += 1

                    if ith == 0:

                        cols.append("ghose")

                if veber is True:

                    log.debug(f"Calculating Veber Filter for molecule {ith}")

                    ret[ith, column_index] = veber_ro2(mol, veber_n_exceptions)

                    column_index += 1

                    if ith == 0:

                        cols.append("veber")

                if ro3 is True:

                    log.debug(f"Calculating Rule of 3 for molecule {ith}")

                    ret[ith, column_index] = congreve_ro3(mol, ro3_n_exceptions)

                    column_index += 1

                    if ith == 0:

                        cols.append("ro3")

                if reos is True:

                    log.debug(f"Calculating REOS Filter for molecule {ith}")

                    ret[ith, column_index] = reos_ro7(mol, reos_n_exceptions)

                    column_index += 1

                    if ith == 0:

                        cols.append("reos")

                if qed is True:

                    log.debug(f"Calculating QED for molecule {ith}")

                    ret[ith, column_index] = qed_threshold(mol, qed_thresh)

                    column_index += 1

                    if ith == 0:

                        cols.append("qed")

            log.debug(f"The raw return is: {ret}")

            if return_numpy is True:

                return ret

            else:

                return pd.DataFrame(ret, columns=cols)


### get_filter_properties

```python3
def get_filter_properties(
    data_df: pandas.core.frame.DataFrame,
    representation_column: str = 'smiles',
    representation_type: str = 'smiles',
    label_column: Optional[str] = None,
    training_data_csv_or_df: Union[str, pandas.core.frame.DataFrame, NoneType] = None,
    tanimoto_similarity_set: Union[str, pandas.core.frame.DataFrame, NoneType] = None,
    synthetic_accessibility: bool = True,
    druglike: bool = True,
    pains: bool = True,
    tanimoto_similarity: bool = True,
    molecular_weight: bool = True,
    largest_ring_system: bool = True,
    logp: bool = True,
    qed_score: bool = True
) -> pandas.core.frame.DataFrame
```

Get the filter properties for a DataFrame of molecules. The filter properties are:

- Synthetic accessibility score
- Druglikeness
- PAINS
- Tanimoto similarity to the training set
- Molecular weight
- Largest ring system
- LogP
- QED

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| data_df | pd.DataFrame | DataFrame containing the data | None |
| representation_column | str | Column name of the representation | None |
| representation_type | str | Representation type (smiles or inchi) | None |
| label_column | Optional[str] | Column name of the labels | None |
| training_data_csv_or_df | Optional[Union[str, pd.DataFrame]] | Training data to compare the Tanimoto similarity to | None |
| synthetic_accessibility | bool | Apply synthetic accessibility score | None |
| druglike | bool | Apply druglikeness filter | None |
| pains | bool | Apply PAINS filter | None |
| tanimoto_similarity | bool | Apply Tanimoto similarity filter | None |
| molecular_weight | bool | Apply molecular weight filter | None |
| largest_ring_system | bool | Apply largest ring system filter | None |
| logp | bool | Apply logP filter | None |
| qed_score | bool | Apply QED score filter | None |

**Returns:**

| Type | Description |
|---|---|
| pd.DataFrame | DataFrame containing the filter properties for the list of molecules |

**Raises:**

| Type | Description |
|---|---|
| ValueError | If the representation type is not recognised |

??? example "View Source"
        def get_filter_properties(

            data_df: pd.DataFrame,

            representation_column: str = "smiles",

            representation_type: str = "smiles",

            label_column: Optional[str] = None,

            training_data_csv_or_df: Optional[Union[str, pd.DataFrame]] = None,

            tanimoto_similarity_set: Optional[Union[str, pd.DataFrame]] = None,

            synthetic_accessibility: bool = True,

            druglike: bool = True,

            pains: bool = True,

            tanimoto_similarity: bool = True,

            molecular_weight: bool = True,

            largest_ring_system: bool = True,

            logp: bool = True,

            qed_score: bool = True,

        ) -> pd.DataFrame:

            """

            Get the filter properties for a DataFrame of molecules. The filter properties are:

            - Synthetic accessibility score

            - Druglikeness

            - PAINS

            - Tanimoto similarity to the training set

            - Molecular weight

            - Largest ring system

            - LogP

            - QED

            Args:

                data_df (pd.DataFrame): DataFrame containing the data

                representation_column (str): Column name of the representation

                representation_type (str): Representation type (smiles or inchi)

                label_column (Optional[str]): Column name of the labels

                training_data_csv_or_df (Optional[Union[str, pd.DataFrame]]): Training data to compare the Tanimoto similarity to

                synthetic_accessibility (bool): Apply synthetic accessibility score

                druglike (bool): Apply druglikeness filter

                pains (bool): Apply PAINS filter

                tanimoto_similarity (bool): Apply Tanimoto similarity filter

                molecular_weight (bool): Apply molecular weight filter

                largest_ring_system (bool): Apply largest ring system filter

                logp (bool): Apply logP filter

                qed_score (bool): Apply QED score filter

            Returns:

                pd.DataFrame: DataFrame containing the filter properties for the list of molecules

            Raises:

                ValueError: If the representation type is not recognised

            """

            df = data_df.copy()

            log.info(f"Cleaning and validating the {representation_type} representation")

            if representation_type == "smiles":

                cleaned_df = smilesmod.clean_and_validate_smiles(

                    df, smiles_column=representation_column

                )

                mols = [

                    Chem.MolFromSmiles(smi) for smi in cleaned_df[f"standardized_smiles"].values

                ]

            elif representation_type == "inchi":

                cleaned_df = inchimod.clean_and_validate_inchi(

                    df, inchi_column=representation_column

                )

                mols = [

                    Chem.MolFromSmiles(smi) for smi in cleaned_df[f"standardized_inchi"].values

                ]

            else:

                raise ValueError(

                    f"Representation type not recognised {representation_type}. Currently this should be on of the following: 'smiles' or 'inchi'"

                )

            if label_column is not None:

                labels = df[label_column].values

            else:

                labels = None

            funcs = []

            args = []

            if synthetic_accessibility is True:

                funcs.append(synthetic_accessibility_score)

                args.append({})

            if druglike is True:

                funcs.append(druglikeness)

                args.append({"qed": False})

            if pains is True:

                funcs.append(conatains_pains)

                catalogue_all = FilterCatalogParams()

                catalogue_all.AddCatalog(FilterCatalogParams.FilterCatalogs.ALL)

                args.append({"catalogue": FilterCatalog(catalogue_all)})

            if tanimoto_similarity is True and tanimoto_similarity_set is not None:

                funcs.append(tanimoto_similarity_metrics)

                args.append(

                    {"similarity_set": tanimoto_similarity_set, "clean_and_validate": True}

                )

            if molecular_weight is True:

                funcs.append(get_molecular_mass)

                args.append({})

            if largest_ring_system is True:

                funcs.append(get_largest_ring_system)

                args.append({})

            if logp is True:

                funcs.append(get_logp)

                args.append({})

            if qed_score is True:

                funcs.append(get_qed_common_versions)

                args.append({})

            mf = MoleculeFilter(funcs, mols, list_of_fx_arg_dicts=args, labels=labels)

            return mf.filter_results


### get_largest_ring_system

```python3
def get_largest_ring_system(
    mols: List[rdkit.Chem.rdchem.Mol],
    return_numpy: bool = False
) -> Union[numpy.ndarray, pandas.core.frame.DataFrame]
```

Calculate the largest ring system for a list of molecules

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mols | List[Chem.rdchem.Mol] | List of RDKit molecule objects | None |
| return_numpy | bool | Return a numpy array | None |

**Returns:**

| Type | Description |
|---|---|
| np.ndarray | Largest ring system for the list of molecules |

??? example "View Source"
        def get_largest_ring_system(

            mols: List[Chem.rdchem.Mol], return_numpy: bool = False

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate the largest ring system for a list of molecules

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objects

                return_numpy (bool): Return a numpy array

            Returns:

                np.ndarray: Largest ring system for the list of molecules

            """

            largest_ring_n_atoms = []

            for ith, mol in enumerate(mols):

                log.debug(f"Calculating largest ring system for molecule {ith} of {len(mols)}")

                try:

                    largest_ring_n_atoms.append(

                        max([len(a) for a in mol.GetRingInfo().AtomRings()])

                    )

                except ValueError:

                    largest_ring_n_atoms.append(0)

            log.debug(largest_ring_n_atoms)

            log.debug(len(largest_ring_n_atoms))

            if return_numpy is True:

                return np.array(largest_ring_n_atoms)

            else:

                return pd.DataFrame(

                    np.array([[ent] for ent in largest_ring_n_atoms]),

                    columns=["largest_ring_system"],

                )


### get_logp

```python3
def get_logp(
    mols: List[rdkit.Chem.rdchem.Mol],
    invalid_value=nan,
    return_numpy: bool = False
) -> Union[numpy.ndarray, pandas.core.frame.DataFrame]
```

Calculate the logP for a list of molecules

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mols | List[Chem.rdchem.Mol] | List of RDKit molecule objects | None |
| invalid_value | float | Value to return if the logP cannot be calculated | None |
| return_numpy | bool | Return a numpy array | None |

**Returns:**

| Type | Description |
|---|---|
| Union[np.ndarray, pd.DataFrame] | logP for the list of molecules |

??? example "View Source"
        def get_logp(

            mols: List[Chem.rdchem.Mol], invalid_value=np.nan, return_numpy: bool = False

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate the logP for a list of molecules

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objects

                invalid_value (float): Value to return if the logP cannot be calculated

                return_numpy (bool): Return a numpy array

            Returns:

                Union[np.ndarray, pd.DataFrame]: logP for the list of molecules

            """

            logp = []

            for mol in mols:

                try:

                    logp.append(Descriptors.MolLogP(mol))

                except Exception as e:

                    logp.append(invalid_value)

            if return_numpy is True:

                return np.array(logp)

            else:

                return pd.DataFrame(np.array([[ent] for ent in logp]), columns=["logp"])


### get_molecular_mass

```python3
def get_molecular_mass(
    mols: List[rdkit.Chem.rdchem.Mol],
    invalid_value=nan,
    return_numpy: bool = False
) -> Union[numpy.ndarray, pandas.core.frame.DataFrame]
```

Calculate the molecular mass for a list of molecules

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mols | List[Chem.rdchem.Mol] | List of RDKit molecule objects | None |
| invalid_value | float | Value to return if the molecular mass cannot be calculated | None |
| return_numpy | bool | Return a numpy array | None |

**Returns:**

| Type | Description |
|---|---|
| Union[np.ndarray, pd.DataFrame] | Molecular mass for the list of molecules |

??? example "View Source"
        def get_molecular_mass(

            mols: List[Chem.rdchem.Mol], invalid_value=np.nan, return_numpy: bool = False

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate the molecular mass for a list of molecules

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objects

                invalid_value (float): Value to return if the molecular mass cannot be calculated

                return_numpy (bool): Return a numpy array

            Returns:

                Union[np.ndarray, pd.DataFrame]: Molecular mass for the list of molecules

            """

            molecular_mass = []

            for mol in mols:

                try:

                    molecular_mass.append(Descriptors.ExactMolWt(mol))

                except Exception as e:

                    molecular_mass.append(invalid_value)

            if return_numpy is True:

                return np.array(molecular_mass)

            else:

                return pd.DataFrame(

                    np.array([[ent] for ent in molecular_mass]), columns=["molecular_mass"]

                )


### get_qed_common_versions

```python3
def get_qed_common_versions(
    mols: List[rdkit.Chem.rdchem.Mol],
    invalid_value=nan,
    return_numpy: bool = False
) -> Union[numpy.ndarray, pandas.core.frame.DataFrame]
```

Calculate QED scores for a list of molecules using the different versions of QED:

- QED weights max (i.e. the maximum weights used)
- QED weights mean (i.e. the original QED score)
- QED weights none (i.e. all weights are set to 1.0)

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mols | List[Chem.rdchem.Mol] | List of RDKit molecule objects | None |
| invalid_value | float | Value to return if the QED score cannot be calculated | None |
| return_numpy | bool | Return a numpy array | None |

**Returns:**

| Type | Description |
|---|---|
| pd.DataFrame | QED scores for the list of molecules |

??? example "View Source"
        def get_qed_common_versions(

            mols: List[Chem.rdchem.Mol], invalid_value=np.nan, return_numpy: bool = False

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate QED scores for a list of molecules using the different versions of QED:

            - QED weights max (i.e. the maximum weights used)

            - QED weights mean (i.e. the original QED score)

            - QED weights none (i.e. all weights are set to 1.0)

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objects

                invalid_value (float): Value to return if the QED score cannot be calculated

                return_numpy (bool): Return a numpy array

            Returns:

                pd.DataFrame: QED scores for the list of molecules

            """

            qed_scores = np.zeros([len(mols), 4], dtype=float)

            for ith, m in enumerate(mols):

                try:

                    qed_scores[ith, 0] = QED.weights_max(m)

                except Exception as e:

                    qed_scores[ith, 0] = invalid_value

                # This looks like it is the same as the original QED score QED.qed(m)

                try:

                    qed_scores[ith, 1] = QED.weights_mean(m)

                except Exception as e:

                    qed_scores[ith, 1] = invalid_value

                try:

                    qed_scores[ith, 2] = QED.weights_none(m)

                except Exception as e:

                    qed_scores[ith, 2] = invalid_value

                try:

                    qed_scores[ith, 3] = QED.qed(m)

                except Exception as e:

                    qed_scores[ith, 3] = invalid_value

            return (

                pd.DataFrame(

                    qed_scores, columns=["qed_max", "qed_mean", "qed_all_weights_one", "qed"]

                )

                if return_numpy is False

                else qed_scores

            )


### ghose_ro2

```python3
def ghose_ro2(
    mol: rdkit.Chem.rdchem.Mol,
    ghose_n_exceptions: int = 0
) -> int
```

Ghose Filter

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mol | Chem.rdchem.Mol | RDKit molecule object | None |
| ghose_n_exceptions | int | Number of exceptions to the Ghose Filter | None |

**Returns:**

| Type | Description |
|---|---|
| bool | True if molecule passes the Ghose Filter, False otherwise |

??? example "View Source"
        def ghose_ro2(mol: Chem.rdchem.Mol, ghose_n_exceptions: int = 0) -> int:

            """

            Ghose Filter

            Args:

                mol (Chem.rdchem.Mol): RDKit molecule object

                ghose_n_exceptions (int): Number of exceptions to the Ghose Filter

            Returns:

                bool: True if molecule passes the Ghose Filter, False otherwise

            """

            molecular_weight = Descriptors.ExactMolWt(mol)

            logp = Descriptors.MolLogP(mol)

            number_of_atoms = Chem.rdchem.Mol.GetNumAtoms(mol)

            molar_refractivity = Chem.Crippen.MolMR(mol)

            ghose = [

                molecular_weight >= 160,

                molecular_weight <= 480,

                logp >= -0.4,

                logp <= 5.6,

                number_of_atoms >= 20,

                number_of_atoms <= 70,

                molar_refractivity >= 40,

                molar_refractivity <= 130,

            ]

            if sum(ghose) >= len(ghose) - ghose_n_exceptions:

                return 1

            else:

                return 0


### lipinski_ro5

```python3
def lipinski_ro5(
    mol: rdkit.Chem.rdchem.Mol,
    ro5_n_exceptions: int = 0
) -> int
```

Lipinski Rule of 5

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mol | Chem.rdchem.Mol | RDKit molecule object | None |
| ro5_n_exceptions | int | Number of exceptions to the Lipinski Rule of 5 | None |

**Returns:**

| Type | Description |
|---|---|
| bool | True if molecule passes the Lipinski Rule of 5, False otherwise |

??? example "View Source"
        def lipinski_ro5(mol: Chem.rdchem.Mol, ro5_n_exceptions: int = 0) -> int:

            """

            Lipinski Rule of 5

            Args:

                mol (Chem.rdchem.Mol): RDKit molecule object

                ro5_n_exceptions (int): Number of exceptions to the Lipinski Rule of 5

            Returns:

                bool: True if molecule passes the Lipinski Rule of 5, False otherwise

            """

            molecular_weight = Descriptors.ExactMolWt(mol)

            logp = Descriptors.MolLogP(mol)

            h_bond_donor = Descriptors.NumHDonors(mol)

            h_bond_acceptors = Descriptors.NumHAcceptors(mol)

            rotatable_bonds = Descriptors.NumRotatableBonds(mol)

            ro5 = [

                molecular_weight <= 500,

                logp <= 5,

                h_bond_donor <= 5,

                h_bond_acceptors <= 10,

                rotatable_bonds <= 5,

            ]

            if sum(ro5) >= len(ro5) - ro5_n_exceptions:

                return 1

            else:

                return 0


### qed_scores

```python3
def qed_scores(
    mols: List[rdkit.Chem.rdchem.Mol],
    invalid_value=nan,
    weights: Tuple[float] = (0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95),
    qed_properties: Optional[namedtuple] = None,
    return_all_common_varients: bool = False,
    return_numpy: bool = False,
    append_weights_to_qed_column_names: bool = False
) -> Union[numpy.ndarray, pandas.core.frame.DataFrame]
```

Calculate QED scores for a list of molecules. See https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/QED.py

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mols | List[Chem.rdchem.Mol] | List of RDKit molecule objects | None |
| invalid_value | float | Value to return if the QED score cannot be calculated | None |
| weights | Tuple[float] | Weights for the QED properties. The default weights are (0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95) which<br>are from the RDKit documentation https://www.rdkit.org/docs/source/rdkit.Chem.QED.html#:~:text=%3DNone)-,%C2%B6,-Calculate%20the%20weighted accessed 10/12/24 | None |
| qed_properties | namedtuple | QED properties | None |
| return_all_common_varients | bool | Return all common varients of QED | None |
| return_numpy | bool | Return a numpy array | None |
| append_weights_to_qed_column_names | bool | Append the weights to the QED column names | None |

**Returns:**

| Type | Description |
|---|---|
| Union[np.ndarray, pd.DataFrame] | QED scores for the list of molecules |

??? example "View Source"
        def qed_scores(

            mols: List[Chem.rdchem.Mol],

            invalid_value=np.nan,

            weights: Tuple[float] = (0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95),

            qed_properties: Optional[namedtuple] = None,

            return_all_common_varients: bool = False,

            return_numpy: bool = False,

            append_weights_to_qed_column_names: bool = False,

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate QED scores for a list of molecules. See https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/QED.py

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objects

                invalid_value (float): Value to return if the QED score cannot be calculated

                weights (Tuple[float]): Weights for the QED properties. The default weights are (0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95) which

                  are from the RDKit documentation https://www.rdkit.org/docs/source/rdkit.Chem.QED.html#:~:text=%3DNone)-,%C2%B6,-Calculate%20the%20weighted accessed 10/12/24

                qed_properties (namedtuple): QED properties

                return_all_common_varients (bool): Return all common varients of QED

                return_numpy (bool): Return a numpy array

                append_weights_to_qed_column_names (bool): Append the weights to the QED column names

            Returns:

                Union[np.ndarray, pd.DataFrame]: QED scores for the list of molecules

            """

            if return_all_common_varients is True:

                qed_versions_df = get_qed_common_versions(mols, invalid_value)

                if return_numpy is True:

                    return qed_versions_df.values

                else:

                    return qed_versions_df

            qed_scores = np.zeros([len(mols)], dtype=float)

            for ith, m in enumerate(mols):

                try:

                    qed_scores[ith] = QED.qed(m, weights, qed_properties)

                except Exception as e:

                    qed_scores[ith] = invalid_value

            if return_numpy is True:

                return qed_scores

            else:

                if append_weights_to_qed_column_names is True:

                    return pd.DataFrame(

                        qed_scores,

                        columns=[

                            f"qed_weights_{'_'.join([str(weight) for weight in weights])}"

                        ],

                    )

                return pd.DataFrame(qed_scores, columns=["qed"])


### qed_threshold

```python3
def qed_threshold(
    mol: rdkit.Chem.rdchem.Mol,
    qed_threshold: float = 0.5
) -> float
```

Quantitative Estimate of Drug-likeness (QED)

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mol | Chem.rdchem.Mol | RDKit molecule object | None |
| qed_threshold | float | QED threshold | None |

**Returns:**

| Type | Description |
|---|---|
| float | QED score |

??? example "View Source"
        def qed_threshold(mol: Chem.rdchem.Mol, qed_threshold: float = 0.5) -> float:

            """

            Quantitative Estimate of Drug-likeness (QED)

            Args:

                mol (Chem.rdchem.Mol): RDKit molecule object

                qed_threshold (float): QED threshold

            Returns:

                float: QED score

            """

            qed_score = QED.qed(mol)

            if qed_score >= qed_threshold:

                return 1

            else:

                return 0


### reos_ro7

```python3
def reos_ro7(
    mol: rdkit.Chem.rdchem.Mol,
    reos_n_exceptions: int = 0
) -> int
```

REOS Filter

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mol | Chem.rdchem.Mol | RDKit molecule object | None |
| reos_n_exceptions | int | Number of exceptions to the REOS Filter | None |

**Returns:**

| Type | Description |
|---|---|
| bool | True if molecule passes the REOS Filter, False otherwise |

??? example "View Source"
        def reos_ro7(mol: Chem.rdchem.Mol, reos_n_exceptions: int = 0) -> int:

            """

            REOS Filter

            Args:

                mol (Chem.rdchem.Mol): RDKit molecule object

                reos_n_exceptions (int): Number of exceptions to the REOS Filter

            Returns:

                bool: True if molecule passes the REOS Filter, False otherwise

            """

            molecular_weight = Descriptors.ExactMolWt(mol)

            logp = Descriptors.MolLogP(mol)

            h_bond_donor = Descriptors.NumHDonors(mol)

            h_bond_acceptors = Descriptors.NumHAcceptors(mol)

            formal_charge = Chem.rdmolops.GetFormalCharge(mol)

            rotatable_bonds = Descriptors.NumRotatableBonds(mol)

            heavy_atoms = Chem.rdchem.Mol.GetNumHeavyAtoms(mol)

            reos = [

                200 <= molecular_weight <= 500,

                -5 <= logp <= 5,

                0 <= h_bond_donor <= 5,

                0 <= h_bond_acceptors <= 10,

                -2 <= formal_charge <= 2,

                0 <= rotatable_bonds <= 8,

                15 <= heavy_atoms <= 50,

            ]

            if sum(reos) >= len(reos) - reos_n_exceptions:

                return 1

            else:

                return 0


### synthetic_accessibility_score

```python3
def synthetic_accessibility_score(
    mols: List[rdkit.Chem.rdchem.Mol],
    invalid_value=nan,
    return_numpy: bool = False
) -> Union[numpy.ndarray, pandas.core.frame.DataFrame]
```

Calculate the synthetic accessibility score for a list of molecules

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mols | List[Chem.rdchem.Mol] | List of RDKit molecule objects | None |
| invalid_value | float | Value to return if the synthetic accessibility score cannot be calculated | None |
| return_numpy | bool | Return a numpy array | None |

**Returns:**

| Type | Description |
|---|---|
| np.ndarray | Synthetic accessibility score for the list of molecules |

??? example "View Source"
        def synthetic_accessibility_score(

            mols: List[Chem.rdchem.Mol], invalid_value=np.nan, return_numpy: bool = False

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate the synthetic accessibility score for a list of molecules

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objects

                invalid_value (float): Value to return if the synthetic accessibility score cannot be calculated

                return_numpy (bool): Return a numpy array

            Returns:

                np.ndarray: Synthetic accessibility score for the list of molecules

            """

            syn_acc_score = []

            for mol in mols:

                try:

                    syn_acc_score.append(sascorer.calculateScore(mol))

                except ZeroDivisionError:

                    syn_acc_score.append(invalid_value)

            return (

                pd.DataFrame(

                    np.array([[ent] for ent in syn_acc_score]),

                    columns=["synthetic_accessibility_score"],

                )

                if return_numpy is False

                else np.array(syn_acc_score)

            )


### tanimoto_bulk_similarity_metrics

```python3
def tanimoto_bulk_similarity_metrics(
    mols: List[rdkit.Chem.rdchem.Mol],
    similarity_set: Union[str, pandas.core.frame.DataFrame, NoneType] = None,
    smiles_col: Optional[str] = 'smiles',
    return_numpy: bool = False,
    clean_and_validate: bool = True
) -> Union[numpy.ndarray, pandas.core.frame.DataFrame]
```

Calculate Tanimoto similarity metrics for a list of molecules

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mols | List[Chem.rdchem.Mol] | List of RDKit molecule objectsd | None |
| similarity_set | Union[str, pd.DataFrame] | Similarity set to compare the molecules to | None |
| smiles_col | str | Column name of the SMILES in the similarity set | None |
| return_numpy | bool | Return a numpy array | None |

**Returns:**

| Type | Description |
|---|---|
| Union[np.ndarray, pd.DataFrame] | Tanimoto similarity metrics for the list of molecules |

??? example "View Source"
        def tanimoto_bulk_similarity_metrics(

            mols: List[Chem.rdchem.Mol],

            similarity_set: Optional[Union[str, pd.DataFrame]] = None,

            smiles_col: Optional[str] = "smiles",

            return_numpy: bool = False,

            clean_and_validate: bool = True,

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate Tanimoto similarity metrics for a list of molecules

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objectsd

                similarity_set (Union[str, pd.DataFrame]): Similarity set to compare the molecules to

                smiles_col (str): Column name of the SMILES in the similarity set

                return_numpy (bool): Return a numpy array

            Returns:

                Union[np.ndarray, pd.DataFrame]: Tanimoto similarity metrics for the list of molecules

            """

            fpgen = AllChem.GetMorganGenerator(radius=2)

            std_smiles_col_name = "standardized_smiles"

            log.info("Calculating Tanimoto similarity metrics stats over set of molecules")

            log.info("prepare the similarity set for comparison ")

            if isinstance(similarity_set, str):

                try:

                    similarity_set = pd.read_csv(similarity_set)

                    if clean_and_validate is True:

                        log.info("Cleaning and validating the similarity set")

                        similarity_set = smilesmod.clean_and_validate_smiles(

                            similarity_set,

                            smiles_column=smiles_col,

                            return_selfies=False,

                            return_inchi=False,

                            return_inchikey=False,

                        )

                    else:

                        log.info(

                            "Using provided similarity set without cleaning and validation"

                        )

                        std_smiles_col_name = smiles_col

                    rmols = [

                        smilesmod.smiles_to_molecule(s)

                        for s in similarity_set[std_smiles_col_name].values

                    ]

                except Exception as e:

                    raise ValueError(f"Similarity set file not found {similarity_set}")

            elif isinstance(similarity_set, pd.DataFrame):

                log.info("Using provided similarity set")

                if clean_and_validate is True:

                    log.info("Cleaning and validating the similarity set")

                    similarity_set = smilesmod.clean_and_validate_smiles(

                        similarity_set,

                        smiles_column=smiles_col,

                        return_selfies=False,

                        return_inchi=False,

                        return_inchikey=False,

                    )

                else:

                    log.info("Using provided similarity set without cleaning and validation")

                    std_smiles_col_name = smiles_col

                rmols = [

                    smilesmod.smiles_to_molecule(s)

                    for s in similarity_set[std_smiles_col_name].values

                ]

            else:

                raise ValueError(

                    f"Similarity set type not recognised {type(similarity_set)} should be one of str (file path) or pd.DataFrame"

                )

            log.info("Calculating Tanimoto similarity metrics")

            reference_set_fingerprints = [fpgen.GetFingerprint(m) for m in rmols]

            scores = np.zeros((len(mols), 7))

            for ith, gen_mol in enumerate(mols):

                fp_test = fpgen.GetFingerprint(gen_mol)

                score = BulkTanimotoSimilarity(fp_test, reference_set_fingerprints)

                scores[ith][0] = ith

                scores[ith][1] = np.min(score)

                scores[ith][2] = np.max(score)

                scores[ith][3] = np.mean(score)

                scores[ith][4] = np.std(score)

                scores[ith][5] = np.median(score)

                quart75, quart25 = np.percentile(score, [75, 25])

                scores[ith][6] = quart75 - quart25

            if return_numpy is True:

                return scores

            else:

                return pd.DataFrame(

                    scores,

                    columns=[

                        "id",

                        "tanimoto_sim_min",

                        "tanimoto_sim_max",

                        "tanimoto_sim_mean",

                        "tanimoto_sim_std",

                        "tanimoto_sim_median",

                        "tanimoto_sim_iqr",

                    ],

                )


### tanimoto_similarity_metrics

```python3
def tanimoto_similarity_metrics(
    mols: List[rdkit.Chem.rdchem.Mol],
    similarity_set: Union[str, pandas.core.frame.DataFrame, NoneType] = None,
    smiles_col: Optional[str] = 'smiles',
    ref_mol: Union[str, rdkit.Chem.rdchem.Mol, NoneType] = None,
    return_numpy: bool = False,
    clean_and_validate: bool = True
) -> Union[numpy.ndarray, pandas.core.frame.DataFrame]
```

Calculate Tanimoto similarity metrics for a list of molecules

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mols | List[Chem.rdchem.Mol] | List of RDKit molecule objectsd | None |
| similarity_set | Union[str, pd.DataFrame] | Similarity set to compare the molecules to | None |
| smiles_col | str | Column name of the SMILES in the similarity set | None |
| ref_mol | Union[str, Chem.rdchem.Mol] | Reference molecule to compare the molecules to | None |
| return_numpy | bool | Return a numpy array | None |
| clean_and_validate | bool | Clean and validate the similarity set | None |

**Returns:**

| Type | Description |
|---|---|
| Union[np.ndarray, pd.DataFrame] | Tanimoto similarity metrics for the list of molecules |

??? example "View Source"
        def tanimoto_similarity_metrics(

            mols: List[Chem.rdchem.Mol],

            similarity_set: Optional[Union[str, pd.DataFrame]] = None,

            smiles_col: Optional[str] = "smiles",

            ref_mol: Optional[Union[str, Chem.rdchem.Mol]] = None,

            return_numpy: bool = False,

            clean_and_validate: bool = True,

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate Tanimoto similarity metrics for a list of molecules

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objectsd

                similarity_set (Union[str, pd.DataFrame]): Similarity set to compare the molecules to

                smiles_col (str): Column name of the SMILES in the similarity set

                ref_mol (Union[str, Chem.rdchem.Mol]): Reference molecule to compare the molecules to

                return_numpy (bool): Return a numpy array

                clean_and_validate (bool): Clean and validate the similarity set

            Returns:

                Union[np.ndarray, pd.DataFrame]: Tanimoto similarity metrics for the list of molecules

            """

            if ref_mol is not None:

                return tanimoto_single_similarity_metrics(mols, ref_mol, return_numpy)

            if similarity_set is not None:

                return tanimoto_bulk_similarity_metrics(

                    mols, similarity_set, smiles_col, return_numpy, clean_and_validate

                )


### tanimoto_single_similarity_metrics

```python3
def tanimoto_single_similarity_metrics(
    mols: List[rdkit.Chem.rdchem.Mol],
    ref_mol: Union[str, rdkit.Chem.rdchem.Mol, NoneType] = None,
    return_numpy: bool = False
) -> Union[numpy.ndarray, pandas.core.frame.DataFrame]
```

Calculate Tanimoto similarity metrics for a list of molecules

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mols | List[Chem.rdchem.Mol] | List of RDKit molecule objectsd | None |
| ref_mol | Union[str, Chem.rdchem.Mol] | Reference molecule to compare the molecules to | None |
| return_numpy | bool | Return a numpy array | None |

**Returns:**

| Type | Description |
|---|---|
| Union[np.ndarray, pd.DataFrame] | Tanimoto similarity metrics for the list of molecules |

??? example "View Source"
        def tanimoto_single_similarity_metrics(

            mols: List[Chem.rdchem.Mol],

            ref_mol: Optional[Union[str, Chem.rdchem.Mol]] = None,

            return_numpy: bool = False,

        ) -> Union[np.ndarray, pd.DataFrame]:

            """

            Calculate Tanimoto similarity metrics for a list of molecules

            Args:

                mols (List[Chem.rdchem.Mol]): List of RDKit molecule objectsd

                ref_mol (Union[str, Chem.rdchem.Mol]): Reference molecule to compare the molecules to

                return_numpy (bool): Return a numpy array

            Returns:

                Union[np.ndarray, pd.DataFrame]: Tanimoto similarity metrics for the list of molecules

            """

            fpgen = AllChem.GetMorganGenerator(radius=2)

            log.info("Calculating Tanimoto similarity metric against one molecule")

            if isinstance(ref_mol, str):

                if not ref_mol.startswith("InChI="):

                    ref_mol = Chem.MolFromSmiles(ref_mol)

                elif ref_mol.startswith("InChI="):

                    ref_mol = Chem.MolFromInchi(ref_mol)

                else:

                    raise ValueError(

                        f"Reference molecule type not recognised {type(ref_mol)} should be one of str (SMILES or InChI) or Chem.rdchem.Mol"

                    )

            elif isinstance(ref_mol, Chem.rdchem.Mol):

                log.info("Using provided reference molecule")

            reference_fingerprints = fpgen.GetFingerprint(ref_mol)

            scores = np.zeros((len(mols), 1))

            for ith, gen_mol in enumerate(mols):

                fp_test = fpgen.GetFingerprint(gen_mol)

                scores[ith] = TanimotoSimilarity(fp_test, reference_fingerprints)

            if return_numpy is True:

                return scores

            else:

                return pd.DataFrame(scores, columns=["tanimoto_sim"])


### veber_ro2

```python3
def veber_ro2(
    mol: rdkit.Chem.rdchem.Mol,
    veber_n_exceptions: int = 0
) -> int
```

Veber Filter

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| mol | Chem.rdchem.Mol | RDKit molecule object | None |
| veber_n_exceptions | int | Number of exceptions to the Veber Filter | None |

**Returns:**

| Type | Description |
|---|---|
| bool | True if molecule passes the Veber Filter, False otherwise |

??? example "View Source"
        def veber_ro2(mol: Chem.rdchem.Mol, veber_n_exceptions: int = 0) -> int:

            """

            Veber Filter

            Args:

                mol (Chem.rdchem.Mol): RDKit molecule object

                veber_n_exceptions (int): Number of exceptions to the Veber Filter

            Returns:

                bool: True if molecule passes the Veber Filter, False otherwise

            """

            rotatable_bonds = Descriptors.NumRotatableBonds(mol)

            topological_surface_area_mapping = Chem.QED.properties(mol).PSA

            veber = [rotatable_bonds <= 10, topological_surface_area_mapping <= 140]

            if sum(veber) >= len(veber) - veber_n_exceptions:

                return 1

            else:

                return 0

## Classes

### MoleculeFilter

```python3
class MoleculeFilter(
    filter_fx: List[Callable],
    molecules: List[rdkit.Chem.rdchem.Mol],
    list_of_fx_arg_dicts: Optional[List[dict]] = None,
    labels: Optional[List[str]] = None
)
```

Class to apply a series of filters to a list of molecules. The filters are applied in order and the results are stored in a pandas DataFrame. Additional arguments can be passed to the filter functions using the list_of_fx_arg_dicts argument.

This argument should be a list of dictionaries with each dictionary containing the arguments for the corresponding filter function in the filter_fx list. Each filter function must accept a list of RDKit molecule objects as the first
argument and the additional arguments as keyword arguments.
