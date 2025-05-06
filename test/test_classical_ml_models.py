#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module of unit test
"""

import logging

import pandas as pd
import pytest
from chemutil import classical_ml_model_build, classical_ml_models, featurization
from sklearn.preprocessing import MinMaxScaler

log = logging.getLogger(__name__)


number_of_defualt_models = 17
number_of_defualt_linear_models = 3
number_of_defualt_kernel_models = 4
number_of_defualt_bayesian_models = 3
number_of_defualt_ensemble_models = 6
number_of_defualt_neural_network_models = 1


@pytest.fixture
def pandas_dataframe() -> pd.DataFrame:
    """
    pandas_dataframe fixture holding the default input dataframe

    Returns:
        pd.DataFrame - pandas data frame
    """
    test_file = pd.DataFrame(
        [
            ["c1ccccc1", "benzene", 0.5, 1.0, 0.25, 1.2],
            ["C1CCCC1C(N)C", "1-cyclopentylethanamine", 0.9, 0.1, 1.2, 0.9],
            ["C1CCCC1C(=O)C", "1-cyclopentylethanone", 0.75, 0.05, 1.2, 0.9],
            ["C1CCCC1C(O)C", "1-cyclopentylethanol", 0.95, 0.12, 1.2, 0.9],
            ["C1CCCCC1C(N)C", "1-cyclohexylethanamine", 0.95, 0.15, 1.22, 0.95],
            ["C1CCCCC1C(=O)C", "1-cyclohexylethanone", 0.79, 0.02, 1.24, 0.97],
            ["C1CCCCC1C(O)C", "1-cyclohexylethanol", 1.1, 1.2, 1.4, 0.95],
            ["NCc1ccccc1", "benzylamine", 1.2, 0.02, 2.2, 0.75],
            ["C", "methane", -1.2, 0.01, 0.02, -10.0],
            ["CC", "ethane", -1.0, 0.2, 0.07, -10.2],
            ["CCC", "propane", -1.0, -0.4, 0.1, -10.7],
            ["CCCC", "butane", -0.7, -0.9, 0.2, -11.0],
        ],
        columns=["smiles", "names", "bind_target_0", "bind_target_1", "tox", "sol"],
    )
    return test_file


def test_get_all_default_models():
    """
    Test the codes can get all of the default models
    """
    all_default_models = classical_ml_models.get_models()

    assert len(all_default_models) == number_of_defualt_models


def test_get_linear_default_models():
    """
    Test the codes can get all of the linear models
    """
    models = classical_ml_models.linear_models()

    assert len(models) == number_of_defualt_linear_models


def test_get_kernel_default_models():
    """
    Test the codes can get all of the kernel models
    """
    models = classical_ml_models.kernel_models()

    assert len(models) == number_of_defualt_kernel_models


def test_get_bayesian_default_models():
    """
    Test the codes can get all of the Bayesian models
    """
    models = classical_ml_models.bayesian_models()

    assert len(models) == number_of_defualt_bayesian_models


def test_get_ensemble_default_models():
    """
    Test the codes can get all of the ensemble models
    """
    models = classical_ml_models.ensemble_models()

    assert len(models) == number_of_defualt_ensemble_models


def test_get_neural_network_default_models():
    """
    Test the codes can get all of the Bayesian models
    """
    models = classical_ml_models.neural_network_models()

    assert len(models) == number_of_defualt_neural_network_models


def test_model_set_up(pandas_dataframe):
    """
    Test the set up of the train test split class
    """

    data_df = featurization.get_count_ecfp(
        data_df=pandas_dataframe, smiles_column="smiles", return_df=True
    )
    linear_regression_models = classical_ml_models.linear_models(
        lasso=False, lars=False
    )[0]

    model = classical_ml_model_build.RXTrainTestGridSearchSklearnRegresssor(
        data_df,
        model=linear_regression_models.model,
        model_name=linear_regression_models.name,
    )

    model.build_train_test_validate_splits(
        smiles_column="smiles",
        task_columns=["bind_target_0", "bind_target_1"],
        feature_column_regex="ecfp_count_bit_",
        splitter_type="mw",
        train=0.8,
        test=0.2,
        holdout=None,
    )

    assert model.X_train.shape == (9, 2048)
    assert model.y_train.shape == (9, 2)

    assert model.X_test.shape == (3, 2048)
    assert model.y_test.shape == (3, 2)

    assert model.model_name == "LinearRegression"


def test_data_splits_produce_the_same_splits_on_repeat_calls(
    pandas_dataframe, repeated_runs_over_data=5
):
    """
    Test that the data splits are consistent on repeated calls to teh data splitters

    Args:
        pandas_dataframe (pd.DataFrame): data set to split
    """

    models = classical_ml_models.get_models()
    ids = []
    for i in range(repeated_runs_over_data):
        model = classical_ml_model_build.RXTrainTestGridSearchSklearnRegresssor(
            pandas_dataframe, model=models[i], model_name=models[i].name
        )

        model.build_train_test_validate_splits(
            smiles_column="smiles",
            task_columns=["bind_target_0"],
            feature_column_regex="ecfp_count_bit_",
            splitter_type="fp",
            train=0.8,
            test=0.2,
            holdout=None,
        )

        ids.append(model.test.ids)

    for i in range(repeated_runs_over_data):
        assert all(ids[0] == ids[i])


def test_linear_regression(pandas_dataframe):
    """
    Test the training of a linear regression model this should be fairly deterministic across OSs
    The assertion is made against the RMSE for the data and checked to 2 decimal places
    """

    data_df = featurization.get_count_ecfp(
        data_df=pandas_dataframe, smiles_column="smiles", return_df=True
    )
    linear_regression_models = classical_ml_models.linear_models(
        lasso=False, lars=False
    )[0]

    model = classical_ml_model_build.RXTrainTestGridSearchSklearnRegresssor(
        data_df,
        model=linear_regression_models.model,
        model_name=linear_regression_models.name,
    )

    model.build_train_test_validate_splits(
        smiles_column="smiles",
        task_columns=["bind_target_0", "bind_target_1"],
        feature_column_regex="ecfp_count_bit_",
        splitter_type="mw",
        train=0.8,
        test=0.2,
        holdout=None,
    )

    model.build_pipline_list(feature_scaling=MinMaxScaler())
    model.build_parameter_grid(
        model_param_grid=linear_regression_models.default_param_grid
    )

    model.make_pipeline()

    grid_search = model.optimize_model(
        scoring=("neg_root_mean_squared_error", "neg_mean_absolute_percentage_error")
    )

    assert abs(grid_search.best_score_ - -0.60) < 1e-2
