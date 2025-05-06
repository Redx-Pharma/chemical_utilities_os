#!/usr/bin.env python3
# -*- coding: utf-8 -*-

"""
Module for deepchem model building
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import lightgbm
import numpy as np
import sklearn
import xgboost
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    Lars,
    Lasso,
    LinearRegression,
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from xgboost import XGBRegressor

from chemutil import helpers

log = logging.getLogger(__name__)


# Data class to contain an sklearn regressor model. The class allows for a default hyper parameter grid to trained over or for a cutom one to be added without loosing the default.
@dataclass
class skmodel:
    name: str
    model: Callable
    default_param_grid: dict
    custom_param_grid: dict
    multi_output_regressor: bool
    more_info: Optional[str] = None
    sklearn_version: str = sklearn.__version__
    date: str = datetime.now()


def get_models(
    linear: bool = True,
    kernel: bool = True,
    bayesian: bool = True,
    ensemble: bool = True,
    neural_network: bool = True,
    n_features: Optional[int] = None,
    n_targets: Optional[int] = 1,
    prepend_parameter_keys_with_model: bool = False,
    multi_output_only: bool = False,
    prepend_with: str = "model__",
    **kwargs,
) -> List[skmodel]:
    """
    Function to return a list of models to train over and trial

    Args:
        linear (bool, optional): Return linear models. Defaults to True.
        kernel (bool, optional): Return kernel models. Defaults to True.
        bayesian (bool, optional): Return Bayesian models. Defaults to True.
        ensemble (bool, optional): Return ensemble models. Defaults to True.
        neural_network (bool, optional): Return neural network models. Defaults to True.
        n_features (Optional[int], optional): The number of input features. Used in Bayesian model Gaussian process to allow the use for anisotropic kernels. Defaults to None.
        n_targets (Optional[int], optional): The number of targets. Used in Bayesian model Gaussian process to set the expected number of outputs. Defaults to 1.

    Returns:
        List: List of sklearn models wrapped in a custom dataclass with meta data
    """

    models = []

    if linear is True:
        models = models + linear_models(**kwargs)

    if kernel is True:
        models = models + kernel_models(**kwargs)

    if bayesian is True:
        models = models + bayesian_models(
            n_targets=n_targets, n_features=n_features, **kwargs
        )

    if ensemble is True:
        models = models + ensemble_models(**kwargs)

    if neural_network is True:
        models = models + neural_network_models(n_input_features=n_features, **kwargs)

    if prepend_parameter_keys_with_model is True:
        for m in models:
            m.default_param_grid = {
                f"{prepend_with}{k}": v for k, v in m.default_param_grid.items()
            }
            m.custom_param_grid = {
                f"{prepend_with}{k}": v for k, v in m.custom_param_grid.items()
            }

    if multi_output_only is True:
        models = [m for m in models if m.multi_output_regressor is True]

    return models


def linear_models(
    lr: bool = True, lasso: bool = True, lars: bool = True, **kwargs
) -> List[skmodel]:
    """
    Function to build a default set of linear models and parameter grids to optimize the models over

    Returns:
        List[skmodel]: List of linear models and meta data
    """
    models = []

    if lr is True:
        # Linear regression
        models.append(
            skmodel(
                type(LinearRegression()).__name__,
                LinearRegression(n_jobs=-1),
                {},
                {},
                True,
                "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html",
                sklearn.__version__,
                datetime.now(),
            )
        )

    if lasso is True:
        # Lasso - tresting selection as a hyper-parameter as random selection can significantly increase convergence time so may enable models to converge that wouldn't otherwise
        models.append(
            skmodel(
                type(Lasso()).__name__,
                Lasso(random_state=helpers.random_seed),
                {"alpha": [0.5, 1.0, 1.5], "selection": ["cyclic", "random"]},
                {},
                True,
                "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html",
                sklearn.__version__,
                datetime.now(),
            )
        )

    if lars is True:
        # Lars
        models.append(
            skmodel(
                type(Lars()).__name__,
                Lars(random_state=helpers.random_seed),
                {"n_nonzero_coefs": [100, 500, np.inf], "jitter": [None, 0.05, 0.1]},
                {},
                True,
                "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html",
                sklearn.__version__,
                datetime.now(),
            )
        )

    return models


def kernel_models(
    kr: bool = True, svr: bool = True, svm: bool = False, **kwargs
) -> List[skmodel]:
    """
    Function to build a default set of kernel models and parameter grids to optimize the models over

    Returns:
        List[skmodel]: List of kernel models and meta data
    """

    models = []

    if kr is True:
        # Pair wise kernel metric strings: ‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’, ‘laplacian’, ‘sigmoid’, ‘cosine’
        # kernel ridge
        models.append(
            skmodel(
                type(KernelRidge()).__name__,
                KernelRidge(),
                {
                    "kernel": ["rbf", "linear"],
                    "alpha": [0.5, 1.0, 1.5],
                    "gamma": [None, 0.5, 1.5],
                },
                {},
                True,
                "https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html",
                sklearn.__version__,
                datetime.now(),
            )
        )

        # kernel ridge - polynomical kernel has the degree parameter so we separate the model
        models.append(
            skmodel(
                type(KernelRidge()).__name__ + "poly_kernel",
                KernelRidge(),
                {
                    "kernel": ["poly"],
                    "alpha": [0.5, 1.0, 1.5],
                    "degree": [2, 3, 4, 5],
                    "gamma": [None, 0.5, 1.5],
                },
                {},
                True,
                "https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html",
                sklearn.__version__,
                datetime.now(),
            )
        )

    if svr is True or svm is True:
        # Support Vector Regression (SVR)
        models.append(
            skmodel(
                type(SVR()).__name__,
                SVR(max_iter=10000),
                {
                    "kernel": ["rbf", "linear"],
                    "C": [0.9, 1.0, 1.1],
                    "epsilon": [0.05, 0.1, 0.2],
                    "gamma": [None, 0.5, 1.5],
                },
                {},
                False,
                "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html",
                sklearn.__version__,
                datetime.now(),
            )
        )

        # Support Vector Regression (SVR) - polynomical kernel has the degree parameter so we separate the model
        models.append(
            skmodel(
                type(SVR()).__name__ + "poly_kernel",
                SVR(max_iter=10000),
                {
                    "kernel": ["poly"],
                    "C": [0.9, 1.0, 1.1],
                    "epsilon": [0.05, 0.1, 0.2],
                    "degree": [2, 3, 4, 5],
                    "gamma": [None, 0.5, 1.5],
                },
                {},
                False,
                "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html",
                sklearn.__version__,
                datetime.now(),
            )
        )

    return models


def bayesian_models(
    br: bool = True,
    ard: bool = True,
    gp: bool = True,
    n_targets: int = 1,
    n_features: Optional[int] = None,
    n_restarts_optimizer: int = 5,
    **kwargs,
) -> List[skmodel]:
    """
    Function to build a default set of Bayesian models and parameter grids to optimize the models over

    Returns:
        List[skmodel]: List of Bayesian models and meta data
    """

    models = []

    if br is True:
        # BayesianRidge - Note this is in effect a linear model with a probablistic view point i.e. we get a mean and a std as model is ineffect covering a range of predictive functions
        # https://scikit-learn.org/stable/auto_examples/linear_model/plot_bayesian_ridge_curvefit.html#sphx-glr-auto-examples-linear-model-plot-bayesian-ridge-curvefit-py
        models.append(
            skmodel(
                type(BayesianRidge()).__name__,
                BayesianRidge(),
                {
                    "alpha_init": [None, 0.1, 0.5, 1.0, 1.5, 2.0],
                    "lambda_init": [0.01, 0.1, 0.5, 1.0, 1.5, 2.0],
                },
                {},
                False,
                "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html",
                sklearn.__version__,
                datetime.now(),
            )
        )

    # Automatic Relevance Determination (Relevance Vector Machine)
    if ard is True:
        models.append(
            skmodel(
                type(ARDRegression()).__name__,
                ARDRegression(),
                {
                    "alpha_1": [1.0e-8, 1.0e-10, 1.0e-4],
                    "alpha_2": [1.0e-8, 1.0e-10, 1.0e-4],
                    "lambda_1": [1.0e-8, 1.0e-10, 1.0e-4],
                    "lambda_2": [1.0e-8, 1.0e-10, 1.0e-4],
                },
                {},
                False,
                "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression",
                sklearn.__version__,
                datetime.now(),
            )
        )

    if gp is True:
        # Gaussian process
        # If we know the number of features we can use an anisotropic kernel amd an isotropic kernel otherwise we can only use an isotropic kernel
        if n_features is not None:
            ls = np.array([1.0] * n_features)
            # NOTE: None uses the default Constant kernel(1.0) * RBF(length_scale=1.0) with fixed length scales. The last kernel allows length scale optimization
            # using the RBF kernel
            kernels = [
                None,
                Matern(length_scale=ls, nu=1.5),
                Matern(length_scale=ls, nu=2.5),
                Matern(length_scale=ls, nu=0.5),
                RBF(length_scale=ls),
                Matern(length_scale=1.0, nu=1.5),
                Matern(length_scale=1.0, nu=2.5),
                Matern(length_scale=1.0, nu=0.5),
                RBF(length_scale=1.0),
            ]
        else:
            kernels = [
                None,
                Matern(length_scale=1.0, nu=1.5),
                Matern(length_scale=1.0, nu=2.5),
                Matern(length_scale=1.0, nu=0.5),
                RBF(length_scale=1.0),
            ]

        models.append(
            skmodel(
                type(GaussianProcessRegressor()).__name__,
                GaussianProcessRegressor(
                    random_state=helpers.random_seed,
                    n_restarts_optimizer=n_restarts_optimizer,
                    n_targets=n_targets,
                ),
                {"kernel": kernels},
                {},
                True,
                "https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html",
                sklearn.__version__,
                datetime.now(),
            )
        )

    return models


def ensemble_models(
    adb: bool = True,
    rf: bool = True,
    et: bool = True,
    hgb: bool = True,
    xgb: bool = True,
    lgbm: bool = True,
    **kwargs,
) -> List[skmodel]:
    """
    Function to build a default set of ensemble models and parameter grids to optimize the models over

    Returns:
        List[skmodel]: List of ensemble models and meta data
    """

    models = []

    if rf is True:
        # Random forest
        models.append(
            skmodel(
                type(RandomForestRegressor()).__name__,
                RandomForestRegressor(
                    random_state=helpers.random_seed, n_jobs=-1, oob_score=True
                ),
                {
                    "n_estimators": [50, 100, 200, 500],
                    "ccp_alpha": [0.0, 0.05],
                    "max_features": [0.1, "sqrt", "log2", 0.3, 0.7, 1.0],
                    "max_depth": [3, 5, 7, 9],
                    "max_samples": [0.3, 0.5, 0.8],
                },
                {},
                True,
                "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",
                sklearn.__version__,
                datetime.now(),
            )
        )

    if et is True:
        # Extra trees regressor
        models.append(
            skmodel(
                type(ExtraTreesRegressor()).__name__,
                ExtraTreesRegressor(random_state=helpers.random_seed, n_jobs=-1),
                {
                    "n_estimators": [50, 100, 200, 500],
                    "ccp_alpha": [0.0, 0.05],
                    "max_features": [0.1, "sqrt", "log2", 0.3, 0.7, 1.0],
                    "max_depth": [3, 5, 7, 9],
                },
                {},
                True,
                "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html",
                sklearn.__version__,
                datetime.now(),
            )
        )

    if adb is True:
        # Adaboost
        models.append(
            skmodel(
                type(AdaBoostRegressor()).__name__,
                AdaBoostRegressor(random_state=helpers.random_seed),
                {
                    "estimator": [
                        DecisionTreeRegressor(max_depth=3),
                        ExtraTreeRegressor(max_depth=3),
                    ],
                    "n_estimators": [30, 50, 70],
                    "learning_rate": [0.2, 0.5, 1.0, 1.2],
                    "loss": ["linear", "square"],
                },
                {},
                False,
                "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html",
                sklearn.__version__,
                datetime.now(),
            )
        )

    if hgb is True:
        # Gradient boosting - This version is similar to LightGBM
        models.append(
            skmodel(
                type(HistGradientBoostingRegressor()).__name__,
                HistGradientBoostingRegressor(random_state=helpers.random_seed),
                {
                    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                    "max_leaf_nodes": [10, 31, 50],
                    "max_depth": [3, 5, None],
                    "l2_regularization": [0.0, 0.5, 1.0],
                    "max_iter": [50, 100, 150],
                    "loss": ["squared_error", "absolute_error"],
                },
                {},
                False,
                "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html",
                sklearn.__version__,
                datetime.now(),
            )
        )

    if xgb is True:
        # EXtreme Gradient Boosting - Model tuning guide https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning
        # early stopping https://xgboosting.com/configure-xgboost-early-stopping-regularization/
        # training https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/#:~:text=The%20number%20of%20trees%20(or,the%20XGBoost%20library%20is%20100.
        models.append(
            skmodel(
                type(XGBRegressor()).__name__,
                XGBRegressor(random_state=helpers.random_seed, n_jobs=-1, verbosity=0),
                {
                    "learning_rate": [0.05, 0.1, 0.2, 0.3],
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7, 10],
                    "min_child_weight": [1, 5, 7],
                    "gamma": [0.1, 0.25, 0.5],
                    "subsample": [0.7, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "reg_alpha": [0.05, 1.0, 10.0],
                    "reg_lambda": [0.05, 1.0, 10.0],
                },
                {},
                False,
                "https://xgboost.readthedocs.io/en/stable/python/python_api.html",
                f"XGBoost version: {xgboost.__version__}",
                datetime.now(),
            )
        )

        # {
        #             "n_estimators": [100, 200, 500],
        #             "max_depth": [3, 5, 7],
        #             "min_child_weight": [1, 2, 3],
        #             "gamma": [0, 0.05, 0.1],
        #             "subsample": [0.6, 0.8, 1.0],
        #             "colsample_bytree": [0.6, 0.8, 1.0],
        #             "reg_alpha": [0.0, 0.1, 0.2],
        #             "reg_lambda": [1.0, 1.1, 1.2],
        #         },

        # {
        #     "n_estimators": [100, 250, 500, 750, 1000],
        #     "learning_rate": [0.2, 0.3, 0.4],
        #     "max_depth": [3, 5, 7],
        #     "min_child_weight": [1, 2, 3],
        #     "gamma": [0, 0.05, 0.1],
        #     "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        #     "colsample_bytree": [0.6, 0.8, 1.0],
        #     "reg_alpha": [0.0, 0.1, 0.25, 0.4],
        #     "reg_lambda": [1.0, 1.1, 1.25, 1.4],
        #     "grow_policy": ["depthwise", "lossguide"],
        # },

    if lgbm:
        # Light Gradient Boosting Machine - Model tuning guide https://www.kaggle.com/code/bextuychiev/lgbm-optuna-hyperparameter-tuning-w-understanding
        models.append(
            skmodel(
                type(LGBMRegressor()).__name__,
                LGBMRegressor(random_state=helpers.random_seed, n_jobs=-1, verbose=-1),
                {
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "n_estimators": [100, 150, 200],
                    "max_depth": [-1, 3, 5, 7, 9],
                    "num_leaves": [20, 31, 100],
                    "min_data_in_leaf": [10, 20, 50, 70],
                    "reg_alpha": [0.0, 1.0, 10.0],
                    "reg_lambda": [0.0, 1.0, 10.0],
                },
                {},
                False,
                "https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html",
                f"LightGBM version: {lightgbm.__version__}",
                datetime.now(),
            )
        )

        # {
        #             "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        #             "n_estimators": [100, 150, 200],
        #             "max_depth": [-1, 3, 5, 7],
        #             "num_leaves": [15, 31, 45, 60],
        #             "min_child_samples": [10, 20, 30],
        #             "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        #             "colsample_bytree": [0.6, 0.8, 1.0],
        #             "reg_alpha": [0.0, 1.0, 10.0, 100.0],
        #             "reg_lambda": [0.0, 1.0, 10.0, 100.0],
        #         },

    return models


def neural_network_models(
    mlp: bool = True,
    layers: Tuple[Tuple[int]] = (
        (20, 10, 5),
        (50, 50, 50),
        (100, 50, 10),
        (100, 50, 25),
        (100, 50, 40),
    ),
    solver: str = "adam",
    max_iter: int = 400,
    n_input_features: Optional[int] = None,
    skip_feature_n_dependent: bool = False,
    **kwargs,
) -> List[skmodel]:
    """
     Function to build a default set of neural network models and parameter grids to optimize the models over

    Returns:
        List[skmodel]: List of neural network models and meta data
    """

    models = []

    if isinstance(layers, tuple):
        layers = list(layers)
    if n_input_features is not None and skip_feature_n_dependent is False:
        if isinstance(layers, tuple):
            layers = list(layers)
        hl1 = [int(n_input_features * elt) for elt in [0.5, 0.7, 0.9]]
        hl2 = [int(n_input_features * elt) for elt in [0.3, 0.4, 0.5]]
        hl3 = [int(n_input_features * elt) for elt in [0.1, 0.2, 0.4]]
        for ith in range(3):
            layers.append((hl1[ith], hl2[ith], hl3[ith]))

    if mlp is True:
        # Multi-layer perceptron (MLP) - This is the only general NN in sklearn currently. It is built as feedforward network.
        models.append(
            skmodel(
                type(MLPRegressor()).__name__,
                MLPRegressor(
                    max_iter=max_iter, solver=solver, random_state=helpers.random_seed
                ),
                {
                    "learning_rate": ["const", "invscaling", "adaptive"],
                    "learning_rate_init": [0.0001, 0.001, 0.01],
                    "alpha": [0.00001, 0.0001, 0.0005, 0.001],
                    "activation": ["tanh", "relu"],
                    "hidden_layer_sizes": layers,
                    "batch_size": ["auto", 16, 32],
                },
                {},
                True,
                "https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor",
                sklearn.__version__,
                datetime.now(),
            )
        )

    return models
