# Module chemutil.classical_ml_model_build

Module for sklearn model building

??? example "View Source"
        #!/usr/bin.env python3

        # -*- coding: utf-8 -*-

        """

        Module for sklearn model building

        """

        import logging

        import pickle as pkl

        from datetime import datetime

        from typing import Callable, List, Optional, Union



        import deepchem as dc

        import joblib

        import numpy as np

        import pandas as pd

        import sklearn

        import yaml

        from sklearn.ensemble import VotingRegressor

        from sklearn.model_selection import GridSearchCV, train_test_split

        from sklearn.pipeline import Pipeline

        from scipy.stats import pearsonr

        from chemutil import classical_ml_models, datasets, helpers

        log = logging.getLogger(__name__)



        def get_pearson_r(x: np.ndarray, y: np.ndarray):

            """

            Function which can be ysed to make a Pearson R scorer

            Args:

                x (numpy.ndarray): Data for the Pearson R correlation from one source

                y (numpy.ndarray): Data for the Pearson R correlation from another source

            Returns:

                float: Pearson R metric

            """

            r, _ = pearsonr(x, y)

            return r



        class RXGridSearchSklearnRegresssor(object):

            """

            A class to encapulate the model building process for scikit learn classical ML models using a train test and validate data splitting

            The idea of this class is to wrap up the sklearn pipeline model building and training in a single container which can be saved.

            For examples and information see the notebooks directory

            It is expected that this class is used in the following manner:

            1. Load you data and featurize from your moleucle representation:

            ```

            df = pd.read_csv("my_data.csv)

            feat_df = featurization.get_count_ecfp(data_df=df, smiles_column="smiles", return_df=True)

            ```

            2. Select your scikit learn model and parameter grid. You can use the default models provided in this libray which provides name, model and parameter grid to search as follows:

            ```

            my_model = classical_ml_models.linear_models(lasso=False, lars=False)[0]

            ```

            Alternatively wrap up your chosen model:

            ```

            from chemutil.classical_ml_models import skmodel

            my_model = skmodel(

                type(RandomForestRegressor()).__name__,

                RandomForestRegressor(random_state=helpers.random_seed, n_jobs=-1),

                {

                    "n_estimators": [50, 100, 200],

                    "min_samples_split": [2, 4],

                    "ccp_alpha": [0.0, 0.05],

                    "max_features": ["sqrt", "log2", 1.0],

                },

                {},

                True,

                "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",

                sklearn.__version__,

                datetime.now(),

            )

            ```



            3. Instantiate this class and load the model and data into the class

            ```

            cls = RXGridSearchSklearnRegresssor(

                feat_df,

                model=linear_regression_models.model,

                model_name=linear_regression_models.name

            )

            ```

            4. Set your data splits this example is based on train test splits and uses tye molecular weight splitter from deepchem

            ```

            cls.build_trian_test_validate_splits(

                smiles_column="smiles",

                task_columns=["taskX"],

                feature_column_regex="ecfp_count_bit_",

                splitter_type="mw",

                train=0.8,

                test=0.2,

                validation=None

            )

            ```

            5. Define your sklearn pipeline to include, in addition to the model, any of the following elements feature_scaling, dimensionality_reduction or feature_selection. provide the word "passthrough" to activate this

            element or if you are fixing to just one method you can add that directly i.e. feature_scaling=MinMaxScaler(). Following this define the parameter grid to search. if you have activated any of the optional pipeline

            elements if you have activated an element with "passthrough" you must provide a key of the element mapping to the method eg {"feature_scaling": MinMaxScaler()}. You can then extend the dictionary with

            hyper-parameters to try i.e. "dimensionality_reduction": PCA(), "n_components": [2, 3, 4], "iterated_power": [2, 5, 7]}. For the model provide a dictionary of hyper parameters to grid search, not the dafualt models

            in this library have a grid which can be used by default.

            ```

            model.build_pipline_list(feature_scaling="passthrough")

            cls.build_parameter_grid(

                feature_scaling_param_grid={"feature_scaling": [MinMaxScaler(), StandardScaler()]},

                model_param_grid=my_model.default_param_grid

            )

            cls.make_pipeline()

            ```

            6. You can now eun the grid search, defining scoring functions to use. Note the first one is used to identify the best model and refit

            ```

            gs = cls.optimize_model(scoring=("neg_root_mean_squared_error", "r2"))

            ```

            """

            log = logging.getLogger(__name__)

            def __init__(

                self,

                data_df: Optional[pd.DataFrame] = None,

                X_train: Optional[

                    Union[np.ndarray, pd.DataFrame, dc.data.datasets.NumpyDataset]

                ] = None,

                y_train: Optional[

                    Union[np.ndarray, pd.DataFrame, dc.data.datasets.NumpyDataset]

                ] = None,

                X_test: Optional[

                    Union[np.ndarray, pd.DataFrame, dc.data.datasets.NumpyDataset]

                ] = None,

                y_test: Optional[

                    Union[np.ndarray, pd.DataFrame, dc.data.datasets.NumpyDataset]

                ] = None,

                X_val: Optional[

                    Union[np.ndarray, pd.DataFrame, dc.data.datasets.NumpyDataset]

                ] = None,

                y_val: Optional[

                    Union[np.ndarray, pd.DataFrame, dc.data.datasets.NumpyDataset]

                ] = None,

                kfolds: Optional[

                    Union[np.ndarray, pd.DataFrame, dc.data.datasets.NumpyDataset]

                ] = None,

                model: Optional[Union[classical_ml_models.skmodel]] = None,

                target_name: Optional[str] = None,

                model_name: Optional[str] = None,

                pipeline: Optional[Pipeline] = None,

                param_grid: Optional[dict] = None,

                _model: Optional[Union[classical_ml_models.skmodel, Callable]] = None,

                **kwargs,

            ):

                """

                _summary_

                Args:

                    data_df (Optional[pd.DataFrame], optional): _description_. Defaults to None.

                    X_train (Optional[Union[np.ndarray, pd.DataFrame, dc.data.datasets.NumpyDataset]], optional): _description_. Defaults to None.

                    y_train (Optional[Union[np.ndarray, pd.DataFrame, dc.data.datasets.NumpyDataset]], optional): _description_. Defaults to None.

                    X_test (Optional[Union[np.ndarray, pd.DataFrame, dc.data.datasets.NumpyDataset]], optional): _description_. Defaults to None.

                    y_test (Optional[Union[np.ndarray, pd.DataFrame, dc.data.datasets.NumpyDataset]], optional): _description_. Defaults to None.

                    X_val (Optional[Union[np.ndarray, pd.DataFrame, dc.data.datasets.NumpyDataset]], optional): _description_. Defaults to None.

                    y_val (Optional[Union[np.ndarray, pd.DataFrame, dc.data.datasets.NumpyDataset]], optional): _description_. Defaults to None.

                    target_name (Optional[str], optional): _description_. Defaults to None.

                    model_name (Optional[str], optional): _description_. Defaults to None.

                    pipeline (Optional[Pipeline], optional): _description_. Defaults to None.

                    param_grid (Optional[dict], optional): _description_. Defaults to None.

                    _model: (Optional[str], optional): if you want to pass a trained and finalized model in at the start use this variable to set it. Defaults to None.

                """

                # User arguments

                self.data_df = data_df

                self.X_train = X_train

                self.y_train = y_train

                self.X_test = X_test

                self.y_test = y_test

                self.X_validate = X_val

                self.y_validate = y_val

                self.target_name = target_name

                self._model_name = model_name  # type(model).__name__

                self.kfolds = kfolds

                self.initial_model = model

                # Internals for reference

                self._random_seed = helpers.random_seed

                self.scikitlearn_version = sklearn.__version__

                self.date_and_time = datetime.now()

                self.pipe_order = None

                # Internal variables

                self.feature_scaling = None

                self.dimensionality_reduction = None

                self.feature_selection = None

                self.param_grid = param_grid

                self.pipeline_element_order = None

                self.pipeline_elts = None

                self.pipeline = pipeline

                self.model = _model

                self.splitter = None

            @property

            def model_name(self):

                """

                Set the model name

                """

                return self._model_name

            @model_name.setter

            def model_name(self, mname: str):

                self._model_name = mname.strip().lower()

            @model_name.deleter

            def model_name(self):

                self._model_name = None

            @property

            def Xtrain(self):

                """

                Set the features for training

                """

                return self.X_train

            @Xtrain.setter

            def Xtrain(self, xtr: np.ndarray):

                self.X_train = xtr

            @Xtrain.deleter

            def Xtrain(self):

                self.X_train = None

            @property

            def Xtest(self):

                """

                Set the features for training

                """

                return self.X_test

            @Xtest.setter

            def Xtest(self, xts: np.ndarray):

                self.X_test = xts

            @Xtest.deleter

            def Xtest(self):

                self.X_test = None

            @property

            def ytrain(self):

                """

                Set the target for training

                """

                return self.y_train

            @ytrain.setter

            def ytrain(self, ytr: np.ndarray):

                self.y_train = ytr

            @ytrain.deleter

            def ytrain(self):

                self.y_train = None

            @property

            def ytest(self):

                """

                Set the features for training

                """

                return self.X_test

            @Xtest.setter

            def Xtest(self, xts: np.ndarray):

                self.X_test = xts

            @Xtest.deleter

            def Xtest(self):

                self.X_test = None

            def __str__(self) -> str:

                """

                Get a descriptive string of the class

                Returns:

                    str: string description of the class

                """

                return f"Model: {self.model_name} # train {len(self.X_train) if self.X_train is not None else None} # test {len(self.X_test) if self.X_test is not None else None } # validate {len(self.X_validate) if self.X_validate is not None else None} sklearn version {self.scikitlearn_version}"

            def _get_default_pipe_order(

                self,

                feature_scaling=1,

                dimensionality_reduction=2,

                feature_selection=3,

                model=4,

            ) -> dict:

                """

                Get a dictionary setting the order of the elements of the pipeline

                Args:

                    feature_scaling (int, optional): The order (1st 2nd 3rd .....) of the feature scaling element if any. Defaults to 1.

                    dimensionality_reduction (int, optional): The order of the dimensionality reduction element if any. Defaults to 2.

                    feature_selection (int, optional): The order of the feature selection element if any. Defaults to 3.

                    model (int, optional): The order of the model element. Defaults to 4.

                Returns:

                    dict : dictionary of element name to the order position of the elements for the pipeline

                """

                return {

                    "feature_scaling": feature_scaling,

                    "dimensionality_reduction": dimensionality_reduction,

                    "feature_selection": feature_selection,

                    "model": model,

                }

            def build_train_test_validate_splits(

                self,

                smiles_column: Optional[str] = None,

                task_columns: Optional[List[str]] = None,

                df: Optional[pd.DataFrame] = None,

                feature_columns: Optional[List[str]] = None,

                feature_column_regex: Optional[str] = None,

                splitter_type: Optional[Union[str, dc.splits.Splitter]] = None,

                train: Optional[Union[float, int]] = 0.8,

                test: Optional[Union[float, int]] = 0.1,

                holdout: Optional[Union[float, int]] = 0.1,

                kfold: Optional[int] = None,  # TODO: set up for kfold

                **kwargs,

            ) -> None:

                """

                This is a function to build the data splits into train test and validate sets

                Args:

                    df (pd.DataFrame): All of the data in one dataframe to split into a dataset for ML/AI

                    smiles_column (str): The column name that contains smiles strings

                    task_columns (List[str]): List of y task column names

                    feature_columns (Optional[List[str]], optional): Must provide this or feature_column_regex. If this is provided it is a list of the column names that correspond to features i.e. the X mat. Defaults to None.

                    fesature_column_regex (Optional[str], optional): Must provide this or feature_columns. If this is provided it is a partial or full string to identify all columns that contain it as feature columns. Defaults to None.

                    df (Optional[pd.DataFrame], optional): Dataframe to convert to a dataset and split. Please note the keyword args for this are passed through kwargs here, they correspond to the arguments for pandas_to_deepchem(). Defaults to None.

                    train (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for training. Defaults to None.

                    test (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for testing. Defaults to None.

                    validation (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for validation. Defaults to None.

                    _kfold (Optional[int], optional): The number of k fold splits. Defaults to None.

                Raises:

                    RunTimeError: data is not found in the class and is not passed to the call

                """

                if isinstance(self.data_df, pd.DataFrame):

                    tmp_dat = datasets.pandas_to_deepchem(

                        self.data_df,

                        smiles_column=smiles_column,

                        task_columns=task_columns,

                        feature_columns=feature_columns,

                        feature_column_regex=feature_column_regex,

                    )

                elif isinstance(df, pd.DataFrame):

                    self.data_df = df

                    tmp_dat = datasets.pandas_to_deepchem(

                        self.data_df,

                        smiles_column=smiles_column,

                        task_columns=task_columns,

                        feature_columns=feature_columns,

                        feature_column_regex=feature_column_regex,

                    )

                else:

                    raise RuntimeError(

                        "Data not found! Please set self.data_df or pass in a dataframe to this function."

                    )

                log.debug(

                    f"Training fraction {train} {type(train)} Testing fraction {test} {type(test)}validation fraction {holdout} {type(holdout)}"

                )

                if holdout is not None and kfold is None and splitter_type is not None:

                    train, holdout, test, self.splitter = datasets.molecular_dataset_split(

                        ds=tmp_dat,

                        splitter_type=splitter_type,

                        train=train,

                        test=test,

                        holdout=holdout,

                        **kwargs,

                    )

                    self.train = train

                    self.test = test

                    self.validate = holdout

                    self.X_train = train.X

                    self.y_train = train.y

                    self.X_test = test.X

                    self.y_test = test.y

                    self.X_validate = holdout.X

                    self.y_validate = holdout.y

                elif holdout is None and kfold is None and splitter_type is not None:

                    train, test, splitter = datasets.molecular_dataset_split(

                        ds=tmp_dat,

                        splitter_type=splitter_type,

                        train=train,

                        test=test,

                        holdout=holdout,

                        **kwargs,

                    )

                    self.train = train

                    self.test = test

                    self.X_train = train.X

                    self.y_train = train.y

                    self.X_test = test.X

                    self.y_test = test.y

                elif kfold is not None and splitter_type is not None:

                    if holdout is None:

                        log.info(f"Generated {kfold} fold")

                        self.kfolds, self.splitter = datasets.molecular_dataset_split(

                            ds=tmp_dat, splitter_type=splitter_type, kfold=kfold, **kwargs

                        )

                    else:

                        log.info(f"Generated {kfold} fold and validation holdout")

                        self.kfolds, self.validate, self.splitter = (

                            datasets.molecular_dataset_split(

                                ds=tmp_dat,

                                splitter_type=splitter_type,

                                holdout=holdout,

                                kfold=kfold,

                                **kwargs,

                            )

                        )

                        self.X_validate = self.validate.X

                        self.y_validate = self.validate.y

                else:

                    log.warning(

                        "Using SciKit-Learn random splitter you may want to consider using molecule splitters is this is for molecular data."

                    )

                    X_train, X_test, y_train, y_test = train_test_split(

                        tmp_dat.X,

                        tmp_dat.y,

                        test_size=0.2,

                        train_size=0.8,

                        random_state=helpers.random_seed,

                        shuffle=True,

                    )

                    self.X_train = X_train

                    self.y_train = y_train

                    self.X_test = X_test

                    self.y_test = y_test

                if splitter_type is not None:

                    log.debug(

                        f"Train: {train}, Validation: {holdout}, Test {test}, kfolds: {kfold}, splitter {self.splitter}"

                    )

                else:

                    log.debug(

                        f"Train: {train}, Validation: {holdout}, Test {test}, kfolds: {kfold}"

                    )

            def build_pipline_list(

                self,

                pipeline_element_order=None,

                feature_scaling: Optional[Union[List[str], Callable, str]] = None,

                dimensionality_reduction: Optional[Union[List[str], Callable, str]] = None,

                feature_selection: Optional[Union[List[str], Callable, str]] = None,

            ) -> list:

                """

                This function builds a list of sklearn pipeline elements. It is stored as list internally so that it can be modified. Once happy call

                self.make_pipeline to set the pipeline in the class. Note you can pass a pipeline and param_grid in to the class directly to use.

                Args:

                    pipeline_element_order (_type_, optional): A dictionary of scikit learn pipeline steps with keys of "feature_scaling",

                    "dimensionality_reduction", "feature_selection" and "model" and values of integers which set the order in which these steps will

                    happen in the pipeline. Defaults to None.

                Raises:

                    ValueError: if an unknown step is trying to be added to the pipeline

                Returns:

                    List: list of tuples of steps for the pipeline

                """

                if pipeline_element_order is None:

                    orders = self._get_default_pipe_order()

                    self.pipe_order = orders

                else:

                    orders = self._get_default_pipe_order(**pipeline_element_order)

                    self.pipe_order = orders

                if feature_scaling is not None:

                    self.feature_scaling = feature_scaling

                if dimensionality_reduction is not None:

                    self.dimensionality_reduction = dimensionality_reduction

                if feature_selection is not None:

                    self.feature_selection = feature_selection

                self.pipeline_elts = list()

                for elt, x in sorted(orders.items(), key=lambda y: y[1]):

                    log.info(f"Pipeline element: {elt} in order {x}")

                    if elt == "feature_scaling" and self.feature_scaling is not None:

                        log.info("\tAdded feature scaling to pipeline")

                        self.pipeline_elts.append((elt, self.feature_scaling))

                    elif (

                        elt == "dimensionality_reduction"

                        and self.dimensionality_reduction is not None

                    ):

                        log.info("\tAdded dimensionality reduction to pipeline")

                        self.pipeline_elts.append((elt, self.dimensionality_reduction))

                    elif elt == "feature_selection" and self.feature_selection is not None:

                        log.info("\tAdded feature selection to pipeline")

                        self.pipeline_elts.append((elt, self.feature_selection))

                    elif elt == "model" and self.initial_model is not None:

                        log.info("\tAdded model to pipeline")

                        self.pipeline_elts.append((elt, self.initial_model))

                    elif all(

                        ent != elt

                        for ent in [

                            "feature_scaling",

                            "dimensionality_reduction",

                            "feature_selection",

                            "model",

                        ]

                    ):

                        raise ValueError(

                            f"Unknown pipeline entry requested {elt}. This class can only accept feature_scaling, dimensionality_reduction, feature_selection, model."

                        )

                    else:

                        log.info(f"No {elt} in the pipeline")

                return self.pipeline_elts

            def _update_dictionary_keys_for_sklearn_pipeline(

                self, d: dict, prepend_with: str

            ) -> dict:

                """

                Function to change the key names of a dictionary by prepending a constant string to them all.

                Args:

                    d (dict): dictionary of key and value for grid searching hyper-parameter options

                    prepend_with (str): The pipeline element to prepend the key with

                Returns:

                    dict: updated dictionary so that the parameters related to a specific pipeline element are noted in the keys

                """

                return {

                    f"{prepend_with}__{k}" if k != prepend_with else k: v for k, v in d.items()

                }

            def build_parameter_grid(

                self,

                feature_scaling_param_grid: Optional[dict] = None,

                dimensionality_reduction_param_grid: Optional[dict] = None,

                feature_selection_param_grid: Optional[dict] = None,

                model_param_grid: Optional[dict] = None,

                add_prepends_for_pipe_to_all: bool = True,

            ):

                """

                Function to build the parameter grid. It will by default prepend the pipeline step to the parameters as needed. Then it combines all input dictionaries, not there should not

                be any overlapping keys in the separete input dictionaries

                Args:

                    feature_scaling_param_grid (Optional[dict], optional): dictionary of options for feature scaling (if any) in the model pipeline. Defaults to None.

                    dimensionality_reduction_param_grid (Optional[dict], optional): dictionary of options for dimensionality reduction (if any) in the model pipeline. Defaults to None.

                    feature_selection_param_grid (Optional[dict], optional): dictionary of options for feature selection (if any) in the model pipeline. Defaults to None.

                    model_param_grid (Optional[dict], optional): dictionary of options for model hyper-parameters in the model pipeline. Defaults to None.

                    add_prepends_for_pipe_to_all (bool, optional): Whether to add the pipeline element prepend lables to each key. Defaults to True.

                """

                if add_prepends_for_pipe_to_all is True:

                    if isinstance(feature_scaling_param_grid, dict):

                        feature_scaling_param_grid = (

                            self._update_dictionary_keys_for_sklearn_pipeline(

                                feature_scaling_param_grid, "feature_scaling"

                            )

                        )

                    if isinstance(dimensionality_reduction_param_grid, dict):

                        dimensionality_reduction_param_grid = (

                            self._update_dictionary_keys_for_sklearn_pipeline(

                                dimensionality_reduction_param_grid, "dimensionality_reduction"

                            )

                        )

                    if isinstance(feature_selection_param_grid, dict):

                        feature_selection_param_grid = (

                            self._update_dictionary_keys_for_sklearn_pipeline(

                                feature_selection_param_grid, "feature_selection"

                            )

                        )

                    if isinstance(model_param_grid, dict):

                        model_param_grid = self._update_dictionary_keys_for_sklearn_pipeline(

                            model_param_grid, "model"

                        )

                # NOTE: it assumes there are no overlapping keys but doesn't check

                self.param_grid = {}

                if isinstance(feature_scaling_param_grid, dict):

                    self.param_grid = {**self.param_grid, **feature_scaling_param_grid}

                if isinstance(dimensionality_reduction_param_grid, dict):

                    self.param_grid = {**self.param_grid, **dimensionality_reduction_param_grid}

                if isinstance(feature_selection_param_grid, dict):

                    self.param_grid = {**self.param_grid, **feature_selection_param_grid}

                if isinstance(model_param_grid, dict):

                    self.param_grid = {**self.param_grid, **model_param_grid}

            def make_pipeline(self, return_pipeline: bool = False):

                """

                Function to build a pipeline if the pipeline elements have been defined

                """

                if self.pipeline_elts is not None:

                    log.info(f"Setting the scikit learn pipeline using {self.pipeline_elts}.")

                    if return_pipeline is True:

                        return Pipeline(self.pipeline_elts)

                    else:

                        self.pipeline = Pipeline(self.pipeline_elts)

            def save_model(self, filename: str = "model.joblib"):

                """

                Save the final model pipeline to a jobloib file

                Args:

                    filename (str): filename to save

                """

                if self.model is not None:

                    joblib.dump(self.model, filename)

            def load_pipeline_into_class(self, filename: str = "model.joblib"):

                """

                Load a model pipeline into this class

                Args:

                    filename (str): joblib file to load the model pipeline from

                """

                self.model = joblib.load(filename)

            def save_self(self, filename: str):

                """

                Save the entire class as a pickle file

                Args:

                    filename (str): file name

                """

                with open(filename, "wb") as fout:

                    pkl.dump(self, fout)

            def optimize_model(

                self,

            ):

                """

                Function to train a model pipeline using the setting defined

                Args:

                    cv (int, optional): The number of corss validation folds used to establish the best hyper-parameters. Defaults to 5.

                    verbose (bool, optional): To print verbose or not trianing information. Defaults to True.

                """

                pass



        class RXTrainTestGridSearchSklearnRegresssor(RXGridSearchSklearnRegresssor):

            """

            Class for building a model using a single train test split. This class encapulates a generalized model building process.

            To fully optimze a model it will need to be taken put of this class. This is a method wrapper to enable several methods to be tested initially.

            It is expected that this class is used in the following manner:

            1. Load you data and featurize from your moleucle representation:

            ```

            df = pd.read_csv("my_data.csv)

            feat_df = featurization.get_count_ecfp(data_df=df, smiles_column="smiles", return_df=True)

            ```

            2. Select your scikit learn model and parameter grid. You can use the default models provided in this libray which provides name, model and parameter grid to search as follows:

            ```

            my_model = classical_ml_models.linear_models(lasso=False, lars=False)[0]

            ```

            Alternatively wrap up your chosen model:

            ```

            from chemutil.classical_ml_models import skmodel

            my_model = skmodel(

                type(RandomForestRegressor()).__name__,

                RandomForestRegressor(random_state=helpers.random_seed, n_jobs=-1),

                {

                    "n_estimators": [50, 100, 200],

                    "min_samples_split": [2, 4],

                    "ccp_alpha": [0.0, 0.05],

                    "max_features": ["sqrt", "log2", 1.0],

                },

                {},

                True,

                "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",

                sklearn.__version__,

                datetime.now(),

            )

            ```



            3. Instantiate this class and load the model and data into the class

            ```

            cls = RXGridSearchSklearnRegresssor(

                feat_df,

                model=linear_regression_models.model,

                model_name=linear_regression_models.name

            )

            ```

            4. Set your data splits this example is based on train test splits and uses tye molecular weight splitter from deepchem

            ```

            cls.build_trian_test_validate_splits(

                smiles_column="smiles",

                task_columns=["taskX"],

                feature_column_regex="ecfp_count_bit_",

                splitter_type="mw",

                train=0.8,

                test=0.2,

                validation=None

            )

            ```

            5. Define your sklearn pipeline to include, in addition to the model, any of the following elements feature_scaling, dimensionality_reduction or feature_selection. provide the word "passthrough" to activate this

            element or if you are fixing to just one method you can add that directly i.e. feature_scaling=MinMaxScaler(). Following this define the parameter grid to search. if you have activated any of the optional pipeline

            elements if you have activated an element with "passthrough" you must provide a key of the element mapping to the method eg {"feature_scaling": MinMaxScaler()}. You can then extend the dictionary with

            hyper-parameters to try i.e. "dimensionality_reduction": PCA(), "n_components": [2, 3, 4], "iterated_power": [2, 5, 7]}. For the model provide a dictionary of hyper parameters to grid search, not the dafualt models

            in this library have a grid which can be used by default.

            ```

            model.build_pipline_list(feature_scaling="passthrough")

            cls.build_parameter_grid(

                feature_scaling_param_grid={"feature_scaling": [MinMaxScaler(), StandardScaler()]},

                model_param_grid=my_model.default_param_grid

            )

            cls.make_pipeline()

            ```

            6. You can now eun the grid search, defining scoring functions to use. Note the first one is used to identify the best model and refit

            ```

            gs = cls.optimize_model(scoring=("neg_root_mean_squared_error", "r2"))

            ```

            Args:

                RXGridSearchSklearnRegresssor (_type_): _description_

            """

            def optimize_model(

                self,

                cv: int = 5,

                verbose: int = 2,

                scoring: Optional[dict[Union[str, Callable]]] = None,

                refit: Optional[str] = None,

                return_model: bool = False,

                **kwargs,

            ):

                """

                Function to train a model pipeline using the setting defined

                Args:

                    cv (int, optional): The number of corss validation folds used to establish the best hyper-parameters. Defaults to 5.

                    verbose (bool, optional): To print verbose or not trianing information. Defaults to True.

                """

                if isinstance(self.param_grid, dict):

                    log.info(f"Will use the parameter grid: {yaml.dump(self.param_grid)}")

                if isinstance(self.pipeline, Pipeline):

                    log.info("Will run the following pipeline to optimize the hyper parmaeters")

                if self.X_train is not None and self.y_train is not None:

                    log.info("Using the trianing data to run a Grid search CV")

                if scoring is None:

                    scoring = (

                        "neg_mean_squared_error",

                        "neg_root_mean_squared_error",

                        "neg_mean_absolute_percentage_error",

                        "r2",

                    )

                    log.info(f"Using default scoring: {scoring}")

                if refit is None:

                    refit = scoring[0]

                grid_search = GridSearchCV(

                    self.pipeline,

                    n_jobs=-1,

                    param_grid=self.param_grid,

                    cv=cv,

                    verbose=verbose,

                    scoring=scoring,

                    refit=refit,

                    **kwargs,

                )

                if self.y_train.shape[1] > 1:

                    grid_search.fit(self.X_train, self.y_train)

                else:

                    grid_search.fit(self.X_train, self.y_train.ravel())

                self.completed_grid_search = grid_search

                self.model = grid_search.best_estimator_

                self.best_parmeters = grid_search.best_params_

                if return_model is False:

                    return grid_search

                else:

                    return grid_search.best_estimator_



        class RXKFoldGridSearchSklearnRegresssor(RXGridSearchSklearnRegresssor):

            """

            Class for building a model using a k fold split and holdout set. This class encapulates a generalized model building process.

            To fully optimze a model it will need to be taken put of this class. This is a method wrapper to enable several methods to be tested initially.

            It is expected that this class is used in the following manner:

            1. Load you data and featurize from your moleucle representation:

            ```

            df = pd.read_csv("my_data.csv)

            feat_df = featurization.get_count_ecfp(data_df=df, smiles_column="smiles", return_df=True)

            ```

            2. Select your scikit learn model and parameter grid. You can use the default models provided in this libray which provides name, model and parameter grid to search as follows:

            ```

            my_model = classical_ml_models.linear_models(lasso=False, lars=False)[0]

            ```

            Alternatively wrap up your chosen model:

            ```

            from chemutil.classical_ml_models import skmodel

            my_model = skmodel(

                type(RandomForestRegressor()).__name__,

                RandomForestRegressor(random_state=helpers.random_seed, n_jobs=-1),

                {

                    "n_estimators": [50, 100, 200],

                    "min_samples_split": [2, 4],

                    "ccp_alpha": [0.0, 0.05],

                    "max_features": ["sqrt", "log2", 1.0],

                },

                {},

                True,

                "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",

                sklearn.__version__,

                datetime.now(),

            )

            ```



            3. Instantiate this class and load the model and data into the class

            ```

            cls = RXGridSearchSklearnRegresssor(

                feat_df,

                model=linear_regression_models.model,

                model_name=linear_regression_models.name

            )

            ```

            4. Set your data splits this example is based on train test splits and uses tye molecular weight splitter from deepchem

            ```

            cls.build_trian_test_validate_splits(

                smiles_column="smiles",

                task_columns=["taskX"],

                feature_column_regex="ecfp_count_bit_",

                splitter_type="mw",

                train=0.8,

                test=0.2,

                validation=None

            )

            ```

            5. Define your sklearn pipeline to include, in addition to the model, any of the following elements feature_scaling, dimensionality_reduction or feature_selection. provide the word "passthrough" to activate this

            element or if you are fixing to just one method you can add that directly i.e. feature_scaling=MinMaxScaler(). Following this define the parameter grid to search. if you have activated any of the optional pipeline

            elements if you have activated an element with "passthrough" you must provide a key of the element mapping to the method eg {"feature_scaling": MinMaxScaler()}. You can then extend the dictionary with

            hyper-parameters to try i.e. "dimensionality_reduction": PCA(), "n_components": [2, 3, 4], "iterated_power": [2, 5, 7]}. For the model provide a dictionary of hyper parameters to grid search, not the dafualt models

            in this library have a grid which can be used by default.

            ```

            model.build_pipline_list(feature_scaling="passthrough")

            cls.build_parameter_grid(

                feature_scaling_param_grid={"feature_scaling": [MinMaxScaler(), StandardScaler()]},

                model_param_grid=my_model.default_param_grid

            )

            cls.make_pipeline()

            ```

            6. You can now eun the grid search, defining scoring functions to use. Note the first one is used to identify the best model and refit

            ```

            gs = cls.optimize_model(scoring=("neg_root_mean_squared_error", "r2"))

            ```

            Args:

                RXGridSearchSklearnRegresssor (_type_): _description_

            """

            def optimize_model(

                self,

                grid_cv: int = 5,

                verbose: bool = True,

                scoring: Optional[dict[Union[str, Callable]]] = None,

                refit: Optional[str] = None,

                return_gs: bool = False,

                **kwargs,

            ):

                """

                Function to train a model pipeline using the setting defined

                Args:

                    grid_cv (int, optional): The number of corss validation folds used to establish the best hyper-parameters. Defaults to 5.

                    verbose (bool, optional): To print verbose or not trianing information. Defaults to True.

                """

                log.info(

                    "This method will fit k modles on the data avaliable (exluding holdout validation data). This is done over k folds i.e. one model is trained using the best hyper-parameters found for each fold. The returned model must be evaluated on unseen held out data."

                )

                if isinstance(self.param_grid, dict):

                    log.info(f"Will use the parameter grid: {yaml.dump(self.param_grid)}")

                if isinstance(self.pipeline, Pipeline):

                    log.info("Will run the following pipeline to optimize the hyper parmaeters")

                if self.X_train is not None and self.y_train is not None:

                    log.info("Using the trianing data to run a Grid search CV")

                if scoring is None:

                    scoring = (

                        "neg_mean_squared_error",

                        "neg_root_mean_squared_error",

                        "neg_mean_absolute_percentage_error",

                        "r2",

                    )

                    log.info(f"Using default scoring: {scoring}")

                if refit is None:

                    refit = scoring[0]

                model_ensemble = []

                self.completed_grid_search = []

                self.best_parameters = []

                for ith, (train, test) in enumerate(self.kfolds):

                    log.info(f"{ith}: {train.X.shape}, {test.X.shape}")

                    pl = self.make_pipeline(return_pipeline=True)

                    grid_search = GridSearchCV(

                        pl,

                        n_jobs=-1,

                        param_grid=self.param_grid,

                        cv=grid_cv,

                        verbose=verbose,

                        scoring=scoring,

                        refit=refit,

                        **kwargs,

                    )

                    if train.y.shape[1] > 1:

                        grid_search.fit(train.X, train.y)

                    else:

                        grid_search.fit(train.X, train.y.ravel())

                    self.completed_grid_search.append(grid_search)

                    model_ensemble.append((f"model_{ith}", grid_search.best_estimator_))

                if return_gs is False:

                    log.info(model_ensemble)

                    m = VotingRegressor(model_ensemble, n_jobs=-1, verbose=True)

                    expected_n_data = sum([d[1].X.shape[0] for d in self.kfolds])

                    X_all_kf_data_concatenated = np.concatenate([d[1].X for d in self.kfolds])

                    log.debug(f"All X concatenate: {X_all_kf_data_concatenated }")

                    y_all_kf_data_concatenated = np.concatenate([d[1].y for d in self.kfolds])

                    log.debug(f"All y concatenate: {y_all_kf_data_concatenated }")

                    try:

                        assert all(

                            ent.shape[0] == expected_n_data

                            for ent in [X_all_kf_data_concatenated, y_all_kf_data_concatenated]

                        )

                    except AssertionError as aerr:

                        raise aerr(

                            f"ERROR - concatenating all kfold data for final fit leads to unequal number of dat points: expected: {expected_n_data} X data {X_all_kf_data_concatenated.shape[0]} y data {y_all_kf_data_concatenated.shape[0]}"

                        )

                    if train.y.shape[1] > 1:

                        m.fit(X_all_kf_data_concatenated, y_all_kf_data_concatenated)

                    else:

                        m.fit(X_all_kf_data_concatenated, y_all_kf_data_concatenated.ravel())

                    self.pipeline = m

                    return self.pipeline

                else:

                    return grid_search



        def dc_sklearn_model_builder():

            pass

## Variables

```python3
log
```

## Functions


### dc_sklearn_model_builder

```python3
def dc_sklearn_model_builder(

)
```

??? example "View Source"
        def dc_sklearn_model_builder():

            pass


### get_pearson_r

```python3
def get_pearson_r(
    x: numpy.ndarray,
    y: numpy.ndarray
)
```

Function which can be ysed to make a Pearson R scorer

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| x | numpy.ndarray | Data for the Pearson R correlation from one source | None |
| y | numpy.ndarray | Data for the Pearson R correlation from another source | None |

**Returns:**

| Type | Description |
|---|---|
| float | Pearson R metric |

??? example "View Source"
        def get_pearson_r(x: np.ndarray, y: np.ndarray):

            """

            Function which can be ysed to make a Pearson R scorer

            Args:

                x (numpy.ndarray): Data for the Pearson R correlation from one source

                y (numpy.ndarray): Data for the Pearson R correlation from another source

            Returns:

                float: Pearson R metric

            """

            r, _ = pearsonr(x, y)

            return r

## Classes

### RXGridSearchSklearnRegresssor

```python3
class RXGridSearchSklearnRegresssor(
    data_df: Optional[pandas.core.frame.DataFrame] = None,
    X_train: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    y_train: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    X_test: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    y_test: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    X_val: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    y_val: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    kfolds: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    model: Optional[chemutil.classical_ml_models.skmodel] = None,
    target_name: Optional[str] = None,
    model_name: Optional[str] = None,
    pipeline: Optional[sklearn.pipeline.Pipeline] = None,
    param_grid: Optional[dict] = None,
    _model: Union[chemutil.classical_ml_models.skmodel, Callable, NoneType] = None,
    **kwargs
)
```

A class to encapulate the model building process for scikit learn classical ML models using a train test and validate data splitting

The idea of this class is to wrap up the sklearn pipeline model building and training in a single container which can be saved.

For examples and information see the notebooks directory

It is expected that this class is used in the following manner:
1. Load you data and featurize from your moleucle representation:
```
df = pd.read_csv("my_data.csv)
feat_df = featurization.get_count_ecfp(data_df=df, smiles_column="smiles", return_df=True)
```

2. Select your scikit learn model and parameter grid. You can use the default models provided in this libray which provides name, model and parameter grid to search as follows:
```
my_model = classical_ml_models.linear_models(lasso=False, lars=False)[0]
```

Alternatively wrap up your chosen model:

```
from chemutil.classical_ml_models import skmodel

my_model = skmodel(
    type(RandomForestRegressor()).__name__,
    RandomForestRegressor(random_state=helpers.random_seed, n_jobs=-1),
    {
        "n_estimators": [50, 100, 200],
        "min_samples_split": [2, 4],
        "ccp_alpha": [0.0, 0.05],
        "max_features": ["sqrt", "log2", 1.0],
    },
    {},
    True,
    "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",
    sklearn.__version__,
    datetime.now(),
)
```

3. Instantiate this class and load the model and data into the class
```
cls = RXGridSearchSklearnRegresssor(
    feat_df,
    model=linear_regression_models.model,
    model_name=linear_regression_models.name
)
```

4. Set your data splits this example is based on train test splits and uses tye molecular weight splitter from deepchem
```
cls.build_trian_test_validate_splits(
    smiles_column="smiles",
    task_columns=["taskX"],
    feature_column_regex="ecfp_count_bit_",
    splitter_type="mw",
    train=0.8,
    test=0.2,
    validation=None
)
```

5. Define your sklearn pipeline to include, in addition to the model, any of the following elements feature_scaling, dimensionality_reduction or feature_selection. provide the word "passthrough" to activate this
element or if you are fixing to just one method you can add that directly i.e. feature_scaling=MinMaxScaler(). Following this define the parameter grid to search. if you have activated any of the optional pipeline
elements if you have activated an element with "passthrough" you must provide a key of the element mapping to the method eg {"feature_scaling": MinMaxScaler()}. You can then extend the dictionary with
hyper-parameters to try i.e. "dimensionality_reduction": PCA(), "n_components": [2, 3, 4], "iterated_power": [2, 5, 7]}. For the model provide a dictionary of hyper parameters to grid search, not the dafualt models
in this library have a grid which can be used by default.

```
model.build_pipline_list(feature_scaling="passthrough")
cls.build_parameter_grid(
    feature_scaling_param_grid={"feature_scaling": [MinMaxScaler(), StandardScaler()]},
    model_param_grid=my_model.default_param_grid
)
cls.make_pipeline()
```

6. You can now eun the grid search, defining scoring functions to use. Note the first one is used to identify the best model and refit
```
gs = cls.optimize_model(scoring=("neg_root_mean_squared_error", "r2"))
```

#### Descendants

* chemutil.classical_ml_model_build.RXTrainTestGridSearchSklearnRegresssor
* chemutil.classical_ml_model_build.RXKFoldGridSearchSklearnRegresssor

#### Class variables

```python3
log
```

#### Instance variables

```python3
Xtest
```

Set the features for training

```python3
Xtrain
```

Set the features for training

```python3
model_name
```

Set the model name

```python3
ytest
```

Set the features for training

```python3
ytrain
```

Set the target for training

#### Methods


#### build_parameter_grid

```python3
def build_parameter_grid(
    self,
    feature_scaling_param_grid: Optional[dict] = None,
    dimensionality_reduction_param_grid: Optional[dict] = None,
    feature_selection_param_grid: Optional[dict] = None,
    model_param_grid: Optional[dict] = None,
    add_prepends_for_pipe_to_all: bool = True
)
```

Function to build the parameter grid. It will by default prepend the pipeline step to the parameters as needed. Then it combines all input dictionaries, not there should not

be any overlapping keys in the separete input dictionaries

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| feature_scaling_param_grid | Optional[dict] | dictionary of options for feature scaling (if any) in the model pipeline. Defaults to None. | None |
| dimensionality_reduction_param_grid | Optional[dict] | dictionary of options for dimensionality reduction (if any) in the model pipeline. Defaults to None. | None |
| feature_selection_param_grid | Optional[dict] | dictionary of options for feature selection (if any) in the model pipeline. Defaults to None. | None |
| model_param_grid | Optional[dict] | dictionary of options for model hyper-parameters in the model pipeline. Defaults to None. | None |
| add_prepends_for_pipe_to_all | bool | Whether to add the pipeline element prepend lables to each key. Defaults to True. | True |

??? example "View Source"
            def build_parameter_grid(

                self,

                feature_scaling_param_grid: Optional[dict] = None,

                dimensionality_reduction_param_grid: Optional[dict] = None,

                feature_selection_param_grid: Optional[dict] = None,

                model_param_grid: Optional[dict] = None,

                add_prepends_for_pipe_to_all: bool = True,

            ):

                """

                Function to build the parameter grid. It will by default prepend the pipeline step to the parameters as needed. Then it combines all input dictionaries, not there should not

                be any overlapping keys in the separete input dictionaries

                Args:

                    feature_scaling_param_grid (Optional[dict], optional): dictionary of options for feature scaling (if any) in the model pipeline. Defaults to None.

                    dimensionality_reduction_param_grid (Optional[dict], optional): dictionary of options for dimensionality reduction (if any) in the model pipeline. Defaults to None.

                    feature_selection_param_grid (Optional[dict], optional): dictionary of options for feature selection (if any) in the model pipeline. Defaults to None.

                    model_param_grid (Optional[dict], optional): dictionary of options for model hyper-parameters in the model pipeline. Defaults to None.

                    add_prepends_for_pipe_to_all (bool, optional): Whether to add the pipeline element prepend lables to each key. Defaults to True.

                """

                if add_prepends_for_pipe_to_all is True:

                    if isinstance(feature_scaling_param_grid, dict):

                        feature_scaling_param_grid = (

                            self._update_dictionary_keys_for_sklearn_pipeline(

                                feature_scaling_param_grid, "feature_scaling"

                            )

                        )

                    if isinstance(dimensionality_reduction_param_grid, dict):

                        dimensionality_reduction_param_grid = (

                            self._update_dictionary_keys_for_sklearn_pipeline(

                                dimensionality_reduction_param_grid, "dimensionality_reduction"

                            )

                        )

                    if isinstance(feature_selection_param_grid, dict):

                        feature_selection_param_grid = (

                            self._update_dictionary_keys_for_sklearn_pipeline(

                                feature_selection_param_grid, "feature_selection"

                            )

                        )

                    if isinstance(model_param_grid, dict):

                        model_param_grid = self._update_dictionary_keys_for_sklearn_pipeline(

                            model_param_grid, "model"

                        )

                # NOTE: it assumes there are no overlapping keys but doesn't check

                self.param_grid = {}

                if isinstance(feature_scaling_param_grid, dict):

                    self.param_grid = {**self.param_grid, **feature_scaling_param_grid}

                if isinstance(dimensionality_reduction_param_grid, dict):

                    self.param_grid = {**self.param_grid, **dimensionality_reduction_param_grid}

                if isinstance(feature_selection_param_grid, dict):

                    self.param_grid = {**self.param_grid, **feature_selection_param_grid}

                if isinstance(model_param_grid, dict):

                    self.param_grid = {**self.param_grid, **model_param_grid}


#### build_pipline_list

```python3
def build_pipline_list(
    self,
    pipeline_element_order=None,
    feature_scaling: Union[List[str], Callable, str, NoneType] = None,
    dimensionality_reduction: Union[List[str], Callable, str, NoneType] = None,
    feature_selection: Union[List[str], Callable, str, NoneType] = None
) -> list
```

This function builds a list of sklearn pipeline elements. It is stored as list internally so that it can be modified. Once happy call

self.make_pipeline to set the pipeline in the class. Note you can pass a pipeline and param_grid in to the class directly to use.

Args:
    pipeline_element_order (_type_, optional): A dictionary of scikit learn pipeline steps with keys of "feature_scaling",
    "dimensionality_reduction", "feature_selection" and "model" and values of integers which set the order in which these steps will
    happen in the pipeline. Defaults to None.

Raises:
    ValueError: if an unknown step is trying to be added to the pipeline

Returns:
    List: list of tuples of steps for the pipeline

??? example "View Source"
            def build_pipline_list(

                self,

                pipeline_element_order=None,

                feature_scaling: Optional[Union[List[str], Callable, str]] = None,

                dimensionality_reduction: Optional[Union[List[str], Callable, str]] = None,

                feature_selection: Optional[Union[List[str], Callable, str]] = None,

            ) -> list:

                """

                This function builds a list of sklearn pipeline elements. It is stored as list internally so that it can be modified. Once happy call

                self.make_pipeline to set the pipeline in the class. Note you can pass a pipeline and param_grid in to the class directly to use.

                Args:

                    pipeline_element_order (_type_, optional): A dictionary of scikit learn pipeline steps with keys of "feature_scaling",

                    "dimensionality_reduction", "feature_selection" and "model" and values of integers which set the order in which these steps will

                    happen in the pipeline. Defaults to None.

                Raises:

                    ValueError: if an unknown step is trying to be added to the pipeline

                Returns:

                    List: list of tuples of steps for the pipeline

                """

                if pipeline_element_order is None:

                    orders = self._get_default_pipe_order()

                    self.pipe_order = orders

                else:

                    orders = self._get_default_pipe_order(**pipeline_element_order)

                    self.pipe_order = orders

                if feature_scaling is not None:

                    self.feature_scaling = feature_scaling

                if dimensionality_reduction is not None:

                    self.dimensionality_reduction = dimensionality_reduction

                if feature_selection is not None:

                    self.feature_selection = feature_selection

                self.pipeline_elts = list()

                for elt, x in sorted(orders.items(), key=lambda y: y[1]):

                    log.info(f"Pipeline element: {elt} in order {x}")

                    if elt == "feature_scaling" and self.feature_scaling is not None:

                        log.info("\tAdded feature scaling to pipeline")

                        self.pipeline_elts.append((elt, self.feature_scaling))

                    elif (

                        elt == "dimensionality_reduction"

                        and self.dimensionality_reduction is not None

                    ):

                        log.info("\tAdded dimensionality reduction to pipeline")

                        self.pipeline_elts.append((elt, self.dimensionality_reduction))

                    elif elt == "feature_selection" and self.feature_selection is not None:

                        log.info("\tAdded feature selection to pipeline")

                        self.pipeline_elts.append((elt, self.feature_selection))

                    elif elt == "model" and self.initial_model is not None:

                        log.info("\tAdded model to pipeline")

                        self.pipeline_elts.append((elt, self.initial_model))

                    elif all(

                        ent != elt

                        for ent in [

                            "feature_scaling",

                            "dimensionality_reduction",

                            "feature_selection",

                            "model",

                        ]

                    ):

                        raise ValueError(

                            f"Unknown pipeline entry requested {elt}. This class can only accept feature_scaling, dimensionality_reduction, feature_selection, model."

                        )

                    else:

                        log.info(f"No {elt} in the pipeline")

                return self.pipeline_elts


#### build_train_test_validate_splits

```python3
def build_train_test_validate_splits(
    self,
    smiles_column: Optional[str] = None,
    task_columns: Optional[List[str]] = None,
    df: Optional[pandas.core.frame.DataFrame] = None,
    feature_columns: Optional[List[str]] = None,
    feature_column_regex: Optional[str] = None,
    splitter_type: Union[str, deepchem.splits.splitters.Splitter, NoneType] = None,
    train: Union[float, int, NoneType] = 0.8,
    test: Union[float, int, NoneType] = 0.1,
    holdout: Union[float, int, NoneType] = 0.1,
    kfold: Optional[int] = None,
    **kwargs
) -> None
```

This is a function to build the data splits into train test and validate sets

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | pd.DataFrame | All of the data in one dataframe to split into a dataset for ML/AI | None |
| smiles_column | str | The column name that contains smiles strings | None |
| task_columns | List[str] | List of y task column names | None |
| feature_columns | Optional[List[str]] | Must provide this or feature_column_regex. If this is provided it is a list of the column names that correspond to features i.e. the X mat. Defaults to None. | None |
| fesature_column_regex | Optional[str] | Must provide this or feature_columns. If this is provided it is a partial or full string to identify all columns that contain it as feature columns. Defaults to None. | None |
| df | Optional[pd.DataFrame] | Dataframe to convert to a dataset and split. Please note the keyword args for this are passed through kwargs here, they correspond to the arguments for pandas_to_deepchem(). Defaults to None. | None |
| train | Optional[Union[float, int]] | the number or fraction (as a decimal) of the data to use for training. Defaults to None. | None |
| test | Optional[Union[float, int]] | the number or fraction (as a decimal) of the data to use for testing. Defaults to None. | None |
| validation | Optional[Union[float, int]] | the number or fraction (as a decimal) of the data to use for validation. Defaults to None. | None |
| _kfold | Optional[int] | The number of k fold splits. Defaults to None. | None |

**Raises:**

| Type | Description |
|---|---|
| RunTimeError | data is not found in the class and is not passed to the call |

??? example "View Source"
            def build_train_test_validate_splits(

                self,

                smiles_column: Optional[str] = None,

                task_columns: Optional[List[str]] = None,

                df: Optional[pd.DataFrame] = None,

                feature_columns: Optional[List[str]] = None,

                feature_column_regex: Optional[str] = None,

                splitter_type: Optional[Union[str, dc.splits.Splitter]] = None,

                train: Optional[Union[float, int]] = 0.8,

                test: Optional[Union[float, int]] = 0.1,

                holdout: Optional[Union[float, int]] = 0.1,

                kfold: Optional[int] = None,  # TODO: set up for kfold

                **kwargs,

            ) -> None:

                """

                This is a function to build the data splits into train test and validate sets

                Args:

                    df (pd.DataFrame): All of the data in one dataframe to split into a dataset for ML/AI

                    smiles_column (str): The column name that contains smiles strings

                    task_columns (List[str]): List of y task column names

                    feature_columns (Optional[List[str]], optional): Must provide this or feature_column_regex. If this is provided it is a list of the column names that correspond to features i.e. the X mat. Defaults to None.

                    fesature_column_regex (Optional[str], optional): Must provide this or feature_columns. If this is provided it is a partial or full string to identify all columns that contain it as feature columns. Defaults to None.

                    df (Optional[pd.DataFrame], optional): Dataframe to convert to a dataset and split. Please note the keyword args for this are passed through kwargs here, they correspond to the arguments for pandas_to_deepchem(). Defaults to None.

                    train (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for training. Defaults to None.

                    test (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for testing. Defaults to None.

                    validation (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for validation. Defaults to None.

                    _kfold (Optional[int], optional): The number of k fold splits. Defaults to None.

                Raises:

                    RunTimeError: data is not found in the class and is not passed to the call

                """

                if isinstance(self.data_df, pd.DataFrame):

                    tmp_dat = datasets.pandas_to_deepchem(

                        self.data_df,

                        smiles_column=smiles_column,

                        task_columns=task_columns,

                        feature_columns=feature_columns,

                        feature_column_regex=feature_column_regex,

                    )

                elif isinstance(df, pd.DataFrame):

                    self.data_df = df

                    tmp_dat = datasets.pandas_to_deepchem(

                        self.data_df,

                        smiles_column=smiles_column,

                        task_columns=task_columns,

                        feature_columns=feature_columns,

                        feature_column_regex=feature_column_regex,

                    )

                else:

                    raise RuntimeError(

                        "Data not found! Please set self.data_df or pass in a dataframe to this function."

                    )

                log.debug(

                    f"Training fraction {train} {type(train)} Testing fraction {test} {type(test)}validation fraction {holdout} {type(holdout)}"

                )

                if holdout is not None and kfold is None and splitter_type is not None:

                    train, holdout, test, self.splitter = datasets.molecular_dataset_split(

                        ds=tmp_dat,

                        splitter_type=splitter_type,

                        train=train,

                        test=test,

                        holdout=holdout,

                        **kwargs,

                    )

                    self.train = train

                    self.test = test

                    self.validate = holdout

                    self.X_train = train.X

                    self.y_train = train.y

                    self.X_test = test.X

                    self.y_test = test.y

                    self.X_validate = holdout.X

                    self.y_validate = holdout.y

                elif holdout is None and kfold is None and splitter_type is not None:

                    train, test, splitter = datasets.molecular_dataset_split(

                        ds=tmp_dat,

                        splitter_type=splitter_type,

                        train=train,

                        test=test,

                        holdout=holdout,

                        **kwargs,

                    )

                    self.train = train

                    self.test = test

                    self.X_train = train.X

                    self.y_train = train.y

                    self.X_test = test.X

                    self.y_test = test.y

                elif kfold is not None and splitter_type is not None:

                    if holdout is None:

                        log.info(f"Generated {kfold} fold")

                        self.kfolds, self.splitter = datasets.molecular_dataset_split(

                            ds=tmp_dat, splitter_type=splitter_type, kfold=kfold, **kwargs

                        )

                    else:

                        log.info(f"Generated {kfold} fold and validation holdout")

                        self.kfolds, self.validate, self.splitter = (

                            datasets.molecular_dataset_split(

                                ds=tmp_dat,

                                splitter_type=splitter_type,

                                holdout=holdout,

                                kfold=kfold,

                                **kwargs,

                            )

                        )

                        self.X_validate = self.validate.X

                        self.y_validate = self.validate.y

                else:

                    log.warning(

                        "Using SciKit-Learn random splitter you may want to consider using molecule splitters is this is for molecular data."

                    )

                    X_train, X_test, y_train, y_test = train_test_split(

                        tmp_dat.X,

                        tmp_dat.y,

                        test_size=0.2,

                        train_size=0.8,

                        random_state=helpers.random_seed,

                        shuffle=True,

                    )

                    self.X_train = X_train

                    self.y_train = y_train

                    self.X_test = X_test

                    self.y_test = y_test

                if splitter_type is not None:

                    log.debug(

                        f"Train: {train}, Validation: {holdout}, Test {test}, kfolds: {kfold}, splitter {self.splitter}"

                    )

                else:

                    log.debug(

                        f"Train: {train}, Validation: {holdout}, Test {test}, kfolds: {kfold}"

                    )


#### load_pipeline_into_class

```python3
def load_pipeline_into_class(
    self,
    filename: str = 'model.joblib'
)
```

Load a model pipeline into this class

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| filename | str | joblib file to load the model pipeline from | None |

??? example "View Source"
            def load_pipeline_into_class(self, filename: str = "model.joblib"):

                """

                Load a model pipeline into this class

                Args:

                    filename (str): joblib file to load the model pipeline from

                """

                self.model = joblib.load(filename)


#### make_pipeline

```python3
def make_pipeline(
    self,
    return_pipeline: bool = False
)
```

Function to build a pipeline if the pipeline elements have been defined

??? example "View Source"
            def make_pipeline(self, return_pipeline: bool = False):

                """

                Function to build a pipeline if the pipeline elements have been defined

                """

                if self.pipeline_elts is not None:

                    log.info(f"Setting the scikit learn pipeline using {self.pipeline_elts}.")

                    if return_pipeline is True:

                        return Pipeline(self.pipeline_elts)

                    else:

                        self.pipeline = Pipeline(self.pipeline_elts)


#### optimize_model

```python3
def optimize_model(
    self
)
```

Function to train a model pipeline using the setting defined

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| cv | int | The number of corss validation folds used to establish the best hyper-parameters. Defaults to 5. | 5 |
| verbose | bool | To print verbose or not trianing information. Defaults to True. | True |

??? example "View Source"
            def optimize_model(

                self,

            ):

                """

                Function to train a model pipeline using the setting defined

                Args:

                    cv (int, optional): The number of corss validation folds used to establish the best hyper-parameters. Defaults to 5.

                    verbose (bool, optional): To print verbose or not trianing information. Defaults to True.

                """

                pass


#### save_model

```python3
def save_model(
    self,
    filename: str = 'model.joblib'
)
```

Save the final model pipeline to a jobloib file

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| filename | str | filename to save | None |

??? example "View Source"
            def save_model(self, filename: str = "model.joblib"):

                """

                Save the final model pipeline to a jobloib file

                Args:

                    filename (str): filename to save

                """

                if self.model is not None:

                    joblib.dump(self.model, filename)


#### save_self

```python3
def save_self(
    self,
    filename: str
)
```

Save the entire class as a pickle file

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| filename | str | file name | None |

??? example "View Source"
            def save_self(self, filename: str):

                """

                Save the entire class as a pickle file

                Args:

                    filename (str): file name

                """

                with open(filename, "wb") as fout:

                    pkl.dump(self, fout)

### RXKFoldGridSearchSklearnRegresssor

```python3
class RXKFoldGridSearchSklearnRegresssor(
    data_df: Optional[pandas.core.frame.DataFrame] = None,
    X_train: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    y_train: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    X_test: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    y_test: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    X_val: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    y_val: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    kfolds: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    model: Optional[chemutil.classical_ml_models.skmodel] = None,
    target_name: Optional[str] = None,
    model_name: Optional[str] = None,
    pipeline: Optional[sklearn.pipeline.Pipeline] = None,
    param_grid: Optional[dict] = None,
    _model: Union[chemutil.classical_ml_models.skmodel, Callable, NoneType] = None,
    **kwargs
)
```

Class for building a model using a k fold split and holdout set. This class encapulates a generalized model building process.

To fully optimze a model it will need to be taken put of this class. This is a method wrapper to enable several methods to be tested initially.

It is expected that this class is used in the following manner:
1. Load you data and featurize from your moleucle representation:
```
df = pd.read_csv("my_data.csv)
feat_df = featurization.get_count_ecfp(data_df=df, smiles_column="smiles", return_df=True)
```

2. Select your scikit learn model and parameter grid. You can use the default models provided in this libray which provides name, model and parameter grid to search as follows:
```
my_model = classical_ml_models.linear_models(lasso=False, lars=False)[0]
```

Alternatively wrap up your chosen model:

```
from chemutil.classical_ml_models import skmodel

my_model = skmodel(
    type(RandomForestRegressor()).__name__,
    RandomForestRegressor(random_state=helpers.random_seed, n_jobs=-1),
    {
        "n_estimators": [50, 100, 200],
        "min_samples_split": [2, 4],
        "ccp_alpha": [0.0, 0.05],
        "max_features": ["sqrt", "log2", 1.0],
    },
    {},
    True,
    "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",
    sklearn.__version__,
    datetime.now(),
)
```

3. Instantiate this class and load the model and data into the class
```
cls = RXGridSearchSklearnRegresssor(
    feat_df,
    model=linear_regression_models.model,
    model_name=linear_regression_models.name
)
```

4. Set your data splits this example is based on train test splits and uses tye molecular weight splitter from deepchem
```
cls.build_trian_test_validate_splits(
    smiles_column="smiles",
    task_columns=["taskX"],
    feature_column_regex="ecfp_count_bit_",
    splitter_type="mw",
    train=0.8,
    test=0.2,
    validation=None
)
```

5. Define your sklearn pipeline to include, in addition to the model, any of the following elements feature_scaling, dimensionality_reduction or feature_selection. provide the word "passthrough" to activate this
element or if you are fixing to just one method you can add that directly i.e. feature_scaling=MinMaxScaler(). Following this define the parameter grid to search. if you have activated any of the optional pipeline
elements if you have activated an element with "passthrough" you must provide a key of the element mapping to the method eg {"feature_scaling": MinMaxScaler()}. You can then extend the dictionary with
hyper-parameters to try i.e. "dimensionality_reduction": PCA(), "n_components": [2, 3, 4], "iterated_power": [2, 5, 7]}. For the model provide a dictionary of hyper parameters to grid search, not the dafualt models
in this library have a grid which can be used by default.

```
model.build_pipline_list(feature_scaling="passthrough")
cls.build_parameter_grid(
    feature_scaling_param_grid={"feature_scaling": [MinMaxScaler(), StandardScaler()]},
    model_param_grid=my_model.default_param_grid
)
cls.make_pipeline()
```

6. You can now eun the grid search, defining scoring functions to use. Note the first one is used to identify the best model and refit
```
gs = cls.optimize_model(scoring=("neg_root_mean_squared_error", "r2"))
```
#### Attributes

| Name | Type | Description | Default |
|---|---|---|---|
| RXGridSearchSklearnRegresssor | _type_ | _description_ | None |

#### Ancestors (in MRO)

* chemutil.classical_ml_model_build.RXGridSearchSklearnRegresssor

#### Class variables

```python3
log
```

#### Instance variables

```python3
Xtest
```

Set the features for training

```python3
Xtrain
```

Set the features for training

```python3
model_name
```

Set the model name

```python3
ytest
```

Set the features for training

```python3
ytrain
```

Set the target for training

#### Methods


#### build_parameter_grid

```python3
def build_parameter_grid(
    self,
    feature_scaling_param_grid: Optional[dict] = None,
    dimensionality_reduction_param_grid: Optional[dict] = None,
    feature_selection_param_grid: Optional[dict] = None,
    model_param_grid: Optional[dict] = None,
    add_prepends_for_pipe_to_all: bool = True
)
```

Function to build the parameter grid. It will by default prepend the pipeline step to the parameters as needed. Then it combines all input dictionaries, not there should not

be any overlapping keys in the separete input dictionaries

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| feature_scaling_param_grid | Optional[dict] | dictionary of options for feature scaling (if any) in the model pipeline. Defaults to None. | None |
| dimensionality_reduction_param_grid | Optional[dict] | dictionary of options for dimensionality reduction (if any) in the model pipeline. Defaults to None. | None |
| feature_selection_param_grid | Optional[dict] | dictionary of options for feature selection (if any) in the model pipeline. Defaults to None. | None |
| model_param_grid | Optional[dict] | dictionary of options for model hyper-parameters in the model pipeline. Defaults to None. | None |
| add_prepends_for_pipe_to_all | bool | Whether to add the pipeline element prepend lables to each key. Defaults to True. | True |

??? example "View Source"
            def build_parameter_grid(

                self,

                feature_scaling_param_grid: Optional[dict] = None,

                dimensionality_reduction_param_grid: Optional[dict] = None,

                feature_selection_param_grid: Optional[dict] = None,

                model_param_grid: Optional[dict] = None,

                add_prepends_for_pipe_to_all: bool = True,

            ):

                """

                Function to build the parameter grid. It will by default prepend the pipeline step to the parameters as needed. Then it combines all input dictionaries, not there should not

                be any overlapping keys in the separete input dictionaries

                Args:

                    feature_scaling_param_grid (Optional[dict], optional): dictionary of options for feature scaling (if any) in the model pipeline. Defaults to None.

                    dimensionality_reduction_param_grid (Optional[dict], optional): dictionary of options for dimensionality reduction (if any) in the model pipeline. Defaults to None.

                    feature_selection_param_grid (Optional[dict], optional): dictionary of options for feature selection (if any) in the model pipeline. Defaults to None.

                    model_param_grid (Optional[dict], optional): dictionary of options for model hyper-parameters in the model pipeline. Defaults to None.

                    add_prepends_for_pipe_to_all (bool, optional): Whether to add the pipeline element prepend lables to each key. Defaults to True.

                """

                if add_prepends_for_pipe_to_all is True:

                    if isinstance(feature_scaling_param_grid, dict):

                        feature_scaling_param_grid = (

                            self._update_dictionary_keys_for_sklearn_pipeline(

                                feature_scaling_param_grid, "feature_scaling"

                            )

                        )

                    if isinstance(dimensionality_reduction_param_grid, dict):

                        dimensionality_reduction_param_grid = (

                            self._update_dictionary_keys_for_sklearn_pipeline(

                                dimensionality_reduction_param_grid, "dimensionality_reduction"

                            )

                        )

                    if isinstance(feature_selection_param_grid, dict):

                        feature_selection_param_grid = (

                            self._update_dictionary_keys_for_sklearn_pipeline(

                                feature_selection_param_grid, "feature_selection"

                            )

                        )

                    if isinstance(model_param_grid, dict):

                        model_param_grid = self._update_dictionary_keys_for_sklearn_pipeline(

                            model_param_grid, "model"

                        )

                # NOTE: it assumes there are no overlapping keys but doesn't check

                self.param_grid = {}

                if isinstance(feature_scaling_param_grid, dict):

                    self.param_grid = {**self.param_grid, **feature_scaling_param_grid}

                if isinstance(dimensionality_reduction_param_grid, dict):

                    self.param_grid = {**self.param_grid, **dimensionality_reduction_param_grid}

                if isinstance(feature_selection_param_grid, dict):

                    self.param_grid = {**self.param_grid, **feature_selection_param_grid}

                if isinstance(model_param_grid, dict):

                    self.param_grid = {**self.param_grid, **model_param_grid}


#### build_pipline_list

```python3
def build_pipline_list(
    self,
    pipeline_element_order=None,
    feature_scaling: Union[List[str], Callable, str, NoneType] = None,
    dimensionality_reduction: Union[List[str], Callable, str, NoneType] = None,
    feature_selection: Union[List[str], Callable, str, NoneType] = None
) -> list
```

This function builds a list of sklearn pipeline elements. It is stored as list internally so that it can be modified. Once happy call

self.make_pipeline to set the pipeline in the class. Note you can pass a pipeline and param_grid in to the class directly to use.

Args:
    pipeline_element_order (_type_, optional): A dictionary of scikit learn pipeline steps with keys of "feature_scaling",
    "dimensionality_reduction", "feature_selection" and "model" and values of integers which set the order in which these steps will
    happen in the pipeline. Defaults to None.

Raises:
    ValueError: if an unknown step is trying to be added to the pipeline

Returns:
    List: list of tuples of steps for the pipeline

??? example "View Source"
            def build_pipline_list(

                self,

                pipeline_element_order=None,

                feature_scaling: Optional[Union[List[str], Callable, str]] = None,

                dimensionality_reduction: Optional[Union[List[str], Callable, str]] = None,

                feature_selection: Optional[Union[List[str], Callable, str]] = None,

            ) -> list:

                """

                This function builds a list of sklearn pipeline elements. It is stored as list internally so that it can be modified. Once happy call

                self.make_pipeline to set the pipeline in the class. Note you can pass a pipeline and param_grid in to the class directly to use.

                Args:

                    pipeline_element_order (_type_, optional): A dictionary of scikit learn pipeline steps with keys of "feature_scaling",

                    "dimensionality_reduction", "feature_selection" and "model" and values of integers which set the order in which these steps will

                    happen in the pipeline. Defaults to None.

                Raises:

                    ValueError: if an unknown step is trying to be added to the pipeline

                Returns:

                    List: list of tuples of steps for the pipeline

                """

                if pipeline_element_order is None:

                    orders = self._get_default_pipe_order()

                    self.pipe_order = orders

                else:

                    orders = self._get_default_pipe_order(**pipeline_element_order)

                    self.pipe_order = orders

                if feature_scaling is not None:

                    self.feature_scaling = feature_scaling

                if dimensionality_reduction is not None:

                    self.dimensionality_reduction = dimensionality_reduction

                if feature_selection is not None:

                    self.feature_selection = feature_selection

                self.pipeline_elts = list()

                for elt, x in sorted(orders.items(), key=lambda y: y[1]):

                    log.info(f"Pipeline element: {elt} in order {x}")

                    if elt == "feature_scaling" and self.feature_scaling is not None:

                        log.info("\tAdded feature scaling to pipeline")

                        self.pipeline_elts.append((elt, self.feature_scaling))

                    elif (

                        elt == "dimensionality_reduction"

                        and self.dimensionality_reduction is not None

                    ):

                        log.info("\tAdded dimensionality reduction to pipeline")

                        self.pipeline_elts.append((elt, self.dimensionality_reduction))

                    elif elt == "feature_selection" and self.feature_selection is not None:

                        log.info("\tAdded feature selection to pipeline")

                        self.pipeline_elts.append((elt, self.feature_selection))

                    elif elt == "model" and self.initial_model is not None:

                        log.info("\tAdded model to pipeline")

                        self.pipeline_elts.append((elt, self.initial_model))

                    elif all(

                        ent != elt

                        for ent in [

                            "feature_scaling",

                            "dimensionality_reduction",

                            "feature_selection",

                            "model",

                        ]

                    ):

                        raise ValueError(

                            f"Unknown pipeline entry requested {elt}. This class can only accept feature_scaling, dimensionality_reduction, feature_selection, model."

                        )

                    else:

                        log.info(f"No {elt} in the pipeline")

                return self.pipeline_elts


#### build_train_test_validate_splits

```python3
def build_train_test_validate_splits(
    self,
    smiles_column: Optional[str] = None,
    task_columns: Optional[List[str]] = None,
    df: Optional[pandas.core.frame.DataFrame] = None,
    feature_columns: Optional[List[str]] = None,
    feature_column_regex: Optional[str] = None,
    splitter_type: Union[str, deepchem.splits.splitters.Splitter, NoneType] = None,
    train: Union[float, int, NoneType] = 0.8,
    test: Union[float, int, NoneType] = 0.1,
    holdout: Union[float, int, NoneType] = 0.1,
    kfold: Optional[int] = None,
    **kwargs
) -> None
```

This is a function to build the data splits into train test and validate sets

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | pd.DataFrame | All of the data in one dataframe to split into a dataset for ML/AI | None |
| smiles_column | str | The column name that contains smiles strings | None |
| task_columns | List[str] | List of y task column names | None |
| feature_columns | Optional[List[str]] | Must provide this or feature_column_regex. If this is provided it is a list of the column names that correspond to features i.e. the X mat. Defaults to None. | None |
| fesature_column_regex | Optional[str] | Must provide this or feature_columns. If this is provided it is a partial or full string to identify all columns that contain it as feature columns. Defaults to None. | None |
| df | Optional[pd.DataFrame] | Dataframe to convert to a dataset and split. Please note the keyword args for this are passed through kwargs here, they correspond to the arguments for pandas_to_deepchem(). Defaults to None. | None |
| train | Optional[Union[float, int]] | the number or fraction (as a decimal) of the data to use for training. Defaults to None. | None |
| test | Optional[Union[float, int]] | the number or fraction (as a decimal) of the data to use for testing. Defaults to None. | None |
| validation | Optional[Union[float, int]] | the number or fraction (as a decimal) of the data to use for validation. Defaults to None. | None |
| _kfold | Optional[int] | The number of k fold splits. Defaults to None. | None |

**Raises:**

| Type | Description |
|---|---|
| RunTimeError | data is not found in the class and is not passed to the call |

??? example "View Source"
            def build_train_test_validate_splits(

                self,

                smiles_column: Optional[str] = None,

                task_columns: Optional[List[str]] = None,

                df: Optional[pd.DataFrame] = None,

                feature_columns: Optional[List[str]] = None,

                feature_column_regex: Optional[str] = None,

                splitter_type: Optional[Union[str, dc.splits.Splitter]] = None,

                train: Optional[Union[float, int]] = 0.8,

                test: Optional[Union[float, int]] = 0.1,

                holdout: Optional[Union[float, int]] = 0.1,

                kfold: Optional[int] = None,  # TODO: set up for kfold

                **kwargs,

            ) -> None:

                """

                This is a function to build the data splits into train test and validate sets

                Args:

                    df (pd.DataFrame): All of the data in one dataframe to split into a dataset for ML/AI

                    smiles_column (str): The column name that contains smiles strings

                    task_columns (List[str]): List of y task column names

                    feature_columns (Optional[List[str]], optional): Must provide this or feature_column_regex. If this is provided it is a list of the column names that correspond to features i.e. the X mat. Defaults to None.

                    fesature_column_regex (Optional[str], optional): Must provide this or feature_columns. If this is provided it is a partial or full string to identify all columns that contain it as feature columns. Defaults to None.

                    df (Optional[pd.DataFrame], optional): Dataframe to convert to a dataset and split. Please note the keyword args for this are passed through kwargs here, they correspond to the arguments for pandas_to_deepchem(). Defaults to None.

                    train (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for training. Defaults to None.

                    test (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for testing. Defaults to None.

                    validation (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for validation. Defaults to None.

                    _kfold (Optional[int], optional): The number of k fold splits. Defaults to None.

                Raises:

                    RunTimeError: data is not found in the class and is not passed to the call

                """

                if isinstance(self.data_df, pd.DataFrame):

                    tmp_dat = datasets.pandas_to_deepchem(

                        self.data_df,

                        smiles_column=smiles_column,

                        task_columns=task_columns,

                        feature_columns=feature_columns,

                        feature_column_regex=feature_column_regex,

                    )

                elif isinstance(df, pd.DataFrame):

                    self.data_df = df

                    tmp_dat = datasets.pandas_to_deepchem(

                        self.data_df,

                        smiles_column=smiles_column,

                        task_columns=task_columns,

                        feature_columns=feature_columns,

                        feature_column_regex=feature_column_regex,

                    )

                else:

                    raise RuntimeError(

                        "Data not found! Please set self.data_df or pass in a dataframe to this function."

                    )

                log.debug(

                    f"Training fraction {train} {type(train)} Testing fraction {test} {type(test)}validation fraction {holdout} {type(holdout)}"

                )

                if holdout is not None and kfold is None and splitter_type is not None:

                    train, holdout, test, self.splitter = datasets.molecular_dataset_split(

                        ds=tmp_dat,

                        splitter_type=splitter_type,

                        train=train,

                        test=test,

                        holdout=holdout,

                        **kwargs,

                    )

                    self.train = train

                    self.test = test

                    self.validate = holdout

                    self.X_train = train.X

                    self.y_train = train.y

                    self.X_test = test.X

                    self.y_test = test.y

                    self.X_validate = holdout.X

                    self.y_validate = holdout.y

                elif holdout is None and kfold is None and splitter_type is not None:

                    train, test, splitter = datasets.molecular_dataset_split(

                        ds=tmp_dat,

                        splitter_type=splitter_type,

                        train=train,

                        test=test,

                        holdout=holdout,

                        **kwargs,

                    )

                    self.train = train

                    self.test = test

                    self.X_train = train.X

                    self.y_train = train.y

                    self.X_test = test.X

                    self.y_test = test.y

                elif kfold is not None and splitter_type is not None:

                    if holdout is None:

                        log.info(f"Generated {kfold} fold")

                        self.kfolds, self.splitter = datasets.molecular_dataset_split(

                            ds=tmp_dat, splitter_type=splitter_type, kfold=kfold, **kwargs

                        )

                    else:

                        log.info(f"Generated {kfold} fold and validation holdout")

                        self.kfolds, self.validate, self.splitter = (

                            datasets.molecular_dataset_split(

                                ds=tmp_dat,

                                splitter_type=splitter_type,

                                holdout=holdout,

                                kfold=kfold,

                                **kwargs,

                            )

                        )

                        self.X_validate = self.validate.X

                        self.y_validate = self.validate.y

                else:

                    log.warning(

                        "Using SciKit-Learn random splitter you may want to consider using molecule splitters is this is for molecular data."

                    )

                    X_train, X_test, y_train, y_test = train_test_split(

                        tmp_dat.X,

                        tmp_dat.y,

                        test_size=0.2,

                        train_size=0.8,

                        random_state=helpers.random_seed,

                        shuffle=True,

                    )

                    self.X_train = X_train

                    self.y_train = y_train

                    self.X_test = X_test

                    self.y_test = y_test

                if splitter_type is not None:

                    log.debug(

                        f"Train: {train}, Validation: {holdout}, Test {test}, kfolds: {kfold}, splitter {self.splitter}"

                    )

                else:

                    log.debug(

                        f"Train: {train}, Validation: {holdout}, Test {test}, kfolds: {kfold}"

                    )


#### load_pipeline_into_class

```python3
def load_pipeline_into_class(
    self,
    filename: str = 'model.joblib'
)
```

Load a model pipeline into this class

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| filename | str | joblib file to load the model pipeline from | None |

??? example "View Source"
            def load_pipeline_into_class(self, filename: str = "model.joblib"):

                """

                Load a model pipeline into this class

                Args:

                    filename (str): joblib file to load the model pipeline from

                """

                self.model = joblib.load(filename)


#### make_pipeline

```python3
def make_pipeline(
    self,
    return_pipeline: bool = False
)
```

Function to build a pipeline if the pipeline elements have been defined

??? example "View Source"
            def make_pipeline(self, return_pipeline: bool = False):

                """

                Function to build a pipeline if the pipeline elements have been defined

                """

                if self.pipeline_elts is not None:

                    log.info(f"Setting the scikit learn pipeline using {self.pipeline_elts}.")

                    if return_pipeline is True:

                        return Pipeline(self.pipeline_elts)

                    else:

                        self.pipeline = Pipeline(self.pipeline_elts)


#### optimize_model

```python3
def optimize_model(
    self,
    grid_cv: int = 5,
    verbose: bool = True,
    scoring: Optional[dict[Union[str, Callable]]] = None,
    refit: Optional[str] = None,
    return_gs: bool = False,
    **kwargs
)
```

Function to train a model pipeline using the setting defined

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| grid_cv | int | The number of corss validation folds used to establish the best hyper-parameters. Defaults to 5. | 5 |
| verbose | bool | To print verbose or not trianing information. Defaults to True. | True |

??? example "View Source"
            def optimize_model(

                self,

                grid_cv: int = 5,

                verbose: bool = True,

                scoring: Optional[dict[Union[str, Callable]]] = None,

                refit: Optional[str] = None,

                return_gs: bool = False,

                **kwargs,

            ):

                """

                Function to train a model pipeline using the setting defined

                Args:

                    grid_cv (int, optional): The number of corss validation folds used to establish the best hyper-parameters. Defaults to 5.

                    verbose (bool, optional): To print verbose or not trianing information. Defaults to True.

                """

                log.info(

                    "This method will fit k modles on the data avaliable (exluding holdout validation data). This is done over k folds i.e. one model is trained using the best hyper-parameters found for each fold. The returned model must be evaluated on unseen held out data."

                )

                if isinstance(self.param_grid, dict):

                    log.info(f"Will use the parameter grid: {yaml.dump(self.param_grid)}")

                if isinstance(self.pipeline, Pipeline):

                    log.info("Will run the following pipeline to optimize the hyper parmaeters")

                if self.X_train is not None and self.y_train is not None:

                    log.info("Using the trianing data to run a Grid search CV")

                if scoring is None:

                    scoring = (

                        "neg_mean_squared_error",

                        "neg_root_mean_squared_error",

                        "neg_mean_absolute_percentage_error",

                        "r2",

                    )

                    log.info(f"Using default scoring: {scoring}")

                if refit is None:

                    refit = scoring[0]

                model_ensemble = []

                self.completed_grid_search = []

                self.best_parameters = []

                for ith, (train, test) in enumerate(self.kfolds):

                    log.info(f"{ith}: {train.X.shape}, {test.X.shape}")

                    pl = self.make_pipeline(return_pipeline=True)

                    grid_search = GridSearchCV(

                        pl,

                        n_jobs=-1,

                        param_grid=self.param_grid,

                        cv=grid_cv,

                        verbose=verbose,

                        scoring=scoring,

                        refit=refit,

                        **kwargs,

                    )

                    if train.y.shape[1] > 1:

                        grid_search.fit(train.X, train.y)

                    else:

                        grid_search.fit(train.X, train.y.ravel())

                    self.completed_grid_search.append(grid_search)

                    model_ensemble.append((f"model_{ith}", grid_search.best_estimator_))

                if return_gs is False:

                    log.info(model_ensemble)

                    m = VotingRegressor(model_ensemble, n_jobs=-1, verbose=True)

                    expected_n_data = sum([d[1].X.shape[0] for d in self.kfolds])

                    X_all_kf_data_concatenated = np.concatenate([d[1].X for d in self.kfolds])

                    log.debug(f"All X concatenate: {X_all_kf_data_concatenated }")

                    y_all_kf_data_concatenated = np.concatenate([d[1].y for d in self.kfolds])

                    log.debug(f"All y concatenate: {y_all_kf_data_concatenated }")

                    try:

                        assert all(

                            ent.shape[0] == expected_n_data

                            for ent in [X_all_kf_data_concatenated, y_all_kf_data_concatenated]

                        )

                    except AssertionError as aerr:

                        raise aerr(

                            f"ERROR - concatenating all kfold data for final fit leads to unequal number of dat points: expected: {expected_n_data} X data {X_all_kf_data_concatenated.shape[0]} y data {y_all_kf_data_concatenated.shape[0]}"

                        )

                    if train.y.shape[1] > 1:

                        m.fit(X_all_kf_data_concatenated, y_all_kf_data_concatenated)

                    else:

                        m.fit(X_all_kf_data_concatenated, y_all_kf_data_concatenated.ravel())

                    self.pipeline = m

                    return self.pipeline

                else:

                    return grid_search


#### save_model

```python3
def save_model(
    self,
    filename: str = 'model.joblib'
)
```

Save the final model pipeline to a jobloib file

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| filename | str | filename to save | None |

??? example "View Source"
            def save_model(self, filename: str = "model.joblib"):

                """

                Save the final model pipeline to a jobloib file

                Args:

                    filename (str): filename to save

                """

                if self.model is not None:

                    joblib.dump(self.model, filename)


#### save_self

```python3
def save_self(
    self,
    filename: str
)
```

Save the entire class as a pickle file

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| filename | str | file name | None |

??? example "View Source"
            def save_self(self, filename: str):

                """

                Save the entire class as a pickle file

                Args:

                    filename (str): file name

                """

                with open(filename, "wb") as fout:

                    pkl.dump(self, fout)

### RXTrainTestGridSearchSklearnRegresssor

```python3
class RXTrainTestGridSearchSklearnRegresssor(
    data_df: Optional[pandas.core.frame.DataFrame] = None,
    X_train: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    y_train: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    X_test: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    y_test: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    X_val: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    y_val: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    kfolds: Union[numpy.ndarray, pandas.core.frame.DataFrame, deepchem.data.datasets.NumpyDataset, NoneType] = None,
    model: Optional[chemutil.classical_ml_models.skmodel] = None,
    target_name: Optional[str] = None,
    model_name: Optional[str] = None,
    pipeline: Optional[sklearn.pipeline.Pipeline] = None,
    param_grid: Optional[dict] = None,
    _model: Union[chemutil.classical_ml_models.skmodel, Callable, NoneType] = None,
    **kwargs
)
```

Class for building a model using a single train test split. This class encapulates a generalized model building process.

To fully optimze a model it will need to be taken put of this class. This is a method wrapper to enable several methods to be tested initially.

It is expected that this class is used in the following manner:
1. Load you data and featurize from your moleucle representation:
```
df = pd.read_csv("my_data.csv)
feat_df = featurization.get_count_ecfp(data_df=df, smiles_column="smiles", return_df=True)
```

2. Select your scikit learn model and parameter grid. You can use the default models provided in this libray which provides name, model and parameter grid to search as follows:
```
my_model = classical_ml_models.linear_models(lasso=False, lars=False)[0]
```

Alternatively wrap up your chosen model:

```
from chemutil.classical_ml_models import skmodel

my_model = skmodel(
    type(RandomForestRegressor()).__name__,
    RandomForestRegressor(random_state=helpers.random_seed, n_jobs=-1),
    {
        "n_estimators": [50, 100, 200],
        "min_samples_split": [2, 4],
        "ccp_alpha": [0.0, 0.05],
        "max_features": ["sqrt", "log2", 1.0],
    },
    {},
    True,
    "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",
    sklearn.__version__,
    datetime.now(),
)
```

3. Instantiate this class and load the model and data into the class
```
cls = RXGridSearchSklearnRegresssor(
    feat_df,
    model=linear_regression_models.model,
    model_name=linear_regression_models.name
)
```

4. Set your data splits this example is based on train test splits and uses tye molecular weight splitter from deepchem
```
cls.build_trian_test_validate_splits(
    smiles_column="smiles",
    task_columns=["taskX"],
    feature_column_regex="ecfp_count_bit_",
    splitter_type="mw",
    train=0.8,
    test=0.2,
    validation=None
)
```

5. Define your sklearn pipeline to include, in addition to the model, any of the following elements feature_scaling, dimensionality_reduction or feature_selection. provide the word "passthrough" to activate this
element or if you are fixing to just one method you can add that directly i.e. feature_scaling=MinMaxScaler(). Following this define the parameter grid to search. if you have activated any of the optional pipeline
elements if you have activated an element with "passthrough" you must provide a key of the element mapping to the method eg {"feature_scaling": MinMaxScaler()}. You can then extend the dictionary with
hyper-parameters to try i.e. "dimensionality_reduction": PCA(), "n_components": [2, 3, 4], "iterated_power": [2, 5, 7]}. For the model provide a dictionary of hyper parameters to grid search, not the dafualt models
in this library have a grid which can be used by default.

```
model.build_pipline_list(feature_scaling="passthrough")
cls.build_parameter_grid(
    feature_scaling_param_grid={"feature_scaling": [MinMaxScaler(), StandardScaler()]},
    model_param_grid=my_model.default_param_grid
)
cls.make_pipeline()
```

6. You can now eun the grid search, defining scoring functions to use. Note the first one is used to identify the best model and refit
```
gs = cls.optimize_model(scoring=("neg_root_mean_squared_error", "r2"))
```
#### Attributes

| Name | Type | Description | Default |
|---|---|---|---|
| RXGridSearchSklearnRegresssor | _type_ | _description_ | None |

#### Ancestors (in MRO)

* chemutil.classical_ml_model_build.RXGridSearchSklearnRegresssor

#### Class variables

```python3
log
```

#### Instance variables

```python3
Xtest
```

Set the features for training

```python3
Xtrain
```

Set the features for training

```python3
model_name
```

Set the model name

```python3
ytest
```

Set the features for training

```python3
ytrain
```

Set the target for training

#### Methods


#### build_parameter_grid

```python3
def build_parameter_grid(
    self,
    feature_scaling_param_grid: Optional[dict] = None,
    dimensionality_reduction_param_grid: Optional[dict] = None,
    feature_selection_param_grid: Optional[dict] = None,
    model_param_grid: Optional[dict] = None,
    add_prepends_for_pipe_to_all: bool = True
)
```

Function to build the parameter grid. It will by default prepend the pipeline step to the parameters as needed. Then it combines all input dictionaries, not there should not

be any overlapping keys in the separete input dictionaries

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| feature_scaling_param_grid | Optional[dict] | dictionary of options for feature scaling (if any) in the model pipeline. Defaults to None. | None |
| dimensionality_reduction_param_grid | Optional[dict] | dictionary of options for dimensionality reduction (if any) in the model pipeline. Defaults to None. | None |
| feature_selection_param_grid | Optional[dict] | dictionary of options for feature selection (if any) in the model pipeline. Defaults to None. | None |
| model_param_grid | Optional[dict] | dictionary of options for model hyper-parameters in the model pipeline. Defaults to None. | None |
| add_prepends_for_pipe_to_all | bool | Whether to add the pipeline element prepend lables to each key. Defaults to True. | True |

??? example "View Source"
            def build_parameter_grid(

                self,

                feature_scaling_param_grid: Optional[dict] = None,

                dimensionality_reduction_param_grid: Optional[dict] = None,

                feature_selection_param_grid: Optional[dict] = None,

                model_param_grid: Optional[dict] = None,

                add_prepends_for_pipe_to_all: bool = True,

            ):

                """

                Function to build the parameter grid. It will by default prepend the pipeline step to the parameters as needed. Then it combines all input dictionaries, not there should not

                be any overlapping keys in the separete input dictionaries

                Args:

                    feature_scaling_param_grid (Optional[dict], optional): dictionary of options for feature scaling (if any) in the model pipeline. Defaults to None.

                    dimensionality_reduction_param_grid (Optional[dict], optional): dictionary of options for dimensionality reduction (if any) in the model pipeline. Defaults to None.

                    feature_selection_param_grid (Optional[dict], optional): dictionary of options for feature selection (if any) in the model pipeline. Defaults to None.

                    model_param_grid (Optional[dict], optional): dictionary of options for model hyper-parameters in the model pipeline. Defaults to None.

                    add_prepends_for_pipe_to_all (bool, optional): Whether to add the pipeline element prepend lables to each key. Defaults to True.

                """

                if add_prepends_for_pipe_to_all is True:

                    if isinstance(feature_scaling_param_grid, dict):

                        feature_scaling_param_grid = (

                            self._update_dictionary_keys_for_sklearn_pipeline(

                                feature_scaling_param_grid, "feature_scaling"

                            )

                        )

                    if isinstance(dimensionality_reduction_param_grid, dict):

                        dimensionality_reduction_param_grid = (

                            self._update_dictionary_keys_for_sklearn_pipeline(

                                dimensionality_reduction_param_grid, "dimensionality_reduction"

                            )

                        )

                    if isinstance(feature_selection_param_grid, dict):

                        feature_selection_param_grid = (

                            self._update_dictionary_keys_for_sklearn_pipeline(

                                feature_selection_param_grid, "feature_selection"

                            )

                        )

                    if isinstance(model_param_grid, dict):

                        model_param_grid = self._update_dictionary_keys_for_sklearn_pipeline(

                            model_param_grid, "model"

                        )

                # NOTE: it assumes there are no overlapping keys but doesn't check

                self.param_grid = {}

                if isinstance(feature_scaling_param_grid, dict):

                    self.param_grid = {**self.param_grid, **feature_scaling_param_grid}

                if isinstance(dimensionality_reduction_param_grid, dict):

                    self.param_grid = {**self.param_grid, **dimensionality_reduction_param_grid}

                if isinstance(feature_selection_param_grid, dict):

                    self.param_grid = {**self.param_grid, **feature_selection_param_grid}

                if isinstance(model_param_grid, dict):

                    self.param_grid = {**self.param_grid, **model_param_grid}


#### build_pipline_list

```python3
def build_pipline_list(
    self,
    pipeline_element_order=None,
    feature_scaling: Union[List[str], Callable, str, NoneType] = None,
    dimensionality_reduction: Union[List[str], Callable, str, NoneType] = None,
    feature_selection: Union[List[str], Callable, str, NoneType] = None
) -> list
```

This function builds a list of sklearn pipeline elements. It is stored as list internally so that it can be modified. Once happy call

self.make_pipeline to set the pipeline in the class. Note you can pass a pipeline and param_grid in to the class directly to use.

Args:
    pipeline_element_order (_type_, optional): A dictionary of scikit learn pipeline steps with keys of "feature_scaling",
    "dimensionality_reduction", "feature_selection" and "model" and values of integers which set the order in which these steps will
    happen in the pipeline. Defaults to None.

Raises:
    ValueError: if an unknown step is trying to be added to the pipeline

Returns:
    List: list of tuples of steps for the pipeline

??? example "View Source"
            def build_pipline_list(

                self,

                pipeline_element_order=None,

                feature_scaling: Optional[Union[List[str], Callable, str]] = None,

                dimensionality_reduction: Optional[Union[List[str], Callable, str]] = None,

                feature_selection: Optional[Union[List[str], Callable, str]] = None,

            ) -> list:

                """

                This function builds a list of sklearn pipeline elements. It is stored as list internally so that it can be modified. Once happy call

                self.make_pipeline to set the pipeline in the class. Note you can pass a pipeline and param_grid in to the class directly to use.

                Args:

                    pipeline_element_order (_type_, optional): A dictionary of scikit learn pipeline steps with keys of "feature_scaling",

                    "dimensionality_reduction", "feature_selection" and "model" and values of integers which set the order in which these steps will

                    happen in the pipeline. Defaults to None.

                Raises:

                    ValueError: if an unknown step is trying to be added to the pipeline

                Returns:

                    List: list of tuples of steps for the pipeline

                """

                if pipeline_element_order is None:

                    orders = self._get_default_pipe_order()

                    self.pipe_order = orders

                else:

                    orders = self._get_default_pipe_order(**pipeline_element_order)

                    self.pipe_order = orders

                if feature_scaling is not None:

                    self.feature_scaling = feature_scaling

                if dimensionality_reduction is not None:

                    self.dimensionality_reduction = dimensionality_reduction

                if feature_selection is not None:

                    self.feature_selection = feature_selection

                self.pipeline_elts = list()

                for elt, x in sorted(orders.items(), key=lambda y: y[1]):

                    log.info(f"Pipeline element: {elt} in order {x}")

                    if elt == "feature_scaling" and self.feature_scaling is not None:

                        log.info("\tAdded feature scaling to pipeline")

                        self.pipeline_elts.append((elt, self.feature_scaling))

                    elif (

                        elt == "dimensionality_reduction"

                        and self.dimensionality_reduction is not None

                    ):

                        log.info("\tAdded dimensionality reduction to pipeline")

                        self.pipeline_elts.append((elt, self.dimensionality_reduction))

                    elif elt == "feature_selection" and self.feature_selection is not None:

                        log.info("\tAdded feature selection to pipeline")

                        self.pipeline_elts.append((elt, self.feature_selection))

                    elif elt == "model" and self.initial_model is not None:

                        log.info("\tAdded model to pipeline")

                        self.pipeline_elts.append((elt, self.initial_model))

                    elif all(

                        ent != elt

                        for ent in [

                            "feature_scaling",

                            "dimensionality_reduction",

                            "feature_selection",

                            "model",

                        ]

                    ):

                        raise ValueError(

                            f"Unknown pipeline entry requested {elt}. This class can only accept feature_scaling, dimensionality_reduction, feature_selection, model."

                        )

                    else:

                        log.info(f"No {elt} in the pipeline")

                return self.pipeline_elts


#### build_train_test_validate_splits

```python3
def build_train_test_validate_splits(
    self,
    smiles_column: Optional[str] = None,
    task_columns: Optional[List[str]] = None,
    df: Optional[pandas.core.frame.DataFrame] = None,
    feature_columns: Optional[List[str]] = None,
    feature_column_regex: Optional[str] = None,
    splitter_type: Union[str, deepchem.splits.splitters.Splitter, NoneType] = None,
    train: Union[float, int, NoneType] = 0.8,
    test: Union[float, int, NoneType] = 0.1,
    holdout: Union[float, int, NoneType] = 0.1,
    kfold: Optional[int] = None,
    **kwargs
) -> None
```

This is a function to build the data splits into train test and validate sets

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| df | pd.DataFrame | All of the data in one dataframe to split into a dataset for ML/AI | None |
| smiles_column | str | The column name that contains smiles strings | None |
| task_columns | List[str] | List of y task column names | None |
| feature_columns | Optional[List[str]] | Must provide this or feature_column_regex. If this is provided it is a list of the column names that correspond to features i.e. the X mat. Defaults to None. | None |
| fesature_column_regex | Optional[str] | Must provide this or feature_columns. If this is provided it is a partial or full string to identify all columns that contain it as feature columns. Defaults to None. | None |
| df | Optional[pd.DataFrame] | Dataframe to convert to a dataset and split. Please note the keyword args for this are passed through kwargs here, they correspond to the arguments for pandas_to_deepchem(). Defaults to None. | None |
| train | Optional[Union[float, int]] | the number or fraction (as a decimal) of the data to use for training. Defaults to None. | None |
| test | Optional[Union[float, int]] | the number or fraction (as a decimal) of the data to use for testing. Defaults to None. | None |
| validation | Optional[Union[float, int]] | the number or fraction (as a decimal) of the data to use for validation. Defaults to None. | None |
| _kfold | Optional[int] | The number of k fold splits. Defaults to None. | None |

**Raises:**

| Type | Description |
|---|---|
| RunTimeError | data is not found in the class and is not passed to the call |

??? example "View Source"
            def build_train_test_validate_splits(

                self,

                smiles_column: Optional[str] = None,

                task_columns: Optional[List[str]] = None,

                df: Optional[pd.DataFrame] = None,

                feature_columns: Optional[List[str]] = None,

                feature_column_regex: Optional[str] = None,

                splitter_type: Optional[Union[str, dc.splits.Splitter]] = None,

                train: Optional[Union[float, int]] = 0.8,

                test: Optional[Union[float, int]] = 0.1,

                holdout: Optional[Union[float, int]] = 0.1,

                kfold: Optional[int] = None,  # TODO: set up for kfold

                **kwargs,

            ) -> None:

                """

                This is a function to build the data splits into train test and validate sets

                Args:

                    df (pd.DataFrame): All of the data in one dataframe to split into a dataset for ML/AI

                    smiles_column (str): The column name that contains smiles strings

                    task_columns (List[str]): List of y task column names

                    feature_columns (Optional[List[str]], optional): Must provide this or feature_column_regex. If this is provided it is a list of the column names that correspond to features i.e. the X mat. Defaults to None.

                    fesature_column_regex (Optional[str], optional): Must provide this or feature_columns. If this is provided it is a partial or full string to identify all columns that contain it as feature columns. Defaults to None.

                    df (Optional[pd.DataFrame], optional): Dataframe to convert to a dataset and split. Please note the keyword args for this are passed through kwargs here, they correspond to the arguments for pandas_to_deepchem(). Defaults to None.

                    train (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for training. Defaults to None.

                    test (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for testing. Defaults to None.

                    validation (Optional[Union[float, int]], optional): the number or fraction (as a decimal) of the data to use for validation. Defaults to None.

                    _kfold (Optional[int], optional): The number of k fold splits. Defaults to None.

                Raises:

                    RunTimeError: data is not found in the class and is not passed to the call

                """

                if isinstance(self.data_df, pd.DataFrame):

                    tmp_dat = datasets.pandas_to_deepchem(

                        self.data_df,

                        smiles_column=smiles_column,

                        task_columns=task_columns,

                        feature_columns=feature_columns,

                        feature_column_regex=feature_column_regex,

                    )

                elif isinstance(df, pd.DataFrame):

                    self.data_df = df

                    tmp_dat = datasets.pandas_to_deepchem(

                        self.data_df,

                        smiles_column=smiles_column,

                        task_columns=task_columns,

                        feature_columns=feature_columns,

                        feature_column_regex=feature_column_regex,

                    )

                else:

                    raise RuntimeError(

                        "Data not found! Please set self.data_df or pass in a dataframe to this function."

                    )

                log.debug(

                    f"Training fraction {train} {type(train)} Testing fraction {test} {type(test)}validation fraction {holdout} {type(holdout)}"

                )

                if holdout is not None and kfold is None and splitter_type is not None:

                    train, holdout, test, self.splitter = datasets.molecular_dataset_split(

                        ds=tmp_dat,

                        splitter_type=splitter_type,

                        train=train,

                        test=test,

                        holdout=holdout,

                        **kwargs,

                    )

                    self.train = train

                    self.test = test

                    self.validate = holdout

                    self.X_train = train.X

                    self.y_train = train.y

                    self.X_test = test.X

                    self.y_test = test.y

                    self.X_validate = holdout.X

                    self.y_validate = holdout.y

                elif holdout is None and kfold is None and splitter_type is not None:

                    train, test, splitter = datasets.molecular_dataset_split(

                        ds=tmp_dat,

                        splitter_type=splitter_type,

                        train=train,

                        test=test,

                        holdout=holdout,

                        **kwargs,

                    )

                    self.train = train

                    self.test = test

                    self.X_train = train.X

                    self.y_train = train.y

                    self.X_test = test.X

                    self.y_test = test.y

                elif kfold is not None and splitter_type is not None:

                    if holdout is None:

                        log.info(f"Generated {kfold} fold")

                        self.kfolds, self.splitter = datasets.molecular_dataset_split(

                            ds=tmp_dat, splitter_type=splitter_type, kfold=kfold, **kwargs

                        )

                    else:

                        log.info(f"Generated {kfold} fold and validation holdout")

                        self.kfolds, self.validate, self.splitter = (

                            datasets.molecular_dataset_split(

                                ds=tmp_dat,

                                splitter_type=splitter_type,

                                holdout=holdout,

                                kfold=kfold,

                                **kwargs,

                            )

                        )

                        self.X_validate = self.validate.X

                        self.y_validate = self.validate.y

                else:

                    log.warning(

                        "Using SciKit-Learn random splitter you may want to consider using molecule splitters is this is for molecular data."

                    )

                    X_train, X_test, y_train, y_test = train_test_split(

                        tmp_dat.X,

                        tmp_dat.y,

                        test_size=0.2,

                        train_size=0.8,

                        random_state=helpers.random_seed,

                        shuffle=True,

                    )

                    self.X_train = X_train

                    self.y_train = y_train

                    self.X_test = X_test

                    self.y_test = y_test

                if splitter_type is not None:

                    log.debug(

                        f"Train: {train}, Validation: {holdout}, Test {test}, kfolds: {kfold}, splitter {self.splitter}"

                    )

                else:

                    log.debug(

                        f"Train: {train}, Validation: {holdout}, Test {test}, kfolds: {kfold}"

                    )


#### load_pipeline_into_class

```python3
def load_pipeline_into_class(
    self,
    filename: str = 'model.joblib'
)
```

Load a model pipeline into this class

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| filename | str | joblib file to load the model pipeline from | None |

??? example "View Source"
            def load_pipeline_into_class(self, filename: str = "model.joblib"):

                """

                Load a model pipeline into this class

                Args:

                    filename (str): joblib file to load the model pipeline from

                """

                self.model = joblib.load(filename)


#### make_pipeline

```python3
def make_pipeline(
    self,
    return_pipeline: bool = False
)
```

Function to build a pipeline if the pipeline elements have been defined

??? example "View Source"
            def make_pipeline(self, return_pipeline: bool = False):

                """

                Function to build a pipeline if the pipeline elements have been defined

                """

                if self.pipeline_elts is not None:

                    log.info(f"Setting the scikit learn pipeline using {self.pipeline_elts}.")

                    if return_pipeline is True:

                        return Pipeline(self.pipeline_elts)

                    else:

                        self.pipeline = Pipeline(self.pipeline_elts)


#### optimize_model

```python3
def optimize_model(
    self,
    cv: int = 5,
    verbose: int = 2,
    scoring: Optional[dict[Union[str, Callable]]] = None,
    refit: Optional[str] = None,
    return_model: bool = False,
    **kwargs
)
```

Function to train a model pipeline using the setting defined

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| cv | int | The number of corss validation folds used to establish the best hyper-parameters. Defaults to 5. | 5 |
| verbose | bool | To print verbose or not trianing information. Defaults to True. | True |

??? example "View Source"
            def optimize_model(

                self,

                cv: int = 5,

                verbose: int = 2,

                scoring: Optional[dict[Union[str, Callable]]] = None,

                refit: Optional[str] = None,

                return_model: bool = False,

                **kwargs,

            ):

                """

                Function to train a model pipeline using the setting defined

                Args:

                    cv (int, optional): The number of corss validation folds used to establish the best hyper-parameters. Defaults to 5.

                    verbose (bool, optional): To print verbose or not trianing information. Defaults to True.

                """

                if isinstance(self.param_grid, dict):

                    log.info(f"Will use the parameter grid: {yaml.dump(self.param_grid)}")

                if isinstance(self.pipeline, Pipeline):

                    log.info("Will run the following pipeline to optimize the hyper parmaeters")

                if self.X_train is not None and self.y_train is not None:

                    log.info("Using the trianing data to run a Grid search CV")

                if scoring is None:

                    scoring = (

                        "neg_mean_squared_error",

                        "neg_root_mean_squared_error",

                        "neg_mean_absolute_percentage_error",

                        "r2",

                    )

                    log.info(f"Using default scoring: {scoring}")

                if refit is None:

                    refit = scoring[0]

                grid_search = GridSearchCV(

                    self.pipeline,

                    n_jobs=-1,

                    param_grid=self.param_grid,

                    cv=cv,

                    verbose=verbose,

                    scoring=scoring,

                    refit=refit,

                    **kwargs,

                )

                if self.y_train.shape[1] > 1:

                    grid_search.fit(self.X_train, self.y_train)

                else:

                    grid_search.fit(self.X_train, self.y_train.ravel())

                self.completed_grid_search = grid_search

                self.model = grid_search.best_estimator_

                self.best_parmeters = grid_search.best_params_

                if return_model is False:

                    return grid_search

                else:

                    return grid_search.best_estimator_


#### save_model

```python3
def save_model(
    self,
    filename: str = 'model.joblib'
)
```

Save the final model pipeline to a jobloib file

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| filename | str | filename to save | None |

??? example "View Source"
            def save_model(self, filename: str = "model.joblib"):

                """

                Save the final model pipeline to a jobloib file

                Args:

                    filename (str): filename to save

                """

                if self.model is not None:

                    joblib.dump(self.model, filename)


#### save_self

```python3
def save_self(
    self,
    filename: str
)
```

Save the entire class as a pickle file

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| filename | str | file name | None |

??? example "View Source"
            def save_self(self, filename: str):

                """

                Save the entire class as a pickle file

                Args:

                    filename (str): file name

                """

                with open(filename, "wb") as fout:

                    pkl.dump(self, fout)
