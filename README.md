[![Python package](https://github.com/Redx-Pharma/chemical_utilities_os/actions/workflows/testing.yaml/badge.svg)](https://github.com/Redx-Pharma/chemical_utilities_os/actions/workflows/testing.yaml)

[![Deploy static content to Pages](https://github.com/Redx-Pharma/chemical_utilities_os/actions/workflows/static.yaml/badge.svg)](https://github.com/Redx-Pharma/chemical_utilities_os/actions/workflows/static.yaml)

# Chemical utilities
Chemical data utilities and methods. See [pages](https://laughing-barnacle-pl8klok.pages.github.io/) for more information.

## Description
This is a set of tools for chemical data science. The key methods are built on top of the open source RDKit library. There are specific modules for:
* SMILES manipulation (smiles.py)
* InChI manupulation (inchi.py)
* Computational analysis of molecule graphs (molecules.py)
* Pareto front determination and visualization (multiopt.py)
* Molecule visualizations (vis.py)
* helpful utils (helpers.py)
* Classic ML models using scikit-learn (classical\_ml\_models.py)
* ML pipeline model builds using scikit-learn (classical\_ml\_model\_build.py)

## Dependencies
For full details and version number please see pyproject.toml
* numpy
* rdkit
* pandas
* paretoset
* pymoo
* jax
* pytorch
* scikit-learn
* deepchem
* matplotlib
* seaborn

## Install

This code is built as a python package which is locally installable.

Editable mode: Run the following from the top level directory
```bash
pip install -e .
```

Fixed mode: Run the following from the top level directory
```bash
pip install .
```

## Examples
Here are some of the functionalities in the library to help to automate and wrap other libraries and processes.

#### Similarity Maps

```python
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import chemutil
from chemutil import vis
import logging

%load_ext autoreload
%autoreload 2

logging.basicConfig(format='%(levelname)-9s : %(message)s')
log=logging.getLogger()
log.setLevel(logging.INFO)

log.info(f"RDKit version: {rdkit.__version__} Chemutil version: {chemutil.__version__}")

smiles1 = "c1ccccc1C"
smiles2 = "c1ccccc1Cl"

vis.molecule_similarity_maps(ref_smiles=smiles1, exp_smiles=smiles2)
```

To use atomic pair FP and dice metric
```python
vis.molecule_similarity_maps(ref_smiles=ref_smiles, exp_smiles=exp_smiles, fp="atompair", metric=DataStructs.DiceSimilarity)
```

You can also use custom property values for the similaririty
```python
mol = Chem.MolFromSmiles(smiles2)
AllChem.ComputeGasteigerCharges(mol)
contribs = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]
vis.molecule_similarity_maps(exp_smiles=smiles2, fp="custom", weights=contribs)
```

#### Filtering
There are default functions used for filtering based on druglikeness and accessibility of synthesis. These can be used directly.
```python
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from chemutil.filtering import (
    druglikeness,
    get_molecular_mass,
    get_logp,
    get_filter_properties,
    MoleculeFilter,
)

data = {'smiles': ['CCO', 'C1CCCCC1']}
df = pd.DataFrame(data)
result = get_filter_properties(df, representation_column='smiles')
result.to_dict()
```

The functions for diltering could also be customized either to include a custom function or use a subset of the defaults.

```python
mols = [Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('C1CCCCC1')]
funcs = [druglikeness, get_logp, get_molecular_mass]
args = [{"ghose": False, "veber": False, "ro3": False, "reos": False}, {}, {}]
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
mf = MoleculeFilter(funcs, mols, list_of_fx_arg_dicts=args)
result = mf.filter_results
result.select_dtypes(include=numerics).values
```

#### Pareto Front Multi-objective Optimization
This section used property data to find the molecules on a defined pareto front.

```python
import chemutil
from chemutil import multiopt
import pandas as pd

data = {'smiles': ['CCO', 'C1CCCCC1', 'c1ccccc1'], 'prop1': [1, 2, 3], 'prop2': [4, 2, 5], 'prop3': [4, 1, 2]}
df = pd.DataFrame(data)

pareto_df = multiopt.get_pareto_efficient_set(
    df,
    minmax=['min', 'max', 'min'],
    objective_columns=df.columns,
    )

pareto_df[pareto_df["pareto_efficent_set"] == 1]
```

This can be plotted on scatter graphs for 2D and 3D or radar plots for 4D and over.
```python
imag = multiopt.plot_pareto_front(
    pareto_df,
    ['min', 'max', 'min'],
    objective_columns=df.columns,
    label_column='smiles',
    filename=output_file,
    smiles_column='smiles'
    )
```

#### Featurization
We can use the library to build features for the molecules.
``` python
import deepchem as dc
import rdkit
import pandas as pd
import numpy as np
import chemutil
from chemutil import featurization, datasets, vis
from sklearn.linear_model import LinearRegression
import logging

%load_ext autoreload
%autoreload 2

logging.basicConfig(format='%(levelname)-9s : %(message)s')
log=logging.getLogger()
log.setLevel(logging.INFO)

log.info(f"RDKit version: {rdkit.__version__} Chemutil version: {chemutil.__version__} Deepchem version: {dc.__version__} RDKit version: {rdkit.__version__} Pandas version: {pd.__version__} Numpy version: {np.__version__}")

data_df = pd.DataFrame({'smiles': ['c1ccccc1', 'CCO'], 'prop2': [1, 2]})

df_ecfp = featurization.get_ecfp(data_df=data_df, smiles_column="smiles",
return_df=True)

df_counts = featurization.get_count_ecfp(data_df=data_df, smiles_column="smiles", return_df=True)

df_maccs_keys = featurization.get_maccs(data_df=data_df, smiles_column="smiles", return_df=True)

df_rdkit_descriptors = featurization.get_rdkit_descriptors(data_df=data_df, smiles_column="smiles", return_df=True)
```

#### Data set splitting
We can also split the data set use deepchem splitters following from the featurization.

```python
ds = datasets.pandas_to_deepchem(df, smiles_column="smiles", task_columns=["prop2"], feature_column_regex="ecfp_bit_")

train, test, val, dcsplitter = datasets.molecular_dataset_split(ds, train=0.8, validation=0.1, test=0.1, splitter_type="fp")

# or for kfold splits
kfold_datasets, dcsplitter = datasets.molecular_dataset_split(ds, kfold=5)
```

It can be helpful to compare the dataset from spliting to determine different the training and test sets really are.

```python
vis.tanimoto_box_plot(train.to_dataframe(), test.to_dataframe(), labels_column_1="ids", smiles_column_1="ids", smiles_column_2="ids")

vis.tanimoto_distrbution_plots(train.to_dataframe(), test.to_dataframe(), labels_column_1="ids", smiles_column_1="ids", smiles_column_2="ids")
```

#### Classical ML
Below we provide a general setup and application approach for a single classical ML model:
```python

data_df = pd.read_csv("data_file.csv")


bayesian_regressor = classical_ml_models.get_models(
    multi_output_only=False,
    n_targets=len(task_columns),
    linear=False,
    bayesian=True,
    gp=False
    ard=False
    kernel=False,
    ensemble=False
    neural_network=False,
    )

model = classical_ml_model_build.RXTrainTestGridSearchSklearnRegresssor(
    data_df,
    model=bayesian_regressor.model,
    model_name=bayesian_regressor.name
)

model.build_train_test_validate_splits(
    smiles_column=smiles_column,
    task_columns=task_columns,
    feature_column_regex=feature_column_regex,
    splitter_type=splitter_type,
    train=train,
    test=test,
    validation=validation
)

model.build_pipline_list()
if len(m.custom_param_grid.keys()) == 0:
    model.build_parameter_grid(model_param_grid=bayesian_regressor.default_param_grid)
else:
    model.build_parameter_grid(model_param_grid=bayesian_regressor.custom_param_grid)
model.make_pipeline()

grid_search = model.optimize_model(scoring=("neg_mean_squared_error", "neg_root_mean_squared_error", "neg_mean_absolute_percentage_error", "r2"), verbose=1)
log.info(f"{grid_search.best_params_}")
model.save_self(f"trainer_and_model_{model.model_name}.pkl")
pd.DataFrame(grid_search.cv_results_).to_csv(f"train_data_{model.model_name}_regression.csv", index=False)

predictions = grid_search.best_estimator_.predict(model.X_test)
test_set_mse = mean_squared_error(predictions, model.y_test.ravel())
test_set_r2_CoD = r2_score(predictions, model.y_test.ravel())

plt.figure(figsize=(10,10))
xymin = np.floor(min(model.y_test.ravel().tolist() + predictions.tolist()))
xymax = np.ceil(max(model.y_test.ravel().tolist() + predictions.tolist()))
plt.plot([xymin, xymax], [xymin, xymax], "k--", label=f"x = y")
plt.scatter(model.y_test.ravel(), predictions, label=f"{model.model_name}")
plt.grid()
plt.legend()
plt.xlabel(f"Known data", fontsize=25)
plt.ylabel("Prediction", fontsize=25)
plt.title(f"Known data Vs. Prediction Using\nA {model.model_name} Model", fontsize=27)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=17)
ax.set_yticks(ticks)
ax.set_xticks(ticks)
plt.tight_layout()
plt.savefig("parity_prediction_known_data.png")
plt.close()

test_set_df = pd.DataFrame(test_set_metics, columns=test_set_metrics_headers)
test_set_df.style.format('{:.2f}')
test_set_df.to_csv(f"test_set_summary_metrics.csv", index=False)
```

## Help/FAQ


## Contributions
Authors:
* Redx Pharma Team

Notable contributions
* S. Yung
* A. Haworth
* J. McDonagh
