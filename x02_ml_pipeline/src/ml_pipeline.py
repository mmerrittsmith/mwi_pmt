import yaml
import typing
import sys
import multiprocessing as mp
from functools import partial
import warnings
from scipy.stats import ConstantInputWarning

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, recall_score, mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator
from sklearn.linear_model import LassoCV



from scipy.stats import spearmanr, pearsonr
import joblib
import statsmodels.api as sm

def min_max_normalize(X: np.ndarray) -> np.ndarray:
    """
    Normalize the input array to a 0-1 range using min-max scaling.
    
    Args:
    X (np.ndarray): Input array to be normalized.
    
    Returns:
    np.ndarray: Normalized array with values scaled to [0, 1] range.
    """
    X = np.array(X)  # Ensure X is a numpy array
    X_min = np.min(X)
    X_max = np.max(X)
    
    # Check if X_min equals X_max to avoid division by zero
    if X_min == X_max:
        return np.zeros_like(X)  # or you could return np.ones_like(X)
    
    return (X - X_min) / (X_max - X_min)

def is_binary_or_nan(column: pd.Series) -> bool:
    """
    Check if a column is binary or contains NaN values.
    
    Args:
    column (pd.Series): Input column to check.
    
    Returns:
    bool: True if the column is binary or contains NaN values, False otherwise.
    """
    return column.isin([0., 1., 0, 1, np.nan]).all()

def load_config() -> dict[str, typing.Any]:
    """
    Load the configuration from the YAML file.
    
    Returns:
    dict[str, typing.Any]: Configuration dictionary.
    """
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def ee_rate(eligible, chosen):
    """
    Exclusion error rate, aka undercoverage or false negative rate, is the proportion of the eligible who are 
    wrongly excluded.
    EER = FN / (FN + TP)
    This measure is equivalent to the Targeting Error Rate, as we fix the number of beneficiaries to the target 
    coverage.
    The function spells the operations to mirror the other metrics, but can be rolled into:
    np.sum(np.logical_and(y_test_binary, 1-np.array(yhat_test_binary))) / np.sum(y_test_binary)
    
    Parameters:
    eligible (0/1 int tuple): 1 households whose true consumption is below the threshold
    chosen  (0/1 int tuple): 1 for households whose estimated consumption is below the threshold
    
    Returns:
    eer (float): Exclusion error rate
    """
    not_chosen = 1 - np.array(chosen)
    wrongly_excluded = np.logical_and(eligible, not_chosen)
    eer = np.sum(wrongly_excluded) / np.sum(eligible)
    return eer

def ie_rate(eligible, chosen):
    """
    Inclusion error rate, or false positive rate, is the proportion of the non-eligible who are wrongly included.
    IER = FP / (FP + TN)
    This definition differs from the IER definition [FP / (FP + TP)], or leakage, that is standard in social 
    protection.
    
    Parameters:
    eligible (0/1 int tuple): 1 households whose true consumption is below the threshold
    chosen  (0/1 int tuple): 1 for households whose estimated consumption is below the threshold
    
    Returns:
    ier (float): Inclusion error rate
    """
    ineligible = 1 - np.array(eligible)
    wrongly_included = np.logical_and(ineligible, chosen)
    ier = np.sum(wrongly_included) / np.sum(ineligible)
    return ier

def fi_getter(fitted_model: typing.Any) -> np.ndarray:
    """
    Get the feature importances from a fitted model, regardless of the model type.
    
    Parameters:
    fitted_model (typing.Any): Fitted model.
    
    Returns:
    np.ndarray: Feature importances.
    """
    if isinstance(fitted_model, sm.regression.linear_model.RegressionResultsWrapper):
        return fitted_model.params
    elif hasattr(fitted_model, 'coef_'):
        return fitted_model.coef_
    elif hasattr(fitted_model, 'feature_importances_'):
        return fitted_model.feature_importances_
    else:
        raise ValueError(f"Model {fitted_model.__class__.__name__} does not have feature importances")

def forward_stepwise_arbitrary_model(X_train: pd.DataFrame,
                                     y_train: pd.Series,
                                     weights_train: pd.Series,
                                     X_test: pd.DataFrame,
                                     y_test: pd.Series,
                                     model: typing.Any, 
                                     config: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """
    Perform forward stepwise feature selection using an arbitrary model.

    This function implements a forward stepwise selection algorithm for feature selection.
    It starts with a base feature ('hhsize') and iteratively adds features that improve
    the model's performance based on mean squared error (MSE).

    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature set.
    y_train : pd.Series
        Training target variable.
    weights_train : pd.Series
        Sample weights for training data.
    X_test : pd.DataFrame
        Test feature set.
    y_test : pd.Series
        Test target variable.
    model : Any
        The machine learning model to use for feature selection.
    config : dict
        Configuration dictionary containing parameters like 'n_folds' and 'random_state'.

    Returns:
    --------
    dict
        A dictionary containing:
        - 'selected': List of selected features.
        - 'train_r2s': Dictionary of training R² scores for each step.
        - 'test_r2s': Dictionary of test R² scores for each step.
        - 'all_selected_vars': Dictionary of selected variables at each step.
        - 'fis': Dictionary of feature importances at each step.

    Notes:
    ------
    - The function uses k-fold cross-validation to evaluate feature candidates.
    - It always includes 'hhsize' as the initial feature.
    - The process stops when no remaining feature improves the model's performance.
    """
    # If you want to test this function, uncomment the following lines
    # It'll be a lot faster. It might not work because the random selection of variables
    # can lead to a quick suboptimal selection and exit, but just run it again until you see
    # a few variables to test the rest of the code.
    # var_list = np.random.choice(X_train.columns.tolist(), size=5, replace=False).tolist()
    # X_train = X_train[var_list + ['const']]   
    # X_test = X_test[var_list + ['const']]

    kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=config['random_state'])
    remaining = list(X_train.columns)
    remaining.remove('const')
    selected, current_vars = ['const'], ['const']
    train_r2s, test_r2s, all_selected_vars, fis = {}, {}, {}, {}

    train_ids, val_ids = train_test_split(range(len(y_train)), random_state=config['random_state'], test_size=0.25)
    if model is None:
        model = sm.WLS(y_train.iloc[train_ids], sm.add_constant(X_train.iloc[train_ids][selected]), weights=weights_train.iloc[train_ids])
        fitted_model = model.fit()
    else:
        fitted_model = model.fit(X_train.iloc[train_ids][selected], y_train.iloc[train_ids], sample_weight=weights_train.iloc[train_ids])
    yhat_val = fitted_model.predict(X_train.iloc[val_ids][selected])
    yhat_test = fitted_model.predict(X_test[selected])
    mse_current = mean_squared_error(y_train.iloc[val_ids], yhat_val)
    mse_best = mse_current.copy()

    train_r2s[1] = r2_score(y_train.iloc[val_ids], yhat_val)
    test_r2s[1] = r2_score(y_test, yhat_test)
    all_selected_vars[1] = current_vars
    fis[1] = pd.Series(fi_getter(fitted_model), index=current_vars)
    keep_going_anyway = True
    while len(remaining) > 0:
        best_mse_this_loop = np.inf
        if len(selected) == 20:
            keep_going_anyway = False
        for var in remaining:
            candidates = selected + [var]

            mse_candidate = np.zeros(config['n_folds'])
            r2_val_candidates, r2_test_candidates, fis_candidate = [], [], []
            for k, (train_ids, val_ids) in enumerate(kfold.split(range(len(X_train)))):
                if isinstance(model, BaseEstimator):
                    fitted_model = model.fit(X_train.iloc[train_ids][candidates], y_train.iloc[train_ids], sample_weight=weights_train.iloc[train_ids])
                else:
                    model = sm.WLS(y_train.iloc[train_ids], X_train.iloc[train_ids][candidates], 
                                   weights=weights_train.iloc[train_ids])
                    fitted_model = model.fit()
                yhat_val = fitted_model.predict(X_train.iloc[val_ids][candidates])
                r2_val_candidates.append(r2_score(y_train.iloc[val_ids], yhat_val))
                yhat_test = fitted_model.predict(X_test[candidates])
                r2_test_candidates.append(r2_score(y_test, yhat_test))
                mse_candidate[k] = mean_squared_error(y_train.iloc[val_ids], yhat_val)
                fis_candidate.append(fi_getter(fitted_model))
            if np.mean(mse_candidate) < mse_current:
                current_vars = candidates
                best_mse_this_loop = np.mean(mse_candidate)
                mse_current = np.mean(mse_candidate)
                current_best_r2_val = np.median(r2_val_candidates)
                current_best_r2_test = np.median(r2_test_candidates)
                current_fis = np.mean(fis_candidate, axis=0)
                # print(f"Current best R2 val: {current_best_r2_val}")
                # print(f"Current best R2 test: {current_best_r2_test}")
                # print(f"Current variables: {current_vars}")
            elif keep_going_anyway:
                if np.mean(mse_candidate) < best_mse_this_loop:
                    current_vars = candidates
                    best_mse_this_loop = np.mean(mse_candidate)
                    mse_current = np.mean(mse_candidate)
                    current_best_r2_val = np.median(r2_val_candidates)
                    current_best_r2_test = np.median(r2_test_candidates)
                    current_fis = np.mean(fis_candidate, axis=0)
                    # print(f"Not an improvement, but keep going anyway. Current best R2 val: {current_best_r2_val}")
                    # print(f"Not an improvement, but keep going anyway. Current best R2 test: {current_best_r2_test}")
                    # print(f"Not an improvement, but keep going anyway. Current variables: {current_vars}")
        if current_vars == selected and not keep_going_anyway:
            break
        train_r2s[len(current_vars)] = current_best_r2_val
        test_r2s[len(current_vars)] = current_best_r2_test
        all_selected_vars[len(current_vars)] = current_vars
        fis[len(current_vars)] = pd.Series(current_fis, index=current_vars)
        remaining.remove(current_vars[-1])
        selected = current_vars
        mse_best = mse_current.copy()
    all_results = {'selected': selected, 'train_r2s': train_r2s, 'test_r2s': test_r2s, 'all_selected_vars': all_selected_vars, 'fis': fis}
    return all_results



def run_single_bootstrap(df: pd.DataFrame, config: dict[str, typing.Any], model_name: str) -> pd.DataFrame:
    """
    Run a single bootstrap iteration for a given model.

    This function performs the following steps:
    1. Splits the data into training and test sets based on panel IDs.
    2. Prepares metrics dictionaries for various performance measures.
    3. Separates features and target variables for both training and test sets.
    4. Identifies binary and continuous columns in the dataset.
    5. Applies min-max normalization to continuous features.
    6. Transforms the training and test data using the normalization pipeline.
    7. Resets indices and ensures all data is in float format.

    The function is designed to work with different models, which are specified
    by the 'model_name' parameter and handled in the calling function.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing all data.
    config (dict[str, typing.Any]): A configuration dictionary containing various
                                    settings and column names.
    model_name (str): The name of the model to be used for this bootstrap iteration.

    Returns:
    pd.DataFrame: A DataFrame containing the results of the bootstrap iteration,
                  including various performance metrics.

    Note:
    The function prepares the data for model fitting and evaluation, but the actual
    model fitting and evaluation steps are expected to be implemented in the code
    that follows this function.
    """
    panel_ids = list(set(df[config['id_col']]))
    train_ids, test_ids = train_test_split(panel_ids, random_state=config["random_state"], test_size=0.25, shuffle=True)
    metrics = {**{'r2':[],
                'spearman':[],
                'pearson':[]},
            **{'recall_' + str(i):[] for i in np.arange(10, 100, 10)}, 
            **{'eer_' + str(i):[] for i in np.arange(10, 100, 10)}, 
            **{'ier_' + str(i):[] for i in np.arange(10, 100, 10)},
            **{'utility_' + str(i):[] for i in np.arange(10, 100, 10)}}    
    x_train = df.loc[train_ids].drop(columns=[config['target_col']] + config['extras'])
    y_train = df.loc[train_ids][f'{config["target_col"]}']
    weights_train = df.loc[train_ids][config['weight_col']]

    if config["evaluate_on_urban_only"]:
        x_test = df.loc[test_ids].drop(columns=[config['target_col']] + config['extras'])
        x_test = x_test[x_test['reside_2.0'] == False]
        y_test = df.loc[test_ids][f'{config["target_col"]}']
        y_test = y_test[df['reside_2.0'] == False]
    else:
        x_test = df.loc[test_ids].drop(columns=[config['target_col']] + config['extras'])
        y_test = df.loc[test_ids][f'{config["target_col"]}']

    binary_columns = [col for col in df.columns if is_binary_or_nan(df[col])]
    continuous_columns = [col for col in df.columns 
                            if df[col].dtype in ['float64', 'int64'] 
                            and col not in binary_columns
                            and col not in [config['target_col']] + config['extras']]

    # Check if all columns are accounted for
    all_columns = set(binary_columns).union(set(continuous_columns)).union(set([config['target_col']] + config['extras']))
    unaccounted_columns = set(df.columns) - all_columns
    assert len(unaccounted_columns) == 0, f"The following columns are unaccounted for: {unaccounted_columns}"
    min_max_normalizer = ColumnTransformer(
        transformers=[
                ('min_max_normalizer', FunctionTransformer(min_max_normalize), continuous_columns)
            ],
            remainder='passthrough'
        )
    pipeline = Pipeline([
        ('min_max_normalizer', min_max_normalizer),
    ])
    X_train_transformed = pd.DataFrame(pipeline.fit_transform(x_train), columns=x_train.columns)
    X_test_transformed = pd.DataFrame(pipeline.transform(x_test), columns=x_train.columns)
    X_train_transformed = X_train_transformed.reset_index(drop=True)
    X_train_transformed = X_train_transformed.astype(float)
    y_train = y_train.reset_index(drop=True)
    y_train = y_train.astype(float)
    weights_train = weights_train.reset_index(drop=True)
    weights_train = weights_train.astype(float)
    y_test = y_test.reset_index(drop=True)
    y_test = y_test.astype(float)
    X_test_transformed = X_test_transformed.reset_index(drop=True)
    X_test_transformed = X_test_transformed.astype(float)

    match model_name:
        case 'Ridge':
            model = Ridge(alpha=0.00464, fit_intercept=True, random_state=config['random_state'])
            all_results = forward_stepwise_arbitrary_model(X_train_transformed, y_train, weights_train, X_test_transformed, y_test, model, config)
            fitted_model = model.fit(X_train_transformed[all_results['selected']], y_train, sample_weight=weights_train)
            if config["evaluate_on_urban_only"]:
                model_filename = f"output/{config['urban_or_rural']}/ridge_eval_on_urban.joblib"
            else:
                model_filename = f"output/{config['urban_or_rural']}/ridge.joblib"
            joblib.dump(fitted_model, model_filename)
        case 'Lasso':
            # Why no alphas in LassoCV call here? https://github.com/emilylaiken/pmt-decay/blob/e655b9d0f66d145a1da0f2355fe79d8ea3ec6533/Machine%20Learning.ipynb#L212
            # I don't know, but I'm following the example in the notebook
            # model = LassoCV(fit_intercept=True, random_state=config['random_state'])
            # Use LassoCV to find the optimal alpha
            param_grid = {'alpha': np.logspace(-3, 3, 100)}
            lasso_cv = GridSearchCV(
                Lasso(random_state=config['random_state'], fit_intercept=True),
                param_grid,
                cv=10,
                scoring='neg_mean_squared_error',
                n_jobs=config['n_cpus']
            )
            lasso_cv.fit(X_train_transformed, y_train, sample_weight=weights_train)
            model = Lasso(alpha=lasso_cv.best_params_['alpha'], fit_intercept=True, random_state=config['random_state'])
            print(f"Optimal alpha: {lasso_cv.best_params_['alpha']}")
            all_results = forward_stepwise_arbitrary_model(X_train_transformed, y_train, weights_train, X_test_transformed, y_test, model, config)
            fitted_model = model.fit(X_train_transformed[all_results['selected']], y_train, sample_weight=weights_train)
            if config["evaluate_on_urban_only"]:
                model_filename = f"output/{config['urban_or_rural']}/lasso_eval_on_urban.joblib"
            else:
                model_filename = f"output/{config['urban_or_rural']}/lasso.joblib"
            joblib.dump(fitted_model, model_filename)
            
        case 'RandomForest':
            # Using parameters found from previous grid search code
            model = RandomForestRegressor(n_jobs=config['n_cpus'], n_estimators=50, max_depth=8, random_state=config['random_state'])
            all_results = forward_stepwise_arbitrary_model(X_train_transformed, y_train, weights_train, X_test_transformed, y_test, model, config)
            fitted_model = model.fit(X_train_transformed[all_results['selected']], y_train, sample_weight=weights_train)
            if config["evaluate_on_urban_only"]:
                model_filename = f"output/{config['urban_or_rural']}/random_forest_eval_on_urban.joblib"
            else:
                model_filename = f"output/{config['urban_or_rural']}/random_forest.joblib"
            joblib.dump(fitted_model, model_filename)
        case 'GradientBoosting':
            # Using parameters found from previous grid search code
            model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=8, min_samples_leaf=1, random_state=config['random_state'])
            all_results = forward_stepwise_arbitrary_model(X_train_transformed, y_train, weights_train, X_test_transformed, y_test, model, config)
            fitted_model = model.fit(X_train_transformed[all_results['selected']], y_train, sample_weight=weights_train)
            if config["evaluate_on_urban_only"]:
                model_filename = f"output/{config['urban_or_rural']}/gradient_boosting_eval_on_urban.joblib"
            else:
                model_filename = f"output/{config['urban_or_rural']}/gradient_boosting.joblib"
            joblib.dump(fitted_model, model_filename)
        case 'Stepwise':
            X_train_transformed['intercept'], X_test_transformed['intercept'] = np.ones(len(y_train)), np.ones(len(y_test))
            model = None
            all_results = forward_stepwise_arbitrary_model(X_train_transformed, y_train, weights_train, X_test_transformed, y_test, model, config)
            x_train_for_stepwise = X_train_transformed.reset_index(drop=True)
            x_train_for_stepwise = x_train_for_stepwise.astype(float)
            y_train_for_stepwise = y_train.reset_index(drop=True)
            y_train_for_stepwise = y_train_for_stepwise.astype(float)
            weights_for_stepwise = weights_train.reset_index(drop=True)
            weights_for_stepwise = weights_for_stepwise.astype(float)
            X_test_for_stepwise = X_test_transformed.reset_index(drop=True)
            X_test_for_stepwise = X_test_for_stepwise.astype(float)
            model = sm.WLS(y_train_for_stepwise, x_train_for_stepwise[all_results['selected']], weights=weights_for_stepwise)
            fitted_model = model.fit()
            y_hat_test = fitted_model.predict(X_test_for_stepwise[all_results['selected']])
            if config["evaluate_on_urban_only"]:
                model_filename = f"output/{config['urban_or_rural']}/stepwise_eval_on_urban.joblib"
            else:
                model_filename = f"output/{config['urban_or_rural']}/stepwise.joblib"
            joblib.dump(fitted_model, model_filename)
        case _:
            raise ValueError("Invalid model type")
    
    y_hat_test = fitted_model.predict(X_test_transformed[all_results['selected']])

    r2_df = pd.DataFrame(pd.Series(all_results['train_r2s'], index=all_results['train_r2s'].keys()))
    r2_df.columns = ['Train R^2']
    r2_df.index.name = 'Number of Features'
    r2_df["Test R^2"] = all_results['test_r2s']
    r2_df["Selected Variables"] = all_results['all_selected_vars']
    all_results["r2_df"] = r2_df

    pearson = pearsonr(y_test, y_hat_test)[0]
    spearman = spearmanr(y_test, y_hat_test)[0]
    metrics['Train R^2'] = [list(all_results['train_r2s'].values())[-1]]
    metrics['Test R^2'] = [list(all_results['test_r2s'].values())[-1]]
    metrics['Selected Variables'] = [list(all_results['all_selected_vars'].values())[-1]]
    metrics['pearson'] = [pearson]
    metrics['spearman'] = [spearman]
    all_results['metrics'] = metrics

    # Remove all the empty lists in the metrics dictionary
    metrics = {k: v for k, v in metrics.items() if len(v) > 0}

    all_results["metrics"] = metrics
    return all_results

def run_ml(df: pd.DataFrame, config: dict[str, typing.Any], model_name: str) -> pd.DataFrame:
    """
    Run the machine learning pipeline on the given dataset.

    This function performs the following steps:
    1. Applies min-max scaling to the household weights.
    2. Runs a single bootstrap iteration of the specified model.
    3. Saves the results (metrics, R-squared values, and feature importances) to CSV files.

    Parameters:
    -----------
    df : pd.DataFrame
        The input dataset containing features, target variable, and weights.
    config : dict[str, typing.Any]
        A dictionary containing configuration parameters for the ML pipeline.
    model_name : str
        The name of the machine learning model to run.

    Returns:
    --------
    pd.DataFrame
        The input DataFrame with scaled weights.

    Side Effects:
    -------------
    - Saves metrics to 'output/{urban_or_rural}/metrics_{model_name}.csv'
    - Saves R-squared values to 'output/{urban_or_rural}/r2_{model_name}.csv'
    - Saves feature importances to 'output/{urban_or_rural}/importances_{model_name}.csv'

    Notes:
    ------
    The function modifies the input DataFrame in-place by scaling the weights.
    """
    # Min-max scaling of HH weights
    df[config['weight_col']] = 100*((df[config['weight_col']] - df[config['weight_col']].min())/(df[config['weight_col']].max() - df[config['weight_col']].min()))
    df[config['weight_col']] = df[config['weight_col']].apply(lambda x: int(x))

    all_results = run_single_bootstrap(df, config, model_name)

    pd.DataFrame(all_results["metrics"]).to_csv(f"output/{config['urban_or_rural']}/metrics_{model_name}.csv")
    importances_df = fis_to_df(all_results['fis'])
    if config["evaluate_on_urban_only"]:
        model_filename = f"output/{config['urban_or_rural']}/importances_{model_name}_eval_on_urban.csv"
        all_results["r2_df"].to_csv(f"output/{config['urban_or_rural']}/r2_{model_name}_eval_on_urban.csv")
    else:
        model_filename = f"output/{config['urban_or_rural']}/importances_{model_name}.csv"
        all_results["r2_df"].to_csv(f"output/{config['urban_or_rural']}/r2_{model_name}.csv")
    importances_df = fis_to_df(all_results['fis'])
    importances_df.to_csv(model_filename)

def fis_to_df(fis: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Convert feature importances dictionary to a DataFrame.

    This function takes a dictionary of feature importances for different numbers
    of variables and converts it into a pandas DataFrame with a structured format.

    Parameters:
    -----------
    fis : dict[str, pd.Series]
        A dictionary where keys are the number of variables (as strings) and
        values are pandas Series containing feature importances.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with columns:
        - 'num_vars': The number of variables used in the model.
        - 'fi_values': The feature importance values, rounded to 3 decimal places.
        - 'var_names': The names of the variables.
    """
    num_vars = []
    fi_values = []
    var_names = []
    for var_num, fis_series in fis.items():
        for var_name, fi_value in fis_series.items():
            num_vars.append(var_num)
            fi_values.append(round(fi_value, 3))
            var_names.append(var_name)
    return pd.DataFrame({"num_vars": num_vars, "fi_values": fi_values, "var_names": var_names})


def main(config: dict[str, typing.Any]):
    """
    Main function to execute the machine learning pipeline.

    This function orchestrates the entire ML pipeline process, including:
    1. Loading the data
    2. Filtering the data based on urban/rural settings
    3. Running the specified machine learning models

    Parameters:
    -----------
    config : dict[str, typing.Any]
        A dictionary containing configuration parameters for the ML pipeline.
        Expected keys include:
        - 'urban_or_rural': str, specifies whether to use urban or rural data
        - 'id_col': str, name of the column used as an identifier
        - 'models_to_run': list[str], names of the models to be executed

    Returns:
    --------
    None
        This function doesn't return any value but generates output files
        for each model run, including metrics, R-squared values, and feature importances.

    Side Effects:
    -------------
    - Reads data from 'input/merged_data.parquet'
    - Filters the data based on the 'urban_or_rural' configuration
    - Executes the specified machine learning models
    - Generates output files in the 'output/{urban_or_rural}/' directory
    """
    df = pd.read_parquet("input/merged_data.parquet")
    df.index = df[config['id_col']]
    df['const'] = 1
    if config['urban_or_rural'].startswith('urban'):
        df = df[df['reside_2.0'] == False]
    elif config['urban_or_rural'].startswith('rural'):
        df = df[df['reside_2.0'] == True]
    for model in config['models_to_run']:
        print(f"Running {model} on {config['urban_or_rural']} data")
        if model == 'Stepwise':
            config["num_cpus"] = 1
        run_ml(df, config, model)


if __name__ == "__main__":
    config = load_config()
    main(config)