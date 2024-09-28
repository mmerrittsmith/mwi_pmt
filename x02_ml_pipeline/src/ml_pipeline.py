import yaml
import typing
import os
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
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, recall_score, mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import spearmanr, pearsonr

from tqdm import tqdm
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
    return column.isin([0., 1., 0, 1, np.nan]).all()

def load_config() -> dict[str, typing.Any]:
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

def forward_stepwise_OLS(x_train, y_train, y_test, weights, random_seed, n_folds):
    """
    Forward stepwise variable selection using OLS, with variable selection on test set MSE
    
    Parameters:
    x_train (pandas DataFrame): training set household characteristics 
    y_train (pandas DataFrame): training set consumption per person
    weights (pandas DataFrame): HH weights from the population sample survey
    random_seed (int): seed for random train/test split
    n_folds (int): 75/25% train/val split if == 2, k-fold crossval if > 2
    
    Returns:
    selected (list): List of predictor variables, in order of selection
    """
    # Resetting indices to avoid issues with statsmodels complaining about mismatched indices
    x_train = x_train.reset_index(drop=True)
    x_train = x_train.astype(float)
    y_train = y_train.reset_index(drop=True)
    y_train = y_train.astype(float)
    weights = weights.reset_index(drop=True)
    weights = weights.astype(float)
    y_test = y_test.reset_index(drop=True)
    y_test = y_test.astype(float)
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    train_ids, val_ids = train_test_split(range(len(y_train)), random_state=random_seed, test_size=0.25)
    remaining = list(x_train.columns)
    remaining.remove('intercept')
    selected, current_vars = ['intercept'], ['intercept']
    r2s = {}


    
    model = sm.WLS(y_train.iloc[train_ids], x_train.iloc[train_ids][selected], weights=weights.iloc[train_ids])
    fitted_model = model.fit()
    yhat_val = fitted_model.predict(x_train.iloc[val_ids][selected])
    mse_current = mean_squared_error(y_train.iloc[val_ids], yhat_val)
    mse_best = mse_current.copy()

    # Randomly select 50 columns from the dataset
    all_columns = list(x_train.columns)
    all_columns.remove('intercept')  # Remove 'intercept' if it's already in the list
    selected_columns = list(np.random.choice(all_columns, min(50, len(all_columns)), replace=False))
    
    # Add 'intercept' to the selected columns
    selected_columns.append('intercept')
    
    # Create new x_train with selected columns
    x_train = x_train[selected_columns]
    
    # Update remaining and selected lists
    remaining = [col for col in selected_columns if col != 'intercept']
    selected = ['intercept']
    
    # Recalculate initial model with new x_train
    model = sm.WLS(y_train.iloc[train_ids], x_train.iloc[train_ids][selected], weights=weights.iloc[train_ids])
    fitted_model = model.fit()
    yhat_val = fitted_model.predict(x_train.iloc[val_ids][selected])
    mse_current = mean_squared_error(y_train.iloc[val_ids], yhat_val)
    mse_best = mse_current

    while len(remaining) > 0:
        for var in remaining:
            candidates = selected + [var]
            if n_folds == 2:
                model = sm.WLS(y_train.iloc[train_ids], x_train.iloc[train_ids][candidates], 
                               weights=weights.iloc[train_ids])
                fitted_model = model.fit()
                yhat_val = fitted_model.predict(x_train.iloc[val_ids][candidates])
                r2 = r2_score(y_train.iloc[val_ids], yhat_val)
                r2s[len(current_vars)] = r2
                mse_candidate = mean_squared_error(y_train.iloc[val_ids], yhat_val)
                if mse_candidate < mse_current:
                    current_vars = candidates
                    mse_current = mse_candidate
                # else:
                #     break
            elif n_folds > 2:
                mse_candidate = np.zeros(n_folds)
                for k, (train_ids, val_ids) in enumerate(kfold.split(range(len(x_train)))):
                    model = sm.WLS(y_train.iloc[train_ids], x_train.iloc[train_ids][candidates], 
                                   weights=weights.iloc[train_ids])
                    fitted_model = model.fit()
                    yhat_val = fitted_model.predict(x_train.iloc[val_ids][candidates])
                    mse_candidate[k] = mean_squared_error(y_train.iloc[val_ids], yhat_val)
                    r2 = r2_score(y_train.iloc[val_ids], yhat_val)
                    r2s[len(current_vars)] = r2
                if np.mean(mse_candidate) < mse_current:
                    current_vars = candidates
                    mse_current = np.mean(mse_candidate)
        if current_vars == selected:
            break
        remaining.remove(current_vars[-1])
        selected = current_vars
        mse_best = mse_current.copy()
    return selected, r2s




def run_single_bootstrap(i: int, df: pd.DataFrame, config: dict[str, typing.Any], model_name: str, calc_importances: bool = False) -> pd.DataFrame:
    panel_ids = list(set(df[config['id_col']]))
    train_ids, test_ids = train_test_split(panel_ids, random_state=config["random_state"]+i, test_size=0.25, shuffle=True)
    metrics = {**{'r2':[],
                'spearman':[],
                'pearson':[]},
            **{'recall_' + str(i):[] for i in np.arange(10, 100, 10)}, 
            **{'eer_' + str(i):[] for i in np.arange(10, 100, 10)}, 
            **{'ier_' + str(i):[] for i in np.arange(10, 100, 10)},
            **{'utility_' + str(i):[] for i in np.arange(10, 100, 10)}}

    r2 = {}
    spearman = {}
    pearson = {}
    recall = {i:{} for i in np.arange(10, 100, 10)}
    eer = {i:{} for i in np.arange(10, 100, 10)}
    ier = {i:{} for i in np.arange(10, 100, 10)}
    utility = {i:{} for i in np.arange(10, 100, 10)}
    

    seed = config["random_state"] + i
    
    x_train = df.loc[train_ids].drop(columns=[config['target_col']] + config['extras'])
    y_train = df.loc[train_ids][f'{config["target_col"]}']
    weights_train = df.loc[train_ids][config['weight_col']]

    x_test = df.loc[test_ids].drop(columns=[config['target_col']] + config['extras'])
    y_test = df.loc[test_ids][f'{config["target_col"]}']
    weights_test = df.loc[test_ids][config['weight_col']]

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

    feature_importances = None

    match model_name:
        case 'Ridge':
            model = RidgeCV(alphas=np.logspace(-3, 3, 10), fit_intercept=True)
            model.fit(X_train_transformed, y_train, sample_weight=weights_train)
            y_hat_test = model.predict(X_test_transformed)
            if calc_importances:
                feature_importances = pd.Series(model.coef_, index=X_train_transformed.columns)
        case 'Lasso':
            # Why no alphas in LassoCV call here? https://github.com/emilylaiken/pmt-decay/blob/e655b9d0f66d145a1da0f2355fe79d8ea3ec6533/Machine%20Learning.ipynb#L212
            # I don't know, but I'm following the example in the notebook
            model = LassoCV(fit_intercept=True)
            model.fit(X_train_transformed, y_train, sample_weight=weights_train)
            y_hat_test = model.predict(X_test_transformed)  
            if calc_importances:
                feature_importances = pd.Series(model.coef_, index=X_train_transformed.columns)
        case 'OLS':
            x_train, x_test = sm.add_constant(X_train_transformed), sm.add_constant(x_test)
            model = sm.WLS(y_train, X_train_transformed, weights=weights_train)
            fitted_model = model.fit()
            y_hat_test = fitted_model.predict(X_test_transformed)
            if calc_importances:
                feature_importances = pd.Series(fitted_model.params, index=X_train_transformed.columns)
        case 'RandomForest':
            model = RandomForestRegressor()
            grid_search = GridSearchCV(model, param_grid=config['grid_search_params']["rf"], cv=3)
            grid_search.fit(X_train_transformed, y_train, sample_weight=weights_train)
            y_hat_test = grid_search.predict(X_test_transformed)
            if calc_importances:
                feature_importances = pd.Series(grid_search.best_estimator_.feature_importances_, index=X_train_transformed.columns)
        case 'GradientBoosting':
            model = GradientBoostingRegressor()
            grid_search = GridSearchCV(model, param_grid=config['grid_search_params']["gb"], cv=3)
            grid_search.fit(X_train_transformed, y_train, sample_weight=weights_train)
            y_hat_test = grid_search.predict(X_test_transformed)
            if calc_importances:
                feature_importances = pd.Series(grid_search.best_estimator_.feature_importances_, index=X_train_transformed.columns)
        case 'Stepwise':
            X_train_transformed['intercept'], X_test_transformed['intercept'] = np.ones(len(y_train)), np.ones(len(y_test))
            selected, r2s = forward_stepwise_OLS(X_train_transformed, y_train, y_test, weights_train, seed, n_folds=2)
            x_train_for_stepwise = X_train_transformed.reset_index(drop=True)
            x_train_for_stepwise = x_train_for_stepwise.astype(float)
            y_train_for_stepwise = y_train.reset_index(drop=True)
            y_train_for_stepwise = y_train_for_stepwise.astype(float)
            weights_for_stepwise = weights_train.reset_index(drop=True)
            weights_for_stepwise = weights_for_stepwise.astype(float)
            X_test_for_stepwise = X_test_transformed.reset_index(drop=True)
            X_test_for_stepwise = X_test_for_stepwise.astype(float)
            model = sm.WLS(y_train_for_stepwise, x_train_for_stepwise[selected], weights=weights_for_stepwise)
            fitted_model = model.fit()
            y_hat_test = fitted_model.predict(X_test_for_stepwise[selected])
            if calc_importances:
                feature_importances = pd.Series(fitted_model.params, index=selected)
            r2_df = pd.DataFrame(pd.Series(r2s, index=r2s.keys()))
            r2_df.columns = ['R^2']
            r2_df.index.name = 'Number of Features'
            r2_df["Bootstrap"] = i
        case _:
            raise ValueError("Invalid model type")

    results = pd.DataFrame([list(y_test), list(y_hat_test), list(weights_test), 
                        list(df.loc[test_ids]['hhsize'])]).T
    results.columns = ['y_test', 'yhat_test', 'weight', 'hh members']

    if np.std(results['yhat_test']) > 100000:
        print('Warning: Crazy prediction SD')
    results = pd.DataFrame(results.values.repeat(results['weight'], axis=0), 
                            columns=results.columns)
    results['random'] = np.random.rand(len(results))
    results['consumption'] = np.exp(results['y_test'])
    r2 = r2_score(results['y_test'], results['yhat_test'])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        spearman = spearmanr(results['y_test'], results['yhat_test'])[0]
        pearson = pearsonr(results['y_test'], results['yhat_test'])[0]

    metrics['r2'].append(r2)
    metrics['spearman'].append(spearman)
    metrics['pearson'].append(pearson)

    for threshold in range(10, 100, 10):
        y_test_cutoff = np.percentile(results['y_test'], threshold)
        yhat_test_cutoff = np.percentile(results['yhat_test'], threshold)
        random_cutoff = np.percentile(results['random'], threshold)
        
        y_test_binary = (results['y_test'] < y_test_cutoff).astype('int')
        yhat_test_binary = (results['yhat_test'] < yhat_test_cutoff).astype('int')
        
        metrics[f'recall_{threshold}'].append(recall_score(y_test_binary, yhat_test_binary))
        metrics[f'eer_{threshold}'].append(ee_rate(y_test_binary, yhat_test_binary))
        metrics[f'ier_{threshold}'].append(ie_rate(y_test_binary, yhat_test_binary))

        # if model_name == 'Stepwise':
        #     utility_grid = {}
        #     for benefits in range(5, 2010, 5):
        #         rho = 3
        #         results['benefits_consumption'] = y_test_binary * benefits / results['hh members']
        #         results['benefits_pmt'] = yhat_test_binary * benefits / results['hh members']
        #         results['benefits_random'] = (results['random'] < random_cutoff).astype('int') * benefits / results['hh members']
                
        #         results['utility_consumption'] = ((results['consumption'] + results['benefits_consumption'])**(1-rho))/(1-rho)
        #         results['utility_pmt'] = ((results['consumption'] + results['benefits_pmt'])**(1-rho))/(1-rho)
        #         results['utility_random'] = ((results['consumption'] + results['benefits_random'])**(1-rho))/(1-rho)
        #         results['utility_none'] = (results['consumption']**(1-rho))/(1-rho)
                
        #         pmt = results['utility_pmt'].mean()
        #         none = results['utility_none'].mean()
        #         utility_grid[benefits] = (pmt - none) / abs(none)
            
        #     metrics[f'utility_{threshold}'].append(utility_grid)
    results_and_metrics = {}
    if model_name == 'Stepwise':
        results_and_metrics["r2_df"] = r2_df
    if calc_importances:
        results_and_metrics["feature_importances"] = feature_importances
    results_and_metrics["metrics"] = metrics
    results_and_metrics["results"] = results
    return results_and_metrics

def run_ml(df: pd.DataFrame, config: dict[str, typing.Any], model_name: str, pool: mp.Pool) -> pd.DataFrame:
    # Min-max scaling of HH weights
    df[config['weight_col']] = 100*((df[config['weight_col']] - df[config['weight_col']].min())/(df[config['weight_col']].max() - df[config['weight_col']].min()))
    df[config['weight_col']] = df[config['weight_col']].apply(lambda x: int(x))

    run_bootstrap_partial = partial(run_single_bootstrap, df=df, config=config, model_name=model_name)
    metrics_and_results = list(tqdm(pool.imap(run_bootstrap_partial, range(config["num_bootstraps"])), total=config["num_bootstraps"], desc=f'Running {model_name} on {config["urban_or_rural"]} data'))

    metrics_df, results_df = list_of_dicts_to_dataframes(metrics_and_results)
    metrics_df.to_csv(f"output/{config['urban_or_rural']}/metrics_{model_name}.csv")
    results_df.to_csv(f"output/{config['urban_or_rural']}/results_{model_name}.csv")
    importances = pd.DataFrame(run_single_bootstrap(0, df, config, model_name, calc_importances=True)['feature_importances'])
    importances.columns = [f"{model_name} FI"]
    importances[f'{model_name} FI'] = importances[f'{model_name} FI'].apply(lambda x: round(x, 3))
    importances.to_csv(f"output/{config['urban_or_rural']}/importances_{model_name}.csv")
    if model_name == 'Stepwise':
        r2_df = pd.concat([m_r['r2_df'] for m_r in metrics_and_results])
        r2_df.to_csv(f"output/{config['urban_or_rural']}/r2_{model_name}.csv")
    return metrics_df, importances

def list_of_dicts_to_dataframes(metrics_and_results: list[dict[str, typing.Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert a list of dictionaries with 'metrics' and 'results' keys into two DataFrames.
    
    Args:
    data_list (list): List of dictionaries, each containing 'metrics' and 'results' keys
    
    Returns:
    tuple: (metrics_df, results_df)
    """
    metrics_list = []
    results_dict =  {"bootstrap_num": []}
    
    for i, item in enumerate(metrics_and_results):
        metrics = item['metrics']
        results = item['results']

        # Process metrics
        processed_metrics = {key: (np.nan if len(value) == 0 else value[0]) for key, value in metrics.items()}
        metrics_list.append(processed_metrics)
        
        # Process results
        results_dict['bootstrap_num'].extend([i]*len(results['y_test']))
        for key, result in results.items():
            if result.name not in results_dict:
                results_dict[result.name] = list(result.values)
            else:
                results_dict[result.name].extend(list(result.values))
    metrics_df = pd.DataFrame(metrics_list)
    results_df = pd.DataFrame(results_dict)
    
    return metrics_df, results_df


def main(config: dict[str, typing.Any]):
    df = pd.read_parquet("input/merged_data.parquet")
    df.index = df[config['id_col']]
    if config['urban_or_rural'] == 'urban':
        df = df[df['reside_2.0'] == False]
    elif config['urban_or_rural'] == 'rural':
        df = df[df['reside_2.0'] == True]
    num_cores = config['n_cpus']
    with mp.Pool(processes=num_cores, initializer=os.nice, initargs=(config['niceness'],)) as pool:
        for model in config['models_to_run']:
            run_ml(df, config, model, pool)


if __name__ == "__main__":
    config = load_config()
    main(config)