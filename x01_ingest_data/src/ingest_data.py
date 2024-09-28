import pandas as pd
import yaml
import typing
import numpy as np
import pyreadstat
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn import decomposition as sklearn_decomposition

#TODO: Featurize shocks more fully to match the survey instrument. 


def main(config: dict[str, typing.Any]) -> None:
    data_dir = Path.cwd() / 'input'
    config["covariate_labels_to_descriptions"] = {}
    config["covariate_labels_to_modules"] = {}
    df, meta = pyreadstat.read_dta(data_dir / config['files_to_use'][0])
    df.columns = df.columns.str.lower()
    df = df[config['keep_cols']['hh_mod_a_filt.dta']]
    for covariate_label in meta.column_names_to_labels.keys():
        config["covariate_labels_to_modules"][covariate_label] = config['files_to_use'][0]
    df, config["covariate_labels_to_modules"] = create_full_df(df, config, config["covariate_labels_to_modules"])
    df = add_roster(df, data_dir / config['roster_file'], config)
    df, summary, missing_pcts = clean_full_data(df, config)
    df, summary = remove_outliers(df, config, summary)
    all_summary = make_post_encoding_summary(df, config, missing_pcts)
    all_summary.to_csv("output/post_encoding_summary.csv", index=False)
    urban_summary = make_post_encoding_summary(df[df['reside_2.0'] == False], config, missing_pcts)
    rural_summary = make_post_encoding_summary(df[df['reside_2.0'] == True], config, missing_pcts)
    urban_summary.to_csv("output/post_encoding_summary_urban.csv", index=False)
    rural_summary.to_csv("output/post_encoding_summary_rural.csv", index=False)
    df.to_parquet('output/merged_data.parquet', index=False)

def make_post_encoding_summary(df: pd.DataFrame, config: dict[str, typing.Any], missing_pcts: pd.Series) -> pd.DataFrame:
    stats = []
    for var in df.columns.tolist():
        if pd.api.types.is_numeric_dtype(df[var]):
            weighted_mean = np.average(df[var], weights=df['hh_wgt'])
            weighted_std = np.sqrt(np.average((df[var] - weighted_mean) ** 2, weights=df['hh_wgt']))
        else:
            weighted_mean = df[var].mode()[0]
            weighted_std = 0
        try:
            missing_fraction = missing_pcts[var]
        except KeyError:
            missing_fraction = 0
        stats.append([var, weighted_mean, weighted_std, missing_fraction])
    summary = pd.DataFrame(stats, columns=['covariate', 'mean', 'std', 'Missing Fraction before Imputation/Dummying'])
    return summary

def remove_outliers(df: pd.DataFrame, config: dict[str, typing.Any], summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Removing outliers")
    original_num_households = len(df)

    # Define columns to exclude from outlier analysis
    # columns_to_omit = set(config.get('columns_to_omit_from_outlier_analysis', []))
    # columns_to_omit.update({
    #     c for c in df.columns if any(keyword in c for keyword in ['mosaiks', 'gap'])
    # })
    # columns_to_omit.update(df.select_dtypes(include=['object', 'category']).columns)
    # columns_to_omit.update(df.columns[df.isin([0, 1, np.nan, None]).all()])

    # for _, row in summary.iterrows():
    #     if row.module in config.get('excluded_modules', set()) or 'ID' in row.description:
    #         columns_to_omit.update(row['columns'])

    # columns_for_outlier_analysis = [c for c in df.columns if c not in columns_to_omit]

    # # Log transform specified columns, handling non-positive values
    # for col in config['to_log_transform']:
    #     if col in columns_for_outlier_analysis:
    #         df[col] = np.where(df[col] > 0, np.log(df[col]), np.nan)

    # # Calculate z-scores, ignoring NaN values
    # z_scores = df[columns_for_outlier_analysis].apply(lambda x: (x - x.mean()) / x.std())

    # # Apply different thresholds for log-transformed and regular columns
    # z_scores_thresholded = z_scores.copy()
    # z_scores_thresholded[z_scores_thresholded.columns.difference(config['to_log_transform'])] /= config['NUM_STDS']
    # z_scores_thresholded[config['to_log_transform']] /= config['NUM_STDS_LOG_TRANSFORMED']

    # mask = (z_scores_thresholded.abs() > 1)
    # outlier_households = mask.any(axis=1)

    # # Prepare outlier information
    # outlier_info = df[outlier_households].copy()
    # outlier_info = pd.concat([outlier_info, z_scores.loc[outlier_households].add_suffix('_z')], axis=1)
    # outlier_info = outlier_info.melt(
    #     id_vars=df.columns, 
    #     var_name='field', 
    #     value_name='z_score'
    # )

    # Calculate asset index
    # pca_input_columns = [c for c in df.columns if c.startswith(('durable_asset', 'ag_asset'))]
    # pca = sklearn_decomposition.PCA(n_components=1, random_state=config['random_state'])
    # asset_data = df[pca_input_columns].fillna(df[pca_input_columns].mean())
    # all_missing_cols = asset_data.columns[asset_data.isnull().all()].tolist()
    # all_missing_cols.extend(asset_data.columns[asset_data.eq(0).all()].tolist())
    # asset_data = asset_data.drop(columns=all_missing_cols)

    # asset_data = (asset_data - asset_data.mean()) / asset_data.std()
    # asset_data.to_csv("output/asset_data.csv", index=False)
    # df['asset_index'] = pca.fit_transform(asset_data)

    # Handle outliers

    # case_ids_to_drop = set()
    # # Winsorizing the home price, and potential rent price
    # for _, row in outlier_info.iterrows():
    #     if row.field in ('outcome', 'asset_index'):
    #         case_ids_to_drop.add(row.case_id)
    #     elif config['winsorize']:
    #         field = row.field.replace('_z', '')
    #         if not pd.isna(row.z_score):
    #             clipped_value = df[field].mean() + config['NUM_STDS'] * np.sign(row.z_score) * df[field].std()
    #             df.loc[df.case_id == row.case_id, field] = clipped_value

    # df = df[~df.case_id.isin(case_ids_to_drop)]

    # # Finalize outlier information
    # column_to_description_map = {'outcome': 'estimated daily consumption (USD 2017)'}
    # column_to_description_map.update({col: row.description for _, row in summary.iterrows() for col in row.columns})

    # outlier_info['field'] = outlier_info['field'].str.replace('_z', '')
    # outlier_info = outlier_info[outlier_info.apply(lambda row: mask.loc[df[df.case_id == row.case_id].index, row.field].any(), axis=1)]
    # outlier_info['value'] = outlier_info.apply(lambda row: row[row['field']], axis=1)
    # outlier_info['description'] = outlier_info.field.map(column_to_description_map)
    # outlier_info = outlier_info[['case_id', 'field', 'description', 'value', 'z_score']]
    # outlier_info = outlier_info.sort_values(by='z_score', ascending=False)
    # outlier_info.to_csv("output/outlier_info.csv", index=False)
    # df = df.drop(columns=['asset_index'])

    # Manually dropping the household with 500 chairs and 50 tables
    # AKA Bobby Tables
    df = df.drop(index=[5594])

    print(f"After dropping outliers, {len(df)} households of the original {original_num_households} remain.")
    return df, summary


def clean_full_data(df: pd.DataFrame, config: dict[str, typing.Any]) -> pd.DataFrame:
    df = df.dropna(axis=1, how='all')
    df = df.drop(columns=df.columns[df.eq(0).all()].tolist())
    df = df.dropna(subset=[config['outcome_column']])

    # columns_to_drop = config['cols_to_drop']
    columns_to_drop = ['adult']
    # Dropping columns that have too many categories to one-hot encode
    columns_to_drop.extend([c for c in df.columns if ('_oth' in c or '2oth' in c) and c not in config['known_categorical_columns']])
    df = df.drop(columns=columns_to_drop)

    # Real or nominal here? Currently using real.
    df["outcome"] = df[config['outcome_column']] * config['currency_conversion_factor'] 
    df["outcome"] /= (df.num_adults + df.num_children)
    df["outcome"] /= 365

    # columns not to be imputed, coerced to numeric, or one-hot encoded.
    # summary table won't include these either - for now, this seems fine. 
    columns_to_reserve = [
        'case_id', 'hh_wgt', 'outcome', 'hhid', 'pid'
    ]
    df_reserved = df[columns_to_reserve]

    df_to_process = df[df.columns.difference(columns_to_reserve)].copy()
    
    # coerce columns to numeric that can be coerced
    for c in df_to_process.columns:
        df_to_process[c] = pd.to_numeric(df_to_process[c], errors='ignore')
    
    # coerce known categorical columns to string
    for c in config['known_categorical_columns']:
        df_to_process[c] = df_to_process[c].astype(str)

    summary, df, missing_pcts = make_pre_encoding_summary(df_to_process, df_reserved, config["covariate_labels_to_descriptions"])
    summary.to_csv("output/pre_encoding_summary.csv", index=False)
    return df, summary, missing_pcts
    

def make_pre_encoding_summary(df: pd.DataFrame, df_reserved: pd.DataFrame, covariate_labels_to_descriptions: dict[str, str]) -> pd.DataFrame:
    # Compile column summary (before imputing and one-hot encoding)
    missing_counts = df.isnull().sum() + (df == "").sum()  
    means = df.mean(skipna=True, numeric_only=True)
    stds = df.std(skipna=True, numeric_only=True)
    missing_pcts = missing_counts / len(df)
    missing_pcts.index = df.columns.tolist()
    
    summary = pd.concat((missing_counts, means, stds), axis=1)
    summary.columns = ['missing_count', 'mean', 'std']
    summary.reset_index(names='covariate', inplace=True)

    summary['missing_fraction'] = summary.missing_count / len(df)

    # TODO: replace with a dict get() with default
    def interpret_column_name(column_name, covariate_labels_to_descriptions):
        if column_name in covariate_labels_to_descriptions:
            return covariate_labels_to_descriptions[column_name]
        return column_name
    
    summary['description'] = summary.covariate.apply(lambda x: interpret_column_name(x, config["covariate_labels_to_descriptions"]))
    summary['module'] = summary.covariate.map(config["covariate_labels_to_modules"])
    
    # Calculate weighted statistics
    weights = df_reserved['hh_wgt']
    
    summary['missing_fraction'] = (df.isnull().sum() + (df == "").sum()) / len(df)
    summary['missing_fraction'] = summary['missing_fraction'].round(2)
    
    summary['median'] = df.median(numeric_only=True).round(2)
    
    try:
        summary['mean'] = (df * weights.values[:, None]).sum() / weights.sum()
    except TypeError:
        summary['mean'] = summary['mean'].round(2)
    try:
        summary['std'] = np.sqrt((df.subtract(summary['mean'], axis=1) ** 2 * weights.values[:, None]).sum() / weights.sum())
    except TypeError:
        summary['std'] = summary['std'].round(2)
    
    # Split into numeric and non-numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    df_non_numeric = df.select_dtypes(exclude=[np.number, np.datetime64])
    
    def get_covariate_type(cov):
        
        if cov in df_numeric.columns:
            return 'numeric'
        elif cov in df_non_numeric.columns:
            return 'categorical'
    
    summary['type'] = summary['covariate'].apply(get_covariate_type)
    covariate_to_columns_map = {
        covariate: [covariate] for covariate in summary.covariate
    }
    
    MISSINGNESS_CUTOFF = 0.15
    covariates_over_cutoff = summary[summary.missing_fraction > MISSINGNESS_CUTOFF].covariate.values
    for covariate in df_numeric.columns:
        if covariate in covariates_over_cutoff:
            dummy_column = f'{covariate}_nan'
            df_numeric[dummy_column] = df_numeric[covariate].isna()
            covariate_to_columns_map[covariate].append(dummy_column)
    
    # This is different from what roshni does: She uses 0 to impute
    # if missingness is >15%. 
    all_missing_cols = df_numeric.columns[df_numeric.isnull().all()].tolist()
    df_numeric = df_numeric.drop(columns=all_missing_cols)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(df_numeric)
    
    columns = df_numeric.columns
    df_numeric = pd.DataFrame(imputer.transform(df_numeric))

    df_numeric.columns = columns


    # one-hot encode categoricals.
    # This is different from what roshni does. I'm encoding missing values
    # with a category regardless of missing fraction.
    
    # Ensure all known categorical columns are in df_non_numeric
    known_categorical_columns = set(config['known_categorical_columns'])
    non_numeric_columns = set(df_non_numeric.columns)
    assert known_categorical_columns.issubset(non_numeric_columns), \
        f"The following known categorical columns are not in df_non_numeric: {known_categorical_columns - non_numeric_columns}"
    one_hot_encoder = OneHotEncoder(
        drop='if_binary', sparse_output=False
    ).fit(df_non_numeric)
    encoded_data = one_hot_encoder.transform(df_non_numeric)
    df_non_numeric_encoded = pd.DataFrame(encoded_data)
    df_non_numeric_encoded.columns = one_hot_encoder.get_feature_names_out()

    df_non_numeric_encoded = df_non_numeric_encoded.merge(df_non_numeric, left_index=True, right_index=True)
    
    # populate the map from original column names to the list of one-hot columns. 
    for i in range(len(one_hot_encoder.feature_names_in_)):
    
        covariate = one_hot_encoder.feature_names_in_[i]
        categories = one_hot_encoder.categories_[i]
    
        if one_hot_encoder.drop_idx_[i] is not None:
            categories = np.delete(categories, one_hot_encoder.drop_idx_[i])
    
        covariate_to_columns_map[covariate] = [
            f'{covariate}_{category}' for category in categories
        ]
    
    df = df_reserved.join(df_numeric).join(df_non_numeric_encoded)
    summary['columns'] = summary.covariate.map(covariate_to_columns_map)
    summary = summary[summary["description"] != "hhid"]
    summary.to_csv("output/pre_encoding_summary.csv", index=False)
    df = df.drop(columns=['hh_f01', 'hh_f07', 'hh_f08', 'hh_f09', 'hh_f11', 
                          'hh_f12', 'hh_f19', 'hh_f36', 'hh_f41', 'hh_f41_4', 
                          'hh_f43', 'hh_h01', 'hh_h04', 'hh_head_education',
                          'hh_head_has_cellphone', 'hh_head_labor_type',
                          'hh_head_sex', 'hh_t01', 'hh_t02', 'hh_t03', 'hh_t04', 
                          'region', 'reside'])
    return summary, df, missing_pcts

def handle_shocks(shocks: pd.DataFrame) -> pd.DataFrame:
    shocks_pivoted = shocks.pivot_table(
        index='case_id',
        columns='hh_u0a',
        values='hh_u01_1',
        aggfunc='sum',
        fill_value=0,
        observed=True
    ).add_prefix('shock_')
    shocks_pivoted.columns.name = None
    shocks_pivoted = shocks_pivoted.loc[:, shocks_pivoted.sum(axis=0) > 0]
    shocks_covariate_to_desciption = dict()

    for covariate in shocks_pivoted.columns:
        shocks_covariate_to_desciption[covariate] = f'Shock experienced: {covariate}'
    return shocks_pivoted, shocks_covariate_to_desciption

def handle_durable_goods(durable_goods: pd.DataFrame) -> pd.DataFrame:
    durable_goods_pivoted = durable_goods.pivot_table(
        index='case_id',
        columns='hh_l02',
        values='hh_l03',
        aggfunc='sum',
        fill_value=0,
        observed=True
    ).add_prefix('durable_asset_')
    durable_goods_pivoted.columns.name = None
    durable_goods_pivoted = durable_goods_pivoted.loc[:, durable_goods_pivoted.sum(axis=0) > 0]
    durable_goods_covariate_to_desciption = dict()

    for covariate in durable_goods_pivoted.columns:
        durable_goods_covariate_to_desciption[covariate] = f'number owned: {covariate}'
    return durable_goods_pivoted, durable_goods_covariate_to_desciption

def handle_ag_goods(ag_goods: pd.DataFrame) -> pd.DataFrame:
    ag_goods.hh_m0b = ag_goods.hh_m0b.astype(str)

    ag_goods.loc[ag_goods.hh_m0b == 'OTHER', 'hh_m0b'] = ag_goods[ag_goods.hh_m0b == 'OTHER']['hh_m0b_oth']

    ag_goods_pivoted = ag_goods.pivot_table(
        index='case_id', 
        columns='hh_m0b', 
        values='hh_m01', 
        aggfunc='sum', 
        fill_value=0,
        observed=True # to avoid a warning
    ).add_prefix('ag_asset_')
    ag_goods_pivoted.columns.name = None
    ag_goods_pivoted = ag_goods_pivoted.loc[:, ag_goods_pivoted.sum(axis=0) > 0]
    ag_goods_covariate_to_description = dict()

    for covariate in ag_goods_pivoted.columns:
        ag_goods_covariate_to_description[covariate] = f'number owned: {covariate}'
    return ag_goods_pivoted, ag_goods_covariate_to_description

def handle_land_ownership(land_ownership_df: pd.DataFrame) -> pd.DataFrame:
    land_ownership_df['land_ownership'] = 1
    land_ownership_df = land_ownership_df[['hhid', 'land_ownership']].groupby('hhid')[["land_ownership"]].count().rename(columns={'land_ownership': 'land_ownership'})
    return land_ownership_df

def create_full_df(df: pd.DataFrame, config: dict[str, typing.Any], covariate_labels_to_modules: dict[str, str]) -> pd.DataFrame:
    extra_modules = {}
    data_dir = Path.cwd() / 'input'
    for file in config['files_to_use'][1:]:
        df_temp, meta = pyreadstat.read_dta(
            data_dir / file, apply_value_formats=True
        )
        print(f"File read in: {file}")
        df_temp.columns = df_temp.columns.str.lower()
        df_temp = df_temp[config['keep_cols'][file]]
        match file:
            case 'HH_MOD_L.dta':
                df_temp, durable_goods_covariate_to_desciption = handle_durable_goods(df_temp)
                extra_modules['HH_MOD_L_durable_goods'] = durable_goods_covariate_to_desciption
            case 'HH_MOD_M.dta':
                df_temp, ag_goods_covariate_to_desciption = handle_ag_goods(df_temp)
                extra_modules['HH_MOD_M_ag_goods'] = ag_goods_covariate_to_desciption
            case "HH_MOD_F1.dta":
                df_temp = handle_land_ownership(df_temp)
            case 'HH_MOD_U.dta':
                df_temp, shocks_covariate_to_desciption = handle_shocks(df_temp)
                extra_modules['HH_MOD_U_shocks'] = shocks_covariate_to_desciption
            case _:
                pass #  Nothing special to do for all other files.
        for covariate_label in meta.column_names_to_labels.keys():
            covariate_labels_to_modules[covariate_label] = file
        df = merge_dfs(df, df_temp, file)
        print(f"Merged file: {file}")
    return df, covariate_labels_to_modules

def add_roster(df: pd.DataFrame, roster_path: Path, config: dict[str, typing.Any]) -> pd.DataFrame:
    roster, meta = pyreadstat.read_dta(roster_path)
    roster = clean_roster(roster, config)
    individual_labor_df, meta = pyreadstat.read_dta('input/HH_MOD_E.dta')
    individual_labor_df.columns = individual_labor_df.columns.str.lower()
    individual_labor_df = individual_labor_df[['pid', 'case_id','hhid', 'hh_e06_8a']]
    roster = merge_dfs(roster, individual_labor_df, 'HH_MOD_E.dta', on=['hhid', 'pid'])
    hh_adult_counts = (
        roster[roster.adult].groupby('case_id')[['hhid']].count().rename(columns={'hhid': 'num_adults'})
    )
    hh_child_counts = (
        roster[~roster.adult].groupby('case_id')[['hhid']].count().rename(columns={'hhid': 'num_children'})
    )
    hh_phone_counts = (
        roster[['hh_b04a', 'case_id', 'hhid']].groupby('case_id')[["hhid"]].count().rename(columns={'hhid': 'num_phones'})
    )
    hh_head_labor_type = (
        roster[roster['hh_b04'] == 1][['hhid', 'case_id', 'hh_e06_8a']].groupby('case_id')[['hh_e06_8a']].max().rename(columns={'hh_e06_8a': 'hh_head_labor_type'})
    )
    # Taking the demographic info of the head of household
    hh_head_info = (
        roster[roster['hh_b04'] == 1]
    ).drop(columns=['hhid'])
    hh_head_info = hh_head_info.drop(columns=['hh_b04'])
    hh_head_info = hh_head_info.rename(columns={
        'hh_b05a': 'hh_head_age',
        'hh_b03': 'hh_head_sex',
        'hh_b21': 'hh_head_education',
        'hh_b04a': 'hh_head_has_cellphone'
    })
    df = (
        df
        .merge(hh_adult_counts, how='left', on='case_id')
        .merge(hh_child_counts, how='left', on='case_id')
        .merge(hh_phone_counts, how='left', on='case_id')
        .merge(hh_head_info, how='left', on='case_id')
        .merge(hh_head_labor_type, how='left', on='case_id')
    )
    df[['num_adults', 'num_children']] = (
        df[['num_adults', 'num_children']].fillna(value=0)
    )
    assert (df.num_adults + df.num_children <= 0).sum() == 0
    return df

def clean_roster(roster: pd.DataFrame, config: dict[str, typing.Any]) -> pd.DataFrame:
    roster.columns = [c.lower() for c in roster.columns]
    roster = roster[config['keep_cols']['HH_MOD_B.dta']]
    roster['adult'] = roster.hh_b05a >= config['ADULT_MIN_AGE']
    return roster

def columns_equal(df: pd.DataFrame, col1: str, col2: str) -> bool:
    c1 = df[col1]
    c2 = df[col2]

    if pd.api.types.is_numeric_dtype(c1) and pd.api.types.is_numeric_dtype(c2):
        return np.isclose(c1, c2, rtol=1e-4, equal_nan=True).all()
    else:
        try:
            eq = (c1 == c2).all()
        except TypeError:
            # mismatched categories -> this comparison raises a type error
            eq = False
        return eq

def merge_dfs(df: pd.DataFrame, df_temp: pd.DataFrame, file_name: str, on='case_id') -> pd.DataFrame:
    df = df.merge(df_temp, on=on, how='outer', suffixes=('_left', '_right'))
    for c in df.columns:
        if c.endswith('_left'):
            c_left = c
            base = c_left[:-5]
            c_right = f'{base}_right'

            match = columns_equal(df, c_left, c_right)
            
            if match:
                df.drop(columns=c_right, inplace=True)
                df.rename(columns={c_left: base}, inplace=True)
            # geographies are sometimes named and sometimes encoded as integers. If we've got one of each,  
            # keep the string name: that way it won't accidentally be treated as numeric later.
            elif (
                (base in ['region', 'district'])
                & (
                    pd.api.types.is_numeric_dtype(df[c_left]) 
                    + pd.api.types.is_numeric_dtype(df[c_right]) 
                    == 1
                    )
            ):
                if pd.api.types.is_numeric_dtype(df[c_left]):
                    df.drop(columns=c_left, inplace=True)
                    df.rename(columns={c_right: base}, inplace=True)
                else:
                    df.drop(columns=c_right, inplace=True)
                    df.rename(columns={c_left: base}, inplace=True)
            else:
                print(f'error merging {file_name}, mismatch in {base}')
                df.drop(columns=c_right, inplace=True)
                df.rename(columns={c_left: base}, inplace=True)
    return df

def load_config() -> dict[str, typing.Any]:
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    config = load_config()
    main(config)