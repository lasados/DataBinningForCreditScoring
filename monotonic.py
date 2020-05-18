import pandas as pd
import numpy as np
from scipy.stats import chi2, chi2_contingency


def create_stats(X, Y, feature_type='numeric', max_bins=20):
    """
    Create DataFrame with statistical features of group_data by "Bucket.
    Arguments:
        X: np.array, feature
        Y: np.array
        feature_type: type 'numeric' or 'categorical'
        max_bins: max number of bins for 'numeric' feature
    Returns:
        df_stat: pd.DataFrame with statistical features such as WOE, IV
            columns = ['VAR_NAME', 'BUCKET", 'MIN_VALUE', 'MAX_VALUE', 'COUNT',
                       'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE',
                       'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV']
    """
    df_init = pd.DataFrame({"X": X, "Y": Y})

    # Split on notmiss and justmiss DataFrames
    df_notmiss = df_init[['X', 'Y']][df_init["X"].notnull()]
    df_justmiss = df_init[['X', 'Y']][df_init["X"].isnull()]

    if feature_type == 'categorical':
        # Bin - just a category
        group_data = df_notmiss.groupby('X', as_index=True)
    else:
        # For each pair numeric x, y -> find bucket
        df_bucket = pd.DataFrame({"X": df_notmiss['X'],
                                  "Y": df_notmiss['Y'],
                                  "Bucket": pd.qcut(df_notmiss['X'], max_bins, duplicates='drop')})

        # Grouping pairs (x, y) by "Bucket of x"
        group_data = df_bucket.groupby('Bucket', as_index=True)

    df_stat = pd.DataFrame({}, index=[])

    df_stat['BUCKET'] = group_data.groups.keys()

    # Compute min, max value of X for each "Bucket"
    if feature_type == 'categorical':
        df_stat["MIN_VALUE"] = group_data.groups.keys()
        df_stat["MAX_VALUE"] = df_stat["MIN_VALUE"]
    else:
        df_stat["MIN_VALUE"] = group_data.min()['X'].values
        df_stat["MAX_VALUE"] = group_data.max()['X'].values

    # Count number of points in "Bucket"
    df_stat["COUNT"] = group_data.count()['Y'].values
    # Count number of positive and negative points in "Bucket
    df_stat["EVENT"] = group_data.sum()['Y'].values
    df_stat["NONEVENT"] = df_stat["COUNT"] - df_stat["EVENT"]
    df_stat = df_stat.reset_index(drop=True)

    df_stat["EVENT_RATE"] = df_stat['EVENT'] / df_stat['COUNT']
    df_stat["NON_EVENT_RATE"] = df_stat['NONEVENT'] / df_stat['COUNT']

    # Compute probabilities of EVENT and NONEVENT for each "Bucket"
    df_stat["DIST_EVENT"] = df_stat['EVENT'] / df_stat.sum()['EVENT']
    df_stat["DIST_NON_EVENT"] = df_stat['NONEVENT'] / df_stat.sum()['NONEVENT']

    # Compute Weight Of Evidence and Information Value
    df_stat["WOE"] = np.log(df_stat['DIST_EVENT'] / df_stat['DIST_NON_EVENT'])
    df_stat["IV"] = (df_stat['DIST_EVENT'] - df_stat['DIST_NON_EVENT']) * df_stat["WOE"]
    df_stat["IV"] = df_stat["IV"].sum()

    # Set order of columns
    df_stat["VAR_NAME"] = "VAR"
    df_stat = df_stat[['VAR_NAME', 'BUCKET', 'MIN_VALUE', 'MAX_VALUE', 'COUNT',
                       'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE',
                       'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV']]
    # Delete infinity
    df_stat = df_stat.replace([np.inf, -np.inf], 0)
    return df_stat


def merge_rows(statistical_data, bottom_id, top_id):
    """
    Merge to rows in statistical_data with indexes bottom_id, top_id.
    Arguments:
        statistical_data: pd.DataFrame
        bottom_id: int, index of row in statistical_data
        top_id: int, index of row in statistical_data
    Returns:
        df_merged: pd.DataFrame

    """
    if bottom_id not in statistical_data.index:
        return statistical_data

    if top_id not in statistical_data.index:
        return statistical_data

    df_merged = statistical_data.copy()

    top_indx = top_id
    bot_indx = bottom_id
    bot_row = df_merged.iloc[bot_indx]  # row with smaller index
    top_row = df_merged.iloc[top_indx]  # row with higher index

    # Compute new boundaries of merged "Bucket"
    left_bnd = bot_row['BUCKET'].left
    right_bnd = top_row['BUCKET'].right
    # Merge top_row to bot_row
    merged_row = bot_row.copy()
    merged_row['BUCKET'] = pd.Interval(left=left_bnd, right=right_bnd)
    merged_row['MIN_VALUE'] = bot_row['MIN_VALUE']
    merged_row['MAX_VALUE'] = top_row['MAX_VALUE']
    merged_row['COUNT'] = bot_row['COUNT'] + top_row['COUNT']
    merged_row['EVENT'] = bot_row['EVENT'] + top_row['EVENT']
    merged_row['EVENT_RATE'] = merged_row['EVENT'] / merged_row['COUNT']
    merged_row['NONEVENT'] = bot_row['NONEVENT'] + top_row['NONEVENT']
    merged_row['NON_EVENT_RATE'] = merged_row['NONEVENT'] / merged_row['NONEVENT']
    merged_row["DIST_EVENT"] = merged_row['EVENT'] / df_merged.sum()['EVENT']
    merged_row["DIST_NON_EVENT"] = merged_row['NONEVENT'] / df_merged.sum()['NONEVENT']
    merged_row["WOE"] = np.log(merged_row['DIST_EVENT'] / merged_row['DIST_NON_EVENT'])

    # Place to table
    df_merged.iloc[bot_indx] = merged_row

    # Drop top_row
    df_merged.drop(top_indx, axis=0, inplace=True)
    df_merged.reset_index(inplace=True, drop=True)

    # Recalculate Information Value
    df_merged["IV"] = (df_merged['DIST_EVENT'] - df_merged['DIST_NON_EVENT']) * df_merged["WOE"]
    df_merged["IV"] = df_merged["IV"].sum()

    return df_merged


def make_monotonic(statistical_data, criteria='IV'):
    """
    Make data monotonic by WOE choosing best direction of monotonicity.
    Arguments:
        statistical_data: pd.DataFrame with column 'WOE'
        criteria: condition to choose best - number_of_bins -> 'bins', information_value -> 'IV"
    Returns:
        best_df: pd.DataFrame, 'increase' or 'decrease' monotonic df with max number of bins
    """

    df_up_down = {'increase': None, 'decrease': None}
    for direction in ['increase', 'decrease']:
        # Init data for current direction
        df_monotonic = statistical_data.copy()
        top_indx = df_monotonic.index[-1]
        bot_indx = top_indx - 1
        while bot_indx >= 0:
            bot_row = df_monotonic.iloc[bot_indx]  # row with smaller index
            top_row = df_monotonic.iloc[top_indx]  # row with higher index
            if direction == 'increase':
                # If WOE of top_row smaller -> merge to save increase monotonicity
                comparison = top_row['WOE'] < bot_row['WOE']
            else:
                # If WOE of top_row larger -> merge to save decrease monotonicity
                comparison = top_row['WOE'] < bot_row['WOE']

            if comparison:
                # Merging
                df_monotonic = merge_rows(df_monotonic, bot_indx, top_indx)
                # Reset top index
                top_indx = df_monotonic.index[-1]
                bot_indx = top_indx - 1
            else:
                top_indx -= 1
                bot_indx -= 1

        # Add current data_frame to collection
        df_up_down[direction] = df_monotonic

    if criteria == 'bins':
        # Choose data_frame with bigger number of bins
        n_bins_up = len(df_up_down['increase'])
        n_bins_down = len(df_up_down['decrease'])
        if n_bins_up >= n_bins_down:
            best_direction = 'increase'
        else:
            best_direction = 'decrease'
    else:
        # Choose data_frame with bigger IV
        IV_up = df_up_down['increase']['IV'][0]
        IV_down = df_up_down['decrease']['IV'][0]
        if IV_up >= IV_down:
            best_direction = 'increase'
        else:
            best_direction = 'decrease'

    #print('Choosed ' + best_direction)
    best_df = df_up_down[best_direction]
    return best_df


def compute_min_p_values(monotonic_df, min_size, min_rate):
    """
    Computes p-value for each pair of bins.
    Arguments:
        monotonic_df: pd.DataFrame with statistical features,  WOE monotonously decreasing
        min_size: int, threshold size of bins to merge (min count in bins)
        min_rate: float, threshold of EVENT_RATE to merge bins
    Returns:
        min_p: minimum p-value between two bins from all buckets
        buckets_min_p: index of two bins with min_p
    """

    df_monoton = monotonic_df.copy()
    n = len(df_monoton)

    n_obs = df_monoton['COUNT'].sum()
    p_values = dict()
    for i in range(n - 1):
        indx_top = i + 1
        indx_bot = i
        top_row = df_monoton.iloc[indx_top]  # row with higher index
        bot_row = df_monoton.iloc[indx_bot]  # row with smaller index

        # A_ij - number of examples of class "j" in interval "i"
        # R_i - number of examples in "i" interval
        # C_j - number of examples of "j" class
        # N - total number of examples
        # E_ij - expected frequency of class "j" in interval "i"

        A = np.array([[bot_row['EVENT'], bot_row['NONEVENT']],
                      [top_row['EVENT'], top_row['NONEVENT']]])

        R = np.array([bot_row['COUNT'], top_row['COUNT']])
        C = np.array([np.sum(A[:, j], axis=0) for j in range(A.shape[1])])
        N = np.sum(A)
        E = np.array([[R[i]*C[j]/N for j in range(A.shape[1])] for i in range(A.shape[0])])

        chi_2_stat = np.sum(np.power((A - E), 2)/E)

        # deg_free = (N_columns - 1)(N_rows - 1)
        deg_free = (A.shape[0] - 1)*(A.shape[1] - 1)
        assert (deg_free == 1), 'Check degree'

        # Compute p-value
        p_val = chi2.cdf(chi_2_stat, df=deg_free)

        # Filtering by size
        if (bot_row['COUNT']/n_obs < min_size) or (top_row['COUNT']/n_obs < min_size):
            p_val -= 1

        # Filtering by event_rate
        if (bot_row['EVENT_RATE'] < min_rate) or (top_row['EVENT_RATE'] < min_rate):
            p_val -= 1
        # Filtering by non_event_rate
        if (bot_row['NON_EVENT_RATE'] < min_rate) or (top_row['NON_EVENT_RATE'] < min_rate):
            p_val -= 1

        # Add final p-value to dictionary
        p_values[(indx_bot, indx_top)] = p_val

    # Find minimum p_value in dict
    buckets_min_p = min(p_values, key=p_values.get)  # index of rows with max p_value
    min_p = min(p_values.values())

    return min_p, buckets_min_p


def monotone_optimal_binning(X, Y,
                             min_bin_size=0.05, min_bin_rate=0.01,
                             min_p_val=0.95, max_bins=20, min_bins=2):
    """
    Algorithm of monotone optimal binning data for numeric feature.
    Arguments:
        X: np.array, numeric feature
        Y: np.array, binary target
        min_bin_size: minimum percentage of observations in each bin
        min_bin_rate: minimum event_rate and non_event_rate in each bin
        min_p_val: threshold to merge bins, if p_value(bin1, bin2) < min_p_val  --> merge bin1, bin2
        max_bins: max number of bins
        min_bins: min number of bins

    """

    df_stat = create_stats(X, Y, 'numeric', max_bins)

    # print('Initial data frame')
    # print(df_stat)
    # input()
    # Make df_stat Monotonic
    df_monotonic = make_monotonic(df_stat)
    n_bins = len(df_monotonic)
    # Merging bins
    while n_bins > min_bins:
        min_p, buckets_min_p = compute_min_p_values(df_monotonic,
                                                    min_size=min_bin_size,
                                                    min_rate=min_bin_rate)
        # print('Monotonic by WOE')
        # print(df_monotonic)
        # print('min_p = {}, indx_buckets = {}'.format(min_p, buckets_min_p))
        # input()

        n_bins = len(df_monotonic)
        if min_p < min_p_val:
            indx_bot, indx_top = buckets_min_p
            df_monotonic = merge_rows(df_monotonic, indx_bot, indx_top)
        else:
            break

    #     print('Merging')
    #     print(df_monotonic)
    #     input()
    #
    # print('FINAL')
    # print(df_monotonic)
    # input()
    return df_monotonic


def replace_feature_by_woe(X, Y):
    """
    Replace numeric feature on WOE by using algorithm of monotone optimal binning.
    Arguments:
        X: np.array, origin numeric feature
        Y: binary target
    Returns:
        X_woe: np.array, WOE of X
        info_value: Information Value of created binning.

    """
    df_optimal_binning = monotone_optimal_binning(X, Y)
    X_woe = X.copy().astype(float)
    for i_x, x in enumerate(X):
        for i_b, bucket in enumerate(df_optimal_binning['BUCKET']):
            if x in bucket:
                X_woe[i_x] = df_optimal_binning['WOE'].iloc[i_b]
                break
    # print('Init X \n', X)
    # print('Woe X \n', X_woe)
    info_value = df_optimal_binning['IV'][0]
    return X_woe, info_value


def create_bins_df(raw_data):
    """ Create data frame with stats of all features in input data."""
    # Init data and target
    data = raw_data.copy()
    Y = (data['y'] == 'yes').astype(int).values
    data.drop(columns='y', inplace=True)

    # Init final df
    df_bins = pd.DataFrame()

    for column in data:
        X = data[column].values
        # Check type of feature
        try:
            # If numeric feature - use algorithm of optimal binning
            X = X.astype('float32')
            curr_stats_df = monotone_optimal_binning(X, Y)
        except ValueError:
            # If categorical feature - just compute stats
            curr_stats_df = create_stats(X, Y, feature_type='categorical')

        curr_stats_df['VAR_NAME'] = column

        # Add to final df
        df_bins = pd.concat([df_bins, curr_stats_df], ignore_index=True)

    # Select inforamtion values from df
    IV = pd.DataFrame({'IV': df_bins.groupby('VAR_NAME')['IV'].max()})
    IV.reset_index(inplace=True)

    return df_bins, IV


def create_woe_df_numeric(raw_data, numeric_columns):
    """
    Create data with WOE-features.
    Args:
        raw_data: pd.DataFrame, initial data
        numeric_columns: list of strings, names of numeric features
    Returns:
        data_woe: pd.DataFrame
        info_values: dictionary with Information Value of each feature.
    """
    data = raw_data.copy()
    Y = (data['y'] == 'yes').astype(int).values
    data_woe = pd.DataFrame()
    data_woe['target'] = Y
    info_values = dict()
    for column in numeric_columns:
        X = data[column].values
        data_woe['WOE_' + column], info_values['WOE_' + column] = replace_feature_by_woe(X, Y)

    return data_woe, info_values


def delete_correlated_features(df_woe, iv_values, method='cramer', cut_off=0.05):
    """
    Drop columns with big correlation and small Information Value.
    Args:
        df_woe: pd.DataFrame
        iv_values: dictionary with Information Value of each feature
        method: method for computing correlation {'cramer', 'pearson', 'spearman'}
        cut_off: float, threshold to drop
    Returns:
        df_uncorr: pd.DataFrame without correlated features
    """
    df_features_woe = df_woe.drop(columns='target')
    n_features = df_features_woe.shape[1]
    to_drop = []
    corr_matrix = pd.DataFrame(columns=df_features_woe.columns, index=df_features_woe.columns)

    # Compute continuous correlation
    if method in ['pearson', 'spearman']:
        # Compute correlation matrix
        corr_matrix = df_features_woe.corr(method).abs()
        for i_row in range(n_features - 1):
            for j_col in range(i_row + 1, n_features):
                # Correlation between feature i, j
                corr_ij = corr_matrix.iloc[i_row, j_col]
                # Deletion
                if corr_ij > cut_off:
                    feature_name_i = corr_matrix.index[i_row]
                    feature_name_j = corr_matrix.columns[j_col]
                    # Don't add feature to drop if already added
                    if (feature_name_i in to_drop) or (feature_name_j in to_drop):
                        continue
                    else:
                        # Delete features with least Information Value
                        i_IV = iv_values[feature_name_i]
                        j_IV = iv_values[feature_name_j]
                        if i_IV < j_IV:
                            to_drop.append(feature_name_i)
                        else:
                            to_drop.append(feature_name_j)
    else:
        # Compute categorical correlation
        assert method == 'cramer', 'Method does not exists'

        # Compute correlation as Cramers Coefficient
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Choose features
                woe_i = df_features_woe.iloc[:, i].values
                woe_j = df_features_woe.iloc[:, j].values
                # Create categories
                unique_woe_i = sorted(list(set(woe_i)))
                unique_woe_j = sorted(list(set(woe_j)))
                confusion_dict = {(w_i, w_j): 0 for w_i in unique_woe_i for w_j in unique_woe_j}

                # Calculate confusion matrix
                for w_i, w_j in zip(woe_i, woe_j):
                    confusion_dict[(w_i, w_j)] += 1
                confusion_matrix = pd.DataFrame(columns=unique_woe_i,
                                                index=unique_woe_j)
                for key in confusion_dict:
                    column = key[0]
                    row = key[1]
                    confusion_matrix.loc[row, column] = confusion_dict[key]

                # Calculate Cramers Coefficient
                k1, k2 = confusion_matrix.shape
                n = np.sum(confusion_matrix.values)
                chi2_stat = chi2_contingency(confusion_matrix)[0]
                phi_cram = np.sqrt(chi2_stat/(n*min(k1, k2) - 1))

                # Add to correlation matrix
                feature_name_i = df_features_woe.iloc[:, i].name
                feature_name_j = df_features_woe.iloc[:, j].name
                corr_matrix.loc[feature_name_i, feature_name_j] = phi_cram

                # Deletion features
                if phi_cram > cut_off:
                    # Don't add feature to drop if already added
                    if (feature_name_i in to_drop) or (feature_name_j in to_drop):
                        continue
                    else:
                        # Delete features with least Information Value
                        i_IV = iv_values[feature_name_i]
                        j_IV = iv_values[feature_name_j]
                        if i_IV < j_IV:
                            to_drop.append(feature_name_i)
                        else:
                            to_drop.append(feature_name_j)

    print('Info_Values = ', iv_values)
    print('Columns to drop = ', to_drop)
    print(corr_matrix)
    input()
    df_uncorr = df_woe.drop(columns=to_drop)
    return df_uncorr


numeric_col = ['age', 'balance', 'day', 'duration', 'pdays']
data = pd.read_excel('data/bank.xlsx')

df_final, iv_final = create_bins_df(data)
print(iv_final.sort_values('IV'))
# data_woe, info_values = create_woe_df_numeric(data, numeric_col)
# data_not_corr = delete_correlated_features(data_woe, info_values)
# print(data_not_corr)


