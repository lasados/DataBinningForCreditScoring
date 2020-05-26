import pandas as pd
import numpy as np
from scipy.stats import chi2, chi2_contingency
import time

# Raise error if division by zero occurs
np.seterr(divide='raise')


def binary_search(arr, low, high, x):
    if high >= low:
        mid = low + (high - low) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] > x:
            return binary_search(arr, low, mid - 1, x)
        else:
            return binary_search(arr, mid + 1, high, x)
    else:
        return low


def is_numeric(array):
    """Check is array contains numeric or categorical data."""
    verdict = False
    for x in array:
        if x != x:
            continue
        else:
            try:
                float(x)
                return True
            except ValueError:
                return False


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
    df_stat["NONEVENT"] = df_stat["COUNT"].values - df_stat["EVENT"].values

    # Add statistics from missed
    if df_justmiss.shape[0] > 0:
        df_stat_miss = pd.DataFrame({'BUCKET': np.nan}, index=[0])
        df_stat_miss["MIN_VALUE"] = np.nan
        df_stat_miss["MAX_VALUE"] = np.nan
        df_stat_miss["COUNT"] = df_justmiss.count()['Y']
        df_stat_miss["EVENT"] = df_justmiss.sum()['Y']
        df_stat_miss["NONEVENT"] = df_stat_miss["COUNT"] - df_stat_miss["EVENT"]
    else:
        df_stat_miss = pd.DataFrame({'BUCKET': np.nan}, index=[0])
        df_stat_miss["MIN_VALUE"] = np.nan
        df_stat_miss["MAX_VALUE"] = np.nan
        df_stat_miss["COUNT"] = 0
        df_stat_miss["EVENT"] = 0
        df_stat_miss["NONEVENT"] = 0

    df_stat = df_stat.append(df_stat_miss, ignore_index=True)

    epsilon = 1e-6
    df_stat["EVENT_RATE"] = df_stat['EVENT'].values / (df_stat['COUNT'].values + epsilon)
    df_stat["NON_EVENT_RATE"] = df_stat['NONEVENT'].values / (df_stat['COUNT'].values + epsilon)

    # Compute probabilities of EVENT and NONEVENT for each "Bucket"
    df_stat["DIST_EVENT"] = df_stat['EVENT'].values / (df_stat.sum()['EVENT'] + epsilon)
    df_stat["DIST_NON_EVENT"] = df_stat['NONEVENT'].values / (df_stat.sum()['NONEVENT'] + epsilon)

    # Compute Weight Of Evidence and Information Value
    df_stat["WOE"] = np.log((df_stat['DIST_NON_EVENT'].values + epsilon) /
                            (df_stat['DIST_EVENT'].values + epsilon))
    df_stat["IV"] = ((df_stat['DIST_NON_EVENT'].values - df_stat['DIST_EVENT'].values)
                     * df_stat["WOE"].values)
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
    merged_row['MIN_VALUE'] = float(bot_row['MIN_VALUE'])
    merged_row['MAX_VALUE'] = float(top_row['MAX_VALUE'])
    merged_row['COUNT'] = float(bot_row['COUNT']) + float(top_row['COUNT'])
    merged_row['EVENT'] = float(bot_row['EVENT']) + float(top_row['EVENT'])
    epsilon = 1e-6
    merged_row['EVENT_RATE'] = float(merged_row['EVENT']) / (float(merged_row['COUNT']) + epsilon)
    merged_row['NONEVENT'] = float(bot_row['NONEVENT']) + float(top_row['NONEVENT'])
    merged_row['NON_EVENT_RATE'] = (float(merged_row['NONEVENT'])
                                    / (float(merged_row['COUNT']) + epsilon))
    merged_row["DIST_EVENT"] = float(merged_row['EVENT']) / (df_merged.sum()['EVENT'] + epsilon)
    merged_row["DIST_NON_EVENT"] = (float(merged_row['NONEVENT'])
                                    / (df_merged.sum()['NONEVENT'] + epsilon))
    merged_row["WOE"] = np.log((float(merged_row['DIST_NON_EVENT']) + epsilon) /
                               (float(merged_row['DIST_EVENT']) + epsilon))

    # Place to table
    df_merged.iloc[bot_indx] = merged_row

    # Drop top_row
    df_merged.drop(top_indx, axis=0, inplace=True)
    df_merged.reset_index(inplace=True, drop=True)

    # Recalculate Information Value
    df_merged["IV"] = (df_merged['DIST_NON_EVENT'] - df_merged['DIST_EVENT']) * df_merged["WOE"]
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
        # Find number of rows with nan
        n_nan_rows = 0
        if df_monotonic.iloc[top_indx]['BUCKET'] != df_monotonic.iloc[top_indx]['BUCKET']:
            n_nan_rows = 1

        top_indx -= n_nan_rows
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
                top_indx = df_monotonic.index[-1] - n_nan_rows
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
    n_not_nan = len(df_monoton)
    n_nan_rows = 0
    if df_monoton.iloc[-1]['BUCKET'] != df_monoton.iloc[-1]['BUCKET']:
        n_nan_rows = 1

    n_not_nan -= n_nan_rows
    n_obs = df_monoton['COUNT'].sum()
    p_values = dict()
    for i in range(n_not_nan - 1):
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
    buckets_min_p = min(p_values, key=p_values.get)  # index of rows with min p_value
    min_p = min(p_values.values())

    return min_p, buckets_min_p


def monotone_optimal_binning(X, Y,
                             min_bin_size=0.05, min_bin_rate=0.01,
                             min_p_val=0.95, max_bins=20, min_bins=3):
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

    # Make df_stat Monotonic
    df_monotonic = make_monotonic(df_stat)
    n_bins = len(df_monotonic)
    # Merging bins
    while n_bins > min_bins:
        min_p, buckets_min_p = compute_min_p_values(df_monotonic,
                                                    min_size=min_bin_size,
                                                    min_rate=min_bin_rate)

        n_bins = len(df_monotonic)
        if min_p < min_p_val:
            indx_bot, indx_top = buckets_min_p
            df_monotonic = merge_rows(df_monotonic, indx_bot, indx_top)
        else:
            break

    # Open boundaries of first and last not None buckets
    bucket_intervals = df_monotonic['BUCKET'].values
    bucket_intervals[0] = pd.Interval(left=-1e9, right=bucket_intervals[0].right)
    try:
        bucket_intervals[-1] = pd.Interval(left=bucket_intervals[-1].left, right=1e9)
    except AttributeError:
        # if bucket in last row is Nan -> open boundaries in previous row
        bucket_intervals[-2] = pd.Interval(left=bucket_intervals[-2].left, right=1e9)

    df_monotonic['BUCKET'] = bucket_intervals

    return df_monotonic


def create_bins_df(raw_data):
    """ Create data frame with stats of all features in input data."""

    # Init data and target
    data = raw_data.copy()
    Y = (data['y'] == 'yes').astype(int).values
    data.drop(columns='y', inplace=True)

    # Init final df
    full_stats_df = pd.DataFrame()

    for column in data:
        X = data[column].values
        # Check type of feature
        if is_numeric(X):
            # If numeric feature - use algorithm of optimal binning
            X = X.astype('float32')
            curr_stats_df = monotone_optimal_binning(X, Y)
        else:
            # If categorical feature - just compute stats
            curr_stats_df = create_stats(X, Y, feature_type='categorical')

        curr_stats_df['VAR_NAME'] = column

        # Add to final df
        full_stats_df = pd.concat([full_stats_df, curr_stats_df], ignore_index=True, sort=False)

    # Select information values from df
    iv_df = pd.DataFrame({'IV': full_stats_df.groupby('VAR_NAME')['IV'].max()}).reset_index()

    return full_stats_df, iv_df


def cut_off_iv(full_stats_df, iv_df, cut_off=0.00):
    """
    Cut data with IV less than cut_off.
    Arguments:
        full_stats_df: pd.DataFrame with stats of all features
        iv_df: pd.DataFrame with IV of all features
        cut_off: threshold to cut
    Returns:
        cut_stats_df: final data_frame
        not_use_names: not used VAR_NAME
    """

    not_use_names = iv_df['VAR_NAME'][iv_df['IV'] < cut_off].values
    use_name = iv_df['VAR_NAME'][iv_df['IV'] >= cut_off].values
    use_idx = [name in use_name for name in full_stats_df['VAR_NAME']]
    cut_stats_df = full_stats_df.iloc[use_idx]
    return cut_stats_df, use_name


def replace_by_woe_optimal(raw_data, cut_stats_df, use_name, fill_na=-999999.0):
    """
    Replace features on WOE.
    Arguments:
        raw_data: pd.DataFrame, origin features
        cut_stats_df: full statistics reduced by IV
        use_name: name of columns which are used
        fill_na: value of WOE for missed feature (not in any bucket)
    Returns:
        data_with_woe: pd.DataFrame, features in WOE representation
    """

    data = raw_data.copy()
    data_with_woe = pd.DataFrame()
    for column in use_name:
        X = data[column].values
        X_woe = np.empty_like(X, dtype=float)
        # Choose df for current feature
        current_stats_df = cut_stats_df[cut_stats_df['VAR_NAME'] == column].reset_index(drop=True)
        buckets = current_stats_df['BUCKET'].iloc[:-1]
        woe_buckets = current_stats_df['WOE'].iloc[:-1]

        bucket_nan = current_stats_df['BUCKET'].iloc[-1]
        woe_bucket_nan = current_stats_df['WOE'].iloc[-1]

        # Replace NaNs on WoE
        mask_miss = pd.isnull(X)
        X_woe[mask_miss] = woe_bucket_nan

        # Choose not NaNs
        X_true = X[np.invert(mask_miss)]
        X_true_woe = np.empty_like(X_true, dtype=float)
        if is_numeric(X_true):
            # Sort values of feature
            sorted_idx = np.argsort(X_true)
            X_sorted = X_true[sorted_idx]

            # Init last index of Sorted X that belongs to previous bucket
            prev_idx_from_bucket = 0
            for bucket, bucket_woe in zip(buckets, woe_buckets):
                # Find last index in sorted X that belongs current bucket
                right = bucket.right + 1e-6
                last_idx_from_bucket = binary_search(X_sorted, low=0, high=len(X_sorted)-1, x=right)

                # Find indexes of X that belongs current bucket and replace on WOE
                curr_bucket_X_idx = sorted_idx[prev_idx_from_bucket: last_idx_from_bucket]
                X_true_woe[curr_bucket_X_idx] = bucket_woe

                # New last index of Sorted X that belongs to previous bucket
                prev_idx_from_bucket = last_idx_from_bucket
        else:
            # Use dict for categories
            bucket_woe_dict = {bucket: woe for bucket, woe in zip(buckets, woe_buckets)}
            for i in range(len(X_true)):
                category_i = X_true[i]

                if category_i in bucket_woe_dict:
                    X_true_woe[i] = bucket_woe_dict[category_i]
                else:
                    X_true_woe[i] = fill_na

        # Add to final df
        X_woe[np.invert(mask_miss)] = X_true_woe
        data_with_woe['WOE_' + column] = X_woe

    data_with_woe['target'] = (data['y'] == 'yes').astype(int).values
    return data_with_woe


def replace_by_woe_naive(raw_data, cut_stats_df, use_name, fill_na=-777777.0):
    """
    Replace features on WOE.
    Arguments:
        raw_data: pd.DataFrame, origin features
        cut_stats_df: full statistics reduced by IV
        use_name: name of columns which are used
        fill_na: value of WOE for missed feature (not in any bucket)
    Returns:
        data_with_woe: pd.DataFrame, features in WOE representation
    """

    data = raw_data.copy()
    data_with_woe = pd.DataFrame()
    for column in use_name:
        X = data[column].values
        X_woe = np.empty_like(X, dtype=float)
        # Choose df for current feature
        current_stats_df = cut_stats_df[cut_stats_df['VAR_NAME'] == column].reset_index(drop=True)
        for i_x, x in enumerate(X):
            # Start from check on NULL
            if x != x:
                X_woe[i_x] = current_stats_df['WOE'].iloc[-1]
                continue

            # Iteration across all buckets
            is_replaced = False
            for i_b, bucket in enumerate(current_stats_df['BUCKET'][:-1]):
                if x in bucket:
                    X_woe[i_x] = current_stats_df['WOE'].iloc[i_b]
                    is_replaced = True
                    break
            if not is_replaced:
                X_woe[i_x] = fill_na

        data_with_woe['WOE_' + column] = X_woe

    data_with_woe['target'] = (data['y'] == 'yes').astype(int).values

    return data_with_woe


def delete_correlated_features(df_woe, iv_df,
                               method='cramer',
                               cut_off=0.75,
                               inplace=True,
                               sort_iv=True):
    """
    Drop columns with big correlation and small Information Value.
    Args:
        df_woe: pd.DataFrame
        iv_df: pd.DataFrame with Information Value of each feature
        method: method for computing correlation {'cramer', 'pearson', 'spearman'}
        cut_off: float, threshold to drop
        inplace: not compute correlation if feature dropped already
        sort_iv: sort features by IV before delete correlations
    Returns:
        df_uncorr: pd.DataFrame without correlated features
    """

    iv_values = {'WOE_' + var_name: iv for var_name, iv in zip(iv_df['VAR_NAME'], iv_df['IV'])}
    df_features_woe = df_woe.drop(columns='target')
    # Sort by IV
    if sort_iv:
        sorted_columns = [(column, iv_values[column]) for column in df_features_woe.columns]
        sorted_columns = sorted(sorted_columns, key=lambda x: x[1], reverse=True)
        sorted_columns = [x[0] for x in sorted_columns]
        df_features_woe = df_features_woe[sorted_columns]

    n_features = df_features_woe.shape[1]
    to_drop = []
    corr_matrix = pd.DataFrame(columns=df_features_woe.columns, index=df_features_woe.columns)

    # Compute continuous correlation
    if method in ['pearson', 'spearman']:
        # Compute correlation matrix
        corr_matrix = df_features_woe.corr(method).abs()
        for i_row in range(n_features - 1):
            for j_col in range(i_row + 1, n_features):
                # Skip if some feature already dropped
                feature_name_i = corr_matrix.index[i_row]
                feature_name_j = corr_matrix.columns[j_col]
                if inplace:
                    if feature_name_i in to_drop:
                        break
                    if feature_name_j in to_drop:
                        continue

                # Correlation between feature i, j
                corr_ij = corr_matrix.iloc[i_row, j_col]
                # Deletion
                if corr_ij > cut_off:
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
                # Skip if some feature already dropped
                feature_name_i = df_features_woe.iloc[:, i].name
                feature_name_j = df_features_woe.iloc[:, j].name
                if inplace:
                    if feature_name_i in to_drop:
                        break
                    if feature_name_j in to_drop:
                        continue

                woe_i = df_features_woe.iloc[:, i].values
                woe_j = df_features_woe.iloc[:, j].values

                # Calculate confusion matrix
                confusion_matrix = pd.crosstab(woe_i, woe_j)
                # Calculate Cramers Coefficient
                k1, k2 = confusion_matrix.shape
                n = np.sum(confusion_matrix.values)
                chi2_stat = chi2_contingency(confusion_matrix)[0]
                phi_cram = np.sqrt(chi2_stat/(n*min(k1, k2) - 1))

                # Add to correlation matrix
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

    df_uncorr = df_woe.drop(columns=to_drop)
    return df_uncorr, corr_matrix, to_drop


def start_pipeline(raw_data, woe_replace='optimal'):
    """
    Prepossess data to binning representation.
    Arguments:
        raw_data: pd.DataFrame with numeric and categorical features.
        woe_replace: implementation of algorithm {'optimal', 'naive', 'compare'}
    Returns:
        df_woe_uncorr: Data with WOE as features
        corr_matrix: correlation matrix
        to_drop: columns that dropped from data
        iv_values: information values of features
        full_stats: full statistics of binning
        """
    data = raw_data.copy()
    t1 = time.time()
    full_stats, iv_values = create_bins_df(data)
    t2 = time.time()
    full_stats_cut, use_name_iv = cut_off_iv(full_stats, iv_values)

    if woe_replace == 'compare':
        t3 = time.time()
        data_woe_naive = replace_by_woe_naive(data, full_stats_cut, use_name_iv)
        t4 = time.time()
        data_woe_opt = replace_by_woe_optimal(data, full_stats_cut, use_name_iv)
        t5 = time.time()
        data_woe = data_woe_naive

        print('Replace on WOE naive = {} seconds'.format(round(t4 - t3, 3)))
        print('Replace on WOE optimal = {} seconds'.format(round(t5 - t4, 3)))
        if data_woe_opt.equals(data_woe_naive):
            print('Optimal and Naive Implementation of WOE replacing is EQUAL')
        else:
            print('ERROR.\n Optimal and Naive Implementation of WOE replacing is DIFFERENT')
            print(data_woe_opt)
            print(data_woe_naive)
    elif woe_replace == 'optimal':
        t3 = time.time()
        data_woe = replace_by_woe_optimal(data, full_stats_cut, use_name_iv)
        t4 = time.time()
        print('Replace on WOE optimal = {} seconds'.format(round(t4 - t3, 3)))
    else:
        t3 = time.time()
        data_woe = replace_by_woe_naive(data, full_stats_cut, use_name_iv)
        t4 = time.time()
        print('Replace on WOE naive = {} seconds'.format(round(t4 - t3, 3)))

    t5 = time.time()
    df_woe_uncorr, corr_matrix, to_drop = delete_correlated_features(data_woe,
                                                                     iv_values,
                                                                     inplace=True)
    t6 = time.time()

    print('Creating full stats = {} seconds'.format(round(t2 - t1, 3)))
    print('Cutting of by IV = {} seconds'.format(round(t3 - t2, 3)))
    print('Delete correlations = {} seconds'.format(round(t6 - t5, 3)))

    return df_woe_uncorr, corr_matrix, to_drop, iv_values, full_stats
