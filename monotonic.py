import pandas as pd
import numpy as np

def create_stats(group_data):
    """
    Create DataFrame with statistical features of group_data by "Bucket.
    group_data: pandas.GroupBy object, groups - Buckets
    """

    df_stat = pd.DataFrame({}, index=[])
    df_stat['BUCKET'] = group_data.groups.keys()

    # Compute min, max value of X for each "Bucket"
    df_stat["MIN_VALUE"] = group_data.min()['X']
    df_stat["MAX_VALUE"] = group_data.max()['X']
    # Count number of points in "Bucket"
    df_stat["COUNT"] = group_data.count()['Y']
    # Count number of positive and negative points in "Bucket
    df_stat["EVENT"] = group_data.sum()['Y']
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


def make_monotonic(statistical_data):
    df_monoton = statistical_data.copy()
    top_indx = df_monoton.index[-1]
    bot_indx = top_indx - 1

    while bot_indx >= 0:
        bot_row = df_monoton.iloc[bot_indx]  # row with smaller index
        top_row = df_monoton.iloc[top_indx]  # row with higher index

        # If WOE of top_row larger -> merge to save monotonic
        if top_row['WOE'] > bot_row['WOE']:
            # Compute new boundaries of merged "Bucket"
            left_bnd = bot_row['BUCKET'].left
            right_bnd = top_row['BUCKET'].right
            # Merge top_row to bot_row
            merged_row = bot_row.copy()
            merged_row['BUCKET'] = pd.Interval(left=left_bnd, right=right_bnd)
            merged_row['MIN_VALUE'] = min(bot_row['MIN_VALUE'], top_row['MIN_VALUE'])
            merged_row['MAX_VALUE'] = max(bot_row['MAX_VALUE'], top_row['MAX_VALUE'])
            merged_row['COUNT'] = bot_row['COUNT'] + top_row['COUNT']
            merged_row['EVENT'] = bot_row['EVENT'] + top_row['EVENT']
            merged_row['EVENT_RATE'] = merged_row['EVENT'] / merged_row['COUNT']
            merged_row['NONEVENT'] = bot_row['NONEVENT'] + top_row['NONEVENT']
            merged_row['NON_EVENT_RATE'] = merged_row['NONEVENT'] / merged_row['NONEVENT']
            merged_row["DIST_EVENT"] = merged_row['EVENT'] / df_monoton.sum()['EVENT']
            merged_row["DIST_NON_EVENT"] = merged_row['NONEVENT'] / df_monoton.sum()['NONEVENT']
            merged_row["WOE"] = np.log(merged_row['DIST_EVENT'] / merged_row['DIST_NON_EVENT'])
            merged_row["IV"] = (merged_row['DIST_EVENT'] - merged_row['DIST_NON_EVENT']) * merged_row["WOE"]
            merged_row["IV"] = df_monoton["IV"].sum()

            # Place to table
            df_monoton.iloc[bot_indx] = merged_row

            # Drop top_row
            df_monoton.drop(top_indx, axis=0, inplace=True)
            df_monoton.reset_index(inplace=True, drop=True)

            # Reset top index
            top_indx = df_monoton.index[-1]
            bot_indx = top_indx - 1
        else:
            top_indx -= 1
            bot_indx -= 1

    return df_monoton


def compute_p_values(monotonic_df):
    """
    Computes p-value for each pair of bins.
    Arguments:
        monotonic_df: pd.DataFrame with statistical features,  WOE monotonously decreasing
    Returns:


    """
    df_monoton = monotonic_df.copy()

    n = len(df_monoton)
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

        # print('top_row\n', top_row)
        # print('bot_row\n', bot_row)
        # print('A = ', A)
        # print('E = ', E)
        chi_2 = np.sum(np.power((A - E), 2)/E)
        print(chi_2, i)
        input()


class MonotoneOptBin:
    """  Class of numeric binarizer. Preprocessor of numeric data. """
    def __init__(self, criteria='spearman'):
        self.criteria = criteria

    def bin(self, X, Y, max_bins=20, min_bins=1):
        """
        Split array X on 'Buckets', and compute statistical features for each 'Bucket'
        Binarize X to groups.
        Arguments:
            X: data for binning, np.array
            Y: target, np.array
        Returns:
            df_stat: pd.DataFrame with statistical features such as WOE, IV
                columns = ['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'COUNT',
                           'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE',
                           'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV']

        """

        # Transfer np.arrays to pd.DataFrame
        df_init = pd.DataFrame({"X": X, "Y": Y})

        # Split on notmiss and justmiss DataFrames
        df_notmiss = df_init[['X', 'Y']][df_init["X"].notnull()]
        df_justmiss = df_init[['X', 'Y']][df_init["X"].isnull()]

        # Compute optimal number of bins

        # For each pair x, y -> find bucket
        df_bucket = pd.DataFrame({"X": df_notmiss['X'],
                                  "Y": df_notmiss['Y'],
                                  "Bucket": pd.qcut(df_notmiss['X'], max_bins)})

        # Grouping pairs (x, y) by "Bucket of x"
        group_bucket = df_bucket.groupby('Bucket', as_index=True)
        df_stat = create_stats(group_bucket)

        # Make df_stat Monotonic
        # print('BEFORE MONOTONIC\n', df_stat)
        df_monoton = make_monotonic(df_stat)
        # print('MONOTONIC\n', df_monoton)

        return df_monoton

binner = MonotoneOptBin()
X = np.arange(0, 100)
Y = np.random.randint(0, 2, 100)

df_monoton = binner.bin(X, Y)
compute_p_values(df_monoton)