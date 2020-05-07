import pandas as pd
import numpy as np
from scipy.stats import stats

# To do
# 1) Boundary cases
#
#

class NumericBinnner:
    """  Class of numeric binarizer. Preprocessor of numeric data. """
    def __init__(self, criteria='spearman'):
        self.criteria = criteria


    def bin(self, X, Y, max_bin=20, min_bin=1):
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
        df_notmiss = df_init[['X','Y']][df_init["X"].notnull()]
        df_justmiss = df_init[['X', 'Y']][df_init["X"].isnull()]

        # Init correlation
        r_corr = 0.0
        n_bins = max_bin

        # For each pair x, y -> find bucket
        df_bucket = pd.DataFrame({"X": df_notmiss['X'],
                                  "Y": df_notmiss['Y'],
                                  "Bucket": pd.qcut(df_notmiss['X'], n_bins)})

        # Grouping pairs (x, y) by "Bucket of x"
        group_bucket = df_bucket.groupby('Bucket', as_index=True)

        # Compute mean_X, mean_Y for each "Bucket"
        mean_x_in_buckets = group_bucket.mean()['X']
        mean_y_in_buckets = group_bucket.mean()['Y']

        # Compute correlation between (mean_x, mean_y)
        r_corr, p_val = stats.spearmanr(mean_x_in_buckets, mean_y_in_buckets)
        print('Correlation = {}, p_value = {}'.format(r_corr, p_val))

        # Create DataFrame with statistical features of group_data by "Bucket"
        df_stat = pd.DataFrame({}, index=[])

        # Compute min, max value of X for each "Bucket"
        df_stat["MIN_VALUE"] = group_bucket.min()['X']
        df_stat["MAX_VALUE"] = group_bucket.max()['X']
        # Count number of points in "Bucket"
        df_stat["COUNT"] = group_bucket.count()['Y']
        # Count number of positive and negative points in "Bucket
        df_stat["EVENT"] = group_bucket.sum()['Y']
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
        df_stat = df_stat[['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'COUNT',
                           'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE',
                           'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV']]
        # Delete infinity
        df_stat = df_stat.replace([np.inf, -np.inf], 0)

        return df_stat

binner = NumericBinnner()
X = np.arange(0, 100)
Y = np.random.randint(0, 2, 100)

print(binner.bin(X, Y))


