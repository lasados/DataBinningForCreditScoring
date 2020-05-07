import pandas as pd
import pandas.core.algorithms as algos
import numpy as np
from scipy.stats import stats

# To do
# 1) Boundary cases
#
#

def optimal_bins(notmiss_X, notmiss_Y, max_bins, min_bins):
    """
    Maximize Spearman correlation and find optimal number of bins.
    Arguments:
        notmiss_X: array or pd.DataFrame with not null values
        notmiss_Y: array or pd.DataFrame with not null values
        max_bins: int, maximum number of bins for split
        min_bins: int, minimal number of bins for split
    Returns:
        n_optimal: optimal number of bins

    """
    r_corr_max = 0.0
    r_corr = 0.0
    n_bins = max_bins
    n_optimal = max_bins
    while n_bins >= min_bins:
        df_bucket = pd.DataFrame({"X": notmiss_X,
                                  "Y": notmiss_Y,
                                  "Bucket": pd.qcut(notmiss_X, n_bins)})

        # Grouping pairs (x, y) by "Bucket of x"
        group_bucket = df_bucket.groupby('Bucket', as_index=True)

        # Compute mean_X, mean_Y for each "Bucket"
        mean_x_in_buckets = group_bucket.mean()['X']
        mean_y_in_buckets = group_bucket.mean()['Y']

        # Compute correlation between (mean_x, mean_y)
        r_corr, p_val = stats.spearmanr(mean_x_in_buckets, mean_y_in_buckets)

        # Update optimal parameters
        if abs(r_corr) > abs(r_corr_max):
            r_corr_max = r_corr
            n_optimal = n_bins

        n_bins -= 1
    print('Correlation = {}, n_optimal = {}'.format(r_corr_max, n_optimal))
    return n_optimal


class NumericBinnner:
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
        n_optimal = optimal_bins(df_notmiss['X'], df_notmiss['Y'], max_bins, min_bins)

        # For each pair x, y -> find bucket
        df_bucket = pd.DataFrame({"X": df_notmiss['X'],
                                  "Y": df_notmiss['Y'],
                                  "Bucket": pd.qcut(df_notmiss['X'], n_optimal)})

        # Grouping pairs (x, y) by "Bucket of x"
        group_bucket = df_bucket.groupby('Bucket', as_index=True)

        # Case of 1 bin
        # If minimum number of bins
        # if len(group_bucket) == 1:
        #     n_bins = min_bins
        #     bins = algos.quantile(df_notmiss['X'], np.linspace(0, 1, n_bins))
        #     if len(np.unique(bins)) == 2:
        #         bins = np.insert(bins, 0, 1)
        #         bins[1] = bins[1] - (bins[1] / 2)
        #     d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y,
        #                        "Bucket": pd.cut(notmiss.X, np.unique(bins), include_lowest=True)})
        #     d2 = d1.groupby('Bucket', as_index=True)

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



