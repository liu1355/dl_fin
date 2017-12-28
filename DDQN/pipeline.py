# Tradition Value Factor, Fundamental Factor Models, Other Alpha Factors, FRED Economic Data
# Note: Plotting not yet included.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
from statsmodels import regression
import scipy
import datetime

from quantopian.pipeline import Pipeline
from quantopian.research import run_pipeline, get_pricing

from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.factors import CustomFactor, Returns, Latest
from quantopian.pipeline.filters import Q1500US

class SPY_proxy(CustomFactor):
    """Synthetic S&P500 proxy"""

    inputs = [Fundamentals.market_cap]
    window_length = 1

    def compute(self, today, assets, out, mc):
        out[:] = mc[-1]

class Div_Yield(CustomFactor):
    """Div Yield = Annual Div per Share / Price per Share"""

    inputs = [Fundamentals.div_yield5_year]
    window_length = 1
    def compute(self, today, assets, out, d_y):
        out[:] = d_y[-1]

class Price_to_Book(CustomFactor):
    """PB Ratio = Price per Share / NAV per Share"""

    inputs = [Fundamentals.pb_ratio]
    window_length = 1

    def compute(self, today, assets, out, pbr):
        out[:] = pbr[-1]


class Price_to_TTM_Sales(CustomFactor):
    """P12S Ratio = Price per Share / TTM Sales"""

    inputs = [Fundamentals.ps_ratio]
    window_length = 1

    def compute(self, today, assets, out, ps):
        out[:] = -ps[-1]


class Price_to_TTM_Cashflows(CustomFactor):
    """P12CF Ratio = Price per Share / TTM Cashflow"""

    inputs = [Fundamentals.pcf_ratio]
    window_length = 1

    def compute(self, today, assets, out, pcf):
        out[:] = -pcf[-1]

class Momentum(CustomFactor):
    """Equity Momentum"""

    inputs = [Returns(window_length=20)]
    window_length = 20

    def compute(self, today, assets, out, lag_returns):
        out[:] = lag_returns[0]

class Returns(CustomFactor):
    """Equity Momentum"""

    inputs = [Returns(window_length=20)]
    window_length = 2

    def compute(self, today, assets, out, ret):
        out[:] = ret[0]

# DEFINE START AND END DATE
s_date = '{}'.format('yyyy-mm-dd')
e_date = '{}'.format('yyyy-mm-dd')

# Limit effect of outliers
# Used to vectorize dataframes to standardized columns
def filter_fn(x):
    if x <= -10:
        x = -10.0
    elif x >= 10:
        x = 10.0
    return x

# Standardize using mean and standard deviation of S& related dataframes
def standard_frame_compute(self):

    # Remove infinite values
    self.df.replace([np.inf, -np.inf], np.nan)
    self.df.dropna()

    # Standardize parameters of S&P500
    df_SPY = self.df.sort(columns = 'SPY Proxy', ascending = False)

    # Create separate dataframe for SPY for storing standardized values
    df_SPY = df_SPY.head(500)

    # Get dataframes into Numpy array
    df_SPY = df_SPY.as_matrix()

    # Store index values
    index = self.df.index.values
    self.df.as_matrix()

    df_standard = np.empty(self.df.shape[0])

    for col_SPY, col_full in zip(df_SPY.T, self.df.T):

        # Summary stats for S&P500
        mu = np.mean(col_SPY)
        sigma = np.std(col_SPY)
        col_standard = np.array(((col_full - mu) / sigma))

        # Create vectorized function (lambda equivalent)
        fltr = np.vectorize(filter_fn)
        col_standard = (fltr(col_standard))

        # Make range between -10 and 10
        col_standard = (col_standard / self.df.shape[1])

        # Attach calculated values as new row in df_standard
        df_standard = np.vstack((df_standard, col_standard))

        # Get rid of first entry (empty scores)
    df_standard = np.delete(df_standard, 0, 0)

    return (df_standard, index)

# Data pull TVF
def Data_Pull_TVF():

    Data_Pipe_TVF = Pipeline()
    Data_Pipe_TVF.add(SPY_proxy(), 'SPY Proxy')
    Data_Pipe_TVF.add(Div_Yield(), 'Dividend Yield')
    Data_Pipe_TVF.add(Price_to_Book(), 'Price to Book')
    Data_Pipe_TVF.add(Price_to_TTM_Sales(), 'Price / TTM Sales')
    Data_Pipe_TVF.add(Price_to_TTM_Cashflows(), 'Price / TTM Cashflow')

    return Data_Pipe_TVF

class TVF:
    """Traditional Value Factor
    Function 1: init
    Function 2: Dataframe Standardization
    Function 3: Sorting
    Function 4: Score Results"""

    def __init__(self, df, index, data_pull_results):
        self.df = df
        self.index = index
        self.data_pull_results = run_pipeline(Data_Pull_TVF(), start_date = s_date, end_date = e_date)

    # Sum up and sort data
    def composite_score(self):

        # Sum up transformed data
        df_composite = self.df.sum(axis = 0)

        # Put into a Pandas dataframe and connect number to equities via reindexing
        df_composite = pd.Series(data = df_composite, index = self.index)

        # Sort descending
        df_composite.sort(ascending = False)

        return df_composite

    def results(self, standard_frame_compute, composite_score):
        # Compute the standardized values
        results_standard, index = standard_frame_compute(self.data_pull_results)

        # Aggregate the scores
        ranked_scores = composite_score(results_standard, index)

        print(ranked_scores)

        # Make scores into list for ease of manipulation
        ranked_scores_list = ranked_scores.tolist()

# Data pull FFM
def Data_Pull_FFM():

    Data_Pipe_FFM = Pipeline()
    Data_Pipe_FFM.add(Momentum(), 'Momentum')
    Data_Pipe_FFM.add(SPY_proxy(), 'Market Cap')
    Data_Pipe_FFM.add(SPY_proxy.rank(mask = Q1500US()), 'Market Cap Rank')
    Data_Pipe_FFM.add(SPY_proxy.rank.top(1000), 'Biggest')
    Data_Pipe_FFM.add(SPY_proxy.rank.bottom(1000), 'Smallest')
    Data_Pipe_FFM.add(Price_to_Book(), 'Price to Book')
    Data_Pipe_FFM.add(Price_to_Book.rank.top(1000), 'Low BP')
    Data_Pipe_FFM.add(Price_to_Book.rank.bottom(1000), 'High BP')
    Data_Pipe_FFM.add(Returns(), 'Returns')
    Data_Pipe_FFM.add(Momentum.rank(mask=Q1500US()), 'Momentum Rank')
    Data_Pipe_FFM.add(Momentum.rank.top(1000), 'Top')
    Data_Pipe_FFM.add(Momentum.rank.bottom(1000), 'Bottom')

    Data_Pipe_FFM.add(Price_to_Book.rank(mask = Q1500US()), 'Price to Book Rank')

    return Data_Pipe_FFM

today = datetime.datetime.today()

class FFM:
    """Fundamental Factor Model
    Function 1: init
    Function 2: Factor Model group results
    Function 3: OLS Regression
    Function 4: Factor Value Normalization"""

    def __init__(self, df, index, data_pull_results):
        self.df = df
        self.index = index
        # DEFINE START AND END DATE
        self.data_pull_results = run_pipeline(Data_Pull_FFM(), start_date = s_date, end_date = e_date)

    def group_results(self):
        # Compute the standardized values
        results_standard = standard_frame_compute(self.data_pull_results)

        # Average return of each day for a particular group of stocks
        R_biggest = self.data_pull_results[self.data_pull_results.biggest]['Returns'].groupby(level = 0).mean()
        R_smallest = self.data_pull_results[self.data_pull_results.smallest]['Returns'].groupby(level = 0).mean()

        R_lowbp = self.data_pull_results[self.data_pull_results.lowbp]['Returns'].groupby(level = 0).mean()
        R_highbp = self.data_pull_results[self.data_pull_results.highbp]['Returns'].groupby(level = 0).mean()

        R_top = self.data_pull_results[self.data_pull_results.top]['Returns'].groupby(level = 0).mean()
        R_bottom = self.data_pull_results[self.data_pull_results.bottom]['Returns'].groupby(level = 0).mean()

        # Risk-free proxy
        R_F = get_pricing('BIL', fields = 'price', start_date = s_date, end_date = e_date).pct_change()[1:]

        M = get_pricing('SPY', start_date = s_date, end_date = e_date, fields='price').pct_change()[1:]

        EXMRKT = M - R_F
        SMB = R_smallest - R_biggest  # Small minus big
        HML = R_lowbp - R_highbp  # High minus low
        MOM = R_top - R_bottom  # Momentum

        data = self.data_pull_results[['Returns']].set_index(self.data_pull_results.index)
        asset_list_sizes = [group[1].size for group in data.groupby(level = 0)]

        # Spreading the factor portfolio data across all assets for each day
        SMB_column = [[SMB.loc[group[0]]] * size for group, size \
                      in zip(data.groupby(level=0), asset_list_sizes)]
        data['SMB'] = list(itertools.chain(*SMB_column))

        HML_column = [[HML.loc[group[0]]] * size for group, size \
                      in zip(data.groupby(level=0), asset_list_sizes)]
        data['HML'] = list(itertools.chain(*HML_column))

        MOM_column = [[MOM.loc[group[0]]] * size for group, size \
                      in zip(data.groupby(level=0), asset_list_sizes)]
        data['MOM'] = list(itertools.chain(*MOM_column))

        EXMRKT_column = [[EXMRKT.loc[group[0]]] * size if group[0] in EXMRKT.index else [None] * size \
                         for group, size in zip(data.groupby(level=0), asset_list_sizes)]

        data['EXMRKT'] = list(itertools.chain(*EXMRKT_column))

        data = sm.add_constant(data.dropna())

        # List of assets from pipeline
        assets = data.index.levels[1].unique()

        Y = [data.xs(asset, level=1)['Returns'] for asset in assets]
        X = [data.xs(asset, level=1)[['EXMRKT', 'SMB', 'HML', 'MOM', 'const']] for asset in assets]

    def regression(self, group_results):
        reg_results = [regression.linear_model.OLS(y, x).fit().params \
                       for y, x in zip(group_results.Y, group_results.X) if not (x.empty or y.empty)]
        indices = [asset for y, x, asset in zip(group_results.Y, group_results.X, group_results.assets)
                   if not (x.empty or y.empty)]

        betas = sm.add_constant(pd.DataFrame(reg_results, index = indices).drop('const', axis = 1))

        R = data['Returns'].mean(axis = 0, level = 1)

        risk_free_rate = np.mean(group_results.R_F)

        OLS_results = regression.linear_model.OLS(R - risk_free_rate, betas).fit()
        OLS_results.summary()

        # REMINDER: Consider portfolio hedging check for environment module

    # Non-essential
    def factor_norm(self):
        PB = scipy.stats.mstats.winsorize(self.data_pull_results['book_to_price'][today], limits = 0.01)
        PB_z = (PB - PB.mean()) / PB.std()
        PB_z.dropna(inplace = True)

        MC = scipy.stats.mstats.winsorize(self.data_pull_results['market_cap'][date], limits=0.01)
        MC_z = (MC - np.mean(MC)) / np.std(MC)
        MC_z.dropna(inplace = True)

        Lag_Ret = scipy.stats.mstats.winsorize(self.data_pull_results['momentum'][date], limits=0.01)
        Lag_Ret_z = (Lag_Ret - np.mean(Lag_Ret)) / np.std(Lag_Ret)
        Lag_Ret_z.dropna(inplace = True

        returns = self.data_pull_results['Returns'][today]

        df_day = pd.DataFrame({'R': returns,
                               'BTP_z': BTP_z,
                               'MC_z': MC_z,
                               'Lag_Ret_z': Lag_Ret_z,
                               'Constant': 1}).dropna()

class AlphaFactors:
    """WIP: Other alpha factors, e.g. AlphaLens"""
    pass

class EconomicData:
    """WIP: Merge date not yet optimized"""
    FRED_df = pd.read_csv('path/merge.csv')

