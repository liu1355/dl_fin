import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quantopian.pipeline import Pipeline
from quantopian.research import run_pipeline
from quantopian.pipeline.data.builtin import USEquityPricing

from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.factors import CustomFactor, Returns, Latest
from quantopian.pipeline.classifiers import Classifier
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

# Limit effect of outliers
# Used to vectorize dataframes to standardized columns
def filter_fn(x):
    if x <= -10:
        x = -10.0
    elif x >= 10:
        x = 10.0
    return x

# Data pull
def Data_Pull_TVF():

    Data_Pipe = Pipeline()
    Data_Pipe.add(SPY_proxy(), 'SPY Proxy')
    Data_Pipe.add(Div_Yield(), 'Dividend Yield')
    Data_Pipe.add(Price_to_Book(), 'Price to Book')
    Data_Pipe.add(Price_to_TTM_Sales(), 'Price / TTM Sales')
    Data_Pipe.add(Price_to_TTM_Cashflows(), 'Price / TTM Cashflow')

    return Data_Pipe

class TVF:
    """Traditional Value Factor
    Function 1: init
    Function 2: Dataframe Standardization
    Function 3: Sorting
    Function 4: Score Results"""

    def __init__(self, df, index, data_pull_results):
        self.df = df
        self.index = index
        # DEFINE START AND END DATE
        self.data_pull_results = run_pipeline(Data_Pull_TVF(), start_date='yyyy-mm-dd', end_date='yyyy-mm-dd')

    # Standardize using mean and standard deviation of S&P500
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