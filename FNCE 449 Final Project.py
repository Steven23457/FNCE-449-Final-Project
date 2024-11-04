import pandas as pd
import numpy as np
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import efficient_frontier
import random
from tqdm import tqdm
import requests
import re
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
import matplotlib.ticker as mtick
import seaborn as sns
sns.set()

ticker_dict = {'XLC':'Communication',
               'XLY':'Cons. Discretionary',
               'XLP':'Cons. Staples',
               'XLE':'Energy',
               'XLF':'Financials',
               'XLV':'Health Care',
               'XLI':'Industrials',
               'XLB':'Materials',
               'XLRE':'Real Estate',
               'XLK':'Technology',
               'XLU':'Utilities',
               'SPY':'S&P 500'}

class RandomWindowSample:
    '''
    Class creates a random window sample from a pandas dataframe
    '''
    def __init__(self) -> None:
        pass
    def sample(self,X,min_length:int=0,max_length:int=None):
        
        n_samples = len(X)
       
        if max_length > n_samples or max_length is None:
            max_length = n_samples
        if min_length >= max_length:
            raise ValueError("min_length must be less than max_length.")
        
        sample_size = random.randrange(min_length,max_length)
        sample_start = random.randrange(0,n_samples - sample_size)

        return X[sample_start:sample_start + sample_size]
        
    
class BlockingTimeSeriesSplit():
    '''
    Class splits a datarame using a blocked time series split, with n splits and a specified train size 
    '''
    def __init__(self, n_splits,train_size):
        self.n_splits = n_splits
        self.train_size = train_size
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(self.train_size * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]


def make_api_request(ticker,av_apikey):
    
    def clean_stock_data(df,ticker):
        
        #--Keep close and date data and add ticker prefix
        df = df[['Date','Close']]
        df = df[::-1].reset_index(drop=True)
        df = df.apply(lambda col: pd.to_datetime(col) if col.name == 'Date' else pd.to_numeric(col))
        
        df= df.set_index('Date').add_prefix(f'{ticker} ',axis=1).reset_index()

        return df
        
    base_url = 'https://www.alphavantage.co/query'

    params = {'function':'TIME_SERIES_DAILY',
              'symbol':ticker,
              'outputsize':'full',
              'apikey':av_apikey}
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        res_data = response.json()
        #--Turn json response to dataframe, strip non letter characters and put it to title case, error handler if data is missing
        if 'Time Series (Daily)' in res_data:
            data = pd.DataFrame.from_dict(res_data['Time Series (Daily)'],orient='index')
            data.columns = [re.sub(r"[^A-Za-z]",'',c).title() for c in data.columns.to_list()]
        
            data = data.reset_index().rename(columns={'index':'Date'})
            final_data = clean_stock_data(data,ticker)
        
            return final_data
    
        else: 
            return None
        
    
    except requests.exceptions.RequestException as e:
            #--Request error handler
            print(f"Request Error {e}")


def get_data():
    av_apikey = "F1HX5A0Z0OOLBG59"
    stock_data = pd.DataFrame()

    for ticker in ticker_dict.keys():
        temp = make_api_request(ticker,av_apikey)
        if temp is not None:
            use_backup = False
            #--Use frist dataset as full dataframe
            if len(stock_data) == 0:
                stock_data = temp
            else:
                stock_data = pd.merge(stock_data,temp,on='Date',how='inner')
        else:
            #--Use backup dataset if api fails for any ticker or hits 25 per day request limit
            use_backup = True
            print(f"API Limit Reached, using backup dataset") 
            break   


    if use_backup == True:
        stock_data = pd.read_excel(r"https://github.com/Steven23457/FNCE-449-Final-Project/blob/main/FNCE%20449%20Final%20Project%20Data%20Backup.xlsx?raw=true")

    returns = stock_data.copy()
    returns = returns.set_index('Date').pct_change().dropna()
    returns.columns = returns.columns.str.replace(' Close','').map(ticker_dict)
    
    return returns


def random_risk_free(min_rate:int,max_rate:int):
    #--Random Risk Free Rate up to 3 decimal places
    return random.randrange(min_rate*10,max_rate*10)/1000


def optimze_portfolio(return_data,risk_free):
    #-- Supress non optimal solution warning, non-optimal solutions do not significantly skew the average result
    warnings.filterwarnings("ignore", message="Solution may be inaccurate", category=UserWarning)
    #--Mean variance portfolio optimization
    cov_matrix = risk_models.sample_cov(return_data,frequency=252,returns_data=True)
    expected_retruns = expected_returns.mean_historical_return(return_data,returns_data=True,frequency=252)
    optimizer = efficient_frontier.EfficientFrontier(expected_retruns,cov_matrix,weight_bounds=(0,0.20))
    
    try: 
        portfolio_weights = dict(optimizer.max_sharpe(risk_free_rate=risk_free))
        return portfolio_weights
    
    except:
        return None


def portfolio_simulation(sample_data, itterations):
    #--Portfolio optimization with random window sampling
    
    sampler = RandomWindowSample()
    all_weights = pd.DataFrame()
    data_len = len(sample_data)
    
    for _ in tqdm(range(itterations)):
        portfolio_sample = sampler.sample(sample_data,min_length=int(data_len/3),max_length=int(data_len/2))
        rf = random_risk_free(min_rate=1,max_rate=4)
        
        weights = optimze_portfolio(portfolio_sample,rf)
        weight_df = pd.DataFrame(data=weights,index=[0,])
        all_weights = pd.concat([all_weights,weight_df]).reset_index(drop=True)

    weights = pd.Series(all_weights.mean(axis=0).mask(all_weights.mean(axis=0) < 0,0))

    return weights


def cross_validate(market_return,sector_returns):
    #--Cross Validate Model

    sectors = sector_returns.columns.to_list()
    tss = BlockingTimeSeriesSplit(n_splits=3,train_size=0.75)
    itterations = 1000

    cv_results = pd.DataFrame()
    cv_weights = pd.DataFrame()
    folds = []
    
    for fold, (train_idx, test_idx) in enumerate(tss.split(sector_returns)):
        print(f"Optimizing Fold: {fold + 1}")
        train_data, test_data = sector_returns.iloc[train_idx, :], sector_returns.iloc[test_idx,:]
        market_return_fold = market_return.iloc[test_idx]
        #--Find portfolio weights for fold test dataset
        fold_weights = portfolio_simulation(train_data,itterations)
        if len(cv_weights) == 0:
            cv_weights = fold_weights
        else:
            cv_weights = pd.concat([fold_weights,cv_weights],axis=1)
        test_data = test_data.copy()

        #--Compute portfolio return on test dataset and merge with market return
        test_data.loc[:, 'Portfolio Return'] = test_data[sectors].dot(fold_weights)

        test_portfolio = pd.merge(test_data[['Portfolio Return']],market_return_fold,right_index=True,left_index=True)
       
        test_portfolio = test_portfolio.reset_index()
        test_portfolio['Fold'] = fold + 1
        cv_results = pd.concat([cv_results,test_portfolio],axis=0)
        folds.append(fold+1)
    
    cv_weights.columns = folds
    cv_weights.index.name = "Folds"

    return cv_results, cv_weights


def evaluate_portfolio(market_return, portfolio_return):

   #--OLS regression to calculate alpha and beta
    x2,y = market_return, portfolio_return
    x = sm.add_constant(x2)
    results = sm.OLS(y,x).fit()

    #--Compute significance of parameters
    alpha, beta = results.params[0], results.params[1]
    r_squared = results.rsquared
    beta_sig = results.t_test("S&P 500 = 1").pvalue
    alpha_sig = results.pvalues[0]

    #--Return, volatility, and sharpe calculations (2% rf assumption)
    market_annual_return = ((1 + market_return.mean())**252 - 1)
    market_volatility = market_return.std() * np.sqrt(252)
    
    portfolio_annual_return = ((1 + portfolio_return.mean())**252 - 1)
    portfolio_volatility = portfolio_return.std() * np.sqrt(252)

    portfolio_sharpe = (portfolio_annual_return - 0.02)/portfolio_volatility
    market_sharpe = (market_annual_return - 0.02) /market_volatility

    metrics = {
    'Alpha': alpha,
    'Alpha P-Value': alpha_sig,
    'Beta': beta,
    'Beta P-Value (Null = 1)': beta_sig,
    'R-Squared Score':r_squared,
    'Market Annual Return': market_annual_return,
    'Market Volatility': market_volatility,
    'Market Sharpe Ratio':market_sharpe,
    'Portfolio Annual Return': portfolio_annual_return,
    'Portfolio Volatility': portfolio_volatility,
    'Portfolio Sharpe Ratio':portfolio_sharpe
    }
    
    portfolio_results = pd.DataFrame([metrics])

    return portfolio_results


def plot_results(plot_df,fold):
    
    #--Plot cumulative return of the portfolio and the S&P 500
    plot_df = plot_df.copy()
    plot_df['Market Cumulative'] = (1 + plot_df['S&P 500']).cumprod()
    plot_df['Portfolio Cumulative'] = (1 + plot_df['Portfolio Return']).cumprod()
    
    plt.plot(plot_df['Date'],plot_df['Market Cumulative'],label='S&P 500')
    plt.plot(plot_df['Date'],plot_df['Portfolio Cumulative'],label='Portfolio')
    plt.title(f'Fold {fold} Results')
    plt.legend(loc='best')
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1,decimals=0, symbol='%'))
    plt.show()

def main():
    
    returns = get_data()
    
    market_return = returns['S&P 500'].copy()
    sector_returns = returns.drop('S&P 500',axis=1).copy()

    cv_results,cv_weights = cross_validate(market_return,sector_returns)
    # cv_weights.to_excel(r"C:\Users\steve\Documents\!Fall 2024 Classes\FNCE 449\Final Project CV Weights.xlsx",index=True)
    print(cv_weights)

    fold_count = cv_results['Fold'].iloc[-1]
    cv_metrics = pd.DataFrame()

    for fold in range(1,fold_count+1):
        
        fold_df = cv_results[cv_results['Fold'] == fold]
        portfolio_metrics = evaluate_portfolio(fold_df['S&P 500'], fold_df['Portfolio Return'])
        plot_results(fold_df, fold)
        portfolio_metrics['Fold'] = fold
        cv_metrics = pd.concat([cv_metrics,portfolio_metrics],axis=0)

    print(cv_metrics)
    # cv_metrics.to_excel(r"C:\Users\steve\Documents\!Fall 2024 Classes\FNCE 449\Final Project Summary Statistics.xlsx",index=False)


if __name__ == "__main__":
    main()