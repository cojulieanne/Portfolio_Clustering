import pandas as pd
import numpy as np
import yfinance as yf


class Portfolio:

    def __init__(self, start_date, end_date,  tickers = None, file = 'data/sp500_data.csv'):
        self.prices = self.fetch_stocks(tickers, start_date, end_date, file)
        self.start_date = min(self.prices.index)
        self.end_date = max(self.prices.index)
        self.daily_returns = self.prices.dropna(axis = 1).pct_change().dropna()
        self.daily_log_returns = np.log(1+self.daily_returns)
        self.get_indiv_metrics()
        

    def fetch_stocks(self, tickers, start_date, end_date, file):
        data = pd.read_csv(file).set_index('Date')
        
        if not tickers:
            tickers = data.columns
        
        data = data[tickers].loc[start_date:end_date]
        
        return data


    def get_portf_return(self, buy_date = None, sell_date = None, tick = None):
        
        if not buy_date:
            buy_date = self.start_date

        if not sell_date:
            sell_date = self.end_date
        
        if not tick:
            tick = self.prices.columns
            
        tick = [t for t in tick if t in self.prices.columns]
        dates = self.prices.index
        
        if buy_date not in dates:
            buy_date = min(dates[dates >= buy_date])
        if sell_date not in dates:
            sell_date = max(dates[dates <= sell_date])

        prices = self.prices[tick].loc[[buy_date, sell_date]].dropna(axis = 1).sum(axis = 1)
        
        returns = (prices.loc[sell_date] - prices.loc[buy_date])/prices.loc[buy_date]
            
        return returns

    def get_portf_sharpe(self, buy_date = None, sell_date = None, tick = None, time = 'daily'):

        if not buy_date:
            buy_date = self.start_date

        if not sell_date:
            sell_date = self.end_date
        
        if not tick:
            tick = self.prices.columns

        tick = [t for t in tick if t in self.prices.columns]
        prices = self.prices[tick].loc[buy_date:sell_date].dropna(axis = 1)

        if time == 'daily':
            agg_price = prices.dropna(axis = 1).sum(axis = 1)
            returns = agg_price.pct_change()
            log_returns = np.log(1 + returns)
            return log_returns.mean()/log_returns.std()

    def get_portf_volatility(self, buy_date = None, sell_date = None, tick = None, time = 'daily'):

        if not buy_date:
            buy_date = self.start_date

        if not sell_date:
            sell_date = self.end_date
        
        if not tick:
            tick = self.prices.columns

        tick = [t for t in tick if t in self.prices.columns]
        prices = self.prices[tick].loc[buy_date:sell_date].dropna(axis = 1)

        if time == 'daily':
            agg_price = prices.dropna(axis = 1).sum(axis = 1)
            returns = agg_price.pct_change()
            log_returns = np.log(1 + returns)
            return log_returns.std()

    def get_portf_sortino(self, buy_date = None, sell_date = None, tick = None, time = 'daily'):
        
        if not buy_date:
            buy_date = self.start_date

        if not sell_date:
            sell_date = self.end_date
        
        if not tick:
            tick = self.prices.columns
    
        tick = [t for t in tick if t in self.prices.columns]
        prices = self.prices[tick].loc[buy_date:sell_date].dropna(axis = 1)
    
        if time == 'daily':
            agg_price = prices.dropna(axis = 1).sum(axis = 1)
            returns = agg_price.pct_change().dropna()
            log_returns = np.log(1 + returns)
            neg_returns = log_returns[log_returns < 0]
            return log_returns.mean()/neg_returns.std()

    def compare_portf_returns(self, port_compare, buy_date = None, sell_date = None):

        if not buy_date:
            buy_date = self.start_date

        if not sell_date:
            sell_date = self.end_date

        if buy_date not in port_compare.prices.index or sell_date not in port_compare.prices.index:

            return None

        prices = pd.DataFrame(self.prices.loc[buy_date:sell_date].sum(axis = 1), columns = ['port'])
        prices['compare'] = port_compare.prices.loc[buy_date:sell_date].sum(axis = 1)
        prices['port_returns'] = (prices['port'] - prices['port'].iloc[0])/prices['port'].iloc[0]
        prices['compare_returns'] = (prices['compare'] - prices['compare'].iloc[0])/prices['compare'].iloc[0]
        prices['diff'] = (prices['port_returns'] - prices['compare_returns'])/abs(prices['compare_returns'])
        prices = prices[1:]

        quartiles_multiplier = prices['diff'].quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
        quartiles_multiplier.append(prices['diff'].mean())
        quartiles_preturns = prices[prices['diff']>0]['port_returns'].quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
        quartiles_preturns.append(prices[prices['diff']>0]['port_returns'].mean())

        return quartiles_multiplier + quartiles_preturns

    def get_indiv_metrics(self):
        self.get_indiv_returns()
        self.get_indiv_sharpe()
        self.get_indiv_sortino()
        self.get_indiv_volatility()
        self.metrics = pd.concat([self.stock_sharpe, self.stock_sortino, self.stock_return, self.stock_volatility], axis = 1)

    def get_indiv_returns(self):

        prices = self.prices.loc[[self.start_date, self.end_date]].dropna(axis = 1)
        returns = (self.prices.loc[self.end_date] - self.prices.loc[self.start_date])/self.prices.loc[self.start_date]
        self.stock_return = pd.DataFrame(returns.dropna(), columns = ['Returns'])
    
    def get_indiv_sharpe(self):
        
        log_returns = self.daily_log_returns
        sharpe = pd.DataFrame(log_returns.mean()/log_returns.std(), columns = ['Sharpe'])
        self.stock_sharpe = sharpe


    def get_indiv_sortino(self):

        log_returns = self.daily_log_returns
        neg_returns = log_returns[log_returns < 0]
        sortino = pd.DataFrame(log_returns.mean()/neg_returns.std(), columns = ['Sortino'])
        self.stock_sortino = sortino

    def get_indiv_volatility(self):
        
        log_returns = self.daily_log_returns
        vol = pd.DataFrame(log_returns.std(), columns = ['Volatility'])
        self.stock_volatility = vol
