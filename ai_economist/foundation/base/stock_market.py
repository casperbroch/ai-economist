import yfinance as yf
import numpy as np
import random
import math

class StockMarket:
    transaction_cost = 0.0075
    
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol
        self.data = yf.download(stock_symbol, progress=False)
        self.sim_price = []
        self.price = 0
            
    def simulate(self, no_days):
        self.sim_price = self.simulate_stock_price(no_days)
        self.price = self.sim_price[-1]
        
    def nextstep(self, supply, demand, stocksQuantity):
        random_return = self.get_random_return()
        if demand+supply > 0:
            p = (demand - supply) / demand+supply
        else:
            p = 0
        
        self.price = self.price * (1 + (0.5*random_return + 0.5*p))
        
        
    def nextsteprandom(self):
        next_step = self.simulate_one_step(self.price)
        self.sim_price = np.concatenate((self.sim_price, [next_step]))    
        self.price = next_step
        
    def getprice(self):
        return self.price
        
    def simulate_stock_price(self, days):
        returns = np.log(1 + self.data['Adj Close'].pct_change())
        mu, sigma = returns.mean(), returns.std()
        sim_returns = np.random.normal(mu, sigma, days)
        start_price = self.data['Adj Close'].iloc[-1]
        sim_price = start_price * (sim_returns + 1).cumprod()
        return sim_price
    
    def simulate_one_step(self, current_price):
        returns = np.log(1 + self.data['Adj Close'].pct_change())
        mu, sigma = returns.mean(), returns.std()
        sim_return = np.random.normal(mu, sigma)
        sim_price = current_price * (sim_return + 1)
        return sim_price
    
    def get_random_return(self):
        returns = np.log(1 + self.data['Adj Close'].pct_change())
        mu, sigma = returns.mean(), returns.std()
        sim_return = np.random.normal(mu, sigma)
        return sim_return
        
market = StockMarket("AAPL")

