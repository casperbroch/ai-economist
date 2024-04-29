import yfinance as yf
import numpy as np
import random
import math

class StockMarket:
    transaction_cost = 0.0075
    
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol
        self.data = yf.download('MSFT')
        self.sim_price = []
        self.price = 0
            
    def simulate(self, no_days):
        self.sim_price = simulate_stock_price(self.data, no_days)
        self.price = self.sim_price[-1]
    
    def nextstep(self):
        next_step = simulate_one_step(self.data, self.price)
        self.sim_price = np.concatenate((self.sim_price, [next_step]))    
        self.price = next_step
        

    def simulate_stock_price(stock_data, days):
        returns = np.log(1 + stock_data['Adj Close'].pct_change())
        mu, sigma = returns.mean(), returns.std()
        sim_returns = np.random.normal(mu, sigma, days)
        start_price = stock_data['Adj Close'].iloc[-1]
        sim_price = start_price * (sim_returns + 1).cumprod()
        return sim_price
    
    def simulate_one_step(stock_data, current_price):
        returns = np.log(1 + stock_data['Adj Close'].pct_change())
        mu, sigma = returns.mean(), returns.std()
        sim_return = np.random.normal(mu, sigma)
        sim_price = current_price * (sim_return + 1)
        return sim_price
        
market = StockMarket("AAPL")

