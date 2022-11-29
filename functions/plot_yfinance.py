import yfinance as yf
from functions.plot_linear_reg import plot_linear_reg

def plot_yfinance(company, period = "1y", interval = "1d"):
  ## import data
  aapl= yf.Ticker(company)
  aapl_historical = aapl.history(period = period, interval=interval)
  ## plot
  plot_linear_reg(aapl_historical, "plots/apple.png")
