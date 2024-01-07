import streamlit as st
import yfinance as yf
import pandas as pd
import cufflinks as cf
import datetime
from simulation_app import prepare_data, predict, backtest, run_simulation, generate_features   

# App title
st.markdown('''
# Stock Direction Forecast App
Shown are the forecast of stock direction either up or down for the period 2023-01-01 till 2023-10-01.
You can select the ticker of your choice in the sidebar.(Companies are technology companies from S&P 500)
Some of the companies don't have extensive data, so the app might not work for all of them.
However companies who have data for the period 2010-01-01 till 2023-10-01 will work.

**Credits**
- Data source: [Yahoo Finance](https://finance.yahoo.com/)
- Shout out to [Data Professor](https://www.youtube.com/channel/UCV8e2g4IWQqK71bbzGDEI4Q) for the app idea
- Shout out to [ritvikmath] (https://www.youtube.com/@ritvikmath) for the insipartion
- App built by [Fatih Karahan](https://github.com/Fatih0234)
''')
st.write('---')

# Sidebar
st.sidebar.subheader('Query parameters')

# let's get the user's amount of money input for the stock
money = st.sidebar.number_input("Money", value=1000, step=100)

# Retrieving tickers data
ticker_list = pd.read_csv("datasets/sp500_companies.txt")
tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list) # Select ticker symbol

# get data on this ticker
data = prepare_data(tickerSymbol, "2010-01-01", "2023-10-01")
data = generate_features(data)
data



run_simulation(data, money, verbose=False, plot=True)

# run the simulation
# run_simulation(data, money, verbose=False, plot=True)

####
#st.write('---')
#st.write(tickerData.info)