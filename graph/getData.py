from os import path
import datetime as dt
import pandas as pd
import pandas_datareader.data as web

def getStockDataFrame(ticker, start, end, api_key):

	# look for data in our current directory, else talk to quandl to get it
	if path.exists('data/stock-data/' + ticker + '.csv'):
		print('data/stock-data/' + ticker + '.csv already exists')
		return pd.read_csv('data/stock-data/' + ticker + '.csv')
	else:
		try:
			# talk to quandl using pandas and store as dataframe
	 		df = web.DataReader(ticker, 'quandl', start, end, api_key = api_key)
		except:
			print('Error: data scraping error while fetching: ', ticker)
			return
		else:
			exportStockDataFrameToCsv(df, ticker)
			return df


def exportStockDataFrameToCsv(df, name):
	try:
		df.to_csv('data/stock-data/' + name + '.csv')
	except:
		print('Error: bad data passed to exportStockDataFrameToCsv')

def getStocksFromFile(tickers, api_key, start, end):

	if (api_key == 'QUANDL API KEY'):
		print('please specify api key')
		return

	if path.exists(tickers):
		# open list of stock tickers from csv file
		stocks = pd.read_csv(tickers, delimiter = ',', usecols=[0])

		# go throught stocks and scrape data
		for ticker in stocks['Symbol']:
			df = getStockDataFrame(ticker, start, end, api_key = api_key)
	else:
		print('Error: please specify stock ticker CSV file')
		return

def getStockTicker(ticker, api_key, start, end):
	if (api_key == 'QUANDL API KEY'):
		print('please specify api key')
		return

	return getStockDataFrame(ticker, start, end, api_key = api_key)
