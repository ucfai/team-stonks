import urllib.request
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
from mpl_finance import candlestick_ohlc
import glob

# Using Alpha Vantage API
class alphaVantageException(Exception):
	# typically thrown when over 5 calls per minute, or 500 calls per day
	pass


class StockData:
	def __init__(self, key):
		#self.key = key
		self.key = key
		return

	def get_stocks(self, filename):
		# takes file name containing stock symbols and returns a list
		f = pd.read_csv(filename, delimiter =',')
		return f['Symbol']

	def download_stock_csv(self, symbol):
		# QUERY: https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey=demo&datatype=csv
		url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=' + symbol + '&apikey=' + self.key + '&outputsize=full&datatype=csv'

		filePath = os.getcwd() + '/data/stock-data/' + symbol + '.csv'

		if os.path.exists(filePath):
			print(symbol + ' already exists')
			return -1

		try:
			urllib.request.urlretrieve(url, filePath)
			print(symbol + ' has been downloaded')
			self.reverse_csv('data/stock-data/' + symbol + '.csv')
			return 1
		except:
			print('Something went wrong downloading ' + symbol)
			raise alphaVantageException

	def reverse_csv(self, filename):
		# because alpha vantage returns the csv from reverse chronological order,
		# and yahoo finance has their csv's in chronological order,
		# we need to reverse the data frame and make it compatible for the model
		frame = pd.read_csv(filename, delimiter=',')
		frame = frame.iloc[::-1, :]
		frame = frame.drop(columns=['dividend_amount', 'split_coefficient'], axis=1)
		frame = frame.rename(columns = {"timestamp" : "Date", "open" : "Open", "high" : "High", "low" : "Low", "close" : "Close", "volume" : "Volume", "adjusted_close" : "Adj Close"})
		frame.to_csv(filename, index=False)

	def get_stock_data(self):
		# download a file for each stock in the file tickers.csv
		stocks = get_stocks(os.getcwd() + '/data/tickers/tickers.csv')

		for symbol in stocks:
			t = time.time()

			try:
				if (download_stock_csv(symbol) == 1):
					# prevent using too many calls to api
					sleepTime = 12
					print('sleeping for ' + str(sleepTime) + ' seconds to avoid the API police')
					time.sleep(sleepTime)
				else:
					# the stock already exists in the stock-data directory
					continue

			except alphaVantageException:
				continue

	def movingAverages(self, filename, averages, ema=True):
		# open csv file
		frame = pd.read_csv(os.getcwd() + '/data/stock-data/' + filename, delimiter=',')

		# create a frame list to merge into a pandas dataframe later
		for i in averages:
			colName = str(i) + 'MA'
			frame[colName] = self.n_moving_average(frame, i)
			if ema == True:
				colName =  str(i) + 'EMA'
				frame[colName] = self.n_day_ema(frame, i)

		# find max average and remove 0 - (max - 1) rows will null values

		# this line will remove the starting indicies that have null values
		#frame = frame[max(averages):]

		frame.to_csv(os.getcwd() + '/data/stock-data/' + filename, index=False)
		return frame


	def n_moving_average(self, frame, n):
		# takes a filename and calculates the n-moving averages
		# ex) n = 5, 5 day moving average is returned

		ma = []
		# pad with None
		for i in range(0, n):
			ma.append(None)

		# iterate through data points and calculate n-average at each point
		for i in range(n, len(frame['Open'])):
			average = 0
			for x in range(i - n, i):
				average += frame['Open'][x]
			ma.append(average / n)

		colName = str(n) + 'MA'
		movingAverage = pd.DataFrame(ma, columns=[colName])

		return movingAverage

	def n_day_ema(self, frame, n):
		# adds a column to the data frame with an n day exponential moving average (ema)
		return frame['Open'].ewm(span=n, adjust=False).mean()


	def n_day_bollinger_bands(self, filename, n):
		# apply n-day bollinger bands and add it to the data frame
		frame = pd.read_csv(os.getcwd() + '/data/stock-data/' + filename)
		tmp = frame.copy()
		# check if n-day moving average is in our data frame
		colNameMA = str(n) + 'MA'
		if colNameMA not in frame.columns:
			frame[colNameMA] = self.n_moving_average(frame, n)


    	# set .std(ddof=0) for population std instead of sample
		tmp[colNameMA + '-STD'] = tmp['Adj Close'].rolling(window=20).std()
		frame[str(n) + 'BB-Upper'] = tmp[colNameMA] + (tmp[colNameMA + '-STD'] * 2)
		frame[str(n) + 'BB-Lower'] = tmp[colNameMA] - (tmp[colNameMA + '-STD'] * 2)
		frame.to_csv(os.getcwd() + '/data/stock-data/' + filename, index=False)

	def bollinger_bands(self, symbol, moving_averages=[20]):
		for i in moving_averages:
			self.n_day_bollinger_bands(symbol, i)

	def drop_nan(self, filename):
		frame = pd.read_csv(os.getcwd() + '/data/stock-data/' + filename)
		frame = frame.dropna()
		frame.to_csv(os.getcwd() + '/data/stock-data/' + filename, index=False)

	def apply_transformations_to_stock(self, filename, moving_averages = [20]):
		#frame = pd.read_csv(os.getcwd() + '/data/stock-data/' + symbol + '.csv')
		self.movingAverages(filename, moving_averages)
		self.bollinger_bands(filename, moving_averages)
		self.drop_nan(filename)

	def apply_transformations_to_all_stocks(self, moving_averages):
		stocks = [os.path.basename(x) for x in glob.glob(os.getcwd() + '/data/stock-data/*.csv')]
		for stock in stocks:
			print('Transforming ' + stock)
			self.apply_transformations_to_stock(stock, moving_averages)


	def plot(self, symbol):
		# read data from csv
		frame = pd.read_csv(os.getcwd() + '/data/stock-data/' + symbol + '.csv')

		# open columns
		ohlc = frame.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]

		# get columns that are note ohlc
		cols = []
		for col in frame.columns:
			if col not in ['Date', 'Open', 'High', 'Low', 'Close']:
				cols.append(col)
				print(col)

		# transform dates so they play nicely with matplotlib
		ohlc['Date'] = pd.to_datetime(ohlc['Date']).apply(mpl_dates.date2num).astype('float')

		plt.style.use('ggplot')
		fig, ax = plt.subplots(1, sharex=True)

		candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red', alpha=0.8)
		ax.set_xlabel('Date')
		ax.set_ylabel('Price')
		fig.suptitle(symbol)

		date_format = mpl_dates.DateFormatter('%d-%m-%Y')
		ax.xaxis.set_major_formatter(date_format)
		fig.autofmt_xdate()
		fig.tight_layout()

		# plot moving averages
		ax.plot(ohlc['Date'], frame['5EMA'], color='blue', label='5EMA')

		# plot Volume
		#plt.subplot(111, sharex=ax)
		#axVol.set_ylabel('Volume', color='blue')
		#plt.plot(ohlc['Date'], frame['Volume'], color='blue')


		plt.show()

def main():
	#movingAverages('AAPL.csv', [5, 10, 15, 30, 50]).to_csv('res.csv', index=False)
	stocks = StockData('2AC5HW9582LL9CZA')
	#stocks.download_stock_csv('')
	stocks.apply_transformations_to_all_stocks(moving_averages=[5, 20, 50, 100, 200])


if __name__ == '__main__':
	main()
