import urllib.request
import pandas
import time
# Using Alpha Vantage API

class alphaVantageException(Exception):
	pass


def get_stocks(filename):
	symbol = 'MSFT'
	f = pandas.read_csv(filename, delimiter =',')
	return f['Symbol']

def download_stock_csv(symbol):
	key = 'ESOGCIA0YJJILED0'
	# QUERY: https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey=demo&datatype=csv
	url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=' + symbol + '&apikey=' + key + '&outputsize=full&datatype=csv'
	try:
		urllib.request.urlretrieve(url,  'data/stock-data/' + symbol + '.csv')
	except:
		print('Something went wrong downloading ' + symbol)
		raise alphaVantageException



def reverse_csv(filename):
	# because alpha vantage returns the csv from reverse chronological order,
	# and yahoo finance has their csv's in chronological order,
	# we need to reverse the data frame and make it compatible for the model
	frame = pandas.read_csv(filename, delimiter=',')
	frame = frame.iloc[::-1, :]
	frame = frame.drop(columns=['dividend_amount', 'split_coefficient'], axis=1)
	frame = frame.rename(columns = {"timestamp" : "Date", "open" : "Open", "high" : "High", "low" : "Low", "close" : "Close", "volume" : "Volume", "adjusted_close" : "Adj Close"})
	frame.to_csv(filename, index=False)

def main():
	stocks = get_stocks('data/tickers/tickers.csv')

	for symbol in stocks:
		t = time.time()

		try:
			download_stock_csv(symbol)
			reverse_csv('data/stock-data/' + symbol + '.csv')

			# prevent using too many calls to api
			while( (t - time.time()) % 60 <= 12):
				time.sleep(1)

		except alphaVantageException:
			continue



if __name__ == '__main__':
	main()
