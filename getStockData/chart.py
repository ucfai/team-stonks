import matplotlib.pyplot as plt
from getstocks import *
import numpy as np
import pandas as pd


def chartStock(ticker, start, end, api_key):
	data = getStockDataFrame(ticker, start, end, api_key)
	data['Date'] = pd.to_datetime(data['Date'])
	fig, ax = plt.subplots()
	ax.plot(data['Date'], data['Close'])

	plt.show()
