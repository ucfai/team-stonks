import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import autograd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
from tqdm import tqdm
import os
import sys
import glob

from LSTM import *

class trainingError:
	pass

class Stonks:
	def __init__(self, device):
		self.device = device
		self.filePath = os.getcwd()
		self.target_index = 0
		self.num_features = 2
		self.batch_size = 64
		self.time_window = 60
		self.forecast_window = 1
		self.year_length = 5
		self.train_proportion = 0.8

		# best parameters: hidden-50, layer-4, learn-0.01
		self.input_dim = self.num_features
		self.hidden_dim = 50
		self.layer_dim 	= 4
		self.output_dim = self.forecast_window
		self.learning_rate = 0.01
		self.num_epochs = 100

		self.model = LSTM(input_dim=self.input_dim, hidden_dim=self.hidden_dim,
						  layer_dim=self.layer_dim, output_dim=self.output_dim,
						  batch_size=self.batch_size, stateful=False, device=device).to(self.device)


	def save_model(self, filename):
		torch.save(self.model.state_dict(), os.getcwd() + '/models/' + filename)

	def backup_model(self):
		torch.save(self.model.state_dict(), self.filePath + '/models/backup.pt')

	def load_model(self, modelFile):
		self.model.load_state_dict(torch.load(self.filePath + '/models/' + modelFile))


	def train_on_all_stocks(self):
		stocks = glob.glob(self.filePath + '/data/stock-data/*.csv')
		for stock_filepath in stocks:
			print('training on ' + stock_filepath)
			try:
				self.train_model(stock_filepath)
				self.backup_model()
			except:
				continue

	def train_on_n_stocks(self, n):
		stocks = glob.glob(self.filePath + '/data/stock-data/*.csv')
		for i in range(n):
			print('training on ' + stocks[i])
			try:
				self.train_model(stocks[i])
				self.backup_model()
			except:
				continue

	def train_on_stock_n_epochs(self, stockFilePath, n):
		self.num_epochs = n
		self.train_model(os.getcwd() + '/data/stock-data/'+ stockFilePath)


	def train_model(self, stockFilePath):

		full_dataset = data_preparation(
		   	stockFilePath, self.year_length, "Open", "Volume")

		train_dataset, val_dataset = train_test_split(
		    full_dataset, train_size = self.train_proportion, shuffle = False)

		datasets = {'Train': train_dataset, 'Validation': val_dataset}

		scaler = MinMaxScaler(feature_range=(0, 1))

		train_dataset_sc = scaler.fit_transform(datasets['Train'])
		val_dataset_sc = scaler.transform(datasets['Validation'])

		datasets_sc = {'Train': train_dataset_sc, 'Validation': val_dataset_sc}

		train_sequence = create_sequence(datasets_sc['Train'],
		                                 self.target_index,
		                                 self.time_window,
		                                 self.forecast_window)

		val_sequence = create_sequence(datasets_sc['Validation'],
		                               self.target_index,
		                               self.time_window,
		                               self.forecast_window)

		dataset_sequences = {'Train': train_sequence, 'Validation': val_sequence}

		# this line will goof up if the stock data isn't exactly correct
		try:
			dataloaders = {x: DataLoader(dataset_sequences[x],
			                             self.batch_size,
			                             shuffle=True if x == 'Train' else True, num_workers=0)
			               for x in ['Train', 'Validation']}
		except:
			print('Error while training on ' + stockFilePath)
			raise trainingError
			return

		criterion = torch.nn.MSELoss()
		optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)

		# train model
		self.model = train(self.model, criterion, optimizer,
		              self.num_epochs, dataloaders, self.device, datasets)


	def forecast_stock(self, stockFile):

		test_dataset = data_preparation(
			self.filePath + '/data/stock-data/' + stockFile,
			self.year_length, "Open", "Volume")

		# normalize data
		target_scaler = MinMaxScaler(feature_range=(0, 1))
		test_scaler = MinMaxScaler(feature_range=(0, 1))

		# Make array that has indices for one column
		# [original_array[indices][column] for index in range(length(original_array))]
		to_scale = np.array([test_dataset[i][self.target_index]
							 for i in range(len(test_dataset))])

		to_scale = to_scale.reshape(-1, 1)
		target_scaler = target_scaler.fit(to_scale)

		test_dataset_sc = []

		for feature in range(self.num_features):
			to_scale = np.array([test_dataset[i][feature]
								 for i in range(len(test_dataset))])

			to_scale = to_scale.reshape(-1, 1)
			test_dataset_sc.append(test_scaler.fit_transform(to_scale))


		test_dataset_sc = np.array(test_dataset_sc)
		test_dataset_sc = test_dataset_sc.swapaxes(0, 1)
		test_dataset_sc = test_dataset_sc.reshape(-1, self.num_features)

		test_sequence = create_sequence(test_dataset_sc,
										self.target_index,
										self.time_window,
										self.forecast_window)


		test_input = [test_sequence[i][0] for i in range(len(test_sequence))]
		test_input = np.array(test_input)
		test_input = test_input.reshape(-1, self.time_window, self.num_features)

		test_target = [test_sequence[i][1] for i in range(len(test_sequence)-1)]

		test_output = test(self.model, test_input, self.device)

		# reshape test target for grpahing
		test_target = np.array(test_target)
		test_target = test_target.reshape(-1, 1)
		test_target = target_scaler.inverse_transform(test_target)

		# reshape test_output for graphing
		test_output = np.array(test_output)
		test_output = test_output.reshape(-1, 1)
		test_output = target_scaler.inverse_transform(test_output)

		error_plot = target_scaler.transform(
			abs(test_target - test_output[1:]))

		plt.figure()
		plt.subplot(311)
		plt.plot(test_output)
		plt.plot(test_target)
		plt.subplot(312)
		plt.plot(error_plot)
		plt.subplot(313)
		plt.plot([test_dataset[i][self.target_index] for i in range(len(test_dataset))])
		plt.show()
