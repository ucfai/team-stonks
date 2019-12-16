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

def data_preparation(csvFile, year_length=5, *columns):
	'''
	Reads a csv file, parses the dates and makes them the index. Then it fills
	for NaNs and will choose the most recent amount of years. Lastly, it shapes the data
	as a numpy array.

	ARGUMENTS:
		csvFile: The file path to be read in.
		year_length: The amount of years to keep for input.
		*columns: The columns you wish to keep for data analysis.
	RETURNS:
		data: The formatted numpy array of the data.
	'''
	data = pd.read_csv(csvFile, index_col="Date", parse_dates=True)
	data = (data.ffill()+data.bfill())/2
	data = data.bfill().ffill()
	data = data.astype("float")
	data = data[list(columns)]
	data = data.values.reshape(-1, len(data.columns))
	data = data[len(data) - year_length*365:]
	return data


def create_sequence(input_data, target_index, time_window, forecast_window=1):
	'''
	Creates a sequence that consists of a time window and a forecast window that will
	be used to input and predict data.

	ARGUMENTS:
		input_data: The numpy array that is to be inputted.
		target_index: The column index of the target to attempt to predict.
		time_window: The length of time to look into the past to base predictions off of.
		forecast_window: The length of time to compare into the future to compare predictions to.
	RETURNS:
		seq: The created sequence that organizes the data in windows for every index.
	'''
	seq = []
	L = len(input_data)
	for i in range(time_window, L):
		inp = input_data[i-time_window:i]
		targ = input_data[i:i+forecast_window][0][target_index]
		seq.append((inp, targ))
	# seq has dimensions [index][inp, targ][window][column]
	return seq


class LSTM(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim=400, layer_dim=4, output_dim=1, batch_size=32, dropout=0.3, stateful=False, device='cpu'):
		super(LSTM, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.layer_dim = layer_dim
		self.output_dim = output_dim
		self.batch_size = batch_size
		self.dropout = dropout
		self.stateful = stateful
		self.device = device

		# manually defined hidden dimensions

		self.hidden = (torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim, dtype=torch.float).requires_grad_(),
					   torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim, dtype=torch.float).requires_grad_())

		self.state_iteration = 0

		# batch_first=True -> input/output tensors = (batch_dim, seq_dim, feature_dim)
		self.lstm = torch.nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim,
								  dropout=self.dropout, batch_first=True)

		self.drop = torch.nn.Dropout(self.dropout)

		# Readout layer
		self.linear = torch.nn.Linear(hidden_dim, output_dim)

	def reset_state(self, x):
		if self.device == 'cuda':
			h0 = torch.zeros(self.layer_dim, x.size(0),
							 self.hidden_dim, dtype=torch.float).requires_grad_().cuda()
			c0 = torch.zeros(self.layer_dim, x.size(0),
							 self.hidden_dim, dtype=torch.float).requires_grad_().cuda()
		else:
			h0 = torch.zeros(self.layer_dim, x.size(0),
							 self.hidden_dim, dtype=torch.float).requires_grad_()
			c0 = torch.zeros(self.layer_dim, x.size(0),
							 self.hidden_dim, dtype=torch.float).requires_grad_()

		return h0, c0

	def forward(self, x):
		if self.stateful and self.state_iteration == 30:
			self.hidden = self.reset_state(x)
			self.state_iteration = 0
		elif not self.stateful:
			self.hidden = self.reset_state(x)
			self.state_iteration = 0

		out, self.hidden = self.lstm(x, (self.hidden[0].detach(),
										 self.hidden[1].detach()))
		out = self.drop(out)
		out = self.linear(out[:, -1, :])
		self.state_iteration += 1
		return out


def run_epoch(model, criterion, dataloaders, device, phase, optimizer, datasets):
	'''
	Runs a single epoch to train/validate the model against input data.

	ARGUMENTS:
		model: The LSTM model to be trained/validated.
		dataloaders: The dataloader.
		device: Whether to train on CPU or GPU.
		phase: Whether to train vs. validate.
	RETURNS:
		epoch_loss: The loss of the model after this epoch.
		epoch_acc: The accuracy of the model to the true value after this epoch.
	'''

	target_index = 0
	num_features = 2
	batch_size = 64
	time_window = 60
	forecast_window = 1
	year_length = 5
	train_proportion = 0.8

	running_loss = 0.0
	running_corrects = 0

	if phase == 'Train':
		model.train()
	else:
		model.eval()

	for i, (inputs, labels) in enumerate(dataloaders[phase]):

		inputs = inputs.to(device)
		labels = labels.to(device)

		inputs = inputs.float()
		labels = labels.float()

		inputs = inputs.view(-1, time_window, num_features)
		labels = labels.view(-1, 1)

		optimizer.zero_grad()

		with torch.set_grad_enabled(phase == 'Train') and autograd.detect_anomaly():
			outputs = model(inputs)

			# BUG here
			loss = criterion(outputs, labels)

			if phase == 'Train':
				loss.backward()
				optimizer.step()

		running_loss += loss.item() * inputs.size(0)
		running_corrects += torch.sum(abs(outputs - labels) <= 0.05 * labels)

	epoch_loss = running_loss / datasets[phase].__len__()
	epoch_acc = running_corrects.double() / datasets[phase].__len__()

	return epoch_loss, epoch_acc


def train(model, criterion, optimizer, num_epochs, dataloaders, device, datasets):
	'''
	Will train the model by using run_epoch several times and keeping the best model to be used
	at the end.

	ARGUMENTS:
		model: The LSTM model to be trained/validated.
		criterion: The loss funtion for the model
		optimizer: The algorithm for updating the model when training.
		num_epochs: The number of epochs to run.
		dataloaders: The dataloaders that contain a "Train" and "Validation" portion.
		device: Whether to train on CPU vs. GPU.
	RETURNS:
		model: The trained model that can be used for evaluating.
	'''
	start = time.time()

	best_model_weights = model.state_dict()
	best_acc = 0.0
	pbar = tqdm(total=num_epochs, unit="epoch")

	for epoch in range(num_epochs):
		train_loss, train_acc = run_epoch(
			model, criterion, dataloaders, device, 'Train', optimizer, datasets)
		val_loss, val_acc = run_epoch(
			model, criterion, dataloaders, device, 'Validation', optimizer, datasets)

		pbar.set_description_str(' Epoch: {:>4} | Train Loss: {:>5.2f}% | Train Acc: {:>5.2f}% | Valid Loss: {:>5.2f}% | Valid Acc: {:>5.2f}% |'.format(
			epoch+1, train_loss * 100, train_acc * 100, val_loss * 100, val_acc * 100), refresh=False)

		if val_acc > best_acc:
			best_acc = val_acc
			best_model_weights = model.state_dict()

		pbar.update()

	total_time = time.time() - start
	pbar.close()

	print('-' * 80)
	print('Training complete in {:.0f}m {:.0f}s'.format(
		total_time // 60, total_time % 60))
	print('Best validaton accuracy: {:.4f}'.format(best_acc))
	print()

	model.load_state_dict(best_model_weights)

	return model


def test(model, inputs, device):
    '''
    Will test the model without training it.
    ARGUMENTS:
        model: The LSTM model to be evaluated.
        inputs: The inputs of the data to be predicted from.
        device: Whether to evaluate on the CPU vs. GPU.
    RETURNS:
        outputs: The predicted values of the model.
    '''

    model.eval()
    inputs = torch.tensor(inputs, dtype=torch.float).to(device)
    outputs = model(inputs).cpu().detach().numpy()
    return outputs
