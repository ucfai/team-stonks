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
    data = data[len(data)-year_length*365:]
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


class LSTMModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=400, layer_dim=4, output_dim=1, batch_size=32, dropout=0.3, stateful=False):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.dropout = dropout
        self.stateful = stateful
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


def run_epoch(model, dataloaders, device, phase):
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

            loss = criterion(outputs, labels)

            if phase == 'Train':
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(abs(outputs - labels) <= 0.05 * labels)

    epoch_loss = running_loss / datasets[phase].__len__()
    epoch_acc = running_corrects.double() / datasets[phase].__len__()

    return epoch_loss, epoch_acc


def train(model, criterion, optimizer, num_epochs, dataloaders, device):
    '''
    Will train the model by using run_epoch several times and keeping the best model to be used
    at the end.

    ARGUMENTS: 
        model: The LSTM model to be trained/validated.
        criterion: The loss funtion for the model.
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
            model, dataloaders, device, 'Train')
        val_loss, val_acc = run_epoch(
            model, dataloaders, device, 'Validation')

        pbar.set_description_str(' Epoch: {:>4} | Train Loss: {:>5.2f}% | Train Acc: {:>5.2f}% | Valid Loss: {:>5.2f}% | Valid Acc: {:>5.2f}% |'.format(
            epoch+1, train_loss * 100, train_acc * 100, val_loss * 100, val_acc * 100), refresh=False)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_weights = model.state_dict()

        pbar.update()

    total_time = time.time() - start
    pbar.close()

    print('-' * 74)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        total_time // 60, total_time % 60))
    print('Best validaton accuracy: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_weights)

    return model


target_index = 0
num_features = 2
batch_size = 64
time_window = 60
forecast_window = 1
year_length = 5
train_proportion = 0.8

# best parameters: hidden-50, layer-4, learn-0.01
input_dim = num_features
hidden_dim = 10
layer_dim = 2
output_dim = forecast_window
learning_rate = 0.01
num_epochs = 10

train_stock = "AAPL.csv"
forecast_stock = "AAPL.csv"

filePath = "D:\\Documents\\AI@UCF\\_team-stonks\\getStockData\\data\\stock-data\\"

full_dataset = data_preparation(
    filePath+train_stock, year_length, "Open", "Volume")

train_dataset, val_dataset = train_test_split(
    full_dataset, train_size=train_proportion, shuffle=False)

datasets = {'Train': train_dataset, 'Validation': val_dataset}

scaler = MinMaxScaler(feature_range=(0, 1))

train_dataset_sc = scaler.fit_transform(datasets['Train'])
val_dataset_sc = scaler.transform(datasets['Validation'])

datasets_sc = {'Train': train_dataset_sc, 'Validation': val_dataset_sc}

train_sequence = create_sequence(datasets_sc['Train'],
                                 target_index,
                                 time_window,
                                 forecast_window)

val_sequence = create_sequence(datasets_sc['Validation'],
                               target_index,
                               time_window,
                               forecast_window)

dataset_sequences = {'Train': train_sequence, 'Validation': val_sequence}

dataloaders = {x: DataLoader(dataset_sequences[x],
                             batch_size,
                             shuffle=True if x == 'Train' else True, num_workers=0)
               for x in ['Train', 'Validation']}

model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim,
                  layer_dim=layer_dim, output_dim=output_dim,
                  batch_size=batch_size, stateful=False)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

if device == 'cuda':
    torch.backends.cudnn.benchmark = True

weight_save_path = 'best.weights.pt'
best_loss = 0


model = train(model, criterion, optimizer,
              num_epochs, dataloaders, device)


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


test_dataset = data_preparation(
    filePath+forecast_stock, year_length, "Open", "Volume")

target_scaler = MinMaxScaler(feature_range=(0, 1))
test_scaler = MinMaxScaler(feature_range=(0, 1))

# Make array that has indices for one column
# [original_array[indices][column] for index in range(length(original_array))]

to_scale = np.array([test_dataset[i][target_index]
                     for i in range(len(test_dataset))])
to_scale = to_scale.reshape(-1, 1)
target_scaler = target_scaler.fit(to_scale)

test_dataset_sc = []

for feature in range(num_features):
    to_scale = np.array([test_dataset[i][feature]
                         for i in range(len(test_dataset))])
    to_scale = to_scale.reshape(-1, 1)
    test_dataset_sc.append(test_scaler.fit_transform(to_scale))


test_dataset_sc = np.array(test_dataset_sc)
test_dataset_sc = test_dataset_sc.swapaxes(0, 1)
test_dataset_sc = test_dataset_sc.reshape(-1, num_features)

test_sequence = create_sequence(test_dataset_sc,
                                target_index,
                                time_window,
                                forecast_window)


test_input = [test_sequence[i][0] for i in range(len(test_sequence))]
test_input = np.array(test_input)
test_input = test_input.reshape(-1, time_window, num_features)

test_target = [test_sequence[i][1] for i in range(len(test_sequence)-1)]

test_output = test(model, test_input, device)

test_target = np.array(test_target)
test_target = test_target.reshape(-1, 1)
test_target = target_scaler.inverse_transform(test_target)

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
plt.plot([test_dataset[i][target_index] for i in range(len(test_dataset))])
plt.show()
