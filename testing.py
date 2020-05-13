import pandas as pd
print("Pandas imported.")
import matplotlib.pyplot as plt
print("Matplotlib imported.")
import modules.preprocess as pre
print("Preproccess imported.")
import modules.models as models
print("Models imported.")
from sklearn.model_selection import train_test_split
print("Train-test-split imported.")
from torch.utils.data import DataLoader
print("DataLoader imported.")
from sklearn.preprocessing import MinMaxScaler
print("MinMaxScaler imported.")

data = pre.data_preparation(f".\\data\\stock-data\\AAPL.csv", 1)
pbw_model = models.MultiSequenceModel()
pbw_model.train_model(data, 10, 0.8, ["Open"], ["Close"])
predictions = pbw_model.predict(data, ["Open"], ["Close"])
print(predictions.head())
data = data.reset_index()
print(data["Close"].head())
predictions["Close"].plot(legend=True)
data["Close"].plot(legend=True)
plt.show()
"""
#perm
past = 5
future = 1
input_dim = 1
hidden_dim = 50
layer_dim = 2
output_dim = 1
batch_size = 32
dropout = 0.3
inputs = ["Open"]
outputs = ["Open"]

model = lstm.LSTM("Test", input_dim, hidden_dim, layer_dim, output_dim, batch_size, dropout, past, future)

model.train_model(data, 10, 0.8, inputs, outputs)

predictions = model.predict(data, inputs, outputs)
predictions.plot()
plt.show()
display_frame = pd.DataFrame(data.loc[data.index[past:],outputs].copy(), index=data.index[past:], columns=outputs)
display_frame["Prediction"] = predictions[outputs].to_numpy()
print(display_frame.head())
display_frame.plot()
plt.show()
"""