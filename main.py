import torch
from Stonks import Stonks
import os

def main():
	device = None
	if torch.cuda.is_available():
		print('GPU enabled')
		device = 'cuda'
		torch.backends.cudnn.benchmark = True

	stonks = Stonks('cuda')
	stonks.train_on_n_stocks(10)
	#torch.save(stonks.model.state_dict(), os.getcwd() + '/fucking-model.pt')
	#stonks.save_model(os.getcwd() + '/models/150-stocks.pt')

	#stonks.save_model(os.getcwd() + 'weights.pt')
	#stonks.load_model('backup.pt')
	stonks.forecast_stock('T.csv')



if __name__ == '__main__':
	main()
