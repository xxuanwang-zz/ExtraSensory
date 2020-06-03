import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler, Adam

from utilize.test import evaluate_model

class MLP(nn.Module):
	def __init__(self, n_hidden_layers, n_features, n_labels):
		super(MLP, self).__init__()
		self.n_features = n_features
		self.n_labels = n_labels
		self.n_hidden_layers = n_hidden_layers

		self.fc1 = nn.Linear(n_features, n_hidden_layers[0])
		self.fc2 = nn.Linear(n_hidden_layers[0], n_hidden_layers[1])
		self.fc3 = nn.Linear(n_hidden_layers[1], n_labels)

	def forward(self, x):

		x = self.fc1(x)
		x = F.prelu(x, torch.tensor(0.1))
		x = self.fc2(x)
		x = F.prelu(x, torch.tensor(0.1))
		x = self.fc3(x)

		return x

	def logist(self, x):

		x = torch.sigmoid(x)

		return x
# Add dropout layers and batch nomarlization layers 
class MLP_model():
	def __init__(self, n_hidden_layers, target_labels):

		self.target_labels = target_labels
		self.n_hidden_layers = n_hidden_layers

	def predict(self, X):
		if type(X) is not torch.Tensor:
			X = torch.tensor(X).float()
		y_out = self.MLP(X)
		y_pred = self.MLP.logist(y_out) > 0.5

		return y_pred

	def fit(self, X_train, y_train, X_test = None, y_test = None, M_train = None, M_test = None, epoches = 20, learning_rate = 0.00001, report = False):

		self.MLP = MLP(self.n_hidden_layers, X_train.shape[1], y_train.shape[1])

		

		if M_train is not None:
			W_train = (y_train/np.sum(y_train, axis = 0)/2 + abs(y_train -1)/(np.sum(abs(y_train -1), axis = 0))/2)*y_train.shape[0]*abs(M_train-1)
		else:
			W_train = (y_train/np.sum(y_train, axis = 0)/2 + abs(y_train -1)/(np.sum(abs(y_train -1), axis = 0))/2)*y_train.shape[0]

		X_train = torch.tensor(X_train).float()
		y_train = torch.tensor(y_train).float()
		W_train = torch.tensor(W_train).float()

		if M_test is not None:
			W_test = torch.tensor(np.abs(M_test-1)).float()
			X_test = torch.tensor(X_test).float()
			y_test = torch.tensor(y_test).float()

		batch_size = 300
		#epoches = 40

		optimizer = Adam(self.MLP.parameters(), lr=learning_rate)
		#optimizer = SGD(self.MLP.parameters(), lr=learning_rate, momentum=0.5)
		#lambda1 = lambda epoch: (learning_rate - epoch*(learning_rate-0.1*learning_rate)/40)
		#scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
		BCE = F.binary_cross_entropy

		for i in range(epoches):
			permutation = np.random.permutation(X_train.shape[0])
			X_train = X_train[permutation, :]
			y_train = y_train[permutation, :]
			W_train = W_train[permutation, :]
			for batch_idx in range(int(X_train.shape[0]/batch_size)):

				X_batch = X_train[batch_idx*batch_size:(batch_idx+1)*batch_size, :]
				y_batch = y_train[batch_idx*batch_size:(batch_idx+1)*batch_size, :]
				W_batch = W_train[batch_idx*batch_size:(batch_idx+1)*batch_size, :]

				y_out = self.MLP(X_batch)
				y_out = self.MLP.logist(y_out)
				loss = BCE(y_out, y_batch, weight = W_batch)

				loss.backward()
				optimizer.step()
				#scheduler.step()

				if (batch_idx+1) % 100 == 0 and report:
					print('Train Epoch: %d [%d/%d (%d%%)]\tLoss: %.6f' %(i, batch_idx * batch_size, X_train.shape[0], 100. * batch_idx * batch_size/ X_train.shape[0], loss.item()))
			if X_test is not None:
				print('Test epoch %d:' %(i))
				evaluate_model(self, X_test, y_test, W_test)

if __name__ == 'main': 

	# Didn't include the code for loading the data
	# Just for your reference about how to use the model

	mlp = MLP_model([64, 64], target_label)
	mlp.fit(X_train, y_train, X_test, y_test, M_train, M_test, epoches = 20, learning_rate = 0.00001, report = True)