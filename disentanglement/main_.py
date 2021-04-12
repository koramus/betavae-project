# https://arxiv.org/abs/1606.05908
# https://arxiv.org/abs/1907.08956
# https://openreview.net/forum?id=Sy2fzU9gl

import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()

		self.fc1 = nn.Linear(784, 400)
		self.fc21 = nn.Linear(400, 20)
		self.fc22 = nn.Linear(400, 20)
		self.fc3 = nn.Linear(20, 400)
		self.fc4 = nn.Linear(400, 784)

	def encode(self, x):
		h1 = F.relu(self.fc1(x))	
		return self.fc21(h1), self.fc22(h1)

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)

		return mu + (eps * std)

	def decode(self, z):
		h3 = F.relu(self.fc3(z))
		return torch.sigmoid(self.fc4(h3))

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		recon = self.decode(z)
		
		return recon, mu, logvar

def loss_function(recon, x, mu, logvar):
	# BETA variable loss
	KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

	# Reconstruction loss
	LP = torch.nn.functional.mse_loss(recon, x, reduction = 'sum')

	return LP + KL


def train(epoch):
	model.train()
	
	train_loss = 0
	for batch_idx, (data, _) in enumerate(train_loader):
		data = data.view(-1, 784)
		optimizer.zero_grad()

		recon, mu, logvar = model(data)
		loss = loss_function(recon, data, mu, logvar)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
	print("{}: {}".format(epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
	model.eval()
	
	test_loss = 0
	with torch.no_grad():
		for i, (data, _) in enumerate(test_loader):
			data = data.view(-1, 784)
			
			recon, mu, logvar = model(data)
			loss = loss_function(recon, data, mu, logvar)

			test_loss += loss.item()

			print("{}: {}".format(epoch, test_loss / len(test_loader.dataset)))
