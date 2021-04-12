import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class View(nn.Module):
	def __init__(self, size):
		super(View, self).__init__()
		self.size = size

	def forward(self, tensor):
		return tensor.view(self.size)

class PCA:
	def __init__(self, device, data, z_dim=10, sz=64):
		self.z_dim = z_dim
		self.sz = sz
		self.cached = True
		d = data.cpu().float().view(-1, sz*sz)
		(U, S, V) = torch.pca_lowrank(d, q=z_dim)
		self.encoded = torch.matmul(d, V[:, :z_dim])

class BetaVAE_Bernoulli(nn.Module):

	def __init__(self, device, z_dim=10, sz=64):
		super(BetaVAE_Bernoulli, self).__init__()
		self.z_dim = z_dim
		self.sz = sz
		self.cached = False
		self.encoder = nn.Sequential(
			View((-1, self.sz * self.sz)),
			nn.Linear(self.sz * self.sz, 1200),
			nn.ReLU(True),
			nn.Linear(1200, 1200),
			nn.ReLU(True),
			nn.Linear(1200, self.z_dim * 2)
		)
		self.decoder = nn.Sequential(
			nn.Linear(z_dim, 1200),
			nn.Tanh(),
			nn.Linear(1200, 1200),
			nn.Tanh(),
			nn.Linear(1200, 1200),
			nn.Tanh(),
			nn.Linear(1200, self.sz * self.sz),
			View((-1, 1, self.sz, self.sz))
		)

		self.weight_init()
		self.to(device)

	def weight_init(self):
		for block in self._modules:
			for m in self._modules[block]:
				kaiming_init(m)

	def loss_fn(self, x, x_recon, params):
		mu = params[0]
		logvar = params[1]
		batch_size = mu.size(0)

		recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(batch_size)
		KL = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
		KL_loss = KL.sum(1).mean(0, True)[0]

		return recon_loss, KL_loss

	def encode(self, x):
		distributions = self.encoder(x)
		mu = distributions[:, :self.z_dim]
		logvar = distributions[:, self.z_dim:]

		std = torch.exp(logvar.mul(0.5))
		eps = torch.randn_like(std)

		z = mu + std*eps

		return z, (mu, logvar)

	def forward(self, x):
		z, params = self.encode(x)
		x_recon = self.decoder(z)

		if self.training == False:
			x_recon = torch.sigmoid(x_recon)
		
		return x_recon, params

class BetaVAE_Gauss(nn.Module):
	"""Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

	def __init__(self, device, z_dim=10, nc=1):
		super(BetaVAE_Gauss, self).__init__()
		self.z_dim = z_dim
		self.nc = nc
		self.cached = False
		self.encoder = nn.Sequential(
			nn.Conv2d(self.nc, 32, 4, 2, 1),
			nn.ReLU(True),
			nn.Conv2d(32, 32, 4, 2, 1),
			nn.ReLU(True),
			nn.Conv2d(32, 64, 4, 2, 1),
			nn.ReLU(True),
			nn.Conv2d(64, 64, 4, 2, 1),
			nn.ReLU(True),
			nn.Conv2d(64, 256, 4, 1),
			nn.ReLU(True),
			View((-1, 256*1*1)),
			nn.Linear(256, z_dim*2)
		)
		self.decoder = nn.Sequential(
			nn.Linear(z_dim, 256),
			View((-1, 256, 1, 1)),
			nn.ReLU(True),
			nn.ConvTranspose2d(256, 64, 4),
			nn.ReLU(True),
			nn.ConvTranspose2d(64, 64, 4, 2, 1),
			nn.ReLU(True),
			nn.ConvTranspose2d(64, 32, 4, 2, 1),
			nn.ReLU(True),
			nn.ConvTranspose2d(32, 32, 4, 2, 1),
			nn.ReLU(True),
			nn.ConvTranspose2d(32, nc, 4, 2, 1)
		)

		self.weight_init()
		self.to(device)

	def weight_init(self):
		for block in self._modules:
			for m in self._modules[block]:
				kaiming_init(m)

	def loss_fn(self, x, x_recon, params):
		mu = params[0]
		logvar = params[1]
		batch_size = mu.size(0)
	
		# Reconstruction loss
		recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum').div(batch_size)
		# KL divergence
		KL = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
		KL_total = KL.sum(1).mean(0, True)

		return recon_loss, KL_total

	def encode(self, x):
		distributions = self.encoder(x)
		mu = distributions[:, :self.z_dim]
		logvar = distributions[:, self.z_dim:]

		std = torch.exp(logvar.mul(0.5))
		eps = torch.randn_like(std)

		z = mu + std*eps

		return z, (mu, logvar)


	def forward(self, x):
		z, params = self.encode(x)
		x_recon = torch.sigmoid(self.decoder(z))

		return x_recon, params


def kaiming_init(m):
	if isinstance(m, (nn.Linear, nn.Conv2d)):
		init.kaiming_normal_(m.weight)
		if m.bias is not None:
			m.bias.data.fill_(0)
	elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
		m.weight.data.fill_(1)
		if m.bias is not None:
			m.bias.data.fill_(0)

from tqdm import tqdm

def train(train_dl, model, optim, beta, epochs, scheduler = None):
	model.train()

	nm_iters = int(len(train_dl.dataset) / train_dl.batch_size)
	for e in range(epochs):
		pbar = tqdm(total=nm_iters)
		train_loss = 0
		for i, x in enumerate(train_dl):
			x_recon, params = model(x)
			recon_loss, kl_total = model.loss_fn(x, x_recon, params)
			
			beta_vae_loss = (recon_loss + beta*kl_total)

			optim.zero_grad()
			beta_vae_loss.backward()
			optim.step()

			loss = beta_vae_loss.item()
			train_loss += loss
			pbar.update(1)
			
		if scheduler is not None:
			scheduler.step()
		pbar.write('[{} - {}/{}] loss:{:.3f}'.format(e, i, nm_iters, loss))
		pbar.close()
		print('')

	return train_loss / nm_iters
