import torch
import torch.nn as nn
import random
import numpy as np

class LinearClassifier(nn.Module):
	def __init__(self, device, n=5):
		super(LinearClassifier, self).__init__()
		self.n = n
		self.classifier = nn.Sequential(
			nn.Linear(self.n, 6),
			nn.Softmax(dim=1)
		)
		self.to(device)

	def forward(self, x):
		y = self.classifier(x)

		return y

def compute_new_index(i, v_old, v_new):
	dims = np.array([1, 3, 6, 40, 32, 32])
	unraveled_v_old = list(np.unravel_index(v_old, dims))
	unraveled_v_new = np.unravel_index(v_new, dims)
	unraveled_v_old[i] = unraveled_v_new[i]
	
	return np.ravel_multi_index(unraveled_v_old, dims)

def disentanglement_epoch(device, model, classifier, data, imgs, batch, i):
	i1 = np.array(random.sample(range(len(data)), batch))
	# v1 = data[i1]

	i2 = np.array(random.sample(range(len(data)), batch))
	i2 = compute_new_index(i, i2, i1)
	# v2 = data[i2]
	with torch.no_grad():
		if model.cached == True:
			x1 = model.encoded[i1].to(device)
			x2 = model.encoded[i2].to(device)
			m1 = x1[:, :model.z_dim]
			m2 = x2[:, :model.z_dim]
		else:
			x1 = model.encoder(torch.from_numpy(imgs[i1]).unsqueeze(1).float().to(device))
			m1 = x1[:, :model.z_dim]
			x2 = model.encoder(torch.from_numpy(imgs[i2]).unsqueeze(1).float().to(device))
			m2 = x2[:, :model.z_dim]

	z_diff = torch.abs(m1 - m2).mean(dim=0).unsqueeze(0)

	return classifier(z_diff)

def disentanglement(device, model, imgs, data, zs, epochs=10, batch=256):
	classifier = LinearClassifier(device, model.z_dim)
	optim = torch.optim.Adam(classifier.parameters(), lr=1e-3 * 0.5)
	loss = torch.nn.NLLLoss()

	for e in range(epochs):
		for b in range(len(imgs) // batch):
			i = random.sample(zs, 1)[0]
			output = loss(disentanglement_epoch(device, model, classifier, data, imgs, batch, i), torch.full((1,), i).to(device))
			optim.zero_grad()
			output.backward()
			optim.step()

	with torch.no_grad():
		corr_predictions = 0
		tot_predictions = 0

		for b in range(2 * len(imgs) // batch):
			i = random.sample(zs, 1)[0]
			output = disentanglement_epoch(device, model, classifier, data, imgs, batch, i)
			pred = output.data.max(1, keepdim=True)[1]
			corr_predictions += pred.eq(torch.full((batch, 1), i).to(device)).sum()
			tot_predictions += batch

		accuracy = corr_predictions / tot_predictions

		return accuracy.item()


