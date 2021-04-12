import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

def variable_traversal(device, model, i, steps=10, start=-2, end=2):
	model.eval()

	with torch.no_grad():
		v = torch.linspace(start, end, steps)
		z = torch.tile(torch.randn((model.z_dim)), (steps, 1))
		z[:, i] = v

		img_recon = torch.sigmoid(model.decoder(z.to(device)))

		_, axs = plt.subplots(1, steps)
		axs = axs.flatten()
		for img, ax in zip(img_recon.cpu(), axs):
				ax.imshow(img.squeeze())
		plt.show()

def random_samples(device, model, imgs, n=2):
	model.eval()

	with torch.no_grad():
		i = np.array(random.sample(range(len(imgs)), n*n))
		data = imgs[i]
		xs = torch.from_numpy(data).unsqueeze(1).float().to(device)
		data_recon, mu = model(xs)

		fig = plt.figure()
		outer = gridspec.GridSpec(1, 2, wspace=0.2)

		for i, imgs in enumerate([data, data_recon.cpu()]):
			inner = gridspec.GridSpecFromSubplotSpec(n, n, subplot_spec=outer[i], wspace=0.05, hspace=0.05)

			for j, img in zip(range(n*n), imgs):
				ax = plt.Subplot(fig, inner[j])
				ax.imshow(img.squeeze())
				fig.add_subplot(ax)
					
		fig.show()
		plt.show()
