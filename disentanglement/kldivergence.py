import torch
import numpy as np
import random
import math

def _matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
	vals, vecs = torch.eig(matrix, eigenvectors=True)
	vals = torch.view_as_complex(vals.contiguous())
	vals_pow = vals.pow(p)
	vals_pow = torch.view_as_real(vals_pow)[:, 0]
	matrix_pow = torch.matmul(vecs, torch.matmul(torch.diag(vals_pow), torch.inverse(vecs)))
	return matrix_pow

def white_denoiser_estimator(s1, s2, k):
	n, m = len(s1), len(s2)
	d = float(s1.shape[1])

	data_mean = (s1.sum(dim=0) + s2.sum(dim=0)) / (n+m)
	s1_avg = s1 - data_mean
	c1 = torch.bmm(s1_avg.unsqueeze(2), s1_avg.unsqueeze(1)).sum(dim=0)
	s2_avg = s2 - data_mean
	c2 = torch.bmm(s2_avg.unsqueeze(2), s2_avg.unsqueeze(1)).sum(dim=0)

	C = _matrix_pow((c1 + c2) / (n + m - 1), -1/2)

	s1 = s1_avg @ C
	s2 = s2_avg @ C

	s1_per_point = s1.tile((n, 1, 1))
	s2_per_point = s2.tile((m, 1, 1))
	k_i = k
	l_i = k
	s2_norms_per_point = torch.norm(s2_per_point.transpose(0, 1) - s1, dim=2)
	nu_per_point = torch.topk(s2_norms_per_point, k_i, largest=False)[0][:, -1]
	s1_norms_per_point = torch.norm(s1_per_point.transpose(0, 1) - s1, dim=2)
	rho_per_point = torch.topk(s1_norms_per_point, l_i + 1, largest=False)[0][:, -1]
	D = torch.log(nu_per_point / (rho_per_point + 1e-20) + 1e-20).mean() * d + np.log(m / (n - 1))
	
	return D.item()

def compute_latent_features(model, dl):
	model.eval()
	r = None

	with torch.no_grad():
		for batch in dl:
			z, params = model.encode(batch)
			r = z if r is None else torch.cat((r, z), 0)

	return r

def compute_kl_divergence(z, i, batch=4096, e=1000, k=2):
	z_i = z[:, i]
	sorted_indices = torch.argsort(z_i)
	sorted_z = z[sorted_indices]
	D_KL = 0

	mask = [True for i in range(sorted_z.shape[1])]
	mask[i] = False
	for i in range(e):
		n1 = random.randint(batch // 2 - 1, len(z) - batch // 2)
		n2 = random.randint(batch // 2 - 1, len(z) - batch // 2)

		s1 = sorted_z[n1 - batch // 2 - 1:n1 + batch // 2 - 1, mask].cuda()
		s2 = sorted_z[n2 - batch // 2 - 1:n2 + batch // 2 - 1, mask].cuda()

		if s1.shape[0] == s2.shape[0]:
			D_KL += white_denoiser_estimator(s1, s2, k)

	return D_KL / e

