import numpy as np
import torch
from kldivergence import compute_latent_features, compute_kl_divergence

if __name__ == '__main__':
  z_dim = 32
  D_KL = 0
  f = 'latents-2.npy'
  z = torch.from_numpy(np.load(f, encoding='bytes'))
  for k in range(z_dim):
    t = compute_kl_divergence(z, k, batch=2048)
    D_KL += t
  
  print(f, D_KL / z_dim)
