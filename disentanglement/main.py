from model import PCA, BetaVAE_Bernoulli, BetaVAE_Gauss, train
from data import Device, Dataloader
from generate import random_samples, variable_traversal
from disentanglement import disentanglement
from kldivergence import compute_latent_features, compute_kl_divergence
from scheduler import Output

import numpy as np
import torch
import datetime
import sys
import getopt
import random

import matplotlib.pyplot as plt

BVAEs = {
  'b': BetaVAE_Bernoulli,
  'g': BetaVAE_Gauss
}

BVAE = 'b'
TRAIN_BATCH = 1024
BETA = 5.0
DIMS = 6
EPOCHS = 100
LR = 1e-4 * 5.0
S_SZ = 20
S_MUL = 0.2

rc = {"axes.spines.left" : False,
			"axes.spines.right" : False,
			"axes.spines.bottom" : False,
			"axes.spines.top" : False,
			"xtick.bottom" : False,
			"xtick.labelbottom" : False,
			"ytick.labelleft" : False,
			"ytick.left" : False}
plt.rcParams.update(rc)

if __name__ == '__main__':
  cmd_line_opts, cmd_line_args = getopt.getopt(sys.argv[1:], 'm:rtae:l:d:b:kpo')

  action = 'train'
  model = None
  device = Device().get()
  output = Output()

  for opt in cmd_line_opts:
    if '-m' in opt:
      DIMS = int(opt[1][:-3].split('_')[-1])
      BVAE = opt[1][0]
      model = BVAEs[BVAE](device, z_dim=DIMS)
      model.load_state_dict(torch.load('./models/' + opt[1]))
    elif '-r' in opt:
      action = 'random-sample'
    elif '-t' in opt:
      action = 'traverse-sample'
    elif '-a' in opt:
      action = 'disentanglement-accuracy'
    elif '-k' in opt:
      action = 'kl-divergence'
    elif '-p' in opt:
      action = 'pca'
    elif '-o' in opt:
      action = 'pca-kl'
    elif '-e' in opt:
      EPOCHS = int(opt[1])
    elif '-l' in opt:
      LR = float(opt[1])
    elif '-d' in opt:
      DIMS = int(opt[1])
    elif '-b' in opt:
      BETA = float(opt[1])
    elif '-g' in opt:
      BVAE = 'g'

  if model is None:
    model = BVAEs[BVAE](device, z_dim=DIMS)

  data_file = np.load('data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding='bytes')

  if action == 'train':
    train_dl = Dataloader(device, data_file['imgs'], TRAIN_BATCH)
    
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, S_SZ, S_MUL, verbose=True)
    name = '{}_{}_b{}_{}.tm'.format(BVAE, datetime.datetime.now().strftime('%m%d_%H_%M'), BETA, DIMS)

    try:
      loss = train(train_dl, model, optim, BETA, EPOCHS, scheduler)
      torch.save(model.state_dict(), './models/' + name)

      output.print('model', name)
      output.print('loss', loss)
    except KeyboardInterrupt:
      s = input('Do you want to save the model? ')
      if s == 'y':
        torch.save(model.state_dict(), './models/' + name)

  elif action == 'random-sample':
    random_samples(device, model, data_file['imgs'], 8)
  elif action == 'traverse-sample':
    for i in range(model.z_dim):
      variable_traversal(device, model, i, 16)
  elif action == 'disentanglement-accuracy':
    accuracy = disentanglement(device, model, data_file['imgs'], data_file['latents_classes'], [2,3,4,5])
    output.print('disentanglement', accuracy)
  elif action == 'kl-divergence':
    dl = Dataloader(device, data_file['imgs'], TRAIN_BATCH)
    z = compute_latent_features(model, dl)
    D_KL = 0
    for k in range(model.z_dim):
      D_KL += compute_kl_divergence(z, k)
    output.print('kldivergence', D_KL / model.z_dim)
  elif action == 'pca':
    model = PCA(device, torch.from_numpy(data_file['imgs']), z_dim=6)
    accuracy = disentanglement(device, model, data_file['imgs'], data_file['latents_classes'], [2,3,4,5])
    output.print('disentanglement', accuracy)
  elif action == 'pca-kl':
    model = PCA(device, torch.from_numpy(data_file['imgs']), z_dim=6)
    D_KL = 0
    for k in range(model.z_dim):
      D_KL += compute_kl_divergence(model.encoded, k)
    output.print('kldivergence', D_KL / model.z_dim)
      
  output.save()