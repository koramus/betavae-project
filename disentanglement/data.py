import torch

class Device:
	def __init__(self):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if self.device == "cuda":
			torch.backends.cudnn.enabled = True
			torch.backends.cudnn.allow_tf32 = True
			torch.backends.cudnn.deterministic = False
			torch.backends.cudnn.benchmark = True

	def get(self):
		return self.device

class Dataloader:
	def __init__(self, _device, _dataset, _batch_size, _shuffle=True, _num_workers=1, _pin_memory=True):
		self.device = _device
		self.dataset = _dataset
		self.batch_size = _batch_size
		self.dataloader = torch.utils.data.DataLoader(_dataset,
			batch_size=_batch_size, shuffle=_shuffle, num_workers=_num_workers, pin_memory=_pin_memory, drop_last=False, persistent_workers=True, prefetch_factor=6)

	def __iter__(self):
		for data in self.dataloader.__iter__():
			yield data.unsqueeze(1).float().to(self.device, non_blocking=True)
	
	def __len__(self):
		return len(self.dataset)
