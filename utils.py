import torch
from torch.autograd import Variable

def clip(x, x0, eps):
	'''
	https://github.com/leiwu0308/adversarial.example
	'''
	dx = x - x0
	dx.clamp_(-eps, eps)
	x.copy_(x0).add_(1, dx).clamp_(0,1)
	return x
