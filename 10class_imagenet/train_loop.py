import torch
from torch.autograd import Variable
import torch.nn.init as init

import numpy as np
import pickle

import os
from glob import glob
from tqdm import tqdm

class TrainLoop(object):

	def __init__(self, model, optimizer, train_loader, valid_loader, patience, checkpoint_path=None, checkpoint_epoch=None, cuda=True):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, 'checkpoint.pt')
		self.cuda_mode = cuda
		self.model = model
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.history = {'train_loss': [], 'valid_loss': [], 'valid_er': []}
		self.total_iters = 0
		self.cur_epoch = 0
		self.its_without_improv = 0
		self.last_best_val_loss = float('inf')
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=patience, verbose=True, threshold=1e-4, min_lr=1e-6)

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))

	def train(self, n_epochs=1):

		while self.cur_epoch < n_epochs:
			print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
			train_iter = tqdm(enumerate(self.train_loader))

			train_loss = 0.0
			valid_loss = 0.0

			# Train step
			for t, batch in train_iter:
				new_train_loss = self.train_step(batch)
				train_loss += new_train_loss

			self.history['train_loss'].append(train_loss/(t+1))
			self.total_iters += 1

			# Validation

			tot_correct = 0.
			
			for t, batch in enumerate(self.valid_loader):
				new_valid_loss, correct = self.valid(batch)
				valid_loss += new_valid_loss
				tot_correct += correct

			self.history['valid_loss'].append(valid_loss/(t+1))
			self.history['valid_er'].append(1.-tot_correct/len(self.valid_loader.dataset))

			print('Total train loss: {}'.format(self.history['train_loss'][-1]))
			print('Total valid loss: {}'.format(self.history['valid_loss'][-1]))

			print('Current and best error rates on validation set, and best epoch: {}, {}, {}'.format(self.history['valid_er'][-1], np.min(self.history['valid_er']), 1+np.argmin(self.history['valid_er'])))

			self.scheduler.step(self.history['valid_er'][-1])

			self.cur_epoch += 1

			if self.history['valid_er'][-1] < self.last_best_val_loss:
				self.checkpointing()
				self.its_without_improv = 0
				self.last_best_val_loss = self.history['valid_er'][-1]

	def train_step(self, batch):

		self.model.train()

		self.optimizer.zero_grad()

		x, y = batch

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()

		x = Variable(x)
		y = Variable(y, requires_grad=False)

		out = self.model.forward(x)

		loss = torch.nn.functional.cross_entropy(out, y)

		loss.backward()

		self.optimizer.step()

		return loss.item()

	def valid(self, batch):

		self.model.eval()

		x, y = batch

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()

		x = Variable(x)
		y = Variable(y, requires_grad=False)

		out = self.model.forward(x)

		pred = out.data.max(1)[1]

		correct = pred.eq(y.data).cpu().sum().item()

		loss = torch.nn.functional.cross_entropy(out, y)

		return loss.item(), correct

	def checkpointing(self):

		# Checkpointing
		print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'optimizer_state': self.optimizer.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'cur_epoch': self.cur_epoch,
		'its_without_improve': self.its_without_improv,
		'last_best_val_loss': self.last_best_val_loss}
		torch.save(ckpt, self.save_epoch_fmt)

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt)
			# Load model state
			self.model.load_state_dict(ckpt['model_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			self.its_without_improv = ckpt['its_without_improve']
			self.last_best_val_loss = ckpt['last_best_val_loss']

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def print_grad_norms(self):
		norm = 0.0
		for params in list(self.model.parameters()):
			norm+=params.grad.norm(2).data[0]
		print('Sum of grads norms: {}'.format(norm))

	def check_nans(self):
		for params in list(self.model.parameters()):
			if np.any(np.isnan(params.data.cpu().numpy())):
				print('params NANs!!!!!')
			if np.any(np.isnan(params.grad.data.cpu().numpy())):
				print('grads NANs!!!!!!')
