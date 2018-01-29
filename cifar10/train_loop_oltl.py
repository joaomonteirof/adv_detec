import torch
from torch.autograd import Variable
import torch.nn.init as init

import numpy as np
import pickle

import os
from glob import glob
from tqdm import tqdm

class TrainLoop(object):

	def __init__(self, model1, model2, optimizer1, optimizer2, train_loader, valid_loader, checkpoint_path=None, checkpoint_epoch=None, cuda=True):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt = os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.cuda_mode = cuda
		self.model_1 = model1
		self.model_2 = model2
		self.optimizer_1 = optimizer1
		self.optimizer_2 = optimizer2
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.history = {'train_loss': [], 'valid_loss': [], 'valid_acc_1': [], 'valid_acc_2': []}
		self.total_iters = 0
		self.cur_epoch = 0
		self.its_without_improv_1 = 0
		self.last_best_val_loss_1 = -np.inf
		self.its_without_improv_2 = 0
		self.last_best_val_loss_2 = -np.inf
		self.last_best_val_loss = np.inf

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))
		else:
			self.initialize_params()

	def train(self, n_epochs=1, patience=10):

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

			tot_correct_1 = 0
			tot_correct_2 = 0
			
			for t, batch in enumerate(self.valid_loader):
				new_valid_loss, correct_1, correct_2 = self.valid(batch)
				valid_loss += new_valid_loss
				tot_correct_1 += correct_1
				tot_correct_2 += correct_2

			self.history['valid_loss'].append(valid_loss/(t+1))
			self.history['valid_acc_1'].append(tot_correct_1/len(self.valid_loader.dataset))
			self.history['valid_acc_2'].append(tot_correct_2/len(self.valid_loader.dataset))

			print('Total train loss: {}'.format(self.history['train_loss'][-1]))
			print('Total valid loss: {}'.format(self.history['valid_loss'][-1]))
			print('Accuracy on validation set on model 1: {}'.format(self.history['valid_acc_1'][-1]))
			print('Accuracy on validation set on model 2: {}'.format(self.history['valid_acc_2'][-1]))

			self.cur_epoch += 1

			if valid_loss < self.last_best_val_loss:
				self.last_best_val_loss = valid_loss
				self.checkpointing()

			if self.history['valid_acc_1'][-1] > self.last_best_val_loss_1:
				self.last_best_val_loss_1 = self.history['valid_acc_1'][-1]
				self.its_without_improv_1 = 0
			else:
				self.its_without_improv_1 += 1

			if self.history['valid_acc_2'][-1] > self.last_best_val_loss_2:
				self.last_best_val_loss_2 = self.history['valid_acc_2'][-1]
				self.its_without_improv_2 = 0
			else:
				self.its_without_improv_2 += 1

			if self.its_without_improv_1 > patience:
				self.its_without_improv_1 = 0
				self.update_lr(opt=1)

			if self.its_without_improv_2 > patience:
				self.its_without_improv_2 = 0
				self.update_lr(opt=2)

		# saving final models
		print('Saving final model...')
		self.checkpointing()

	def train_step(self, batch):

		self.model_1.train()
		self.model_2.train()

		self.optimizer_1.zero_grad()
		self.optimizer_2.zero_grad()

		x, y = batch

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()

		x = Variable(x)
		y = Variable(y, requires_grad=False)

		out1_log, out1 = self.model_1.forward_oltl(x)
		out2_log, out2 = self.model_2.forward_oltl(x)

		target_1 = Variable(out1.data, requires_grad=False)
		target_2 = Variable(out2.data, requires_grad=False)

		loss1 = torch.nn.functional.nll_loss(out1_log, y) + torch.nn.functional.kl_div(out1_log, target_2)
		loss2 = torch.nn.functional.nll_loss(out2_log, y) + torch.nn.functional.kl_div(out2_log, target_1) 
		loss = loss1+loss2

		loss.backward()

		self.optimizer_1.step()
		self.optimizer_2.step()

		return loss.data[0]

	def valid(self, batch):

		self.model_1.eval()
		self.model_2.eval()

		x, y = batch

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()

		x = Variable(x)
		y = Variable(y, requires_grad=False)

		out1_log, out1 = self.model_1.forward_oltl(x)
		out2_log, out2 = self.model_2.forward_oltl(x)

		target_1 = Variable(out1.data, requires_grad=False)
		target_2 = Variable(out2.data, requires_grad=False)

		loss1 = torch.nn.functional.nll_loss(out1_log, y) + torch.nn.functional.kl_div(out1_log, target_2)
		loss2 = torch.nn.functional.nll_loss(out2_log, y) + torch.nn.functional.kl_div(out2_log, target_1) 
		loss = loss1 + loss2

		pred_1 = out1_log.data.max(1)[1]
		correct_1 = pred_1.eq(y.data).cpu().sum()
		pred_2 = out2_log.data.max(1)[1]
		correct_2 = pred_2.eq(y.data).cpu().sum()

		return loss.data[0], correct_1, correct_2

	def checkpointing(self):

		# Checkpointing
		print('Checkpointing...')
		ckpt = {'model_1_state': self.model_1.state_dict(),
		'model_2_state': self.model_2.state_dict(),
		'optimizer_1_state': self.optimizer_1.state_dict(),
		'optimizer_2_state': self.optimizer_2.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'cur_epoch': self.cur_epoch,
		'its_without_improve_1': self.its_without_improv_1,
		'last_best_val_loss_1': self.last_best_val_loss_1,
		'its_without_improve_2': self.its_without_improv_2,
		'last_best_val_loss_2': self.last_best_val_loss_2,
		'last_best_val_loss': self.last_best_val_loss}
		torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt)
			# Load model state
			self.model_1.load_state_dict(ckpt['model_1_state'])
			self.model_2.load_state_dict(ckpt['model_2_state'])
			# Load optimizer state
			self.optimizer_1.load_state_dict(ckpt['optimizer_1_state'])
			self.optimizer_2.load_state_dict(ckpt['optimizer_2_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			self.its_without_improv_1 = ckpt['its_without_improve_1']
			self.last_best_val_loss_1 = ckpt['last_best_val_loss_1']
			self.its_without_improv_2 = ckpt['its_without_improve_2']
			self.last_best_val_loss_2 = ckpt['last_best_val_loss_2']
			self.last_best_val_loss = ckpt['last_best_val_loss']

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def update_lr(self, opt=1):
		if opt==1:
			for param_group in self.optimizer_1.param_groups:
				param_group['lr'] = max(param_group['lr']/10., 0.000001)
			print('updating lr_1 to: {}'.format(param_group['lr']))
		elif opt==2:
			for param_group in self.optimizer_2.param_groups:
				param_group['lr'] = max(param_group['lr']/10., 0.000001)
			print('updating lr_2 to: {}'.format(param_group['lr']))
		else:
			print('opt shoud be set to 1 for optimizer 1 or 2 for optimizer 2. Value received was {}'.format(opt))

	def print_grad_norms(self):
		norm = 0.0
		for params in list(self.model_1.parameters()):
			norm+=params.grad.norm(2).data[0]
		print('Sum of grads norms 1: {}'.format(norm))

		norm = 0.0
		for params in list(self.model_2.parameters()):
			norm+=params.grad.norm(2).data[0]
		print('Sum of grads norms 2: {}'.format(norm))

	def check_nans(self):
		for params in list(self.model_1.parameters()):
			if np.any(np.isnan(params.data.cpu().numpy())):
				print('params NANs!!!!!  1')
			if np.any(np.isnan(params.grad.data.cpu().numpy())):
				print('grads NANs!!!!!!  1')

		for params in list(self.model_2.parameters()):
			if np.any(np.isnan(params.data.cpu().numpy())):
				print('params NANs!!!!!  2')
			if np.any(np.isnan(params.grad.data.cpu().numpy())):
				print('grads NANs!!!!!!  2')

	def initialize_params(self):
		for layer in self.model_1.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal(layer.weight.data)
			elif isinstance(layer, torch.nn.BatchNorm2d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()

		for layer in self.model_2.modules():
			if isinstance(layer, torch.nn.Conv2d):
				init.kaiming_normal(layer.weight.data)
			elif isinstance(layer, torch.nn.BatchNorm2d):
				layer.weight.data.fill_(1)
				layer.bias.data.zero_()
