import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import copy
from utils import clip

def fgsm(model, ct, x, y, lr=0.1, eps=0.1, niter=1):

	'''
	Adapted from https://github.com/kawine/atgan/tree/master/attacks
	'''
	x0 = x.clone()
	x, y = Variable(x, requires_grad=True), Variable( y )
	model.eval()

	if x.grad is not None:
		x.grad.data.zero_()
	y_ = model(x)
	loss = ct(y_, y)
	loss.backward()

	dx = x.grad

	adv = x0 + eps*torch.sign(dx)
	adv = torch.clamp(adv, 0.0, 1.0)

	return adv.data[0]

def igsm(model, ct, x, y, lr=0.1, eps=0.1, niter=1):

	'''
	Adapted from https://github.com/leiwu0308/adversarial.example
	'''
	x0 = x.clone()
	x, y = Variable(x, requires_grad=True), Variable( y )
	model.eval()

	for i in range(niter):
		if x.grad is not None:
			x.grad.data.zero_()
		y_ = model(x)
		loss = ct(y_, y)
		loss.backward()

		dx = x.grad
		x.add_(lr, dx.sign())
		x = clip(x.data, x0, eps)

	return x.data[0]

def deepfool(model, x, num_classes=None, overshoot=0.02, max_iter=50):

	"""
	Adapted from https://github.com/LTS4/DeepFool
	"""

	x0 = Variable(x, requires_grad=True)

	f_image = model.forward(x0)

	I = f_image.data.cpu().numpy().flatten().argsort()[::-1]

	if num_classes:
		I = I[0:num_classes]
	label = I[0]

	pert_image = x0.copy()
	w = torch.zeros_like(x0)
	r_tot = torch.zeros_like(x0)

	loop_i = 0

	x = Variable(pert_image, requires_grad=True)
	fs = model(x)
	k_i = label

	while k_i == label and loop_i < max_iter:

		pert = np.inf
		fs[0, I[0]].backward(retain_graph=True)
		grad_orig = x.grad.copy()

		for k in range(1, num_classes):
			zero_gradients(x)

			fs[0, I[k]].backward(retain_graph=True)
			cur_grad = x.grad.copy()

			# set new w_k and new f_k
			w_k = cur_grad - grad_orig
			f_k = (fs[0, I[k]] - fs[0, I[0]])

			pert_k = f_k.abs()/w_k.norm(2)

			# determine which w_k to use
			if pert_k.data[0] < pert.data[0]:
				pert = pert_k
				w = w_k

		# compute r_i and r_tot
		# Added 1e-4 for numerical stability
		r_i = (pert + 1e-4) * w / w.norm(2)
		r_tot = r_tot + r_i

		pert_image = image + (1 + overshoot)*r_tot

		x = pert_image.copy()
		fs = model.forward(x)
		_, k_i = fs.max(1)[0].data[0]

		loop_i += 1

	r_tot = (1+overshoot)*r_tot

	return pert_image.data

def c_w_L2(model, x, y, confidence=0, learning_rate=1e-3, binary_search_steps=5, max_iterations=1000, initial_const=1, num_classes=10, clip_min=-1, clip_max=1, abort_early = True):
	'''
	Adapted from https://github.com/kawine/atgan/blob/master/attacks/attacks.py
	'''

	x0 = x.clone()
	label = y.clone()

	# re-scale instances to be within [0, 1]
	x0 = (x0 - clip_min) / (clip_max - clip_min)
	x0 = torch.clamp(x0, 0., 1.)
	# now convert to [-1, 1]
	x0 = (x0 * 2) - 1
	# convert to tanh-space
	x0 = x0 * .999999
	x0 = (torch.log((1 + x0) / (1 - x0))) * 0.5 # arctanh		### CHECK

	lower_bound = 0.0
	scale_const = initial_const					### CHECK
	upper_bound = 1e10

	pertubation = torch.zeros_like(x)

	one_hot_labels = torch.zeros( (1, num_classes) )
	one_hot_labels[0, label[0]] = 1.

	label = Variable(one_hot_labels, requires_grad=False)
	pertubation = Variable(pertubation, requires_grad=True)

	if x.is_cuda():
		label = label.cuda()
		pertubation = pertubation.cuda()

	optimizer = optim.Adam(pertubation, lr=learning_rate)

	for outer_step in range(binary_search_steps):
		best_loss = np.inf
		best_attack = None
		best_pred = None

		pertubation.data.zero_()

		# last iteration (if we run many steps) repeat the search once
		if binary_search_steps >= 10 and outer_step == binary_search_steps - 1:
			scale_const = upper_bound

		scale_const_tensor = torch.from_numpy(scale_const).float()	# .float() needed to conver to FloatTensor
		scale_const_var = Variable(scale_const_tensor.cuda() if self.cuda else scale_const_tensor, requires_grad=False)

		prev_loss = np.inf 	# for early abort

		for step in range(max_iterations):

			x_pert = (torch.tanh(pertubation + x0) + 1) * 0.5
			x_pert = x_pert * (clip_max - clip_min) + clip_min

			# outputs BEFORE SOFTMAX
			predicted = model(inputs_adv)

			# before taking the L2 distance between the original and perturbed inputs,
			# transform the original inputs in the same way (arctan, then clip)
			unmodified = (torch.tanh(x0) + 1) * 0.5
			unmodified = unmodified * (clip_max - clip_min) + clip_min
			loss2 = reduce_sum((x_pert - unmodified)**2)
			loss2 = loss2.sum()

			# compute probability of label class and maximum other
			real = predicted[label.data[0]]
			other = predicted.sum() - real

			# the greater the likelihood of label, the greater the loss
			loss1 = torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
			loss1 = torch.sum(scale_const * loss1)
			loss = loss1 + loss2

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# abort early if loss is too small
			if  abort_early and step % (max_iterations // 10) == 0:
				if loss > prev_loss * 0.9999:
					break

				prev_loss = loss


			prediction = predicted.data.cpu().numpy()

			prediction[label] += confidence
			y_hat = np.argmax(prediction)

			success = True if y_hat != label.data[0] else False

			# if smaller perturbation and still different predicted class ...
			if loss2 < best_loss and success:
				best_l2 = dist
				best_pred = y_hat
				best_attack = pertubation

		if best_pred is not None and best_pred != label.data[0]:
			# successful, do binary search and divide const by two
			upper_bound = min(upper_bound, scale_const)

			if upper_bound < 1e9:
				scale_const = (lower_bound + upper_bound) / 2
		else:
			# failure, multiply by 10 if no solution found
			# or do binary search with the known upper bound
			lower_bound = max(lower_bound, scale_const)
			upper_bound = (lower_bound + upper_bound) / 2 if (upper_bound < 1e9) else (scale_const * 10)

	return x+best_attack

	def _compare(self, prediction, label):
		"""
		Return True if label is not the most likely class.
		If there is a prediction for each class, prediction[label] should be at least
		self.confidence from being the most likely class.
		"""
		if not isinstance(prediction, (float, int, np.int64)):
			prediction = np.copy(prediction)
			prediction[label] += self.confidence
			prediction = np.argmax(prediction)

		return prediction != label

	def _optimize(self, model, optimizer, modifier, inputs, labels, scale_const):
		"""
		Calculate loss and optimize for modifier here. Return the loss, adversarial inputs,
		and predicted classes. Since the attack is untargeted, aim to make label the least
		likely class.
		modifier is the variable we're optimizing over (w in the paper).
		Don't think of it as weights in a NN; there is a unique w for each x in the batch.
		"""
		inputs_adv = (torch.tanh(modifier + inputs) + 1) * 0.5
		inputs_adv = inputs_adv * (self.clip_max - self.clip_min) + self.clip_min
		# outputs BEFORE SOFTMAX
		predicted = model(inputs_adv)

		# before taking the L2 distance between the original and perturbed inputs,
		# transform the original inputs in the same way (arctan, then clip)
		unmodified = (torch.tanh(inputs) + 1) * 0.5
		unmodified = unmodified * (self.clip_max - self.clip_min) + self.clip_min
		dist = L2_dist(inputs_adv, unmodified)
		loss2 = dist.sum()

		# compute probability of label class and maximum other
		real = (labels * predicted).sum(1)
		other = ((1. - labels) * predicted - labels * 10000.).max(1)[0]

		# the greater the likelihood of label, the greater the loss
		loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
		loss1 = torch.sum(scale_const * loss1)
		loss = loss1 + loss2

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# convert to numpy form before returning it
		loss = loss.data.cpu().numpy()[0]
		dist = dist.data.cpu().numpy()
		predicted = predicted.data.cpu().numpy()
		# inputs_adv = inputs_adv.data.permute(0, 2, 3, 1).cpu().numpy()

		return loss, dist, predicted, inputs_adv
