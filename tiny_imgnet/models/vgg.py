import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


cfg = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
	def __init__(self, vgg_name, soft=True):
		super(VGG, self).__init__()
		self.features = self._make_layers(cfg[vgg_name])
		self.classifier = nn.Linear(2048, 200)
		self.soft = soft

	def forward(self, x):
		out = self.features(x)
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		if self.soft:
			out = F.softmax(out, dim=1)
			return F.log_softmax(out, dim=1)
		else:
			return F.log_softmax(out, dim=1)

	def forward_oltl(self, x):
		out = self.features(x)
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		return F.log_softmax(out, dim=1), F.softmax(out, dim=1)

	def _make_layers(self, cfg):
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
						   nn.BatchNorm2d(x),
						   nn.ReLU(inplace=True)]
				in_channels = x
		layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		return nn.Sequential(*layers)
