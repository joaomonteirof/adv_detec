import torch
import torch.nn as nn
import torch.nn.functional as F

class resnet(nn.Module):
	def __init__(self, feat_model):
		super(resnet, self).__init__()

		self.features = feat_model

		self.fc = nn.Linear(2048, 10)

	def forward(self, x):
		x = self.features.conv1(x)
		x = self.features.bn1(x)
		x = self.features.relu(x)
		x = self.features.maxpool(x)
		x = self.features.layer1(x)
		x = self.features.layer2(x)
		x = self.features.layer3(x)
		x = self.features.layer4(x)
		x = self.features.avgpool(x)

		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x

class vgg(nn.Module):
	def __init__(self, feat_model):
		super(vgg, self).__init__()

		self.features = feat_model

		self.fc = nn.Sequential( nn.Linear(512*7*7, 4096),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(4096, 10) )
			

	def forward(self, x):
		x = self.features.features(x)

		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x
