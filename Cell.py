import torch
from torch import nn
import math
from torchvision.models import resnet
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable


class ResCell(nn.Module):
	# size: (1, 64, 74, 74)
	def __init__(self, size, node_num):
		super(ResCell, self).__init__()

		# from inplances
		_, c, h, w = size
		self.inplanes = size[1] * 2
		outc = size[1]

		self.cell = self._make_layer(resnet.BasicBlock, outc, 1)

		self.out = nn.Sequential(nn.Linear(c*h*w, 128),
		                         nn.ReLU(inplace=True),
		                         nn.Linear(128, node_num))

		self.table = [-1] * node_num # all unknown


		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
				          kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, rin):
		"""

		:param rin: [batch, c, h, w]
		:param rout:  [batch, c, h, w]
				logits:
		:return:
		"""
		# => [b, c, h, w], this is the feature for next cell to reuse
		rout = self.cell(rin)
		# [b, node_num]
		logits = self.out(rout.view(rout.size(0), -1))

		return rout, logits


if __name__ == '__main__':
	model = ResCell(3, 64, 2)
	print(model)
	print(model(Variable(torch.rand(1, 3, 24, 24))).size())
