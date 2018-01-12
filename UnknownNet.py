import torch
from torch import nn
from torchvision.models import resnet
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from Cell import ResCell


class UnknownNet(nn.Module):

	# node number for each cell, it should include a unknown node
	node_num = 2

	def __init__(self):
		super(UnknownNet, self).__init__()

		# build pre-process network
		conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=3, padding=1, bias=False)
		bn1 = nn.BatchNorm2d(16)
		relu1 = nn.ReLU(inplace=True)
		maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
		self.pre = nn.Sequential(conv1, bn1, relu1, maxpool1)
		self.pre_sz = list(self.pre(Variable(torch.rand(1,3,84,84))).size())
		print('pre net size:', self.pre_sz) # (1,64,27,27)

		# save the tree information for cells
		mod = ResCell(self.pre_sz, self.node_num).cuda()
		self.add_module('node:0', mod)
		self.tree = [mod] # append the first cell
		self.table = [-1] * self.node_num # all initialized as unknown
		self.table_p = 0 # record current label assignment position


		self.criteon = nn.CrossEntropyLoss()

		print('current network:', self)

	def predict(self, input):
		assert input.size(0) == 1

		x = self.pre(input)
		rout = x
		prob = [0] * self.node_num

		for cellctr, cell in enumerate(self.tree):
			rin = torch.cat([x, rout], dim = 1)

			rout, logits = cell(rin)
			prob = F.softmax(logits, dim = 1)
			_, cellidx = torch.max(prob, dim = 1)
			cellidx = cellidx.data[0]

			if cellidx % self.node_num != (self.node_num - 1): # not pointer to the last
				return [self.table[cellctr * self.node_num + cellidx]], prob

			# else continue to next cell

		# if it iterates to the end unknown node, it's unknown actually.
		return [-1], prob


	def forward(self, input, label, train = True):
		"""
		label in network: train and backprop
		label new to network: assign label and train, backprop
		label as unknown: label = -1 and should pred as -1
		currently only support 1-element training.
		:param input: [1, c, h, w]
		:param label: [1]
		:return:
		"""
		assert input.size(0) == 1

		# [1, c_, h_, w_] => pre_sz: [1, c, h, w]
		x = self.pre(input)
		# for the first cell, set rout = x
		rout = x

		## check whether new label existed in label.
		# NOTICE: for the new label, if its label is -1, that means it should be treated as unknown and will not be added into current table.
		label = label[0]
		if label != -1 and label not in self.table: # known and not added into current table
			if self.table_p != 0 and self.table_p % self.node_num == (self.node_num - 1): # when current table_p point to unknown node, the last node is 6-1
				self.table_p += 1 # skip unknown node
				mod = ResCell(self.pre_sz, self.node_num).cuda()
				self.add_module('node:%d'%len(self.tree), mod)
				self.tree.append(mod) # append new cell
				self.table.extend([-1] * self.node_num) # initialize new cell's table

				print('add new cell, current network:', self)
			# now write down label info
			self.table[self.table_p] = label
			self.table_p += 1

		## now forward recursively
		# the forward process will terminate early if the appropriate condition triggered.
		for cellctr, cell in enumerate(self.tree):
			# [1, c1, h, w] cat with [1, c2, h, w]
			rin = torch.cat([x, rout], dim = 1)
			# print('cell:', cellctr, 'rin:',rin.size(), 'rout:', rout.size())
			# logits: [1, node_num], rout: [1, c2, h, w]
			rout, logits = cell(rin)
			# prob: [1, 6]
			prob = F.softmax(logits, dim= 1)
			# get pred
			_, cellidx = torch.max(prob, dim = 1)
			cellidx = cellidx.data[0] # [1] => scalar
			# get its real label info
			pred = self.table[cellctr * self.node_num + cellidx]

			"""
			return loss:
				1. pred is unknown, but label is known and in currently layer
				2. pred is known, but label is unknown
				3. pred is known and label is in current cell as well
				4. pred is known, but label is not in current cell
				
			go deeper:
				1. pred is unknown, label is known but not in current cell
				
			train former cell:
				none
				
			The forward process will terminate once it reach the right situation. What I mean the right sitation is the 
			either it predicts right or the loss function can be calucuated already.
			once it forward process need go deeper as the case: pred is unknown and label also not in current cell.
			In this case, currect cell make a right decision and the network need forward utils it read a judgable stage.
				
			"""
			# 1. check pred in known node or unknown node
			# for unknown pred, if label is unknown, then loss = 0, if label is known, it should go deeper.
			if pred == -1: # 1. if pred is unknown, the pred can occur not only in last node, but also in other -1 position, we treat it as equal
				if label == -1: # 1.1 pred is unknow and groundtruth is unknown
					loss = self.criteon(prob, Variable(torch.LongTensor([cellidx])).cuda())
					return loss, prob
				# check the groundtruth whether in current cell table, then decide whether to go ahead
				elif label in self.table[cellctr * self.node_num : cellctr * self.node_num + self.node_num]:
					# 1.2 pred is unknown, label is known node in current cell
					# HERE to stop going deeper, since its label resides in currently layer
					# we should train current cell only
					labelidx = self.table.index(label)
					labelidx = labelidx % self.node_num # offset in current cell
					loss = self.criteon(prob, Variable(torch.LongTensor([labelidx])).cuda())
					return loss, prob
				else: # 1.3 pred is unknown, label is known but not in current layer, that's, in next layer.
					# in this case, it must have next cell.
					continue # just forward


			else: # 2.pred is known
				# 2.1 pred is known, label is unknown, node last.
				# just train curent cell
				if label == -1:
					loss = self.criteon(prob, Variable(torch.LongTensor([self.node_num - 1])).cuda())
					return loss, prob
				# 2.2 pred is known and label is in current cell as well.
				# just use cross-entory to calculate its loss
				# its loss maybe 0 or not , depending on its label
				elif label in self.table[cellctr * self.node_num : cellctr * self.node_num + self.node_num]:
					# pred is known, groundtruth in current cell and groundtruth is known
					labelidx = self.table.index(label)
					labelidx = labelidx % self.node_num
					loss = self.criteon(prob, Variable(torch.LongTensor([labelidx])).cuda())
					return loss, prob
				# 2.3 pred is known, but label is not in current cell.
				else:
					# should train current cell
					# pred is known, groundtruth is NOT in current cell and groundtruth is known
					labelidx = self.table.index(label)
					labelidx = labelidx % self.node_num
					loss = self.criteon(prob, Variable(torch.LongTensor([self.node_num - 1])).cuda())
					return loss, prob



if __name__ == '__main__':
	from MiniImgOnDemand import MiniImgOnDemand
	import random
	import numpy as np

	db = MiniImgOnDemand('../mini-imagenet/', type='train')
	net = UnknownNet().cuda()

	optimizer = optim.Adam(net.parameters(), lr= 2e-5)

	model_parameters = filter(lambda p: p.requires_grad, net.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('total params:', params)

	num_cls = 64
	for label_status in range(num_cls):
		step = 0
		accuracy = 0
		total_loss = 0

		while accuracy < 0.9:
			label = np.random.choice([label_status, num_cls-1], 1, p = [0.5, 0.5])
			if label == label_status: # select training data
				label = random.randint(0, label_status)
				img = Variable(db.get(label).unsqueeze(0)).cuda()
				loss, prob = net(img, [label])
			else: # select unknown data
				label = random.randint(label_status + 1, num_cls - 1)
				img = Variable(db.get(label).unsqueeze(0)).cuda()
				loss, prob = net(img, [-1])

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			step += 1
			total_loss += loss.data[0]

			if step % 100 == 0:
				print('current progress:',label_status, 'step:', step, 'loss:', total_loss)
				total_loss = 0

			if step % 500 == 0:
				total = 100
				right = 0
				for i in range(total):
					label_test = np.random.choice([label_status, num_cls-1], 1, p = [0.5, 0.5])
					if label_test == label_status: # select training data
						label_test = random.randint(0, label_status)
						img = Variable(db.get(label_test).unsqueeze(0)).cuda()
						pred, prob = net.predict(img) # [1, 6]
						pred = pred[0]
						if pred == label_test:
							right += 1
					else: # select unknown data
						label_test = random.randint(label_status + 1, num_cls - 1)
						img = Variable(db.get(label_test).unsqueeze(0)).cuda()
						pred, prob = net.predict(img) # [1, 6]
						pred = pred[0]
						if pred == -1:
							right += 1

				accuracy = right / total
				print('current progress:', label_status, 'step:', step, 'accuracy:', accuracy)




