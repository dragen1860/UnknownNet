import torch
from torch import nn
from torchvision.models import resnet18
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from Cell import ResCell
import pickle
import numpy as np
import random

class UnknownNet(nn.Module):

	# node number for each cell, it should include a unknown node
	node_num = 4

	def __init__(self):
		super(UnknownNet, self).__init__()

		# build pre-process network
		conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=3, padding=1, bias=False)
		relu1 = nn.ReLU(inplace=True)
		conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
		relu2 = nn.ReLU(inplace=True)
		# maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
		self.pre = nn.Sequential(conv1, relu1, conv2, relu2)


		# resnet = resnet18(pretrained=True)
		# modules = list(resnet.children())[:-3]
		# self.pre = nn.Sequential(*modules) # [c, d, d]

		self.pre_sz = list(self.pre(Variable(torch.rand(1, 3, 84, 84))).size())
		print('Pre net size:', self.pre_sz) # (1,64,27,27)

		## save the tree information for cells
		# create the first cell
		mod = ResCell(self.pre_sz, self.node_num).cuda()
		self.add_module('node:0', mod)
		self.tree = [mod] # append the first cell
		self.table = [-1] * self.node_num # all initialized as unknown
		self.table_p = 0 # record current label assignment position

		# this criteon is used for several purpose
		self.criteon = nn.CrossEntropyLoss()
		print('initial network:', self)

	def save_mdl(self, filename):
		"""
		As the network is highly dynamic, we need to known current learning status when saveing model.
		:param label_status:
		:param filename:
		:return:
		"""
		with open(filename, 'wb') as f:
			torch.save(self.state_dict(),f)
		with open(filename+'.meta', 'wb') as f:
			meta = {
				'table': self.table,
				'table_p': self.table_p
			}
			pickle.dump(meta, f)
			print('saved meta:', meta)
			print('saved model to :', filename, 'and meta:', filename + '.meta')

	def load_mdl(self, filename, optimizer):
		"""
		To recover the network, we need to learn current learning progress, that's: label_status
		You should call this function after __init__
		:param filename:
		:return:
		"""
		meta = filename + '.meta'
		with open(meta, 'rb') as f:
			meta = pickle.load(f)
			self.table_p = meta['table_p']
			self.table = meta['table']
			print('load meta:', meta)

		num = self.table_p // self.node_num + 1
		for i in range(num -1): # as we have one cell in __init__ stage.
			mod = ResCell(self.pre_sz, self.node_num).cuda()
			self.add_module('node:%d' % len(self.tree), mod)
			optimizer.add_param_group({'params': mod.parameters()})
			self.tree.append(mod)  # append new cell
		print('recover network:', self)

		with open(filename, 'rb') as f:
			self.load_state_dict(torch.load(f))
			print('loaded weights done.')

		# return the maximum label in current table to indicate the learner should learn
		# from this label_status next label
		return np.array(self.table).max() + 1






	def predict(self, input):
		"""

		:param input: [1, c, h, w]
		:return: predicted label, not index. and its prob distribution
		"""
		assert input.size(0) == 1

		# push to the pre-process network to get feature
		x = self.pre(input)
		# since no resnet out is avaliable, just use x as rout
		rout = x
		prob = [0] * self.node_num

		for cellctr, cell in enumerate(self.tree):
			# [1, c, h, w] with [1, c, h, w] => [1, 2c, h, w]
			rin = torch.cat([x, rout], dim = 1)
			# rout: features, [1, c, h, w], logits: [1, node_num]
			rout, logits = cell(rin)
			prob = F.softmax(logits, dim = 1)
			_, cellidx = torch.max(prob, dim = 1)
			cellidx = cellidx.data[0] # variable => scalar

			if cellidx % self.node_num != (self.node_num - 1):
				# not pointer to the last, we can return already though its label maybe -1 as well.
				return [self.table[cellctr * self.node_num + cellidx]], prob

			# else continue to next cell

		# if it iterates to the end unknown node, it's unknown actually.
		return [-1], prob


	def forward(self, input, label, optimizer):
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
		# randomly print some inter-mediate result.
		if random.randint(1, 4000) < 2:
			need_print = True
		else:
			need_print = False

		# [1, c_, h_, w_] => pre_sz: [1, c, h, w]
		x = self.pre(input)
		# for the first cell, set rout = x
		rout = x

		## 0. check whether new label existed in label.
		# NOTICE: for the new label, if its label is -1, that means it should be treated as unknown and will not be added into current table.
		label = label[0]
		if label != -1 and label not in self.table: # label is known and not added into current table yet, add it
			if self.table_p != 0 and self.table_p % self.node_num == (self.node_num - 1):
				# when current table_p point to last unknown node, skip the last unknown node and create a cell to place new label
				self.table_p += 1 # skip the last unknown node
				mod = ResCell(self.pre_sz, self.node_num).cuda()
				self.add_module('node:%d'%len(self.tree), mod) # add to self module
				optimizer.add_param_group({'params': mod.parameters()}) # update optimizer's parameters
				self.tree.append(mod) # append new cell
				self.table.extend([-1] * self.node_num) # initialize new cell's table, and will write new label following

				# print('ADD new cell, current network:', self)
			# now write down label info
			self.table[self.table_p] = label
			self.table_p += 1

		# loss chain, we have several losses to return since we need to optimize the tree
		losses = []
		## now forward recursively
		# the forward process will terminate early if the appropriate condition triggered.
		for cellctr, cell in enumerate(self.tree):
			# [1, c, h, w] cat with [1, c, h, w]
			rin = torch.cat([x, rout], dim = 1)
			# print('cell:', cellctr, 'rin:',rin.size(), 'rout:', rout.size())
			# logits: [1, node_num], rout: [1, c, h, w]
			rout, logits = cell(rin)
			# prob: [1, 6]
			prob = F.softmax(logits, dim= 1)
			# get pred
			_, cellidx = torch.max(prob, dim = 1)
			cellidx = cellidx.data[0] # [1] => scalar
			# get its real label info
			pred = self.table[cellctr * self.node_num + cellidx]

			if need_print:
				print('  prob:', prob.cpu().data[0].numpy(), 'label idx:',cellidx)



			# 1. check pred in known node or unknown node
			# for unknown pred, if label is unknown, then loss = 0, if label is known, it should go deeper.
			if pred == -1: # 1. if pred is unknown, the pred can occur not only in last node, but also in other -1 position, we treat it as equal
				if label == -1: # 1.1 pred is unknow and groundtruth is unknown
					if cellctr == len(self.tree) - 1: # this is the last cell
						# TODO: decide multiple -1
						loss = self.criteon(logits, Variable(torch.LongTensor([self.node_num - 1])).cuda()) # as the last cell may have several -1 node, we can NOT apply crossentropy loss here.
						losses.append(loss)
						if need_print: print('\tpred: -1, label: -1, last cell, loss:', loss.data[0])
						return losses, prob
					else: # this is not the last cell
						# we calcuate loss by ourself.
						# TODO: decide multiple -1
						loss = self.criteon(logits, Variable(torch.LongTensor([self.node_num - 1])).cuda()) # cellidx == node_num - 1
						if need_print: print('\tpred: -1, label: -1, mid loss: %f go NEXT'%loss.data[0])
						losses.append(loss)
						continue
				# check the label whether in current cell table, then decide whether to go ahead
				elif label in self.table[cellctr * self.node_num : cellctr * self.node_num + self.node_num]:
					# 1.2 pred is unknown, label is known node in current cell
					# STOP going deeper, since its label resides in current layer
					# we should train current cell only
					labelidx = self.table.index(label)
					labelidx = labelidx % self.node_num # offset in current cell
					loss = self.criteon(logits, Variable(torch.LongTensor([labelidx])).cuda())
					losses.append(loss)
					if need_print: print('\tpred: -1, label: %d, in cur cell, loss:'%label, loss.data[0])
					return losses, prob
				else: # 1.3 pred is unknown, label is known but not in current layer, that's, in next layer.
					# in this case, it must have next cell.
					loss = self.criteon(logits, Variable(torch.LongTensor([self.node_num - 1])).cuda())
					losses.append(loss)
					if need_print: print('\tpred: -1, label: %d, loss: %f, in next cell, go NEXT'%(label,loss.data[0]))
					continue # just forward

			## 2.pred is known
			else:
				# 2.1 pred is known, label is unknown, it maybe have several unknown position .
				# just train curent cell
				if label == -1:
					loss = self.criteon(logits, Variable(torch.LongTensor([self.node_num - 1])).cuda())
					losses.append(loss)
					if need_print: print('\tpred: %d, label: -1, loss:'%pred, loss.data[0])
					return losses, prob
				# 2.2 pred is known and label is in current cell as well.
				# just use cross-entory to calculate its loss
				# its loss maybe 0 or not , depending on its label
				elif label in self.table[cellctr * self.node_num : cellctr * self.node_num + self.node_num]:
					# pred is known, groundtruth in current cell and groundtruth is known
					labelidx = self.table.index(label)
					labelidx = labelidx % self.node_num
					loss = self.criteon(logits, Variable(torch.LongTensor([labelidx])).cuda())
					losses.append(loss)
					if need_print: print('\tpred: %d, label: %d, in cur cell, loss:'%(pred, label), loss.data[0])
					return losses, prob
				# 2.3 pred is known, but label is not in current cell.
				else:
					# should train current cell
					# pred is known, groundtruth is NOT in current cell and groundtruth is known
					labelidx = self.table.index(label)
					labelidx = labelidx % self.node_num
					loss = self.criteon(logits, Variable(torch.LongTensor([self.node_num - 1])).cuda())
					losses.append(loss)
					if need_print: print('\tpred: %d, label: %d, NEXT cell, loss:'%(pred, label), loss.data[0])
					return losses, prob



if __name__ == '__main__':
	from MiniImgOnDemand import MiniImgOnDemand
	import random
	import numpy as np
	from tensorboardX import SummaryWriter
	import time,os

	db = MiniImgOnDemand('../mini-imagenet/', type='train')
	net = UnknownNet().cuda()
	tb = SummaryWriter('runs')

	optimizer = optim.Adam(net.parameters(), lr= 1e-4)

	if os.path.exists('unknown.mdl'):
		label_status_start = net.load_mdl('unknown.mdl', optimizer)
		print('learning from label:', label_status_start)
	else:
		label_status_start = 0

	model_parameters = filter(lambda p: p.requires_grad, net.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('total params:', params)

	# this is the total training dataset label, from [0, 64)
	num_cls = 64


	## 1. test previous model performance, exclusive, [0, label_status_start)
	for label_test in range(label_status_start):
		total = 100
		right = 0
		for i in range(total):
			# all choose current label
			# use db_prev as it's testing checkpoint performance
			img = Variable(db.get_test(label_test).unsqueeze(0)).cuda()
			pred, prob = net.predict(img)  # [1, 6]
			pred = pred[0]
			if pred == label_test:
				right += 1

		accuracy = right / total
		print('>> verify learning:', label_test,  'accuracy:', accuracy)


	## 2. training from label_status_start, exclusive to learn
	for label_status in range(label_status_start , num_cls):
		step = 0  # steps for single class
		accuracy = 0 # total accuracy of all learned labels
		label_status_accuracy = 0 # current learning label accuracy
		total_loss = 0
		start_time = time.time()

		# complete learning once total accuracy and current learning label accuracy satisfy some threshold
		while accuracy < 0.9 or label_status_accuracy < 0.9:
			# here have 0.9 prob to select learned data and 0.1 prob to select unknown ata.
			label = np.random.choice([label_status, num_cls-1], 1, p = [0.9, 0.1])[0]
			if label == label_status: # select training data, including previous learned data and current learning data.
				learn_current = random.randint(0, 1)
				# it has 0.5 prob to learning current learning label and 0.5 prob to randomly sample previous data.
				if learn_current == 0:
					label = random.randint(0, label_status)

				img = Variable(db.get(label).unsqueeze(0)).cuda()
				losses, prob = net(img, [label], optimizer)
			else: # select unknown data
				label = random.randint(label_status + 1, num_cls - 1)
				img = Variable(db.get(label).unsqueeze(0)).cuda()
				losses, prob = net(img, [-1], optimizer)

			# iterate loss from last to end
			optimizer.zero_grad()
			for i, loss in enumerate(reversed(losses)):
				if i == len(losses) -1 : # for the last backward, remove all tmp result
					loss.backward()
				else:
					loss.backward(retain_graph=True)
			optimizer.step()

			step += 1
			total_loss += losses[-1].data[0]

			if step % 100 == 0:
				tb.add_scalar('loss', losses[-1].data[0])

			if step % 1000 == 0:
				print('current progress:',label_status, 'step:', step, 'loss:', total_loss/1000)
				total_loss = 0

			if step % 2000 == 0:
				total = 500
				right = 0
				label_status_right = 0
				label_status_total = 0
				for i in range(total):
					label_test = np.random.choice([label_status, num_cls-1], 1, p = [0.9, 0.1])[0]
					if label_test == label_status: # select training data
						# sample label_status has prob of 0.5
						label_test = random.sample([0, label_status], 1)[0]
						# when prob 0.5 not sampled label_status, we random select a label from previous learned.
						if label_test != label_status:
							label_test = random.randint(0, label_status)
						img = Variable(db.get_test(label_test).unsqueeze(0)).cuda()
						pred, prob = net.predict(img) # [1, 6]
						pred = pred[0]
						if pred == label_test:
							right += 1

						# this is for statistics of current learning label
						if label_test == label_status: # sampled label is current learning label
							label_status_total += 1
							if pred == label_test: # and pred is right
								label_status_right += 1

					else: # select unknown data
						label_test = random.randint(label_status + 1, num_cls - 1)
						img = Variable(db.get_test(label_test).unsqueeze(0)).cuda()
						pred, prob = net.predict(img) # [1, 6]
						pred = pred[0]
						if pred == -1:
							right += 1

				accuracy = right / total
				label_status_accuracy = label_status_right / label_status_total
				print('>> current progress:', label_status, 'step:', step,
				      'accuracy:', accuracy, 'learning precision: %f'%label_status_accuracy, 'in', label_status_total)
				tb.add_scalar('accuracy', accuracy)


		print('**time for progress:', label_status, time.time() - start_time)
		net.save_mdl('unknown.mdl')

