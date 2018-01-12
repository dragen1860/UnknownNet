from UnknownNet import UnknownNet
import torch
from torch import nn
from torchvision.models import resnet
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
import pickle
from MiniImgOnDemand import MiniImgOnDemand
import random
import numpy as np
from tensorboardX import SummaryWriter
import time,os


if __name__ == '__main__':
	# we will use test set to fine-tune the algorithm and then test its performance
	db = MiniImgOnDemand('../mini-imagenet/', type='test')
	net = UnknownNet().cuda()
	tb = SummaryWriter('runs')
	optimizer = optim.Adam(net.parameters(), lr= 1e-5)

	if os.path.exists('unknown.mdl'):
		label_status_start = net.load_mdl('unknown.mdl', optimizer)
		print('learning from label:', label_status_start)
	else:
		raise FileNotFoundError

	model_parameters = filter(lambda p: p.requires_grad, net.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('total params:', params)

	num_cls = 64
	# 1. test previous model performance
	for label_status in range(label_status_start + 1):
		total = 500
		right = 0
		for i in range(total):
			label_test = np.random.choice([label_status, num_cls - 1], 1, p=[0.5, 0.5])
			if label_test == label_status:  # select training data
				label_test = random.randint(0, label_status)
				img = Variable(db.get_test(label_test).unsqueeze(0)).cuda()
				pred, prob = net.predict(img)  # [1, 6]
				pred = pred[0]
				if pred == label_test:
					right += 1
			else:  # select unknown data
				label_test = random.randint(label_status + 1, num_cls - 1)
				img = Variable(db.get_test(label_test).unsqueeze(0)).cuda()
				pred, prob = net.predict(img)  # [1, 6]
				pred = pred[0]
				if pred == -1:
					right += 1

		accuracy = right / total
		print('>> verify learning:', label_status,  'accuracy:', accuracy)


	# from label_status_start, exclusive to learn
	for label_status in range(label_status_start + 1, num_cls):
		step = 0
		accuracy = 0
		total_loss = 0
		start_time = time.time()

		while accuracy < 0.9:
			label = np.random.choice([label_status, num_cls-1], 1, p = [0.5, 0.5])
			if label == label_status: # select training data
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
				for i in range(total):
					label_test = np.random.choice([label_status, num_cls-1], 1, p = [0.5, 0.5])
					if label_test == label_status: # select training data
						label_test = random.randint(0, label_status)
						img = Variable(db.get_test(label_test).unsqueeze(0)).cuda()
						pred, prob = net.predict(img) # [1, 6]
						pred = pred[0]
						if pred == label_test:
							right += 1
					else: # select unknown data
						label_test = random.randint(label_status + 1, num_cls - 1)
						img = Variable(db.get_test(label_test).unsqueeze(0)).cuda()
						pred, prob = net.predict(img) # [1, 6]
						pred = pred[0]
						if pred == -1:
							right += 1

				accuracy = right / total
				print('>> current progress:', label_status, 'step:', step, 'accuracy:', accuracy)
				tb.add_scalar('accuracy', accuracy)


		print('**time for progress:', label_status, time.time() - start_time)
		net.save_mdl('unknown.mdl')
