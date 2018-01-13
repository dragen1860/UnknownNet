from UnknownNet import UnknownNet
from torch import optim
from torch.autograd import Variable
from MiniImgOnDemand import MiniImgOnDemand
import random
import numpy as np
import time,os


if __name__ == '__main__':
	# we will use test set to fine-tune the algorithm and then test its performance
	db = MiniImgOnDemand('../mini-imagenet/', type='test')
	db_prev = MiniImgOnDemand('../mini-imagenet/', type='train')
	net = UnknownNet().cuda()
	optimizer = optim.Adam(net.parameters(), lr= 1e-5)

	# checkpoint is a must
	if os.path.exists('unknown.mdl'):
		label_status_start = net.load_mdl('unknown.mdl', optimizer)
		print('learning from label:', label_status_start)
	else:
		raise FileNotFoundError

	# calculate the total network size
	model_parameters = filter(lambda p: p.requires_grad, net.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('total params:', params)


	# 1. test previous model performance, exclusive, [0, label_status_start)
	for label_test in range(label_status_start):
		total = 100
		right = 0
		for i in range(total):
			# all choose current label
			# use db_prev as it's testing checkpoint performance
			img = Variable(db_prev.get_test(label_test).unsqueeze(0)).cuda()
			pred, prob = net.predict(img)  # [1, 6]
			pred = pred[0]
			if pred == label_test:
				right += 1

		accuracy = right / total
		print('>> verify learning:', label_test,  'accuracy:', accuracy)

	# in test.csv, the label is from 0-19
	num_cls = 20
	# 2.from label_status_start, exclusive to learn
	# NOTICE: for the following stage, we only have several images to few-shot learning, we need to make a tradeoff,
	# to learn but not overfitting.
	# although we can learn all data, but we only learn n_way data for better accuracy.
	n_way = 5
	k_shot = 5
	for label_status in range(label_status_start , label_status_start + n_way):
		step = 0 # steps of training
		accuracy = 0 # validation accuracy
		total_loss = 0 # total_loss per print period
		label_status_accuracy = 0 # validation accracy for current learning class
		start_time = time.time() # time elapsed per print time

		while accuracy < 0.8 or label_status_accuracy < 0.8:
			explore = np.random.choice([label_status, label_status_start - 1], 1, p = [0.5, 0.5])
			if explore == label_status: # select current few-shot img, it has up-to k_shot number of imgs
				# we need to understand our test dataset also index from 0
				# index in network: label or label_status
				# index in db: label - label_status_start
				# if not explore, we just use current few-shot imgs and previous learned few-shot imgs
				label = random.randint(label_status_start, label_status)
				# we must use db.fewshot_get since it will return back fixed k_shot number of imgs.
				img = Variable(db.fewshot_get(label - label_status_start, k_shot).unsqueeze(0)).cuda()
				losses, prob = net(img, [label], optimizer)
			else: # use other data as unknown data, we don't use its label
				# use the following data as unknown data, it's not limited to k_shot, but all num_cls
				label = random.randint(label_status + 1, label_status_start + num_cls - 1)
				# here we can use db.get to access all data since its label is unknown
				img = Variable(db.get(label - label_status_start).unsqueeze(0)).cuda()
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
			if step % 30 == 0:
				print('learning:',label_status, 'step:', step, 'loss:', total_loss/30)
				total_loss = 0

			# validation now
			# as we only have k_shot imgs and no other data to help us decide whether continue to learn to terminate.
			# here as a stage of validation
			if step % 50 == 0:
				total = 100
				right = 0
				label_status_total = 0
				label_status_right = 0

				for i in range(total):
					explore = np.random.choice([0, 1], 1, p = [0.5, 0.5])
					if explore == 0: # select training data, no exploration
						# from current learning label random select one
						# NOTICE: must use fewshot_get HERE
						label_test = random.randint(label_status_start, label_status)
						img = Variable(db.fewshot_get(label_test - label_status_start, k_shot).unsqueeze(0)).cuda()
						pred, prob = net.predict(img) # [1, 6]
						pred = pred[0]
						if pred == label_test:
							right += 1

						# this is for statistics of current learning label
						if label_test == label_status:  # sampled label is current learning label
							label_status_total += 1
							if pred == label_test:  # and pred is right
								label_status_right += 1
					else: # select unknown data, the unknown data means not learned data
						# NOTICE: can use db.get here since its label is unknown
						label_test = random.randint(label_status + 1, label_status_start + num_cls - 1)
						img = Variable(db.get(label_test - label_status_start).unsqueeze(0)).cuda()
						pred, prob = net.predict(img) # [1, 6]
						pred = pred[0]
						if pred == -1:
							right += 1

				accuracy = right / total
				label_status_accuracy = label_status_right / label_status_total
				print('>> learning:', label_status, 'step:', step, 'accuracy:', accuracy, \
				      'learning accuracy:',label_status_accuracy,label_status_total)



	print('End of fine-tuning.')

	## 3. test stage
	print('='*20, 'test now', '='*20)
	accuracy = 0
	total = 40
	right = 0
	for i in range(total):
		# we random select total number of imgs.
		# randomly select img from [64, 64+5-1]
		label_test = random.randint(label_status_start, label_status_start + k_shot -1)
		# convert index of network to index of db
		# must use fewshot_get_test to access un-polluted data.
		img = Variable(db.fewshot_get_test(label_test - label_status_start, k_shot).unsqueeze(0)).cuda()
		pred, prob = net.predict(img)  # [1, 6]
		pred = pred[0]
		print(i, 'pred:', pred, 'label:', label_test)
		if pred == label_test:
			right += 1

	accuracy = right / total
	print('way:',n_way, 'shot:', k_shot, 'accuracy:', accuracy, 'num:',total)


