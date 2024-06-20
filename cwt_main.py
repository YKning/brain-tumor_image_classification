import argparse, json
import datetime
import os
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch, random


from server import *
from client import *
import models, datasets
from torchvision.datasets import ImageFolder
 
import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from log import get_log

from torch import randperm


import os 



logger = get_log('/home/ykn/cds/chapter03_Python_image_classification/log/log.txt')
#logger.info("MSE: %.6f" % (mse))
#logger.info("RMSE: %.6f" % (rmse))
#logger.info("MAE: %.6f" % (mae))
#logger.info("MAPE: %.6f" % (mape))


transforms = transforms.Compose([
    transforms.Resize(256),    # 将图片短边缩放至256，长宽比保持不变：
    transforms.CenterCrop(224),   #将图片从中心切剪成3*224*224大小的图片
    transforms.ToTensor()          #把图片进行归一化，并把数据转换成Tensor类型
]) 



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('--conf', default = '/home/ykn/cds/chapter03_Python_image_classification/utils/conf.json', dest='conf')
	args = parser.parse_args()
	

	with open(args.conf, 'r') as f:
		conf = json.load(f)	
	
	path1 = '/home/ykn/cds/chapter03_Python_image_classification/data/Brain Tumor MRI Dataset/archive/Training'
	path2 = '/home/ykn/cds/chapter03_Python_image_classification/data/Brain Tumor MRI Dataset/archive/Testing'
	data_train = datasets.ImageFolder(path1, transform=transforms)
	data_test = datasets.ImageFolder(path2, transform=transforms)
	print(data_train)
	# data_loader = DataLoader(data_train, batch_size=64, shuffle=True)
 
	# for i, data in enumerate(data_loader):
	# 	images, labels = data
	# img = torchvision.utils.make_grid(images).numpy()
	# plt.imshow(np.transpose(img, (1, 2, 0)))
	# #plt.show()	

	# train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
	# data_train_shuffle = DataLoader(data_train, batch_size=64, shuffle=True)
	# data_test_shuffle = DataLoader(data_test, batch_size=64, shuffle=True)
	# print(data_train_shuffle)

	lenth_train = randperm(len(data_train)).tolist() # 生成乱序的索引
	data_train_shuffle = torch.utils.data.Subset(data_train, lenth_train)
	lenth_test = randperm(len(data_test)).tolist() # 生成乱序的索引
	data_test_shuffle = torch.utils.data.Subset(data_test, lenth_test)


	train_datasets, eval_datasets = data_train_shuffle, data_test_shuffle
	server = Server(conf, eval_datasets)
	clients = []
	
	for c in range(conf["no_models"]):
		clients.append(Client(conf, server.global_model, train_datasets, c))
		
	print("\n\n")
	for e in range(conf["global_epochs"]):
		random.shuffle(clients)
		for client in clients[:conf['k']]:
			print(client.client_id)		
			weight_accumulator = {}
				
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name] = torch.zeros_like(params)
				

			diff = client.local_train(server.global_model)
					
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name])
					
				
			server.model_aggregate(weight_accumulator)
				
			acc, loss = server.model_eval()
				
			print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

	
				
			
		
		
	
		
		
	