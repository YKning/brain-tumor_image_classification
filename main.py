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



logger = get_log('/home/cds/brain-tumor_image_classification/log/logkashi.txt')
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
	parser.add_argument('--conf', default = '/home/cds/brain-tumor_image_classification/utils/conf.json', dest='conf')
	args = parser.parse_args()
	

	with open(args.conf, 'r') as f:
		conf = json.load(f)	
	
	path1 = '/home/cds/brain-tumor_image_classification/data/Brain Tumor MRI Dataset/archive/Training'
	path2 = '/home/cds/brain-tumor_image_classification/data/Brain Tumor MRI Dataset/archive/Testing'
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
	#
	# class Client:
	# 	def __init__(self, conf, global_model, train_datasets, client_id):
	# 		self.conf = conf
	# 		self.global_model = global_model
	# 		self.train_datasets = train_datasets
	# 		self.client_id = client_id

    #     # 计算并打印数据分布
	# 		self.print_data_distribution()

	# 	def print_data_distribution(self):
    #     # 创建一个字典来存储每个类别的数量
	# 		class_counts = {}

    #     # 遍历数据集
	# 		for _, label in self.train_datasets:
    #         # 获取类别的名称
	# 			class_name = self.train_datasets.dataset.classes[label]

    #         # 如果这个类别还没有被计数过，就添加到字典中
	# 			if class_name not in class_counts:
	# 				class_counts[class_name] = 0

    #         # 增加这个类别的计数
	# 				class_counts[class_name] += 1

    #     # 打印结果
	# 		print(f"Client {self.client_id} data distribution:")
	# 		for class_name, count in class_counts.items():
	# 			print(f"Class {class_name}: {count} samples")

 
	
	for c in range(conf["no_models"]):
		clients.append(Client(conf, server.global_model, train_datasets, c))
		test_client = Client(conf, server.global_model, train_datasets, c)
		class_num_total = np.array([0,0,0,0])
		for batch_id, batch in enumerate(test_client.train_loader):
			data, target = batch
			unique_elements, counts = np.unique(target, return_counts=True)
			# 打印每个元素及其计数
			for element, count in zip(unique_elements, counts): 
				class_num_total[element] += count
		print(class_num_total)
   
		# for element, count in zip(unique_elements, counts): 
		# 	print("Client:", c, "Class:", test_client.train_loader.dataset.classes[element], "Count:", count)
		
		#client = Client(conf, server.global_model, train_datasets, c)
    
		#clients.append(client)
	print("\n\n")


	for e in range(conf["global_epochs"]):
	
		candidates = random.sample(clients, conf["k"])
		
		weight_accumulator = {}
	
		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)
		
		for c in candidates:
			diff = c.local_train(server.global_model)
			
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name])
				
		
		server.model_aggregate(weight_accumulator)
		
		# acc, loss = server.model_eval()
		
		# print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
		acc, loss, average_precision, average_recall, average_f1_score = server.model_eval()
		print("Epoch {}, Accuracy: {:.2f}%, Loss: {:.2f}, Average Precision: {:.2f}%, Average Recall: {:.2f}%, Average F1-Score: {:.2f}%".format(
			e, acc, loss, average_precision * 100, average_recall * 100, average_f1_score * 100))
			
						
