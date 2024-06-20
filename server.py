
import models, torch


class Server(object):
	
	def __init__(self, conf, eval_dataset):
	
		self.conf = conf 
		
		self.global_model = models.get_model(self.conf["model_name"]) 
		
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
		
	
	def model_aggregate(self, weight_accumulator):
		for name, data in self.global_model.state_dict().items():
			
			update_per_layer = weight_accumulator[name] * self.conf["lambda"]
			
			if data.type() != update_per_layer.type():
				data.add_(update_per_layer.to(torch.int64))
			else:
				data.add_(update_per_layer)
				
	# def model_eval(self):
	# 	self.global_model.eval()
		
	# 	total_loss = 0.0
	# 	correct = 0
	# 	dataset_size = 0
	# 	for batch_id, batch in enumerate(self.eval_loader):
	# 		data, target = batch 
	# 		dataset_size += data.size()[0]
			
	# 		if torch.cuda.is_available():
	# 			data = data.cuda()
	# 			target = target.cuda()
				
			
	# 		output = self.global_model(data)
	# 		# print(output)
	# 		print("Targets: ",target)
			
	# 		total_loss += torch.nn.functional.cross_entropy(output, target,
	# 										  reduction='sum').item() # sum up batch loss
	# 		pred = output.data.max(1)[1]  # get the index of the max log-probability
	# 		print("pred: ",pred)

	# 		correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

	# 	acc = 100.0 * (float(correct) / float(dataset_size))
	# 	total_l = total_loss / dataset_size

	# 	torch.save(self.global_model.state_dict(), "./data/model_parameter.h5")

	# 	return acc, total_l
				
	def model_eval(self):
		self.global_model.eval()
		
		total_loss = 0.0
		correct = 0
		dataset_size = 0
		confusion_matrix = torch.zeros(4, 4)  # 初始化混淆矩阵

		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch 
			dataset_size += data.size()[0]
			
			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()
			
			output = self.global_model(data)

			total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
			pred = output.data.max(1)[1]  # get the index of the max log-probability

			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

			# 更新混淆矩阵
			for t, p in zip(target.view(-1), pred.view(-1)):
				confusion_matrix[t.long(), p.long()] += 1

		acc = 100.0 * (float(correct) / float(dataset_size))
		total_l = total_loss / dataset_size

		# 计算精确度、召回率和F1分数
		precision = torch.diag(confusion_matrix) / (confusion_matrix.sum(0) + 1e-9)
		average_precision = torch.mean(precision)
		recall = torch.diag(confusion_matrix) / (confusion_matrix.sum(1) + 1e-9)
		average_recall = torch.mean(recall)
		f1_score = 2 * precision * recall / (precision + recall + 1e-9)
		average_f1_score = torch.mean(f1_score)

		# 打印平均精确度、召回率和F1分数
		#print("Average Precision: {:.2f}%, Average Recall: {:.2f}%, Average F1-Score: {:.2f}%".format(average_precision.item() * 100, average_recall.item() * 100, average_f1_score.item() * 100))

		torch.save(self.global_model.state_dict(), "./data/model_parameter.h5")

		return acc, total_l, average_precision, average_recall, average_f1_score  # 返回评估指标