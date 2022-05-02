import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tigonshuju import dataxxx
from torch.utils.data import DataLoader
class Test:
	def __init__(self, train_x, train_y, test_x, test_y):
		self.train_x, self.train_y, self.test_x, self.test_y = train_x, train_y, test_x, test_y
		self.criterion = torch.nn.CrossEntropyLoss()
	#导入训练数据，确定损失函数	
	def single_model(self, path):
		model = torch.load(path)
		#导入模型
		model.eval()
		#使用eval时候框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值
		with torch.no_grad():
			#停止梯度计算
			train_loss, train_acc = 0, 0

			for batch_x, target in zip(self.train_x, self.train_y):
				#zip 将数据打包成元组
				output = model(batch_x)
				loss = self.criterion(output, target)
				output = torch.argmax(output, dim=1)
				acc = torch.sum(output == target)/target.shape[0]
				#测试准确率
				loss = loss.item()
				acc = acc.item()
				train_loss += loss
				train_acc += acc

			train_loss, train_acc = train_loss/self.train_x.shape[0], train_acc/self.train_x.shape[0]

			test_loss, test_acc = 0, 0

			for batch_x, target in zip(self.test_x, self.test_y):
				output = model(batch_x)
				loss = self.criterion(output, target)
				output = torch.argmax(output, dim=1)
				acc = torch.sum(output == target)/target.shape[0]
				loss = loss.item()
				acc = acc.item()
				test_loss += loss
				test_acc += acc

		test_loss, test_acc = test_loss/self.test_x.shape[0], test_acc/self.test_x.shape[0]

		return [train_loss, train_acc, test_loss, test_acc]
		#返回测试集和训练集的准确率和损失数值，（一个模型的测试集和损失数值）
	def analyse_type(self, filename):
		savemodel='results/performance_metrics'
		if os.path.exists(savemodel):
			print("path exit")
			pass
		else:
			os.mkdir(savemodel)
			print("path not exit")
		model_names = os.listdir(filename)
		#读取文件夹下面的文件，并且将文件名保存再一个list中
		print("there is {} models in filename {}".format(len(model_names),filename))
		#os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表，数目和本地训练的次数有关
		models = np.array([filename+i for i in model_names])
		#模型的相对路径
		model_names_index = np.argsort(np.array([int(i[len('model_epoch_'):-3]) for i in model_names]))
		#对模型的下标的位置，下标i代表第i轮聚合产生的模型
		models = models[model_names_index]
		#让models中的模型按照列表的指数进行排序
		performance = []
		round=0
		for model in models:
			print(" *** round is {}".format(round))
			round=round+1
			performance.append(self.single_model(model))
			#返回该model的训练loss，测试loss，训练acc，测试acc 每一个模型只返回一个数据
			#但是，一个模型只是聚合一次的模型而一般聚合10次
		print("************{}*************".format(filename))
		print(np.array(performance).shape)
		performance = pd.DataFrame(np.array(performance))
		performance.columns = ['train_loss', 'train_acc', 'test_loss', 'test_acc']
		
		performance.to_csv('results/performance_metrics/'+filename[len('models/'):-1]+'.csv', index=False)
		performance = performance.values
		best_index = np.argmin(performance.T[2])
		#给出测试损失最小值的下标
		return performance[best_index]
		#返回测试损失最小值下标的一路数据[训练loss，测试loss，训练acc，测试acc]

	def analyse_all(self):
		bar = tqdm(total=3*3*3)
		#用于手动控制进度条
		all_best_performances = []
		with bar:
			for chanshu in [[20,7,1.0],[20,7,0.8],[20,7,0.6],[20,6,1.0],[20,5,1.0],[17,7,1.0],[14,7,1.0]]:
				local_epochs=chanshu[0]
				r=chanshu[2]
				precision=chanshu[1]
				#filename = "./results/models/saved_models_local_epochs_"+str(local_epochs)+"_r_"+str(r).replace('.', '_')+"_precision_"+str(precision)+"/"
				filename = "models/saved_models_local_epochs_"+str(local_epochs)+"_r_"+str(r).replace('.', '_')+"_precision_"+str(precision)+"/"
				best = self.analyse_type(filename)
				#best是该条件下的最好的表现
				#并且analyse_type保存了每一个模型的数据情况[train_loss,train_acc,test_loss,test_acc]
				best = [local_epochs, r, precision] + [i for i in best]
				all_best_performances.append(best)
				bar.update(1)
		all_best_performances = pd.DataFrame(np.array(all_best_performances))
		all_best_performances.columns = ['local_epochs', 'r', 'precision', 'train_loss', 'train_acc', 'test_loss', 'test_acc']
		all_best_performances.to_csv('results/best_performances.csv', index=False)

	def image_beautifier(self):

		image_names = sorted(['./results/'+i for i in os.listdir('./results/') if '.png' in i])
		for names in [image_names[i:i+4] for i in range(0, 12, 4)]:
			images = [Image.open(x) for x in names]
			widths, heights = zip(*(i.size for i in images))

			total_width = sum(widths)
			max_height = max(heights)

			new_im = Image.new('RGB', (total_width, max_height))

			x_offset = 0
			for im in images:
				new_im.paste(im, (x_offset,0))
				x_offset += im.size[0]

			name = names[0][len('./results/'):names[0].index('__')]
			new_im.save(name+'_variations.png')

		### Resizing for actual use

		for image in [i for i in os.listdir() if '_variations.png' in i]:
			img = cv2.resize(cv2.imread(image), (1280, 240))
			cv2.imwrite(image, img)

	def image(self, performances, names, pic_name):
		for i,name in enumerate(['test1_loss', 'test1_acc', 'test2_loss', 'test2_acc']):
			plt.cla() #用于清除当前轴
			for j,performance in enumerate(performances):
				print(len(performance.T[i]))
				plt.plot(np.arange(20), performance.T[i], label=names[j])
			plt.legend()
			if i%2==1:
				plt.ylim([-0.01, 1.01])
			plt.title(name+' - '+pic_name)
			plt.savefig('./results/'+pic_name+'__'+name+'_analysis.png')

	def image_generator(self):
		performance = pd.read_csv('./results/best_performances.csv')
		features = performance.columns
		performance = performance.values
		best_index = np.argmin(performance.T[-2])
		#testloss 的最小值指数
		print("Best Performance By: ", {i:j for i,j in zip(features, performance[best_index])})
		#

		local_epochs, r, precision =20,1.0,7
		#3 1 7
		print("local_epochs is {}".format(local_epochs))
		print("r is {}".format(r))
		print("precision is {}".format(precision))
		#表现最好的模型的3个参数
		### Local Epochs
		print("Analysing local_epochs with r and precision fixed to", r, precision, "respectively.")
		performances = []
		for local_epoch in [14, 17, 20]:
			filename = "./results/performance_metrics/saved_models_local_epochs_"+str(local_epoch)+"_r_"+str(r).replace('.', '_')+"_precision_"+str(precision)+".csv"
			performance = pd.read_csv(filename)
			performance = performance.values
			
			performances.append(performance)
			
		performances = np.array(performances)
		
		self.image(performances, names=['local_epochs='+str(i) for i in [14, 17, 20]], pic_name='local_epochs')

		### r
		print("Analysing local_epochs with local_epochs and precision fixed to", local_epochs, precision, "respectively.")
		performances = []
		for r_id in [0.6,  0.8, 1.0]:
			filename = "./results/performance_metrics/saved_models_local_epochs_"+str(local_epochs)+"_r_"+str(r_id).replace('.', '_')+"_precision_"+str(precision)+".csv"
			performance = pd.read_csv(filename)
			performance = performance.values
			performances.append(performance)
		performances = np.array(performances)
		self.image(performances, names=['r='+str(i) for i in [0.6,  0.8, 1.0]], pic_name='r')

		### r
		print("Analysing local_epochs with local_epochs and r fixed to", local_epochs, r, "respectively.")
		performances = []
		for p in [5, 6, 7]:
			filename = "./results/performance_metrics/saved_models_local_epochs_"+str(local_epochs)+"_r_"+str(r).replace('.', '_')+"_precision_"+str(p)+".csv"
			performance = pd.read_csv(filename)
			performance = performance.values
			performances.append(performance)
		performances = np.array(performances)
		self.image(performances, names=['precision='+str(i) for i in [5, 6, 7]], pic_name='precision')

		self.image_beautifier()

if __name__ == '__main__':
	train_x, train_y, test_x, test_y = dataxxx()
	batch_size=10
	train_x = np.array([train_x[n:n+batch_size] for n in range(0, len(train_x)-batch_size, batch_size)])
	train_y = np.array([train_y[n:n+batch_size] for n in range(0, len(train_y)-batch_size, batch_size)])
	test_x = np.array([test_x[n:n+batch_size] for n in range(0, len(test_x)-batch_size, batch_size)])
	test_y = np.array([test_y[n:n+batch_size] for n in range(0, len(test_y)-batch_size, batch_size)])
	test = Test(train_x, train_y, test_x, test_y)
	#test.analyse_all()
	test.image_generator()