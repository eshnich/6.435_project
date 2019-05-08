import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from IPython import embed
from load_sprites import sprites_act
import time
import math
import matplotlib.pyplot as plt

if torch.cuda.is_available():
        print("CUDA IS AVAILABLE")
        device = torch.device("cuda:0")
        print("DEVICE:", device)
else:
        print("CUDE IS NOT AVAILABLE")
        device = torch.device("cpu")
        print(device)



class CNN(nn.Module):
	def __init__(self, image_channels, x_height, x_width, no_of_frames):
		super(CNN, self).__init__()

		self.image_channels = image_channels # channels - 3 for RBG images
		self.x_height = x_height # height of input image
		self.x_width = x_width # width of the input image
		self.frames = no_of_frames # number of frames

		self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=(4, 4), stride=2)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2)
		self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=2)	
		self.hidden = 512
		self.fc1 = nn.Linear(in_features=256*2*2*no_of_frames, out_features=self.hidden)
		self.fc2 = nn.Linear(in_features = self.hidden, out_features = 6)

	def encode(self, x):
		h = x
		h = h.view(-1, self.image_channels, self.x_height, self.x_width)
		h = F.elu(self.conv1(h))
		h = F.elu(self.conv2(h))
		h = F.elu(self.conv3(h))
		h = F.elu(self.conv4(h))
		h = h.view(-1, 256*2*2)
		return h

	def forward(self, x):
		h = torch.cat([self.encode(frame) for frame in x], 1)
		h = F.elu(self.fc1(h))
		h = self.fc2(h)
		return h

def get_accuracy(data, labels, net):
	pred = [net(data[i].view(1,8,64,64,3).transpose(0,1)).max(1)[1].item() == labels[i].item() for i in range(len(data))]
	return sum(pred)/float(len(data))

def train(input_data, labels, net):
	start = time.time()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=1e-4)

	losses = []
	data = input_data.transpose(0,1)
	current = time.time()
	for epoch in range(1000):
		total_loss = 0

		print(epoch)

		optimizer.zero_grad()
		outputs = net(data)
		loss = criterion(outputs, labels)
	
		loss.backward()
		optimizer.step()

		total_loss += loss.data.item()
		print(total_loss/len(data))
		losses.append(total_loss/len(data))
		
		if (epoch + 1)%10 == 0:
			print("test accuracy = ", get_accuracy(input_data, labels, net))
		#	torch.save(net.state_dict(), 'SavedModels/train_epoch_%d.pt' % epoch)

	return losses

X_train, X_test, A_train, A_test, D_train, D_test = sprites_act('', return_labels=True)
X_train = torch.from_numpy(X_train)
X_train = X_train.transpose(3,4).transpose(2,3)
num_samples = 100
data = torch.from_numpy(X_train[:num_samples]).cuda()
print(data.size())
attr = 0
labels = [[i for i in range(6) if A_train[j][0][attr][i] ==1][0] for j in range(len(A_train))]
labels = torch.LongTensor(labels[:num_samples]).cuda()
net = CNN(3, 64, 64, 8)
net.to(device)
print("pre-train accuracy = ", get_accuracy(data, labels, net))
train(data, labels, net)
print("final test accuracy = ", get_accuracy(data, labels, net))
torch.save(net.state_dict(), 'SavedModels/classifier_attr_%d.pt' % attr)
